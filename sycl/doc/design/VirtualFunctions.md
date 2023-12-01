# Implementation design for sycl_ext_oneapi_virtual_functions

Corresponding language extension specification:
[sycl_ext_oneapi_virtual_functions][1]

## Overview

Main complexity of the feature comes from its co-existence with optional kernel
features ([SYCL 2020 spec][sycl-spec-optional-kernel-features],
[implementaiton design][optional-kernel-features-design]) mechanism. Consider
the following example:

```c++
using syclext = sycl::ext::oneapi::experimental;

struct set_fp64;

struct Base {
  virtual SYCL_EXT_ONEAPI_INDIRECTLY_CALLABLE_PROPERTY() void foo() {}
  virtual SYCL_EXT_ONEAPI_INDIRECTLY_CALLABLE_PROPERTY(set_fp64) void bar() {
    // this virtual function uses double
    double d = 3.14;
  }
};

class Constructor;
class Use;
class UseFP64;

int main() {
  // Selected device may not support 'fp64' aspect
  sycl::queue Q;

  Base *Obj = sycl::malloc_device<Base>(1, Q);
  int *Result = sycl::malloc_shared<int>(2, Q);

  Q.single_task<Constructor>([=] {
    // Only placement new can be used within device functions.
    // When an object of a polymorphic class is created, its vtable is filled
    // with pointer to virtual member functions. However, we don't always know
    // featuures supported by a target device (in case of JIT) and therefore
    // can't decide whether both 'foo' and 'bar' should be both included in the
    // resulting device image - the decision must be made at runtime when we
    // know the target device.
    new (Obj) Derived;
  });

  // The same binary produced by thy sycl compiler should correctly work on both
  // devices with and without support for 'fp64' aspect.
  Q.single_task<Use>(syclext::properties{syclext::calls_indirectly<>}, [=] {
    Obj->foo();
  });

  if (Q.get_device().has(sycl::aspect::fp64)) {
    Q.single_task<Use>(syclext::properties{syclext::calls_indirectly<set_fp64>},
        [=] {
      Obj->bar();
    });
  }

  return 0;
}
```

As comments in the snippet say the main issue is with vtables: at compile time
it may not be clear which exact functions can be safely included in there and
which are not in order to avoid speculative compilation and fulfill optional
kernel features requirements from the SYCL 2020 specificaiton.

To solve this, the following approach is used: all virtual functions marked with
`indirectly_callable` property are grouped by set they belong to and outlined
into separate device images (i.e. device images with kernels using them are left
with declarations only of those virtual functions).

For each device image with virtual functinos that use optional features we also
create a "dummy" version of it where bodies of all virtual functions are
emptied.

Dependencies between deivce images are recorded in properties based on
`calls_indirectly` and `indirectly_callable` properties. They are used later by
runtime to link them together. Device images which depend on optional kernel
features are linked only if those features are supported by a target device and
dummyy versions of those device images are used otherwise.

This way we can emit single unified version of LLVM IR where vtables reference
all device virtual functions, but their definitions are outlined and linked
back dynamically based on device capabilities.

For AOT flow, we don't do outlining and dynamic linking, but instead do direct
cleanup of virtual functions which are incompatible with a target device.

## Design

### Changes to the SYCL header files

New compile-time properties `indirectly_callable` and `calls_indirectly` should
be implemented in accordance with the corresponding [design document][2]:

- `indirectly_callable` property should lead to emission of
  `"indirectly-callable"="set"` function attribute, where "set" is a string
  representation of the property template parameter.
- `calls_indirectly` property should lead to emission of
  `"calls-indirectly"="set1,set2"`, where "set1" and "set2" are string
  representations of the property template parameters.

In order to convert a type to a string, [\__builtin_sycl_unique_stable_name][3]
could be used.

**TODO**: `calls_indirectly` requires compile-time concatenation of strings.
Document how it should be done.

`indirectly_callable` property is applied to functions using "custom" (comparing
to other properties) `SYCL_EXT_ONEAPI_INDIRECTLY_CALLABLE_PROPERTY` macro. This
is done to allow implementations to attach some extra attributes alongside the
property. In particular, functions marked with the macro should be considered
SYCL device functions and compiler should emit diagnostics if those functions
do not conform with the SYCL 2020 specification. To achieve that and avoid
extending FE to parse strings within properties, the aforementioned macro should
also set `sycl_device` attribute:

```
#define SYCL_EXT_ONEAPI_INDIRECTLY_CALLABLE_PROPERTY(SetId)                    \
  __attribute__((sycl_device)) [[__sycl_detail__::add_ir_attribute_function(   \
      "indirectly-callable", __builtin_sycl_unique_stable_name(SetId))]]
```

### Changes to the compiler front-end

Most of the handling for virtual functions happens in middle-end and thanks to
compile-time properties, no extra work is required to propagate necessary
information down to passes from headers.

However, we do need to filter out those virtual functions which are not
considered to be device  as defined by the [extension specifiction][1], such as:

- virtual member functions annotated with `indirectly_callable` compile-time
  property should be emitted into device code;
- virtual member function *not* annotated with `indirectly_callable`
  compile-time property should *not* be emitted into device code;

There is no need to actually check which exact property is applied to a
function, it is enough to check if `add_ir_attribute_function` attribute was
applied and that we are in SYCL device mode to decide whether or not a virtual
function should be emitted into vtable and device code.

**TODO:** any extra diagnostics we would like to emit? Like kernel without
`calls_indirectly` property performing virtual function call.

### Changes to the compiler middle-end

#### Aspects propagation

Aspects propagation pass should be extended to not only gather aspects which are
used directly, but also aspects that are used indirectly, through virtual
functions.

For that the pass should compile a list of aspects used by each set of
indirectly callable functions (as defined by `indirectly_callable` property set
by user) and then append those aspects to every kernel which use those sets (as
defiend by `calls_indirectly` property set by user).

NOTE: if the aspects propagation pass is ever extended to track function
pointers, then aspects attached to virtual functions **should not** be attached
to kernels using this mechanism. For example, if a kernel uses a variable,
which is initialized with a function pointer to a virtual function which uses
an aspect, then such kernel **should not** be considered as using that aspect.
Properties-based mechanism which is described above should be used for aspects
propagation for virtual functions.

#### New compiler diagnostics

**TBD**

#### Device code split and device images

The extension specification restricts implementation from raising a diagnostic
when a kernel not marked with `calls_indirectly` kernel property creates an
object of a polymorphic class where some virtual functions use optional kernel
features incompatible with a target device.

Consider the following example:

```c++
using syclext = sycl::ext::oneapi::experimental;

struct fp64_set;
struct regular_set;

struct Foo {
virtual SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(
    syclext::indirectly_callable<fp64_set>) void foo() {
  // uses double
  double d = 3.14;
}

virtual SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(
    syclext::indirectly_callable<regular_set>) void bar() {}
};

sycl::queue q;

auto *Storage = sycl::malloc_device<Foo>(1, q);

q.single_task([=] {
  // The kernel is not submitted with 'calls_indirectly' property and therefore
  // it is not considered to be using any of virtual member functions of 'Foo'.
  // This means that the object of 'Foo' can be successfully created by this
  // kernel, regardless of whether a target device supports 'fp64' aspect which
  // is used by 'Foo::foo'.
  // No exceptions are expected to be thrown.
  new (Storage) Foo;
});

if (q.get_device().has(sycl::aspect::fp64)) {
  auto props = syclext::properties{syclext::calls_indirectly<fp64_set>};
  q.single_task(props, [=] {
    Storage->foo();
  });
} else {
  auto props = syclext::properties{syclext::calls_indirectly<regular_set>};
  q.single_task(props, [=] {
    Storage->bar();
  });
}

```

This example should work regardless of whether target device supports 'fp64'
aspect or not. To achieve that, virtual member functions are outlined into
separate device images which are linked at runtime depending on whether they are
compatible with a target device.

Regardless of device code split mode selected by a user, functions marked with
`indirectly_callable` property should be outlined into a separate device images
by `sycl-post-link` tool based on the property argument.

Additionally, if any virtual function in such device image uses any optional
kernel features, then the whole image should be cloned with all function bodies
emptied. This cloned device image will be further referred to as "dummy virtual
functions device image".

This dummy device image is needed to support the example showed above when a
kernel creates an object of a polymorhpic class where some of virtual functions
use optional features. LLVM IR generated by front-end will contain a vtable,
which references all methods of the class. However, not all of them can be
directly included into kernel's device image to avoid speculative compilation.

When such kernel is submitted to a device, runtime will check which optional
features are supported and link one or another device image with virtual
functions.

#### New device image properties

To let runtime know which device images should be linked together to get virtual
functions working, new property set is introduced: "SYCL/virtual functions".

For device images, which contain virtual functions (i.e. ones produced by
outlining `indirectly_callable` functions into a separate device image), the
following properties are set within the new property set:
- "virtual-functions-set" with a string value containing name of virtual
  functions set contained within the image (value of the property argument);
- "dummy-image=1" if an image is a dummy virtual functions device image;

For other device images, the following properties are set within the new
property set:
- "calls-virtual-functions-set" with a string value containing comma-separated
  list of names of virtual function sets used by kernels in the image (as
  indicated by `calls_indirectly` kernel property);
- "creates-virtual-functions-set" with a string value containing comma-separate
  list of names of virtual function sets which are referenced from functions
  included into vtables used by a kernel within a device image;
  **TODO:** this item definitely needs better description

### Changes to the runtime

When a kernel submitted to a device comes from a device image with some
properties set in "SYCL/virtual functions" property set, then runtime does some
extra actions to link several device images together to ensure that the kernel
can be executed.

Algorithm for discovery of device images which has to be linked:
- if device image has property "calls-virtual-functions-set=A,B,...,N" on it,
  then all device images with "virtual-functions-set" property equal to "A",
  "B", ..., "N" are taken to be linked with the initial device image;
- if device image has property "creates-virtual-functions-set=A,B,...,N" on it,
  then for each device image with "virtual-functions-set" property equal to "A",
  "B", ..., "N" and *without* "dummy-image=1" property on it:
  - if that device image is compatible with device, it is taken to be linked
    with the initial device image;
  - otherwise, runtime looks for a device image with the same
    "virtual-functions-set" property, but *with* "dummy-image=1" property on it
    and takes that device image to be linked with the initial device image;

Produced list of device images is then linked together and used to enqueue a
kernel.

**TODO:** do we need to say anything about in-memory and on-disk cache
functionality here?

[1]: <../extensions/proposed/sycl_ext_intel_virtual_functions.asciidoc>
[2]: <CompileTimeProperties.md>
[3]: https://clang.llvm.org/docs/LanguageExtensions.html#builtin-sycl-unique-stable-name
[sycl-spec-optional-kernel-features]: https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#sec:optional-kernel-features
[optional-kernel-features-design]: <OptionalDeviceFeatures.md>

