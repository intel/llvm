# Implementation design for sycl_ext_oneapi_virtual_functions

Corresponding language extension specification:
[sycl_ext_oneapi_virtual_functions][1]

## Overview

Main complexity of the feature comes from its co-existence with optional kernel
features ([SYCL 2020 spec][sycl-spec-optional-kernel-features],
[implementation design][optional-kernel-features-design]) mechanism. Consider
the following example:

```c++
using syclext = sycl::ext::oneapi::experimental;

struct set_fp64;

struct Base {
  virtual SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(syclext::indirectly_callable<>)
  void foo() {}

  virtual SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(syclext::indirectly_callable<set_fp64>)
  void bar() {
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
    // features supported by a target device (in case of JIT) and therefore
    // can't decide whether both 'foo' and 'bar' should be included in the
    // resulting device image - the decision must be made at runtime when we
    // know the target device.
    new (Obj) Base;
  });

  // The same binary produced by a sycl compiler should correctly work on both
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
kernel features requirements from the SYCL 2020 specification.

To solve this, the following approach is used: all virtual functions marked with
`indirectly_callable` property are grouped by set they belong to and outlined
into separate device images (i.e. device images with kernels using them are left
with declarations only of those virtual functions).

For each device image with virtual functions that use optional features we also
create a "dummy" version of it where bodies of all virtual functions are
emptied.

Dependencies between device images are recorded in properties based on
`calls_indirectly` and `indirectly_callable` properties. They are used later by
runtime to link them together. Device images which depend on optional kernel
features are linked only if those features are supported by a target device and
dummy versions of those device images are used otherwise.

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

The `calls_indirectly` compile-time property accepts a list of types which
identify virtual functions set. It can be handled using metaprogramming magic to
compile-time concatenate strings to produce a single value out of a set of
parameters. Similar approach is used to handle `reqd_work_group_size` and other
compile-time properties that accept integers:

```c++
// Helper to hide variadic list of arguments under a single type
template <char... Chars> struct CharList {};

// Helper to concatenate several lists of characters into a single string.
// Lists are separated from each other with comma within the resulting string.
template <typename List, typename... Rest> struct ConcatenateCharsToStr;

// Specialization for a single list
template <char... Chars> struct ConcatenateCharsToStr<CharList<Chars...>> {
  static constexpr char value[] = {Chars..., '\0'};
};

// Specialization for two lists
template <char... Chars, char... CharsToAppend>
struct ConcatenateCharsToStr<CharList<Chars...>, CharList<CharsToAppend...>>
    : ConcatenateCharsToStr<CharList<Chars..., ',', CharsToAppend...>> {};

// Specialization for the case when there are more than two lists
template <char... Chars, char... CharsToAppend, typename... Rest>
struct ConcatenateCharsToStr<CharList<Chars...>, CharList<CharsToAppend...>,
                             Rest...>
    : ConcatenateCharsToStr<CharList<Chars..., ',', CharsToAppend...>,
                            Rest...> {};

// Helper to convert type T to a list of characters representing the type (its
// mangled name).
template <typename T, size_t... Indices> struct StableNameToCharsHelper {
  using chars = CharList<__builtin_sycl_unique_stable_name(T)[Indices]...>;
};

// Wrapper helper for the struct above
template <typename T, typename Sequence> struct StableNameToChars;

// Specialization of that wrapper helper which accepts sequence of integers
template <typename T, size_t... Indices>
struct StableNameToChars<T, std::integer_sequence<size_t, Indices...>>
    : StableNameToCharsHelper<T, Indices...> {};

// Top-level helper, which should be used to convert list of typenames into a
// string that contains comma-separated list of their string representations
// (mangled names).
template <typename... Types> struct PropertyValueHelper {
  static constexpr const char *name = "my-fancy-attr";
  static constexpr const char *value =
      ConcatenateCharsToStr<typename StableNameToChars<
          Types,
          std::make_index_sequence<__builtin_strlen(
              __builtin_sycl_unique_stable_name(Types))>>::chars...>::value;
};

// Example usage:
SYCL_EXTERNAL
[[__sycl_detail__::add_ir_attributes_function(
    PropertyValueHelper<void, int>::name,
    PropertyValueHelper<void, int>::value)]] void
foo() {
  // Produced LLVM IR:
  // define void @_Z3foov() #0 { ... }
  // attributes #0 = { "my-fancy-attr"="_ZTSv,_ZTSi" ... }
}

```

### Changes to the compiler front-end

Most of the handling for virtual functions happens in middle-end and thanks to
compile-time properties, no extra work is required to propagate necessary
information down to passes from headers.

However, we do need to filter out those virtual functions which are not
considered to be device  as defined by the [extension specification][1], such
as:

- virtual member functions annotated with `indirectly_callable` compile-time
  property should be emitted into device code;
- virtual member function *not* annotated with `indirectly_callable`
  compile-time property should *not* be emitted into device code;

To achieve that, the front-end should implicitly add `sycl_device` attribtue to
each function which is marked with the `indirectly_callable` attribute. This
can be done during handling of `[[__sycl_detail__::add_ir_attributes_function]]`
attribute by checking if one of string literals passed in there as a property
name is equal to "indirectly_callable". Later the `sycl_device` attribute can be
used to decide if a virtual function should be emitted into device code.

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
defined by `calls_indirectly` property set by user).

**TODO**: should we consider outlining "indirectly used" aspects into a separate
metadata and device image property? This should allow for more precise and
user-friendly exceptions at runtime

NOTE: if the aspects propagation pass is ever extended to track function
pointers, then aspects attached to virtual functions **should not** be attached
to kernels using this mechanism. For example, if a kernel uses a variable,
which is initialized with a function pointer to a virtual function which uses
an aspect, then such kernel **should not** be considered as using that aspect.
Properties-based mechanism which is described above should be used for aspects
propagation for virtual functions.

To illustrate this, let's once again consider the example from Overview section
which is copied below for convenience:

```c++
using syclext = sycl::ext::oneapi::experimental;

struct set_fp64;

struct Base {
  virtual SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(syclext::indirectly_callable<>)
  void foo() {}

  virtual SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(syclext::indirectly_callable<set_fp64>)
  void bar() {
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
    // Even though at LLVM IR level this kernel does reference 'Base::foo'
    // and 'Base::bar' through global variable containing `vtable` for `Base`,
    // we do not consider the kernel to be using `fp64` optional feature.
    new (Obj) Base;
  });

  Q.single_task<Use>(syclext::properties{syclext::calls_indirectly<>}, [=] {
    // This kernel is not considered to be using any optional features, because
    // virtual functions in default set do not use any.
    Obj->foo();
  });

  if (Q.get_device().has(sycl::aspect::fp64)) {
    Q.single_task<UseFP64>(syclext::properties{syclext::calls_indirectly<set_fp64>},
        [=] {
      // This kernel is considered to be using 'fp64' optional feature, because
      // there is a virtual function in 'set_fp64' which uses double.
      Obj->bar();
    });
  }

  return 0;
}
```

This way, "Constructor" kernel(s) won't pull optional features
requirements from virtual functions it may reference through vtable, making it
independent from those. This allows to launch such kernels on wider list of
devices, even though there could be virtual functions which require optional
features.

"Use" kernel(s) do pull optional features requirements from virtual functions
they may call through `calls_indirectly` property and associated sets. This
enables necessary runtime diagnostics that a kernel is not submitted to a device
which doesn't support all required optional features.

#### New compiler diagnostics

**TBD**

#### Device code split and device images

The extension specification restricts implementation from raising a diagnostic
when a kernel that is not marked with `calls_indirectly` kernel property creates
an object of a polymorphic class where some virtual functions use optional
kernel features incompatible with a target device.

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
aspect or not. Implementation differs for JIT and AOT flows.

##### JIT flow

Regardless of device code split mode selected by a user, functions marked with
`indirectly_callable` property should be outlined into separate device images
by `sycl-post-link` tool based on the argument of the `indirectly_callable`
property, i.e. all functions from the same set should be bundled into a
dedicated device image.

**TODO**: as an optimization, we can consider preserving virtual functions from
sets that do not use any optional kernel features.

Virtual functions in the original device image should be turned into
declarations instead of definitions.

Additionally, if any virtual function in such device image uses any optional
kernel features, then the whole image should be cloned with all function bodies
emptied. This cloned device image will be further referred to as "dummy virtual
functions device image".

This dummy device image is needed to support the example showed above when a
kernel creates an object of a polymorphic class where some of virtual functions
use optional features. LLVM IR generated by front-end will contain a vtable,
which references all methods of the class. However, not all of them can be
directly included into kernel's device image to avoid speculative compilation.

When such kernel is submitted to a device, runtime will check which optional
features are supported and link one or another device image with virtual
functions.

##### AOT flow

In AOT mode, there will be no dynamic linking, but at the same time we know the
list of supported optional features by a device thanks to
[device config file][device-config-file-design].

Therefore, `sycl-post-link` should read the device config file to determine list
of optional features supported by a target and based on that drop all virtual
functions from sets that use unsupported optional features.

Note that we are making decisions not based on which aspects are used by each
individual virtual functions, but based on which aspects are used by a set of
virtual functions (as identified by the `indirectlly_callable` property
argument). The latter is computed as conjunction of aspects used by each
virtual function within a set.

The behavior is defined this way to better match the extension specification
which defines virtual functions availability in terms of whole sets and not
individual functions.

#### New device image properties

To let runtime know which device images should be linked together to get virtual
functions working, new property set is introduced: "SYCL/virtual functions".

NOTE: in AOT mode, every device image is already self-contained and contains
the right (supported by a device) set of virtual functions in it. Therefore, we
do not need to emit any of those properties when we are in AOT mode.

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

There is a reason why we need to separate properties and can't just use one for
both kinds of relationships:

When a kernel only creates an object of a polymorphic class, we should only use
virtual functions which are compatible with a target device. Virtual functions
that use unsupported optional features are expected to be outlined into separate
sets in that case and we need to ensure that we are still able to create an
object so that virtual functions that use supported optional features are
usable.

However, when a kernel actually makes calls to virtual functions, we assert
that all optional features used by virtual functions in all sets used by the
kernel are supported on a target device. All those aspects have been already
attached to the kernel as part of aspects propagation phase and therefore at
runtime we will unconditionally pull all device images with virtual functions
which are used by a kernel to make calls to them.

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

NOTE: when shared libraries are involved, they could also provide some
`indirectly_callable` functions in the same sets as application. This means that
there could be more than one image registered with the same value of
"virtual-functions-set" property.

NOTE: No changes are needed for both in-memory and on-disk caches, because they
take both kernel and device as keys and for that pair list of device images
which needs to be linked together does not change from launch to launch.

[1]: <../extensions/proposed/sycl_ext_intel_virtual_functions.asciidoc>
[2]: <CompileTimeProperties.md>
[3]: https://clang.llvm.org/docs/LanguageExtensions.html#builtin-sycl-unique-stable-name
[sycl-spec-optional-kernel-features]: https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#sec:optional-kernel-features
[optional-kernel-features-design]: <OptionalDeviceFeatures.md>
[device-config-file-design]: <DeviceConfigFile.md>

