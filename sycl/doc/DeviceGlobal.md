# Implementation design for "device\_global"

This document describes the implementation design for the DPC++ extension
[SYCL\_EXT\_ONEAPI\_DEVICE\_GLOBAL][1], which allows applications to declare
global variables in device code.

[1]: <extensions/proposed/SYCL_EXT_ONEAPI_DEVICE_GLOBAL.asciidoc>


## Requirements

The extension specification document referenced above contains the full set of
requirements for this feature, but some requirements that are particularly
relevant to the design are called out here.

The first issue relates to the mechanism for integrating host and device code.
Like specialization constants, device global variables are referenced in both
host and device code, so they require some mechanism to correlate the variable
instance in device code with the variable instance in host code.  The API for
referencing a device global variable, however, is different from the API for
specialization constants.  Whereas specialization constants are referenced
through a templated member function:

```
sycl::specialization_id<int> spec_var;

void func(sycl::queue q) {
  q.submit([&](sycl::handler &cgh) {
    cgh.set_specialization_constant<spec_var>(42);
    cgh.single_task([=](sycl::kernel_handler kh) {
      int i = kh.get_specialization_constant<spec_var>();
    });
  });
}
```

Device global variables, by contrast, are referenced by their address:

```
sycl::ext::oneapi::device_global<int> dev_var;

void func(sycl::queue q) {
  int val = 42;
  q.copy(&val, dev_var).wait();      // The 'dev_var' parameter is by reference
  q.submit([&](sycl::handler &cgh) {
    cgh.single_task([=] {
      int i = dev_var;
    });
  });
}
```

This is a key difference because the compiler does not statically know which
device global variable is being referenced; we only know the address at
runtime.  As we will see later, this has a ramification on the integration
headers and on the mechanism that connects instances of device global variables
in host code with their corresponding instances in device code.

Another issue relates to the `device_image_scope` property which can be applied
to a device global variable declaration.  The intent of this property is to
allow a device global variable to be implemented directly on top of a SPIR-V
module scope global variable.  When this property is **not** present, an
instance of a device global variable is shared across all device images that
are loaded onto a particular device.  By contrast, when this property **is**
present, each device image has its own instance of the device global variable.
However, since multiple variable instances have confusing semantics, the API
requires the user to ensure that each such variable exists in exactly one
device image.  The extension specification has more details on this property.

The important impact on the design, though, is that device global variables
declared with the `device_image_scope` property have an implementation that is
quite different from device global variables that are not declared with this
property.  The sections below describe both implementations.


## Design

### Changes to DPC++ headers

#### Class specializations based on `device_image_scope`

The headers, of course, include the declaration of the new `device_global`
class, which is described in the [extension specification][1].  The declaration
of this class uses partial specialization to define the class differently
depending on whether it has the `device_image_scope` property.  When the
property is not present, the class has a member variable which is a pointer to
the underlying type.  Member functions which return a reference to the value
(e.g. `get`) return the value of this pointer:

```
template<typename T, typename PropertyListT>
class device_global {
  T *usmptr;
 public:
  T& get() noexcept { return *usmptr; }
  /* other member functions */
};
```

However, when the property is present, it has a member variable which is the
type itself, and member functions return a reference to this value.

```
template<typename T, typename PropertyListT>
class device_global {
  T val{};
 public:
  T& get() noexcept { return val; }
  /* other member functions */
};
```

Note that the `val` member has a default initializer that causes it to be
"value initialized".  Since the type `T` is limited to types that are trivially
constructible, this means that `val` will be zero initialized.

In both cases the member variable (either `usmptr` or `val`) must be the first
member variable in the class.  As we will see later, the runtime assumes that
the address of the `device_global` variable itself is the same as the address
of this member variable.

#### Attributes attached to the class

The `device_global` class declaration contains three C++ attributes which
convey information to the front-end.  These attributes are only needed for the
device compiler, and the `#ifdef __SYCL_DEVICE_ONLY__` allows the customer to
use another host compiler, even if it does not recognize these attributes.
Also note that these attributes are all in the `__sycl_detail__` namespace, so
they are considered implementation details of DPC++.  We do not intend to
support them as general attributes that customer code can use.

```
template <typename T, typename PropertyListT = property_list<>>
class device_global {/*...*/};

// Partial specialization to make PropertyListT visible as a parameter pack
// of properties.
template <typename T, typename ...Props>
class
#ifdef __SYCL_DEVICE_ONLY__
  [[__sycl_detail__::add_ir_global_variable_attributes(
    "sycl-device-global-size",
    Props::meta_name...,
    sizeof(T),
    Props::meta_value...
    )]]
  [[__sycl_detail__::global_variable_allowed]]
  [[__sycl_detail__::device_global]]
#endif
  device_global<T, property_list<Props...>> {/*...*/};
```

The `[[__sycl_detail__::add_ir_global_variable_attributes()]]` attribute is
used to convey information about the compile-time properties to the front-end,
and it is described more fully by the [compile-time properties][2] design
document.  This attribute is also used for other classes that have properties,
so it is not specific to the `device_global` class.

[2]: <CompileTimeProperties.md>

Note that the parameter list to
`[[__sycl_detail__::add_ir_global_variable_attributes()]]` contains one
additional property named `"sycl-device-global-size"`.  The `sycl-post-link`
tool uses this property to distinguish device global variables from other
module scope variables, and the property tells the size of the underlying data
type of the device global variable.

The `[[__sycl_detail__::global_variable_allowed]]` attribute informs the
front-end that global variables of this type are allowed to be referenced in
device code.  By default, the front-end diagnoses an error if device code
references a global variable unless the variable is `constexpr` or `const` and
constant initialized.  However, the presence of this attribute informs the
front-end that variables of this type are an exception to this rule, so the
front-end does not diagnose an error when device code references a
`device_global` variable.  This attribute could also be used by other types,
so it is also not specific to the `device_global` class.

**NOTE**: The implementation of [device-side asserts][3] recently introduced a
new C++ attribute [`sycl_global_var`][4] for a similar purpose.  The design for
device global variables cannot use that attribute because `sycl_global_var` is
intended to be specified on the variable definition (not the type declaration),
and we do not want to force users to add an attribute to each definition of a
`device_global` variable.  However, the implementation of device-side asserts
could be changed to use `[[__sycl_detail__::global_variable_allowed]]`.  We
could then remove the support for `sycl_global_var`.

[3]: <https://github.com/intel/llvm/pull/3767>
[4]: <https://github.com/intel/llvm/pull/3746>

The last attribute `[[__sycl_detail__::device_global]]` controls error
reporting for variables declared of this type.  The device global extension
specification places restrictions on where a `device_global` variable can be
declared.  Rather than have the front-end recognize the name of the
`device_global` type, the front-end uses this attribute to know which
restrictions to enforce for this type.

#### Declaration of member functions to copy device global variables

The headers are also updated to add the new `copy()` and `memcpy()` member
functions to `handler` and `queue` which copy data to or from a device global
variable.  These declarations use SFINAE such that they are conditionally
available depending on the `host_access` property.

### Changes to the DPC++ front-end

There are several changes to the device compiler front-end:

* The front-end adds a new LLVM IR attribute `sycl-unique-id` to the definition
  of each `device_global` variable, which provides a unique string identifier
  for each device global variable.  The rules for creating this string are the
  same as `__builtin_sycl_unique_stable_id`, so the front-end can use the same
  algorithm when generating the string.

* The front-end checks for restrictions on variable declarations using the
  `device_global` type.  As described above, the front-end uses the
  `[[__sycl_detail__::device_global]]` attribute (rather than the class name)
  to know which set of restrictions to check.  The restrictions specific to
  device global variables are documented in the [extension specification][1].

* The front-end *avoids* diagnosing an error when variables of type
  `device_global` are referenced in device code because the type is decorated
  with the `[[__sycl_detail__::global_variable_allowed]]` attribute.

* The front-end generates new content in both the integration header and the
  integration footer, which is described in more detail below.

#### New content in the integration header and footer

New content in the integration header and footer provides a mapping from the
host address of each device global variable to the unique string for that
variable.  To illustrate, consider a translation unit that defines two
`device_global` variables:

```
#include <sycl/sycl.hpp>

static sycl::device_global<int> Foo;
namespace inner {
  sycl::device_global<double[2]> Bar;
} // namespace inner

// ...
```

The corresponding integration header defines a namespace scope variable of type
`__sycl_device_global_registration` whose sole purpose is to run its
constructor before the application's main() function:

```
namespace sycl::detail {
namespace {

class __sycl_device_global_registration {
 public:
  __sycl_device_global_registration() noexcept;
};
__sycl_device_global_registration __sycl_device_global_registerer;

} // namespace (unnamed)
} // namespace sycl::detail
```

The integration footer contains the definition of the constructor, which calls
a function in the DPC++ runtime with the following information for each device
global variable that is defined in the translation unit:

* The (host) address of the variable.
* The variable's string from the `sycl-unique-id` attribute.

```
namespace sycl::detail {
namespace {

__sycl_device_global_registration::__sycl_device_global_registration() noexcept {
  device_global_map::add(&::Foo,
    /* same string returned from __builtin_sycl_unique_stable_id(::Foo) */);
  device_global_map::add(&::inner::Bar,
    /* same string returned from __builtin_sycl_unique_stable_id(::inner::Bar) */);
}

} // namespace (unnamed)
} // namespace sycl::detail
```

Note that a SYCL application can legally call SYCL APIs even before `main()` by
calling them from a global constructor.  However, the integration headers have
been designed to ensure that the address of each device global variable is
registered with the DPC++ runtime before the user's application could legally
use the variable, even if that use occurs before `main()` executes.

The user's application cannot legally use a device global variable until the
variable's constructor has been called, otherwise the application would be
using an unconstructed object which has undefined behavior by C++ rules.  Since
all device globals must be defined at namespace scope, the C++ rules for the
order of global constructors only guarantee that the device global will be
constructed before subsequent global variables in the same translation unit.
Therefore, a user application could reference a device global from another
global constructor only if that global constructor is for an object defined
*after* the device global in the same translation unit.  However, the
integration header defines `__sycl_device_global_registerer` *before* all
device globals in the user's translation unit.  Therefore, the address of all
device global variables in the translation unit will be registered with the
DPC++ runtime before any user code could legally use them.

#### Handling shadowed device global variables

The example above shows a simple case where the user's device global variables
can all be uniquely referenced via fully qualified lookup (e.g.
`::inner::Bar`).  However, it is possible for users to construct applications
where this is not the case, for example:

```
sycl::device_global<int> FuBar;
namespace {
  sycl::device_global<int> FuBar;
}
```

In this example, the `FuBar` variable in the global namespace shadows a
variable with the same name in the unnamed namespace.  The integration footer
can reference the variable in the global namespace as `::FuBar`, but there is
no way to reference the variable in the unnamed namespace using fully qualified
lookup.

Such programs are still legal, though.  The integration footer can support
cases like this by defining a shim function that returns a reference to the
shadowed device global:

```
namespace {
namespace __sycl_detail {

static constexpr decltype(FuBar) &__shim_1() {
  return FuBar;   // References 'FuBar' in the unnamed namespace
}

} // namespace __sycl_detail
} // namespace (unnamed)

namespace sycl::detail {

__sycl_device_global_registration::__sycl_device_global_registration() noexcept {
  device_global_map::add(&::FuBar,
    /* same string returned from __builtin_sycl_unique_stable_id(::FuBar) */);
  device_global_map::add(&::__sycl_detail::__shim_1(),
    /* same string returned from __builtin_sycl_unique_stable_id(::(unnamed)::FuBar) */);
}

} // namespace sycl::detail
```

The `__shim_1()` function is defined in the same namespace as the second
`FuBar` device global, so it can reference the variable through unqualified
name lookup.  Furthermore, the name of the shim function is globally unique, so
it is guaranteed not to be shadowed by any other name in the translation unit.
This problem with variable shadowing is also a problem for the integration
footer we use for specialization constants.  See the [specialization constant
design document][5] for more details on this topic.

[5]: <SpecializationConstants.md>

### Changes to the DPC++ driver

A new command line argument, `--device-globals` must be passed to the 
`sycl-post-link` tool to enable processing device global variables.

### Changes to the `sycl-post-link` tool

The `sycl-post-link` tool performs its normal algorithm to identify the set of
kernels and device functions that are bundled together into each module.  Once
it identifies the functions in each module, it scans those functions looking
for references to global variables that are decorated with the LLVM IR
attribute `"sycl-device-global-size"` (these are the variables of type
`device_global`).  If any device global variable decorated with the LLVM IR
attribute corresponding to the `device_image_scope` property appears in more
than one module, the `sycl-post-link` tool issues an error diagnostic:

```
error: device_global variable <name> with property "device_image_scope"
       is contained in more than one device image.
```

Assuming that no error diagnostic is issued, the `sycl-post-link` tool includes
the IR definition of each `device_global` variable in the modules that
reference that variable.

As described in the design for [compile-time properties][2], the
`sycl-post-link` tool is responsible for generating idiomatic LLVM IR for any
compile-time properties that need to be generated in SPIR-V.

The **HostAccessINTEL** decoration is required for all device global variables
because it provides the name that the DPC++ runtime uses to access the
variable.  Therefore, the `sycl-post-link` tool always generates idiomatic LLVM
IR for this decoration.  The first SPIR-V operand is set according to the
`host_access` property (or set to **Read/Write** if the device global doesn't
have that property).  The second SPIR-V operation is set to the value of the
device global's `sycl-unique-id`.

The `sycl-post-link` tool also generates idiomatic LLVM IR for the
**InitModeINTEL** decoration (if the device global has the `init_mode`
property) and for the **ImplementInCSRINTEL** decoration (if the device global
has the `implement_in_csr` property).  See the
[SPV\_INTEL\_global\_variable\_decorations][6] SPIR-V extension for details
about all of these decorations.

[6]: <extensions/DeviceGlobal/SPV_INTEL_global_variable_decorations.asciidoc>

The `sycl-post-link` tool also create a "SYCL/device globals" property set for
each device code module that contains at least one device global variable.

### New "SYCL/device globals" property set

Each device code module that references one or more device global variables
has an associated "SYCL/device globals" property set.  The name of each
property in this set is the `sycl-unique-id` string of a `device_global`
variable that is contained by the module.  The value of each property has
property type `PI_PROPERTY_TYPE_BYTE_ARRAY` and contains a structure with the
following fields:

```
struct {
  uint32_t size;
  uint8_t device_image_scope;
};
```

The `size` field contains the size (in bytes) of the underlying type `T` of the
device global variable.  The `sycl-post-link` tool gets this value from the
LLVM IR attribute `"sycl-device-global-size"`.

The `device_image_scope` field is either `1` (true) or `0` (false), telling
whether the device global variable was declared with the `device_image_scope`
property.

### Changes to the DPC++ runtime

Several changes are needed to the DPC++ runtime

* As noted in the requirements section, an instance of a device global variable
  that does not have the `device_image_scope` property is shared by all device
  images on a device.  To satisfy this requirement, the device global variable
  contains a pointer to a buffer allocated from USM device memory, and the
  content of the variable is stored in this buffer.  All device images on a
  particular device point to the same buffer, so the variable's state is
  shared.  The runtime, therefore, must allocate this USM buffer for each such
  device global variable.

* As we noted above, the front-end generates new content in the integration
  footer which calls the function `sycl::detail::device_global_map::add()`.
  The runtime defines this function and maintains information about all the
  device global variables in the application.  This information includes:

  - The host address of the variable.
  - The string which uniquely identifies the variable.
  - The size (in bytes) of the underlying `T` type for the variable.
  - A boolean telling whether the variable is decorated with the
    `device_image_scope` property.
  - The associated per-device USM buffer pointer, if this variable does not
    have the `device_image_scope` property.

  We refer to this information as the "device global database" below.

* The runtime also implements the new `copy` and `memcpy` functions in the
  `queue` and `handler` classes which copy to or from `device_global`
  variables.

#### Initializing the device global variables in device code

When a DPC++ application submits a kernel, the runtime constructs a
`pi_program` containing this kernel that is compiled for the target device, if
such a `pi_program` does not yet exist.  If the kernel resides in a device code
module that calls into a shared library, the runtime identifies a set of device
code modules that need to be online-linked together in order to construct the
`pi_program`.

After creating a `pi_program` and before invoking any kernel it contains, the
runtime does the following:

* Scan the entries of the "SYCL/device globals" property sets of each device
  code module that contributes to the the `pi_program` to get information about
  each device global variable that is used by the `pi_program`.  This
  information is added to device global database.

* For each device global variable that is not decorated with the
  `device_image_scope` property:

    - Check if a USM buffer has already been created for the variable on this
      target device.  If not, the runtime allocates the buffer from USM device
      memory using the size from the database and zero-initializes the content
      of the buffer.  The pointer to this buffer is saved in the database for
      future reuse.

    - Regardless of whether the USM buffer has already been created for the
      variable, the runtime initializes the `usmptr` member in the *device
      instance* of the variable by using a new [PI interface][7] which copies
      data from the host to a global variable in a `pi_program`.  It is a
      simple matter to use this interface to overwrite the `usmptr` member with
      the address of the USM buffer.

[7]: <#new-pi-interface-to-copy-to-or-from-a-module-scope-variable>

Note that the runtime does not need to initialize the `val` member variable of
device global variables that are decorated with `device_image_scope` because
the `val` default initializer already guarantees that this member variable is
zero initialized.

#### Implementing the `copy` and `memcpy` functions in `queue` and `handler`

Each of these functions accepts a (host) pointer to a device global variable as
one of its parameters, and the runtime uses this pointer to find the associated
information for this variable in the device global database.  The code in the
integration footer ensures that the database will at least contain the address
of the variable and its unique string, even if no kernel referencing this
variable has been submitted yet.

Each of these functions is templated on the variable's underlying type `T`, so
it knows the size of this type.  Each function is also templated on the
variable's property list, so it knows whether the variable has the
`device_image_scope` property.

The remaining behavior depends on whether the variable is decorated with the
`device_image_scope` property.

If the variable is not decorated with this property, the runtime uses the
database to determine if a USM buffer has been allocated yet for this variable
on this device.  If not, the runtime allocates the buffer using `sizeof(T)`
and zero-initializes the buffer.  Regardless, the runtime implements the `copy`
/ `memcpy` function by copying to or from this USM buffer, using the normal
mechanism for copying to / from a USM pointer.

The runtime avoids the future cost of looking up the variable in the database
by caching the USM pointer in the host instance of the variable's `usmptr`
member.

If the variable is decorated with the `device_image_scope` property, the
variable's value exists directly in the device code module, not in a USM
buffer.  The runtime first uses the variable's unique string identifier to see
if there is a `pi_program` that contains the variable.  If there is more than
one such `pi_program`, the runtime diagnoses an error by throwing
`errc::invalid`.  If there is no such `pi_program`, the runtime scans all
"SYCL/device globals" property sets to find the device code module that
contains this variable and uses its normal mechanism for creating a
`pi_program` from this device code module.  (The algorithm for creating device
code modules in the `sycl-post-link` tool ensures that there will be no more
than one module that contains the variable.)  Finally, the runtime uses the
new [PI interface][7] to copy to or from the contents of the variable in this
`pi_program`.

It is possible that a device global variable with `device_image_scope` is not
referenced by _any_ kernel, in which case the variable's unique string will not
exist in any property set.  In this case, the runtime simply uses the host
instance of the `device_global` variable to hold the value and copies to or
from the `val` member.

In all cases, the runtime uses `sizeof(T)` to determine if the copy operation
will read or write beyond the end of the device global variable's storage.  If
so, the runtime diagnoses an error by throwing `errc::invalid`.

#### New PI interface to copy to or from a module scope variable

As noted above, we need new PI interfaces that can copy data to or from an
instance of a device global variable in a `pi_program`.  This functionality is
exposed as two new PI interfaces:

```
pi_result piextCopyToDeviceVariable(pi_device Device, pi_program Program,
  const char *name, const void *src, size_t count, size_t offset);

pi_result piextCopyFromDeviceVariable(pi_device Device, pi_program Program,
  const char *name, void *dst, size_t count, size_t offset);
```

In both cases the `name` parameter is the same as the `sycl-unique-id` string
that is associated with the device global variable.

The Level Zero backend has existing APIs that can implement these PI
interfaces.  The plugin first calls [`zeModuleGetGlobalPointer()`][8] to get a
device pointer for the variable and then calls
[`zeCommandListAppendMemoryCopy()`][9] to copy to or from that pointer.
However, the documentation (and implementation) of `zeModuleGetGlobalPointer()`
needs to be extended slightly.  The description currently says:

> * The application may query global pointer from any module that either
>   exports or imports it.
>
> * The application must dynamically link a module that imports a global before
>   the global pointer can be queried from it.

This must be changed to say something along these lines:

> * The interpretation of `pGlobalName` depends on how the module was created.
>   If the module was created from SPIR-V that declares the
>   **GlobalVariableDecorationsINTEL** capability, the implementation looks
>   first for an **OpVariable** that is decorated with **HostAccessINTEL**
>   where the *Name* operand is the same as `pGlobalName`.  If no such variable
>   is found, the implementation then looks for an **OpVariable** that is
>   decorated with **LinkageAttributes** where the *Name* operand is the same
>   as `pGlobalName`.  (The implementation considers both exported and imported
>   variables as candidates.)
>
>   If the module was created from native code that came from a previous call
>   to `zeModuleGetNativeBinary` and that other module was created from SPIR-V,
>   then the interpretation of `pGlobalName` is the same as the SPIR-V case.
>
> * If `pGlobalName` identifies an imported SPIR-V variable, the module must be
>   dynamically linked before the variable's pointer may be queried.

[8]: <https://spec.oneapi.io/level-zero/latest/core/api.html#zemodulegetglobalpointer>
[9]: <https://spec.oneapi.io/level-zero/latest/core/api.html#zecommandlistappendmemorycopy>

The OpenCL backend has a proposed extension
[`cl_intel_global_variable_access`][10] that defines functions
`clEnqueueReadGlobalVariableINTEL()` and `clEnqueueWriteGlobalVariableINTEL()`
which can be easily used to implement these PI interfaces.  This DPC++ design
depends upon implementation of that OpenCL extension.

[10]: <extensions/DeviceGlobal/cl_intel_global_variable_access.asciidoc>

The CUDA backend has existing APIs `cudaMemcpyToSymbol()` and
`cudaMemcpyFromSymbol()` which can be used to implement these PI interfaces.


## Design choices

This section captures some of the discussions about aspects of the design.

### Should the value be zero-initialized

There was some debate about whether the value in the `device_global` should
always be zero-initialized.  We decided to require this in order to be
consistent with C++ rules for global variables.  We want `device_global` to
model the normal rules for global variables.  Since C++ guarantees that a
global variable with a trivial constructor is zero-initialized, we want that
behavior too.

The downside is that some applications may allocate a very large storage for
the underlying type `T` of a device global variable, and they may not want to
pay the cost of zero initializing it.  We agree that this is a theoretical
problem, but we aren't sure if this will be an issue for real applications. If
it turns out to be a real problem, we propose adding a new property that
prevents initialization of the device global value.  For example, we could add
a new parameter to the `init_mode` property called `none`.

### Why not include both `val` and `usmptr` member variables

Rather than using partial specialization to define `device_global` differently
based on the `device_image_scope` property, we could instead define both member
variables regardless of the properties.  This would make the header file
implementation easier, but it would lead to wasted space in the case when the
`device_image_scope` property was not specified since the `val` member is
unused in this case.  Wasting space on the host may not be such a big problem,
but the space would also be wasted on every device that reference the device
global variable, and this seems like a bigger problem.  We decided that the
extra header file complexity of partial specialization is worth avoiding this
wasted memory.
