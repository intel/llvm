# Implementation design for "device\_global"

This document describes the implementation design for the DPC++ extension
[SYCL\_EXT\_ONEAPI\_DEVICE\_GLOBAL][1], which allows applications to declare global
variables in device code.

[1]: <https://github.com/intel/llvm/pull/4233>


## Requirements

The extension specification document referenced above contains the full set of
requirements for this feature, but some requirements that are particularly
relevant to the design are called out here.

The first issue relates to the mechanism for integrating host and device code.
Like specialization constants, device global variables are referenced in both
host and device code, so they require some mechanism to correlate the variable
instance in device code with the variable name in host code.  The API for
reading a device global variable from device code, however, is different from
the API for specialization constants.  Whereas specialization constants are
read through a templated member function:

```
sycl::specialization_id<int> spec_var;

void func(sycl::kernel_handler kh) {
  int i = kh.get_specialization_constant<spec_var>();
}
```

Device global variables are read directly:

```
sycl::ext::oneapi::device_global<int> dev_var;

void func() {
  int i = dev_var;
}
```

As we will see later, this difference has a ramification on the integration
mechanism.

The second issue relates to the semantics of device global variables relative
to SPIR-V module scope global variables.  The semantics are similar, but not
quite the same.  In particular, a device global variable must retain its value
even if a module is recompiled (e.g. to change the value of a specialization
constant), whereas a SPIR-V module scope variable would not retain its value
in this case.  Therefore, device global variables cannot be implemented solely
via SPIR-V module scope global variables.  Instead, the design uses a
combination of SPIR-V module scope variables and also USM device allocated
memory.


## Design

### Changes to DPC++ headers

The headers must be changed, of course, to declare the new `device_global`
class, which is described in the [extension specification][1].  However,
there is no special magic required in the headers beyond the declaration of
this type.

### New LLVM IR attributes

As described below, the device compiler front-end decorates each
`device_global` variable with two attributes which convey information to the
`sycl-post-link` tool.

The `sycl-unique-id` attribute contains a string which uniquely identifies the
variable instance.  If the variable has external linkage, the string must be
the same for all translation units that define the variable (e.g. if the
variable is defined as `inline`).  If the variable has internal linkage, the
string must not be shared by any other `device_global` variable, even in
another translation unit.  (These rules for the identifying string are the
same as the rules we have for `specialization_id` variables.)

The `sycl-device-global-size` attribute contains the size in bytes of the
underlying data type `T` of the `device_global` variable.  As with all
attributes, the value is a string, so the size is represented as a string in
decimal format.

Note that language rules ensure that `device_global` variables are always
declared at namespace scope (i.e. a global variable), and LLVM IR [allows
attributes to be attached to global variables][2].

[2]: <https://llvm.org/docs/LangRef.html#global-attributes>

### Changes to the DPC++ front-end

The device compiler front-end must be changed in two ways: it must generate new
content in the integration footer and it must add the `sycl-unique-id` and
`sycl-device-global-size` attributes to the IR definition of of any
`device_global` variable.  These two tasks are related because the integration
footer contains the same string that is stored in the `sycl-unique-id`
attribute.

**TODO**: See also the "Unresolved issues" section at the bottom of this
document for another change that is needed in the front-end.

#### New content in the integration footer

New content in the integration footer provides a mapping from a host instance
of a `device_global` variable to its unique ID string.  This is done through
partial specialization of a template function in much the same way that we do
for `specialization_id` variables.  To illustrate, consider a translation unit
that defines two `device_global` variables:

```
#include <sycl/sycl.hpp>

sycl::device_global<int> Foo;
static sycl::device_global<double[2]> Bar;

// ...
```

The corresponding integration footer looks like this:

```
inline namespace cl {
namespace sycl::detail {

template<>
inline const char *get_device_global_symbolic_ID_impl<::Foo>() {
  return /* unique string for "Foo" */;
}

template<>
inline const char *get_device_global_symbolic_ID_impl<::Bar>() {
  return /* globally unique string because "Bar" has internal linkage */
}

}
}

#include <CL/sycl/detail/integration_post_footer.hpp>
```

As with the integration footer for `specialization_id` variables, the generated
code is more complex when `device_global` variables are defined in an unnamed
namespace.  See the [specialization constant][3] specification for details.

[3]: <SpecializationConstants.md>

The `<CL/sycl/detail/integration_post_footer.hpp>` file contains the definition
of the wrapper function which calls the partial specializations.  This must be
last in the translation unit to satisfy the C++ requirement that references to
the template function must occur after all partial specializations are defined.

```
inline namespace cl {
namespace sycl::detail {

template <auto &SpecName> const char *get_device_global_symbolic_ID() {
  return get_device_global_symbolic_ID_impl<SpecName>();
}

}
}
```

#### Decorating the IR with new attributes

The device compiler front-end also adds the new `sycl-unique-id` and
`sycl-device-global-size` attribute to the IR definition of any `device_global`
variables.  The `sycl-unique-id` attribute must contain the same string that is
emitted in the integration footer.

### Changes to the `sycl-post-link` tool

The `sycl-post-link` tool performs its normal algorithm to identify the set of
kernels and device functions that are bundled together into each module.  Once
it identifies the functions in each module, it scans those functions looking
for references to global variables of type `device_global`.  The
`sycl-post-link` tool then includes the following additional IR into each
module:

1. The IR definition of each `device_global` variable that is referenced by
   that module.

2. If the module references at least one `device_global` variable, the IR
   definition of a synthesized kernel function that initializes each of those
   `device_global` variables.  The following example shows the structure of
   this kernel function, where `Foo` and `Bar` match the code example above:

   ```
   void __sycl_detail_UNIQUE_STRING(void *p1, void *p2) {
     Foo.usmptr = p1;
     Bar.usmptr = p2;
   }
   ```

   The kernel takes one argument for each `device_global` variable and assigns
   the `usmptr` field of each of those variables to its corresponding argument.
   Note that the name of the kernel must be some unique string.  Otherwise,
   there is a danger that it will conflict with the name of another synthesized
   initialization function if this module is online-linked with device code in
   a shared library.  For example, the implementation can construct a name using
   a GUID.

The `sycl-post-link` tool also emits new property set information as described
below.

### New property in "SYCL/misc properties"

If a device code module has one or more device global variables, a new property
is added to the "SYCL/misc properties" set named "device-global-initializer".
The value of this property has property type `PI_PROPERTY_TYPE_STRING`
containing the name of the synthesized kernel that initializes the device
global variables.

### New "SYCL/device globals" property set

Each device code module that references one or more device global variables
must have an associated "SYCL/device globals" property set.  The name of each
property in this set is the `sycl-unique-id` string of a `device_global`
variable that is referenced in the module.  The value of each property has
property type `PI_PROPERTY_TYPE_UINT32` which tells the size (in bytes) from
the `sycl-device-global-size` attribute for the `device_global` variable.

The order of the properties in this set is important.  The order matches the
order of the parameters accepted by the `__sycl_detail_UNIQUE_STRING` kernel
that is synthesized by the `sycl-post-link` tool.

### Changes to the DPC++ runtime

Several changes are needed to the DPC++ runtime

* The runtime must allocate a buffer from USM device memory for each
  `device_global` variable for each device that accesses that variable.  As
  noted in the requirements, the value of a device global variable must be
  shared even across different device code modules that are loaded onto the
  same device.  As a result, we can't store the value in a SPIR-V module
  scope global variable, which isn't shared across different modules.  All
  modules that access the same variable on a given device must share the same
  USM buffer for that variable.

* We need to call the synthesized `__sycl_detail_UNIQUE_STRING` kernel for each
  device code module to initialize the `device_global` variables.

* We need to implement the new `copy` and `memcpy` functions in the `queue` and
  `handler` classes which copy to or from `device_global` variables.

### Initializing the device global variables in device code

When a DPC++ application submits a kernel, the runtime constructs a
`pi_program` containing this kernel that is compiled for the target device, if
such a `pi_program` does not yet exist.  If the kernel resides in a device code
module that calls into a shared library, the runtime identifies a set of device
code modules that need to be online-linked together in order to construct the
`pi_program`.

After creating a `pi_program` and before invoking any kernel it contains, the
runtime must do the following:

* Scan the entries in the "SYCL/device globals" property sets for each device
  code module that contributes to the `pi_program` to get the full set of
  device global variables used by the `pi_program`.  For each of the device
  global variables, the runtime checks to see if a USM buffer has already been
  created for that variable on this target device.  If not, the runtime
  allocates the buffer from USM device memory, using the size from the
  "SYCL/device globals" property set.  The runtime maintains a mapping from
  the device global's unique string and this USM pointer.

* Scan the "SYCL/misc properties" property set for "device-global-initializer"
  properties.  Each such property names a kernel in the `pi_program` which the
  runtime must call to initialize the device global variables it contains.  The
  runtime uses the contents of the "SYCL/device globals" property set to
  determine the number and order of USM device pointers to pass as arguments to
  this kernel.  The runtime waits for these kernel calls to complete before
  submitting any application kernels from this `pi_program`.

### Implementing the `copy` and `memcpy` functions in `queue` and `handler`

Each of these functions is templated on a reference to a device global variable
like so:

```
template<auto &DeviceGlobal>
event queue::copyto(/*...*/) {/*...*/}
```

The implementation can use the template parameter to obtain the variable's
unique string by calling the mapping function from the integration footer:

```
const char *name = detail::get_device_global_symbolic_ID<DeviceGlobal>();
```

Once the runtime has this name, it is a simple matter to check if a USM buffer
has already been allocated for this device global variable on this device.
If it has not yet been allocated, this means that the application has not yet
submitted any kernels to this device that come from a module that defines this
device global variable.  In this case, the runtime must allocate a buffer from
USM device memory using the size from the template parameter.  The runtime
maintains a mapping from the unique string to this new USM pointer.

```
size_t numBytes = sizeof(decltype(DeviceGlobal)::type);
void *usmptr = malloc_device(numBytes, dev, ctxt);
```

The runtime can now copy to / from this USM buffer using any of the standard
USM explicit copy functions in the `queue` or `handler` class.

Note that the runtime can avoid the cost of subsequent lookups of this
variable's unique string by caching the variable's USM pointer in the host
instance of the `device_global` variable:

```
template<auto &DeviceGlobal>
event queue::copyto(/*...*/) {
  if (!DeviceGlobal.usmptr) {
    const char *name = detail::get_device_global_symbolic_ID<DeviceGlobal>();
    /* etc. */
    DeviceGlobal.usmptr = usmptr;
  }
  /* copy to / from the USM pointer */
}
```

### Accessing the device global from device code

Accessing the value of a `device_global` variable from device code is a simple
matter of accessing the memory from the USM pointer, which is available in the
variable's `usmptr` member.  For example, the implementation of
`device_global::get()` might look like this:

```
T& get() noexcept {
  return *usmptr;
}
```


## Unresolved issues

### Need some way to avoid errors referencing `device_global` variables

The device compiler front-end currently diagnoses an error if device code
references a global variable, unless it is `constexpr` or `const` and constant
initialized.  This is consistent with the SYCL 2020 specification, but the new
device global feature is an exception to this rule.  Device code, of course,
can reference a `device_global` variable even if it is not declared `constexpr`
or `const`.  We need some way to avoid the error diagnostic in this case.

The [newly added][4] `sycl_global_var` attribute is almost what we need,
however that attribute is only allowed to decorate a variable declaration.
This doesn't help us because we don't want to force users to add an attribute
to each declaration of a `device_global` variable.  Instead, we want to
decorate the class declaration of `device_global` with some attribute which
allows any variables of that type to be accessible from device code.

[4]: <https://github.com/intel/llvm/pull/3746>

Since the `sycl_global_var` attribute is currently used only as an
implementation detail for [device-side asserts][5], one options is to repurpose
this attribute.  Rather than applying it to a variable declaration, we could
allow it only on a type declaration.  The implementation of device-side asserts
could be changed to use the attribute on a new type, rather than on a variable
declaration.

[5]: <https://github.com/intel/llvm/pull/3767>

### Need some way to force `device_global` variables into global address space

Although the underlying `T` type of a device global variable is stored in a USM
buffer, the `device_global` variable itself is a module scope global variable.
Unless we decorate these variables in some special way, the current behavior of
the `llvm-spirv` tool is to generate these variables in the private address
space, even though they are declared at module scope.

The [existing OpenCL attribute][6] `[[clang::opencl_global]]` is almost what we
need, but again this attribute can only be applied to a variable declaration.
Instead, we want some attribute that can be applied to the type declaration of
`class device_global`.  We could invent some new attribute with this semantic,
but there is another problem.

[6]: <https://clang.llvm.org/docs/AttributeReference.html#global-global-clang-opencl-global>

Applying `[[clang::opencl_global]]` to a variable of class type currently
raises an error message saying there is no candidate "global" constructor for
the type.  Apparently, the compiler expects a constructor to be defined with
the `__global` keyword:

```
class device_global {
 public:
  device_global() __global;
};
```

We could add a default constructor like that, but the compiler only recognizes
this syntax when it is in OpenCL C++ mode, which is not the case when compiling
SYCL applications.  Therefore, if we invented a new attribute that added
"global address space" semantics to a type, we would probably want that
attribute to cause any constructors to behave as though they were implicitly
declared with the `__global` keyword.

Another option entirely is to change the default behavior of the SYCL device
compiler so that namespace scope variables are implicitly treated as though
they are in the global address space (as opposed to the private address space
as is currently the case).  This behavior would be consistent with the way the
compiler works in OpenCL C 2.0 mode.
