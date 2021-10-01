# Implementation design for "device\_global"

This document describes the implementation design for the DPC++ extension
[SYCL\_EXT\_ONEAPI\_DEVICE\_GLOBAL][1], which allows applications to declare
global variables in device code.

[1]: <https://github.com/intel/llvm/pull/4675>


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

Another issue relates to the `device_image_life` property which can be applied
to a device global variable declaration.  The intent of this property is to
allow a device global variable to be implemented directly on top of a SPIR-V
module scope global variable.  When this property is **not** present, an
instance of a device global variable is shared across all device images that
are loaded onto a particular device.  This makes it easy for the user to reason
about the scope of a variable because the user need not understand which device
image contains each kernel.  However, this semantic makes the implementation
less efficient, especially on FPGA targets.

By contrast, the `device_image_life` property changes the semantic of a device
global variable such that the user must understand which device image contains
each kernel, which is difficult to reason about.  For example, changing the
value of a specialization constant may cause a kernel to be recompiled into a
separate device image on some targets.  As a result, a device global variable
referenced in a kernel may actually have several disjoint instances if the
kernel uses specialization constants.  This problem is more tractable on FPGA
targets because specialization constants are not implemented via separate
device images on those targets, however, there are other factors that FPGA
users need to be aware of when using the `device_image_life` property.  These
are documented more throughly in the extension specification.

The important impact on the design, though, is that device global variables
declared with the `device_image_life` property have an implementation that is
quite different from device global variables that are not declared with this
property.  The sections below describe both implementations.


## Design

### Changes to DPC++ headers

The headers, of course, include the declaration of the new `device_global`
class, which is described in the [extension specification][1].  The declaration
of this class uses partial specialization to define the class differently
depending on whether is has the `device_image_life` property.  When the
property is not present, the class has a member variable which is a pointer to
the underlying type.  Member functions which return a reference to the value
(e.g. `get`) return the value of this pointer:

```
template<typename T>
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
template<typename T>
class device_global {
  T val;
 public:
  T& get() noexcept { return val; }
  /* other member functions */
};
```

In both cases the member variable (either `usmptr` or `val`) must be the first
member variable in the class.  As we will see later, the runtime assumes that
the address of the `device_global` variable itself is the same as the address
of this member variable.

The headers are also updated to add the new `copy()` and `memcpy()` member
functions to `handler` and `queue` which copy data to or from a device global
variable.  These declarations use SFINAE such that they are conditionally
available depending on the `copy_access` property.

### New LLVM IR attributes

Two new attributes are added to communicate information about device global
variable to the `sycl-post-link` tool: `sycl-unique-id` and
`sycl-device-global-image-life`.  As described below, the device compiler
front-end is responsible for adding the attributes to the LLVM IR.

Each device global variable is decorated with `sycl-unique-id`, which provides
a unique string identifier for each device global variable.  This string will
also be used to name the variable in SPIR-V, so it's better for debuggability
if the string matches the mangled name for variables with external linkage.
This is not possible, though, for variables with internal linkage because the
mangled name is not unique in this case.  For these variables, we use the
mangled name and append a unique suffix.

Each device global variable that has the `device_image_life` property is also
decorated with the `sycl-device-global-image-life` attribute.

Note that language rules ensure that `device_global` variables are always
declared at namespace scope (i.e. a global variable), and LLVM IR [allows
attributes to be attached to global variables][2].

[2]: <https://llvm.org/docs/LangRef.html#global-attributes>

### Changes to the DPC++ front-end

The device compiler front-end is changed in two ways: it generates new content
in both the integration header and the integration footer, and it adds the
`sycl-unique-id` and `sycl-device-global-image-life` attributes to the IR
definitions of `device_global` variables as defined above.  These two tasks are
related because the integration footer contains the same string that is stored
in the `sycl-unique-id` attribute.

**NOTE**: See also the "Unresolved issues" section at the bottom of this
document for other changes that are needed in the front-end.

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

The corresponding integration header defines a namespace scope variable of
class type whose sole purpose is to run its constructor before the
application's `main()` function:

```
namespace sycl::detail {
namespace {

class __sycl_device_global_registration {
 public:
  __sycl_device_global_registration() noexcept;
};
__sycl_device_global_registration __sycl_device_global_registerer;

} // namepsace (unnamed)
} // namespace sycl::detail
```

The integration footer contains the definition of the constructor, which calls
a function in the DPC++ runtime with the following information for each device
global variable that is defined in the translation unit:

* The (host) address of the variable.
* The variable's string from the `sycl-unique-id` attribute.
* The size (in bytes) of the underlying `T` type for the variable.
* A boolean telling whether the variable is decorated with the
  `device_image_life` property.

```
namespace sycl::detail {
namespace {

__sycl_device_global_registration::__sycl_device_global_registration() noexcept {
  device_global_map::add(&::Foo,
    /* mangled name of '::Foo' with unique suffix appended */,
    /* size of underlying 'T' type */,
    /* bool telling whether variable has 'device_image_life` property */);
  device_global_map::add(&::inner::Bar,
    /* mangled name of '::inner::Bar' */,
    /* size of underlying 'T' type */,
    /* bool telling whether variable has 'device_image_life` property */);
}

} // namepsace (unnamed)
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
could reference the variable in the global namespace as `::FuBar`, but there is
no way to reference the variable in the unnamed namespace using fully qualified
lookup.

Such programs are still legal, though.  The integration footer can support
cases like this by defining a temporary variable that holds the address of the
shadowed device global:

```
namespace {
const void *__sycl_UNIQUE_STRING = &FuBar;  // References 'FuBar' in the
                                            // unnamed namespace
}

namespace sycl::detail {
namespace {

__sycl_device_global_registration::__sycl_device_global_registration() noexcept {
  device_global_map::add(&::FuBar,
    /* mangled name of '::FuBar' */,
    /* size of underlying 'T' type */,
    /* bool telling whether variable has 'device_image_life` property */);
  device_global_map::add(::__sycl_UNIQUE_STRING,
    /* mangled name of '::(unnamed)::FuBar' with unique suffix appended */,
    /* size of underlying 'T' type */,
    /* bool telling whether variable has 'device_image_life` property */);
}

} // namepsace (unnamed)
} // namespace sycl::detail
```

The `__sycl_UNIQUE_STRING` variable is defined in the same namespace as the
second `FuBar` device global, so it can reference the variable through
unqualified name lookup.  Furthermore, the name of the temporary variable
(`__sycl_UNIQUE_STRING`) is globally unique, so it is guaranteed not to be
shadowed by any other name in the translation unit.  This problem with variable
shadowing is also a problem for the integration footer we use for
specialization constants.  See the [specialization constant design document][3]
for more details on this topic.

[3]: <SpecializationConstants.md>

### Changes to the `sycl-post-link` tool

The `sycl-post-link` tool performs its normal algorithm to identify the set of
kernels and device functions that are bundled together into each module.  Once
it identifies the functions in each module, it scans those functions looking
for references to global variables of type `device_global`.  If any device
global variable decorated with `sycl-device-global-image-life` appears in more
than one module, the `sycl-post-link` tool issues an error diagnostic:

```
error: device_global variable <name> with property "device_image_life"
       is contained in more than one device image.
```

Assuming that no error diagnostic is issued, the `sycl-post-link` tool includes
the IR definition of each `device_global` variable in the modules that
reference that variable.

The [backend functions described below][4] that allow the host to copy to or
from a device global require the variable to have "export" linkage in SPIR-V.
Therefore, the `sycl-post-link` tool needs to make the following IR
transformations for any `device_global` variable that has internal linkage:

[4]: <#back-end-specific-function-to-copy-to--from-a-device-symbol>

* The linkage is changed to be external.
* The name of the variable is changed to be the string from the
  `sycl-unique-id` attribute.

**NOTE**: It seems likely that changing the name of internal linkage variables
will be bad for debuggability of the code.  The user may attempt to print the
value of a variable in the debugger, but the debugger won't know the variable
by that name.  See the "Unresolved issues" section below for more discussion
on this.

The `sycl-post-link` tool also adds the new "device-globals" property to the
"SYCL/misc properties" set, as described below.

### New property in "SYCL/misc properties"

If a device code module has one or more device global variables, a new property
is added to the "SYCL/misc properties" set named "device-globals".  The value
of this property has property type `PI_PROPERTY_TYPE_BYTE_ARRAY` and contains
the `sycl-unique-id` strings for each device global variable that the module
contains.  The value of the property is the concatenation of all these
strings, where each string ends with a null character (`\0`).

### Changes to the DPC++ runtime

Several changes are needed to the DPC++ runtime

* As noted in the requirements section, an instance of a device global variable
  that does not have the `device_image_life` property is shared by all device
  images on a device.  To satisfy this requirement, the device global variable
  contains a pointer to a buffer allocated from USM device memory, and the
  content of the variable is stored in this buffer.  All device images point to
  the same buffer, so the variable's state is shared.  The runtime, therefore,
  must allocate this USM buffer for each such device global variable.

* As we noted above, the front-end generates new content in the integration
  footer which calls the function `sycl::detail::device_global_map::add()`.
  The runtime defines this function and maintains information about all the
  device global variables in the application.  This information includes:

  - The host address of the variable.
  - The string which uniquely identifies the variable.
  - The size (in bytes) of the underlying `T` type for the variable.
  - A boolean telling whether the variable is decorated with the
    `device_image_life` property.
  - The associated per-device USM buffer pointer, if this variable does not
    have the `device_image_life` property.

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

* Scan the strings in the "device-globals" properties of the
  "SYCL/misc properties" sets of each device code module that contributes to
  the `pi_program` to get the unique string associated with each device global
  variable that is used by the `pi_program`.  For each of these strings, the
  runtime uses the device global database to see if the variable was decorated
  with `device_image_life`.  If it was not so decorated and if a USM buffer has
  not already been created for the variable on this target device, the runtime
  allocates the buffer from USM device memory using the size from the database.
  The pointer to this buffer is saved in the database for future reuse.

* For each device global variable that is not decorated with
  `device_image_life`, the runtime initializes the `usmptr` member in the
  *device instance* of the variable by using a backend-specific function which
  copies data from the host to a device variable.  It is a simple matter to use
  this function to overwrite the `usmptr` member with the address of the USM
  buffer.  The details of this device-specific function are described below.

#### Implementing the `copy` and `memcpy` functions in `queue` and `handler`

Each of these functions accepts a (host) pointer to a device global variable as
one of its parameters, and the runtime uses this pointer to find the associated
information for this variable in the device global database.  The remaining
behavior depends on whether the variable is decorated with `device_image_life`.

If the variable is not decorated with this property, the runtime uses the
database to determine if a USM buffer has been allocated yet for this variable
on this device.  If not, the runtime allocates the buffer using the size from
the database.  Regardless, the runtime implements the `copy` / `memcpy` by
copying to or from this USM buffer, using the normal mechanism for copying
to / from a USM pointer.

The runtime avoids the future cost of looking up the variable in the database
by caching the USM pointer in the host instance of the variable's `usmptr`
member.

If the variable is decorated with the `device_image_life` property, the runtime
gets the unique string identifier for the variable from the database and uses
a backend-specific function to copy to or from the variable with that
identifier.  Again, the details of this function are described below.

In all cases, the runtime diagnoses invalid calls that write beyond the device
global variable's size by using the size in the database.

#### Back-end specific function to copy to / from a device symbol

As noted above, we need a backend-specific function copy to / from the device
instance of a variable.  All backends provide this functionality, which is
abstracted with these new PI interfaces:

```
pi_result piextCopyToDeviceVariable(pi_device Device, const char *name,
  const void *src, size_t count, size_t offset);

pi_result piextCopyFromDeviceVariable(pi_device Device, const char *name,
  void *dst, size_t count, size_t offset);
```

In both cases the `name` parameter is the same as the "unique string
identifier" for the device global variable.

On the Level Zero backend, these PI interfaces are implemented by first calling
[`zeModuleGetGlobalPointer()`][5] to get a device pointer for the variable and
then calling [`zeCommandListAppendMemoryCopy()`][6] to copy to or from that
pointer.

[5]: <https://spec.oneapi.io/level-zero/latest/core/api.html#zemodulegetglobalpointer>
[6]: <https://spec.oneapi.io/level-zero/latest/core/api.html#zecommandlistappendmemorycopy>

On the OpenCL backend, these PI interfaces are implemented by first calling
`clGetDeviceGlobalVariablePointerINTEL()` to get a device pointer for the
variable.  This function is provided by the
[`cl_intel_global_variable_pointers`][7] extension which is not yet
productized.  Once we get a pointer, the PI layer calls
`clEnqueueMemcpyINTEL()` to copy to or from that pointer.

[7]: <extensions/DeviceGlobal/cl_intel_global_variable_pointers.asciidoc>

On the CUDA backend, these PI interfaces are implemented on top of
`cudaMemcpyToSymbol()` and `cudaMemcpyFromSymbol()`.


## Unresolved issues

### Need some way to avoid errors referencing `device_global` variables

The device compiler front-end currently diagnoses an error if device code
references a global variable, unless it is `constexpr` or `const` and constant
initialized.  This is consistent with the SYCL 2020 specification, but the new
device global feature is an exception to this rule.  Device code, of course,
can reference a `device_global` variable even if it is not declared `constexpr`
or `const`.  We need some way to avoid the error diagnostic in this case.

The [newly added][8] `sycl_global_var` attribute is almost what we need,
however that attribute is only allowed to decorate a variable declaration.
This doesn't help us because we don't want to force users to add an attribute
to each declaration of a `device_global` variable.  Instead, we want to
decorate the class declaration of `device_global` with some attribute which
allows any variables of that type to be accessible from device code.

[8]: <https://github.com/intel/llvm/pull/3746>

Since the `sycl_global_var` attribute is currently used only as an
implementation detail for [device-side asserts][10], one option is to repurpose
this attribute.  Rather than applying it to a variable declaration, we could
allow it only on a type declaration.  The implementation of device-side asserts
could be changed to use the attribute on a new type, rather than on a variable
declaration.

[10]: <https://github.com/intel/llvm/pull/3767>

### Need to diagnose invalid declarations of `device_global` variables

The device global extension specification places restrictions on where a
`device_global` variable can be declared.  These restrictions are similar to
ones we have already for variables of type `specialization_id`:

* A `device_global` variable can be declared at namespace scope.
* A `device_global` variable can be declared as a static member variable in
  class scope, but only if the declaration has public visibility from namespace
  scope.
* No other declarations are allowed for a variable of type `device_global`.

The device compiler front-end needs to emit a diagnostic if a `device_global`
variable is declared in a way that violates these restrictions.  We do not have
agreement yet, though, on how this should be done.  For example, should the
front-end recognize these variable declarations by the name of their type, or
should we decorate the type with some C++ attribute that helps the front-end
recognize them?

### Need some way to force `device_global` variables into global address space

Although the underlying `T` type of a device global variable is stored in a USM
buffer, the `device_global` variable itself is a module scope global variable.
Unless we decorate these variables in some special way, the current behavior of
the `llvm-spirv` tool is to generate these variables in the private address
space, even though they are declared at module scope.

The [existing OpenCL attribute][11] `[[clang::opencl_global]]` is almost what
we need, but again this attribute can only be applied to a variable
declaration.  Instead, we want some attribute that can be applied to the type
declaration of `class device_global`.  We could invent some new attribute with
this semantic, but there is another problem.

[11]: <https://clang.llvm.org/docs/AttributeReference.html#global-global-clang-opencl-global>

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

### Need some way to propagate properties to SPIR-V

The following three device global properties must be propagated from DPC++
source code, through LLVM IR, and into SPIR-V where they are represented as
SPIR-V decorations (defined in the
[SPV\_INTEL\_global\_variable\_decorations][12] extension).

[12]: <extensions/DeviceGlobal/SPV_INTEL_global_variable_decorations.asciidoc>

* `copy_access`
* `init_via`
* `implement_in_csr`

It's not clear how this should work.  One of the goals of the new property
mechanism is to make it easy to propagate information like this through the
compiler toolchain, so hopefully we can leverage some common infrastructure
rather than hard-coding support for these three properties.  However, there is
not yet a design document for the new properties mechanism, so it's not yet
clear what this infrastructure will be.

### Will changing the name of internal symbols be bad for debugging?

The [backend functions for copying to / from a device symbol][4] currently
require the symbol to have export linkage in SPIR-V.  (This is the case for the
Level Zero and OpenCL functions.  We are not sure about the CUDA functions, but
it seems likely they have the same limitation.)  However, the device global
extension allows these variables to also have internal linkage, and this seems
like a useful feature.  The current strategy is to convert internal linkage
variables to external linkage at the IR level and also rename the symbol in a
way that is globally unique.

This should result in functionally correct code, but it seems likely to make
debugging more difficult.  If the debugger uses the name from SPIR-V, this name
will not match what the user expects.  We attempt to mitigate this somewhat by
preserving the user's name and appending a unique suffix, but this seems like a
weak mitigation.

Do we think the debugging experience will be so bad that we should change the
strategy?  The fundamental requirement is that we need some unique way to
identify each device code variable when using these backend functions.
Currently, we use the variable's mangled name, but this could be changed.
An alternative solution would be to augment the SPIR-V with some new decoration
that gives a unique name to each `OpVariable` that needs to be accessed from
the host.  We could then use that name with the backend functions, and avoid
renaming variables with internal linkage.  This would be more effort, though,
because we would need a new SPIR-V extension, and we would need to change the
implementation of the Level Zero and OpenCL backends.

### Does compiler need to be deterministic?

The compiler is normally deterministic.  If you compile the exact same source
file twice specifying the same command line options each time, you get exactly
the same object file.  However, this will no longer be the case.

The design in this document generates a GUID and uses that GUID to rename
device global variable with internal linkage.  Since the GUID is different each
time the compiler is executed, the resulting object file is different even if
the source file did not change.  The existing design for specialization
constants has exactly the same issue because it also uses a GUID to generate a
unique string for `specialization_id` variables that have internal linkage.

Is this a problem?  If we want to preserve determinism, we could generate
a unique ID as a hash (e.g. SHA-256) from the content of the source file
**and** the command line arguments passed to the compiler.  However, this would
require reading the content of the source file, which would have an impact on
compilation time.  It's not clear how significant this impact would be, though.

Note that the non-determinism will cause a problem with the FPGA `-reuse-exe`
compiler option.  That option uses the result of a previous compilation to
avoid regenerating FPGA native code if the device code in a translation unit
did not change.  (For example, this option avoids regenerating device native
code if the only change in the translation unit was to the host code.)  The
option is implemented by comparing device IR from the previous compilation with
the IR in the new compilation.  Native code is regenerated only if the IR is
different.  This logic will break, though, if the compiler is
non-deterministic because the IR will always be different, so native code will
always be regenerated.  This is a showstopper issue for FPGA because native
code generation takes a very long time.

I see two ways to solve the problem with `-reuse-exe`:

1. We could change the GUID to be a deterministic hash as outlined above.

2. We could change SPIR-V as proposed above to give a unique name to each
   `OpVariable` which needs to be referenced from the host.  This would avoid
   the need to change the exported variable name to be a GUID, thus the IR will
   be deterministic.  (It is also possible to generate the unique `OpVariable`
   names in a deterministic way, so this won't cause a problem.)
