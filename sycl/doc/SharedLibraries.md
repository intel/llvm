# Shared DPC++ libraries

This document describes purpose and design of Shared DPC++ libraries feature.

## Background
Sometimes users want to provide *device* functions via shared libraries.
Simple source example:
```
// App:

CGH.parallel_for<app_kernel>(/* ... */ {
  library_function();
});


// Shared library:
SYCL_EXTERNAL void library_function() {
  // do something
}
```
It is possible to manually create `sycl::program` in both app and shared
library, then use `link` SYCL API to get a single program and launch kernels
using it. But it is not user-friendly and it is very different from regular
C/C++ workflow.

The main purpose of this feature is to provide a mechanism which allows to
provide *device* functions via shared libraries and works as close as possible
to regular shared libraries.

## Requrements:
User's code is compiled into a shared library which consists of some host API,
device code and device API (`SYCL_EXTERNAL` functions). The library is linked to
a user's application which also contains some device code and performs
computations using DPC++/SYCL.
For this combination the following statements must be true:

- `SYCL_EXTERNAL` functions from library can be called (directly or indirectly)
  from device code of the application.
- Function pointers taken in application should work inside the library.
- Specific code changes are not required, i.e. the mechanism of linking works
  as close as possible to regular shared libraries.

## Design
The overall idea is simple:

- Each device image is supplied with an information about exported and imported
  symbols using device image properties
- DPC++ RT performs *device images collection* task by grouping all device
  images required to execute a kernel based on the list of exports/imports
  - Besides symbol names, additional attributes are taken into account (like
    device image format: SPIR-V or device asm)
- Actual linking is performed by underlying backend (OpenCL/L0/etc.)

Next sections describe details of changes in each component.

### DPC++ front-end changes

DPC++ front-end generates `module-id` attribute on each `SYCL_EXTERNAL` function.
It was generated only on kernels earlier. There are two reasons to start
generating this attribute on `SYCL_EXTERNAL` functions:

- Later in pipeline, this attribute will be used by `sycl-post-link` tool to
  separate `SYCL_EXTERNAL` functions from non-`SYCL_EXTERNAL` functions with
  external linkage.
- `module-id` attribute also contains information about source file where the
  function comes from. This information will be used to perform device code
  split on device images that contain only exported functions.

### sycl-post-link changes

`sycl-post-link` performs 3 important tasks:
- Arranges `SYCL_EXTERNAL` functions into a separate device image(s)
- Supplies device images containing exports with an information about exported
  symbols
- Supplies each device image with an information about imported symbols

`sycl-post-link` outlines `SYCL_EXTERNAL` functions with all their reachable
dependencies (functions with definitions called from `SYCL_EXTERNAL` ones)
into a separate device image(s) in order to create minimal self-contained
device images that can be linked from the user's app. There are several
notable moments though.

If a `SYCL_EXTERNAL` function is used within a kernel defined in a shared
library, it will be duplicated: one instance will be stored in the kernel's
device image and the function won't exported from this device image, while the
other will be stored in a special device image for other `SYCL_EXTERNAL`
functions and will be marked as exported there. Such duplication is need for
two reasons:
- We aim to make device images with kernels self-contained so no JIT linker
  invocations would be needed if we have definitions of all called functions.
  Also note that if AOT is requested, it would be impossible to link anything
  at runtime.
- We could export `SYCL_EXTERNAL` functions from device images with kernels,
  but it would mean that when user's app calls `SYCL_EXTERNAL` function, it has
  to link a whole kernel and all its dependencies - not only it increases the
  amount of unnecessary linked code, but might also lead to build errors if the
  kernel uses some features, which are not supported by target device (and they
  are not used in the `SYCL_EXTERNAL` function).
Besides separating `SYCL_EXTERNAL` functions from kernels, they can be further
split into separate device images if device code split is requested. This is
done by grouping them using `module-id` attribute. Non-`SYCL_EXTERNAL` functions
used by `SYCL_EXTERNAL` functions with different `module-id` attributes are
copied to device images corresponding to those `SYCL_EXTERNAL` functions
to make them self-contained
In case one `SYCL_EXTERNAL` function uses another `SYCL_EXTERNAL` function
with different `module-id` attribute, the second one is not copied to the
device image with the first function, but dependency between those device images
is recorder instead.

After `SYCL_EXTERNAL` functions are arranged into a separate device image(s),
all non-`SYCL_EXTERNAL` functions are internalized to avoid multiple definition
errors during runtime linking.
Device images with `SYCL_EXTERNAL` functions will also have a list of names
of exported functions.

**NOTE**: If device code split is enabled, it seems reasonable to perform
exports arrangement before device code split procedure.

In orger to collect information about imported symbols `sycl-post-link` looks
through LLVM IR and for each declared but not defined symbol records its name,
except the following cases:
- Declarations with `__` prefix in demangled name are not recorded as imported
  functions
  - Declarations with `__spirv_*` prefix should not be recorded as dependencies
  since they represent SPIR-V operations and will be transformed to SPIR-V
  instructions during LLVM->SPIR-V translation.
- Based on some attributes which could be defined later
  - This is needed to have possibility to call device-specific builtins not
    starting with `__` by forward-declaring them in DPC++ code

**NOTE**: If device code split is enabled, imports collection is performed after
split and it is performed on splitted images.

All collected information is attached to a device image via properties
mechanism.

Each device image is supplied with an array of property sets:
```
struct pi_device_binary_struct {
...
  // Array of property sets
  pi_device_binary_property_set PropertySetsBegin;
  pi_device_binary_property_set PropertySetsEnd;
};

```
Each property set is represent by the following struct:
```
// Named array of properties.
struct _pi_device_binary_property_set_struct {
  char *Name;                                // the name
  pi_device_binary_property PropertiesBegin; // array start
  pi_device_binary_property PropertiesEnd;   // array end
};
```
It contains name of property set and array of properties. Each property is
represented by the following struct:
```
struct _pi_device_binary_property_struct {
  char *Name;       // null-terminated property name
  void *ValAddr;    // address of property value
  uint32_t Type;    // _pi_property_type
  uint64_t ValSize; // size of property value in bytes
};
```

List of imported symbols is represented as a single property set with name
`ImportedSymbols` recorded in the `Name` field of property set.
Each property in this set holds name of the particular imported symbol recorded
in the `Name` field of the property.
List of exported symbols is represented in the same way, except the
corresponding set has the name `ExportedSymbols`.

### DPC++ runtime changes

DPC++ RT performs *device images collection* task by grouping all device
images required to execute a kernel based on the list of exports/imports and
links them together using PI API.

Given that all exports will be arranged to a separate device images without
kernels it is reasonable to store device images with exports in a separate data
structure.

## Corner cases and limitations

It is not guaranteed that behaviour of host shared libraries and device shared
libraries will always match. There are several cases when it can occur, the
next sections will cover details of such cases.

### ODR violations

C++ standard defines One Definition Rule as:
> Every program shall contain exactly one definition of every non-inline
  function or variable that is odr-used in that program outside of a discarded
  statement; no diagnostic required.
  The definition can appear explicitly in the program, it can be found in the
  standard or a user-defined library, or (when appropriate) it is implicitly
  defined.


Here is an example:

![ODR violation](images/ODR-shared-libraries.svg)

Both libraries libB and libC provide two different definitions of function
`b()`, so this example illustrates ODR violation. Technically this case has
undefined behaviour, however it is possible to run and compile this example on
Linux and Windows. Whereas on Linux only function `b()` from library libB is
called, on Windows both versions of function `b()` are used.
Most of backends online linkers act like static linkers, i.e. just merge
device images with each other, so it is not possible to correctly imitate
Windows behaviour in device code linking because attempts to do it will result
in multiple definition errors.

Given that, it is not guaranteed that behaviour of shared host libraries and
shared device libraries will always match in case of such ODR violations.

#### LD_PRELOAD

Another way to violate ODR is `LD_PRELOAD` environment variable on Linux. It
allows to load specified shared library before any other shared libraries so it
will be searched for symbols before other shared libraries. It allows to
substitute functions from regular shared libraries by functions from preloaded
library.
Device code registration is implemented using global constructors. Order of
global constructors calling is not defined across different translation units,
so with current design of device shared libraries and device code registration
mechanism it is not possible to understand which device code comes from
preloaded library and which comes from regular shared libraries.

Here is an example:

![LD_PRELOAD](images/LD-preload-shared-libraries.svg)

"libPreload" library is preloaded using `LD_PRELOAD` environment variable.
In this example, device code from "libPreload" might be registered after
device code from "libA".

To implement basic support, for each device image we can record name of the
library where this device image comes from and parse content of `LD_PRELOAD`
environment variable to choose the proper images. However such implementation
will only allow to substitute a whole device image and not an arbitrary
function (unless it is the only function in a device image), because partial
substitution will cause multiple definition errors during runtime linking.

### Run-time libraries loading

It is possible to load shared library during run-time. Both Linux and Windows
provide a way to do so (for example `dlopen()` on Linux or `LoadLibrary` on
Windows).
In case run-time loading is used to load some shared library, the symbols from
this shared library do not appear in the namespace of the main program. It means
that even though shared library is loaded successfully in run-time, it is not
possible to use symbols from it directly. The symbols from run-time loaded
library can be accessed by address which can be obtained using corresponding
OS-dependent API (for example `dlsym()` on Linux).

The problem here is that even though symbols from run-time loaded shared
library are not part of application's namespace, the library is loaded through
standard mechanism, i.e. its global constructors are invoked which means that
device code from it is registered, so it is not possible to
understand whether device code comes from run-time loaded library or not.
If such run-time loaded library exports device symbols and they
somehow match with symbols that actually directly used in device code
somewhere, it is possible that symbols from run-time loaded library
will be unexpectedly used.

To resolve this problem we need to ensure that device code registered from
run-time loaded library appears at the end of symbols search list, however
having that device code registration is triggered by global constructors, it
doesn't seem possible.

One more possible mitigation would be to record name of the library from which
each symbol should be imported, but it still won't resolve all potential
issues with run-time library loading, because user can load the library with the
same name as one of the explicitly linked libraries.
