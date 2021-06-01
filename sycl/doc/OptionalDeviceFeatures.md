# Behavior for optional kernel features

This design document describes the changes that are needed in DPC++ in order to
conform to the SYCL 2020 specification regarding the behavior of applications
that use optional kernel features.  An optional kernel feature is any feature
that is supported by some devices and not by others.  For example, not all
devices support 16-bit floating point operations, so the `sycl::half` data type
is an optional kernel feature.  Some DPC++ extensions like AMX are also
optional kernel features.

The requirements for this design come mostly from the SYCL 2020 specification
[section 5.7 "Optional kernel features"][1] but they also encompass the C++
attribute `[[sycl::requires()]]` that is described in [section 5.8.1 "Kernel
attributes"][2] and [section 5.8.2 "Device function attributes"][3].

[1]: <https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html#sec:optional-kernel-features>
[2]: <https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html#sec:kernel.attributes>
[3]: <https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html#_device_function_attributes>


## Requirements

There are several categories of requirements covered by this design.  Each of
these is described in more detail in the sections that follow:

* The front-end compiler must issue a diagnostic in some cases when a kernel or
  device function uses an optional feature.  However, the front-end compiler
  must **not** generate a diagnostic in other cases.

* The runtime must raise an exception when a kernel using optional features
  is submitted to a device that does not support those features.  This
  exception must be raised synchronously from the kernel invocation command
  (e.g. `parallel_for()`).

* The runtime must not raise an exception (or otherwise fail) merely due to
  speculative compilation of a kernel for a device, when the application does
  not specifically submit the kernel to that device.

### Clarification of a kernel's static call tree

Some of the requirements below refer to the "static call tree" of a kernel.  In
standard SYCL, device code is not allowed to contain function pointers, virtual
function, or indirect function calls.  It is therefore easy to compute the
static call tree of a kernel.  By starting at the kernel function itself (e.g.
the function passed to `parallel_for`), the compiler can identify all functions
called by that function, then it can find all functions called by those
functions, etc.  Depending on the tool which does the analysis, the "static
call tree" could include only those functions that reside in the same
translation unit as the kernel, or it could include all functions that reside
in the same executable image (or shared library) as the kernel.  In the
sections below, we try to make the distinction clear whenever we refer to a
kernel's static call tree.

We are contemplating a DPC++ extension that would allow some limited use of
function pointers in device code.  This feature is not yet fully defined or
supported.  We expect that the semantics of this feature will include some way
for the compiler to deduce a limited set of possible targets for each indirect
function call.  Therefore, it is still possible for the compiler to construct a
"static call tree" for each kernel, the only difference is that each call site
now adds a set of possible target functions to a kernel's static call tree.
The details about how this will work are expected to be included in the DPC++
extension specification that enables indirect function calls.

### Diagnostics from the front-end compiler

By "front-end compiler", we mean the DPC++ compiler which parses DPC++ source
code, not the JIT compiler that translates SPIR-V into native code.

In general, the front-end compiler does not know which kernels the application
will submit to which devices.  Therefore, the front-end compiler does not
generally know which optional features a kernel can legally use.  Thus, in
general, the front-end compiler must not issue any diagnostic simply because a
kernel uses an optional feature.

The only exception to this rule occurs when the application uses the C++
attribute `[[sycl::requires()]]`.  When the application decorates a kernel or
device function with this attribute, it is an assertion that the kernel or
device function is allowed to use only those optional features which are listed
by the attribute.  Therefore, the front-end compiler must issue a diagnostic if
the kernel or device function uses any other optional kernel features.

The SYCL 2020 specification only mandates this error when a kernel or device
function that is decorated with `[[sycl::requires()]]` uses an optional kernel
feature (not listed in the attribute), **and** when that use is in the kernel's
static call tree as computed for the translation unit that contains the kernel
function.  Thus, the compiler is not required to diagnose an error if the use
is in a `SYCL_EXTERNAL` function that is defined in another translation unit.

It turns out, though, that DPC++ can diagnose this case at link time, when we
have visibility into device functions that are defined in other translation
units.  Since the design proposed below allows this error to be checked with
minimal extra effort, it is desirable to do so.

Note that this behavior does not change when the compiler runs in AOT mode.
Even if the user specifies a target device via "-fsycl-targets", that does not
necessarily mean that the user expects all the code in the application to be
runnable on that device.  Consider an application that uses some middleware
library, where the library's header contains kernels optimized for different
devices.  An application should be able to compile in AOT mode with this
library without getting errors.  Therefore the AOT compiler must not fail
simply because the middleware header contains device code for devices that
are not being compiled for.

### Runtime exception if device doesn't support feature

When the application submits a kernel to a device via one of the kernel
invocation commands (e.g. `parallel_for()`), the runtime must check if the
kernel uses optional features that are not supported on that device.  If the
kernel uses an unsupported feature, the runtime must throw a synchronous
`errc::kernel_not_supported` exception.

When doing these checks, the runtime must consider all uses of optional
features in the kernel's static call tree, regardless of whether those uses are
in the same translation unit as the kernel and regardless of whether those uses
come from device code in a shared library.

This exception, however, is only required for features that are exposed via a
C++ type or function.  Examples of this include `sycl::half` or instantiating
`sycl::atomic_ref` for a 64-bit type.  If the kernel relies on optional
features that are more "notional" such as sub-group independent forward
progress (`info::device::sub_group_independent_forward_progress`), no exception
is required.

To further clarify, this exception must be thrown in the following
circumstances:

* For a kernel that is not decorated with `[[sycl::requires()]]`, the exception
  must be thrown if the kernel uses a feature that the device does not support.

* For a kernel that is decorated with `[[sycl::requires()]]`, the exception
  must be thrown if the device does not have the aspects listed in that
  attribute.  Note that the exception must be thrown even if the kernel does
  not actually use a feature corresponding to the aspect, and it must be
  thrown even if the aspect does not correspond to any optional feature.

* For a kernel that is decorated with `[[sycl::requires()]]`, the compiler will
  mostly check (at compile time) whether the kernel uses any features that are
  not listed in the attribute.  The only case not checked at compile time is
  when a kernel calls a device function that is defined in a shared library.
  Therefore, the runtime is responsible for throwing the exception if a
  kernel's function (defined in a shared library) uses an optional feature
  that the device does not support.

* For a kernel that is decorated with the `[[sycl::reqd_work_group_size(W)]]`
  or `[[sycl::reqd_sub_group_size(S)]]` attribute, the exception must be thrown
  if the device does not support the work group size `W` or the sub-group size
  `S`.

Note that the exception must be thrown synchronously, not delayed and thrown on
the queue's asynchronous handler.

### No runtime exception for speculative compilation

It is currently common for the runtime to speculatively compile some kernels.
For example, DPC++ may bundle all kernels from the same translation unit
together into a single device image.  When the application submits one kernel K
to a device D, the runtime actually compiles all kernels in K's device image
for device D.  Let's assume in this example that the kernel K uses only
features that are supported by D.  It would be illegal for the runtime to throw
an exception in such a case just because some other kernel in the same device
image uses a feature that is not supported by device D.


## Design to implement required diagnostics

### Changes to DPC++ headers

With the exception of the `[[sycl::reqd_work_group_size()]]` and
`[[sycl::reqd_sub_group_size()]]` attributes, all kernel optional features
are associated with some device aspect.  For example, the `sycl::half` type
is an optional feature which is only supported on devices that have the
`aspect::fp16` aspect.  We can therefore use device aspects as a way to
describe the set of optional features that a kernel uses (with the exception
of the required work-group or sub-group size).

As we will see later, it will be very helpful to decorate all APIs in DPC++
headers that correspond to optional kernel features with the
`[[sycl::requires()]]` attribute.  For example, the declaration of the
`sycl::half` type would look like this:

```
using half [[sycl::requires(has(aspect::fp16))]] = cl::sycl::detail::half_impl::half;
```

In cases where the optional feature corresponds to use of a class (e.g.
`sycl::atomic_ref`), the declaration can look like this:

```
template</*...*/>
class [[sycl::requires(has(aspect::atomic64))]] atomic_ref {
  /* ... */
};
```

(In reality, we can use partial specialization tricks to decorate `atomic_ref`
with the attribute only when the underlying type is 64-bits.  See ["Appendix:
Adding an attribute to 8-byte `atomic_ref`"][4].)

[4]: <#appendix-adding-an-attribute-to-8-byte-atomic_ref>

In cases where the optional feature corresponds to a function, we can decorate
the function's declaration with the attribute like so (demonstrating a
hypothetical AMX multiplication extension):

```
[[sycl::requires(has(aspect::ext_intel_amx))]]
void amx_multiply();
```

These attributes provide an explicit mapping between each optional kernel
feature and its associated aspect.

Unfortunately, the fundamental type `double` is also an optional kernel
feature.  Since there is no type alias for `double`, there is no convenient
place to add an attribute.  Instead, the front-end device compiler must behave
as though there was an implicit `[[sycl::requires(has(aspect::fp64))]]`
attribute for any device code that uses the `double` type.

Note that the usage of `[[sycl::requires()]]` is slightly expanded here beyond
the specified usage in the SYCL 2020 specification because we allow the
attribute also on type alias declarations and class declarations.  If a device
function does any of the following with a type alias or class that was so
decorated, the function is assumed to "use the aspects" listed in the
attribute:

* Declares a variable of that type.
* Has a formal parameter declared with that type.
* Returns that type.

This also includes any qualified version of the type.

**TODO**: This language is not very precise.  The intent is to include most
uses of the type, except for cases like `sizeof(T)` or `decltype(T)`.  Help
appreciated on tightening the wording here.

**TODO**: Would it be better to use a different attribute name when decorating
types, rather than expanding the meaning of `[[sycl::requires()]]`?  If we did
this, the new attribute would become an internal DPC++ implementation detail;
we would not add it to the SYCL specification.

### Implementing diagnostics in the DPC++ front-end

As noted above, the front-end device compiler must behave as though there is an
implicit `[[sycl::requires(has(aspect::fp64))]]` attribute on any use of the
`double` type in device code.

Aside from this, the front-end compiler can implement the required diagnostics
purely from the C++ attributes that have been added to the DPC++ headers.
There is no need for the front-end compiler to know which device features are
optional.

When the front-end compiler sees a kernel or device function that is decorated
with `[[sycl::requires()]]`, it forms the set of allowed aspects for that
kernel or device function using aspects listed in the attribute.  Let's call
this the `Allowed` set.  The front-end then computes the static call tree of
that kernel or device function (examining only code within this translation
unit) and forms the union of all aspects in any `[[sycl::requires()]]`
attributes that decorate any of these functions or any of the types used inside
these functions.  Let's call this the `Used` set.  If the `Used` set contains
any aspects not in the `Allowed` set, the front-end issues a diagnostic.

In order to be user-friendly, the diagnostic should point the user to the
location of the problem.  Therefore, the diagnostic message should include the
following information:

* The source position of the `[[sycl::requires()]]` attribute that decorates
  the kernel or device function which provides the `Allowed` aspect set.  This
  tells the user which aspects the kernel or device function intends to use.

* The source position of the call to a function that is decorated with
  `[[sycl::requires()]]` or the source position of the use of a type that is
  decorated with `[[sycl::requires()]]`.  This tells the user where in the
  kernel a particular aspect is actually used.

Note that this analysis can be done in the front-end compiler even when a
kernel makes a call to a function that is in another translation unit.
Language rules require the application to declare such a function with
`SYCL_EXTERNAL` in the calling TU, and the `SYCL_EXTERNAL` declaration must be
decorated with the `[[sycl::requires()]]` attribute.  Therefore, the front-end
can diagnose errors with aspect usage even without seeing the definition of the
`SYCL_EXTERNAL` function.


## Design to raise required exceptions (and avoid forbidden errors)

As described above the runtime must raise an `errc::kernel_not_supported`
exception when a kernel is submitted to a device that does not support the
optional features that the kernel uses.  Likewise, the runtime must **not**
raise an exception (or otherwise produce an error) due to speculative
compilation of a kernel for a device, unless the application actually submits
the kernel to that device.  The solution is largely the same for both JIT and
AOT cases.

### JIT case

The JIT case requires some change to the way kernels and device functions are
bundled together into device images.  Currently, kernels and device functions
are bundled together regardless of the features they use, and this can lead to
JIT errors due to speculative compilation.  Consider a device image that
contains two kernels: `K1` uses no optional features and `K2` uses an optional
feature that corresponds to aspect `A`.  Now consider that the application
submits kernel `K1` to a device that does not support aspect `A`.  Since the
two kernels are bundled together into one device image, the runtime really
compiles both kernels for the device.  Currently, this will raise a JIT
exception because the compilation of kernel `K2` will fail when compiled for a
device that does not support aspect `A`.

There are two ways to solve this problem.  One is to change the way kernels and
device functions are bundled into device images such that we never bundled two
kernels or device functions together unless they require exactly the same set
of device aspects.  Doing this would avoid the error described above.  However,
we have elected for a different solution.

Instead, we will allow kernels and device functions to be bundled together as
they currently are, but we will introduce extra decorations into the generated
SPIR-V that allow the JIT compiler to discard kernels and device functions
which require aspects that the device does not support.  Although this solution
requires an extension to SPIR-V, we think it is the better direction because it
is aligned with the [device-if][5] feature, which will also requires this same
SPIR-V extension.

[5]: <https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/DeviceIf/device_if.asciidoc>

The idea is to emit a SPIR-V specialization constant for each aspect that is
required by a kernel or device function in the device image.  We then introduce
a new SPIR-V "decoration" that tells the JIT compiler to discard a function if
a specialization constant is `False`.  The DPC++ runtime will set the values of
the specialization constants according to the target device, thus the JIT
compiler discards (and does not compile) any kernels or device functions that
use features which are not supported on that device.  This avoids errors due to
speculative compilation of kernels or device functions.

#### Representation in SPIR-V

To illustrate how kernels using optional features are represented in SPIR-V,
consider a kernel `K` that requires aspects `A1` and `A2`.  The SPIR-V module
will contain three boolean specialization constants: one representing `A1`, one
representing `A2`, and one representing the expression `A1 && A2`.  All of
these can be represented without any extension to SPIR-V.

```
OpDecorate %11 SpecId 1                       ; External ID for spec const A1
OpDecorate %12 SpecId 2                       ; External ID for spec const A2

%10 = OpTypeBool
%11 = OpSpecConstantFalse %10                 ; Represents A1
%12 = OpSpecConstantFalse %10                 ; Represents A2
%13 = OpSpecConstantOp %10 LogicalAnd %11 %12 ; Represents A1 && A2
```

In order to make it easy for the JIT compiler to discard all functions in a
kernel, each function in the kernel's static call tree (including the function
representing the kernel's entry point) is decorated with a new extended SPIR-V
decoration `ConditionalINTEL` whose operand is the `<id>` of the specialization
constant representing `A1 && A2`.  The semantic of this decoration is that the
JIT compiler must discard the function unless the value of the specialization
constant is `True`.  Augmenting the example from above:

```
OpDecorate %11 SpecId 1                       ; External ID for spec const A1
OpDecorate %12 SpecId 2                       ; External ID for spec const A2
OpDecorate %16 ConditionalINTEL %13           ; Says to discard the function
                                              ; below when (A1 && A2) is False
%10 = OpTypeBool
%11 = OpSpecConstantFalse %10                 ; Represents A1
%12 = OpSpecConstantFalse %10                 ; Represents A2
%13 = OpSpecConstantOp %10 LogicalAnd %11 %12 ; Represents A1 && A2
%14 = OpTypeVoid
%15 = OpTypeFunction %14

%16 = OpFunction %14 None %15                 ; Definition of function that is
...                                           ; discarded when (A1 && A2) is False
OpFunctionEnd
```

See the extension specification of [SpecConditional][6] for a full
description of this new SPIR-V decoration.

[6]: <extensions/SPIRV/SPV_INTEL_spec_conditional.asciidoc>

#### Representation in LLVM IR

**TODO**: I need some help here on how to represent the `[[sycl::requires()]]`
attributes in LLVM IR.  I suspect there is already some mechanism for
representing SYCL attributes in LLVM IR, so hopefully we can mostly reuse that
mechanism.

#### Modifications to the post-link tool

The post-link tool must be modified to add the SPIR-V `ConditionalINTEL`
decorations to the appropriate functions and to emit the specialization
constants that these decorations reference.  This can be done with two passes
over the static call tree for each kernel and each exported device function.
When this phase computes the static call tree, it considers all code in any of
the translation units that are being linked together.  This may not be the
complete call tree, however, in cases where a kernel calls out to a device
function that is defined in a different shared library.

**NOTE**: In this context, "exported device function" means a device function
that is exported from a shared library as defined by [Device Code Dynamic
Linking][7].

[7]: <https://github.com/intel/llvm/pull/3210>

The first pass operates on the static call tree for each kernel and each
exported device function, propagating the aspects that are used up from the
leaves of the call tree.  The result of this pass is that each function in
the call tree is labeled with the union of all aspects that are used in that
function or in any of the functions it calls.  We call this the `Used` set of
aspects.

The error checking in the front-end of the compiler has already verified that
a function decorated with the `[[sycl::requires()]]` attribute does not use
any optional features other than those listed in the attribute.  However, the
static call tree constructed by the front-end may not be as complete as the
call tree constructed by the post-link tool, for example when a kernel calls
a `SYCL_EXTERNAL` device function that is defined in another translation unit.
Since the effort is minimal, we do the error checking again in the post-link
tool in order to catch more errors.

If any of the device functions was annotated with the `[[sycl::requires()]]`
attribute, we call the set of aspects in that attribute the `Allowed` set.
If the `Used` set contains any aspects not in the `Allowed` set, we issue a
diagnostic.  Note that we do this analysis for every device function, not
just the ones that correspond to kernels or to exported device functions.

**TODO**: Can this diagnostic include the source position of the attribute and
the source position of the code that uses optional feature?  It think this
depends on the information in the LLVM IR, which is not defined yet.

After checking for diagnostics, we compute the union of the `Used` set and the
`Allowed` set (if any) for each kernel and each exported device function.  We
call this the function's `Required` set of aspects.  We then do the following
for each kernel and each exported device function:

* For each aspect in the `Required` set, emit an `OpSpecConstantFalse`
  instruction to represent this requirement.  We maintain a set of "required
  specialization constants" for each kernel or exported device function, which
  is used later.  Add this specialization constant to that set.  In addition,
  add an "aspect" entry to the device image's "SYCL/requirements" property set,
  as described below.

* If the kernel function is decorated with the `[[reqd_work_group_size()]]`
  attribute, emit an `OpSpecConstantFalse` op to represent this requirement and
  add this also to the kernel's set of required specialization constants.  In
  addition, add a "reqd\_work\_group\_size" entry to the device image's
  "SYCL/requirements" property set.

* If the kernel function is decorated with the `[[reqd_sub_group_size()]]`
  attribute, emit an `OpSpecConstantFalse` op to represent this requirement and
  add this also to the kernel's set of required specialization constants.  In
  addition, add a "reqd\_sub\_group\_size" entry to the device image's
  "SYCL/requirements" property set.

* If the kernel or exported device function's set of required specialization
  constants is not empty, emit a series of `OpSpecConstantOp` ops with the
  `OpLogicalAnd` opcode to compute the expression `S1 && S2 && ...`, where
  `S1`, `S2`, etc. are the specialization constants in that set.  In addition,
  emit a `ConditionalINTEL` decoration for the function which references the
  `S1 && S2 && ...` specialization constant.

The second pass propagates each kernel or exported device function's required
specialization constants back down the static call tree.  This pass starts by
assigning each each kernel entry function and each exported device function the
set of required specialization constants that were computed above.  The set of
required specialization constants for each remaining function `F` is computed
as `P1 || P2 || ...`, where `P1`, `P2`, etc. are the parent functions of `F` in
the static call tree.  (Obviously, a `Pn` term can be omitted if the parent
function has no required specialization constants.)  Once we have this
information, we do the following for each function `F` that has a non-empty set
of required specialization constants:

* Emit a series of `OpSpecConstantOp` ops with the `OpLogicalAnd` and
  `OpLogicalOr` opcodes to compute the expression `P1 || P2 || ...` described
  above.

* Emit a `ConditionalINTEL` decoration for the function, referencing this
  computed specialization constant.

In all cases above, we should keep track of the specialization constants that
are emitted and reuse them when possible, rather than emitting duplicates.

#### New device image property set

A new device image property set is needed to inform the DPC++ runtime of the
aspects that each kernel or exported device function requires and the
work-group or sub-group sizes that each kernel requires.  This property set is
named "SYCL/requirements".  The name of each property in the set is the name of
a kernel or the name of an exported device function in the device image.  The
value of each property is a `BYTE_ARRAY` that has the following form:

```
[entry_count (uint32)]
[entry_type (uint32)] <variable parameters>
[entry_type (uint32)] <variable parameters>
...
[entry_type (uint32)] <variable parameters>
```

Where `entry_count` tells the number of subsequent entries.  Each entry has a
variable number of parameters according to its type.  The allowable types are:

```
enum {
  aspect,
  reqd_work_group_size,
  reqd_sub_group_size
};
```

The format of each entry type is as follows:

```
[aspect (uint32)]               [aspect_id (uint32)] [spec_id (uint32)]
[reqd_work_group_size (uint32)] [dim_count (uint32)] [dim0 (uint32)] ... [spec_id (uint32)]
[reqd_sub_group_size (uint32)]  [dim (uint32)] [spec_id (uint32)]
```

Where the parameter names have the following meaning:

Parameter   | Definition
---------   | ----------
`aspect_id` | The value of the aspect from the `enum class aspect` enumeration.
`dim_count` | The number of work group dimensions (1, 2, or 3).
`dim0` ...  | The value of a dimension from the `[[reqd_work_group_size]]` attribute.
`dim`       | The value of the sub-group size from the `[[reqd_sub_group_size]]` attribute.
`spec_id`   | The SPIR-V `SpecId` decoration for the specialization constant that the post-link tool generated for this requirement.

Note that the post-link tool will generate a series of `OpSpecConstantOp` ops
when the kernel or exported device function has multiple requirements.
However, each property list entry contains only the `SpecId` of the
`OpSpecConstantFalse` op that is associated with a single requirement.

#### Modifications to the DPC++ runtime

Modifications are also required to the DPC++ runtime in order to set the values
of the specialization constants that correspond to each kernel requirement.  In
addition, the runtime needs to check if the target device supports each of the
kernel's requirements, and it must raise an `errc::kernel_not_supported`
exception if it does not.

When a kernel is submitted to a device, the runtime finds the device image that
contains the kernel and also finds the kernel's entry in the
"SYCL/requirements" property set.  This entry tells the set of requirements for
the kernel.  If the target device does not support all of these requirements,
then the runtime raises `errc::kernel_not_supported`.  This check can be done
before the device image is JIT compiled, so the exception can be thrown
synchronously.

If the kernel imports device function symbols from a shared library as defined
in [Device Code Dynamic Linking][7], the runtime first identifies all the
device images that define these exported device functions.  Before attempting
to link them together, the runtime finds the entries for the exported device
functions in their "SYCL/requirements" property sets and checks that the device
supports all these requirements.  If it does not, the runtime throws
`errc::kernel_not_supported`.

Whenever the runtime submits a SPIR-V image to the backend for online
compilation, it must do the following additional steps:

* Compute the union of all requirements from all entries in the image's
  "SYCL/requirements" property set.

* Query the target device to see whether it supports each of these
  requirements, yielding either `True` or `False` for each one.

* Set the value of each associated specialization constant when JIT compiling
  the device image for this target device.

Note that the runtime's cache of compiled device images does not need any
special modification because the cache already needs to know the values of all
the specialization constants that were used to compile the device image.  We
just need to make sure the cache is also aware of the specialization constants
which correspond to the requirements from the "SYCL/requirements" property set.

#### Modifications to the GEN compiler

The GEN compiler, of course, needs to be modified to implement the new
`ConditionalINTEL` SPIR-V decoration.  It must discard any function with this
decoration (unless the corresponding specialization constant is `True`), and it
must not raise any sort of error due to compilation of these discarded
functions.

### AOT case

The AOT case uses exactly the same solution as the JIT case described above,
but there is one extra steps.  For the AOT case, the post-link tool must set
the values of the specialization constants that correspond to the requirements
for the kernel or exported device function, using the device named in the
"-fsycl-targets" command line option.  After doing this, the post-link tool
calls the AOT compiler to generate native code from SPIR-V as it normally does.
If more than one target device is specified, the post-link tool sets the
specialization constants separately for each device before generating native
code for that device.

Note that the native device image may not contain all kernels or all exported
device functions if they use optional features.  Nevertheless, the
"SYCL/requirements" property set still has entries for all kernel functions and
all exported device functions.  If the application attempts to invoke one of
the discarded kernels on a device (which does not support the kernel's
features), the runtime will see that the kernel is not supported by using
information from the "SYCL/requirements" property set, and the runtime will
raise an exception.  Thus, the runtime will never attempt to invoke one of
these discarded kernels.  Likewise, if a kernel imports a discarded device
function, the runtime will see that the device function is unsupported and
will raise an exception before attempting to perform the dynamic link.


## Appendix: Adding an attribute to 8-byte `atomic_ref`

As described above under ["Changes to DPC++ headers"][8], we need to decorate the
`atomic_ref` type with the `[[sycl::requires()]]` attribute only when it is
specialized with an 8-byte type.  This can be accomplished with some template
partial specialization tricks.  The following code snippet demonstrates (best
read from bottom to top):

[8]: <#changes-to-dpc-headers>

```
namespace sycl {
namespace detail {

template<typename T>
class atomic_ref_impl_base {
 public:
  atomic_ref_impl_base(T x) : x(x) {}

  // All the member functions for atomic_ref go here

 private:
  T x;
};

// Template class which can be specialized based on the size of the underlying
// type.
template<typename T, size_t S>
class atomic_ref_impl : public atomic_ref_impl_base<T> {
 public:
  using atomic_ref_impl_base<T>::atomic_ref_impl_base;
};

// Explicit specialization for 8-byte types.  Only this specialization has the
// attribute.
template<typename T>
class [[sycl::requires(has(aspect::atomic64))]] atomic_ref_impl<T, 8> :
    public atomic_ref_impl_base<T> {
 public:
  using atomic_ref_impl_base<T>::atomic_ref_impl_base;
};

} // namespace detail

// Publicly visible atomic_ref class.
template<typename T>
class atomic_ref : public detail::atomic_ref_impl<T, sizeof(T)> {
 public:
  atomic_ref(T x) : detail::atomic_ref_impl<T, sizeof(T)>(x) {}
};

} // namespace sycl
```
