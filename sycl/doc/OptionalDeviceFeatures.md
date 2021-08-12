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

**NOTE**: At the time this document was written, there is a
[proposed change][4] to the SYCL 2020 specification that will rename
`[[sycl::requires()]]` to `[[sycl::device_has()]]`.  Since that proposal has
not yet been adopted, this design document continues to use the
`[[sycl::requires()]]` name.

[4]: <https://github.com/KhronosGroup/SYCL-Docs/pull/171>


## Definition of terms

### A kernel's static call graph

The term "static call graph" of a kernel means the set of all functions that
a kernel may call, including functions that are called transitively from other
functions.  In standard SYCL, device code is not allowed to contain function
pointers, virtual functions, or indirect function calls.  It is therefore easy
to compute the static call graph of a kernel.  By starting at the kernel
function itself (e.g.  the function passed to `parallel_for`), the compiler can
identify all functions called by that function, then it can find all functions
called by those functions, etc.  Depending on the tool which does the analysis,
the "static call graph" could include only those functions that reside in the
same translation unit as the kernel, or it could include all functions that
reside in the same executable image (or shared library) as the kernel.  In the
sections below, we try to make the distinction clear whenever we refer to a
kernel's static call graph.

We are contemplating a DPC++ extension that would allow some limited use of
function pointers in device code.  This feature is not yet fully defined or
supported.  We expect that the semantics of this feature will include some way
for the compiler to deduce a limited set of possible targets for each indirect
function call.  Therefore, it is still possible for the compiler to construct a
"static call graph" for each kernel, the only difference is that each call site
now adds a set of possible target functions to a kernel's static call graph.
The details about how this will work are expected to be included in the DPC++
extension specification that enables indirect function calls.

### An exported device function

The term "exported device function" means a device function that is exported
from a shared library as defined by [Device Code Dynamic Linking][5].

[5]: <https://github.com/intel/llvm/blob/sycl/sycl/doc/SharedLibraries.md>

### The FE compiler

The term "FE compiler" refers to the entire DPC++ compiler chain that runs
when the user executes the `clang++` command.  This includes the clang
front-end itself, all passes over LLVM IR, the post-link tool, and any AOT
compilation phases (when the user compiles in AOT mode).  The FE compiler does
not include the JIT compiler which translates SPIR-V (or another IL format)
into native code when the application executes.


## Requirements

There are several categories of requirements covered by this design.  Each of
these is described in more detail in the sections that follow:

* The FE compiler must issue a diagnostic in some cases when a kernel or device
  function uses an optional feature.  However, the FE compiler must **not**
  generate a diagnostic in other cases.

* The runtime must raise an exception when a kernel using optional features
  is submitted to a device that does not support those features.  This
  exception must be raised synchronously from the kernel invocation command
  (e.g. `parallel_for()`).

* The runtime must not raise an exception (or otherwise fail) merely due to
  speculative compilation of a kernel for a device, when the application does
  not specifically submit the kernel to that device.


### Diagnostics from the FE compiler

In general, the FE compiler does not know which kernels the application will
submit to which devices.  Therefore, the FE compiler does not generally know
which optional features a kernel can legally use.  Thus, in general, the FE
compiler must not issue any diagnostic simply because a kernel uses an optional
feature.

The only exception to this rule occurs when the application uses the C++
attribute `[[sycl::requires()]]`.  When the application decorates a kernel or
device function with this attribute, it is an assertion that the kernel or
device function is allowed to use only those optional features which are listed
by the attribute.  Therefore, the FE compiler must issue a diagnostic if the
kernel or device function uses any other optional kernel features.

The SYCL 2020 specification only mandates this diagnostic when a kernel or
device function that is decorated with `[[sycl::requires()]]` uses an optional
kernel feature (not listed in the attribute), **and** when that use is in the
kernel's static call graph as computed for the translation unit that contains
the kernel function.  Thus, the compiler is not required to issue a diagnostic
if the use is in a `SYCL_EXTERNAL` function that is defined in another
translation unit.

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
features in the kernel's static call graph, regardless of whether those uses
are in the same translation unit as the kernel and regardless of whether those
uses come from device code in a shared library.

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

* For a kernel that is decorated with `[[sycl::requires()]]`, the FE compiler
  will mostly check (at compile time) whether the kernel uses any features that
  are not listed in the attribute.  However, this check only results in a
  warning, so the runtime is still responsible for throwing the exception if
  any of the functions called by the kernel uses an optional feature that the
  device does not support.

* For a kernel that is decorated with the `[[sycl::reqd_work_group_size(W)]]`
  or `[[sycl::reqd_sub_group_size(S)]]` attribute, the exception must be thrown
  if the device does not support the work group size `W` or the sub-group size
  `S`.

Note that the exception must be thrown synchronously, not delayed and thrown on
the queue's asynchronous handler.


### No runtime exception for speculative compilation

It is currently common for the runtime to speculatively compile some kernels.
For example, DPC++ may bundle all kernels from the same translation unit
together into a single device image.  When the application submits one kernel
*K* to a device *D*, the runtime actually compiles all kernels in *K*'s device
image for device *D*.  Let's assume in this example that the kernel *K* uses
only features that are supported by *D*.  It would be illegal for the runtime
to throw an exception in such a case just because some other kernel in the same
device image uses a feature that is not supported by device *D*.


## Design

### Changes to DPC++ headers

With the exception of the `[[sycl::reqd_work_group_size()]]` and
`[[sycl::reqd_sub_group_size()]]` attributes, all kernel optional features
are associated with some device aspect.  For example, the `sycl::half` type
is an optional feature which is only supported on devices that have the
`aspect::fp16` aspect.  We can therefore use device aspects as a way to
describe the set of optional features that a kernel uses (with the exception
of the required work-group or sub-group size).

As we will see later, it is helpful to decorate all APIs in DPC++ headers that
correspond to optional kernel features with a C++ attribute that identifies the
associated aspect.  We cannot use the `[[sycl::requires()]]` attribute for this
purpose, though, because that attribute is allowed only for functions.
Instead, we invent a new internal attribute `[[sycl_detail::uses_aspects()]]`
that can be used to decorate both functions and types.  This attribute is not
documented for user code; instead it is an internal implementation detail of
DPC++.

Like all use of C++ attributes in the DPC++ headers, the headers should spell
the attribute using initial and trailing double underscores
(`[[__sycl_detail__::__uses_aspects__()]]`).  We show that form in the code
samples below, but this design document uses the form without the underbars
elsewhere.  Both forms refer to the same attribute.

To illustrate, the type `sycl::half` is an optional feature whose associated
aspect is `aspect::fp16`.  We therefore decorate the declaration like this:

```
using half [[__sycl_detail__::__uses_aspects__(aspect::fp16)]] =
  cl::sycl::detail::half_impl::half;
```

If an optional feature is expressed as a class type, it can be similarly
decorated (here illustrating a hypothetical AMX type):

```
class [[__sycl_detail__::__uses_aspects__(aspect::ext_intel_amx)]] amx_type {
  /* ... */
};
```

This attribute is also used to decorate function declarations that correspond
to optional features.  Again, illustrating a hypothetical AMX extension:

```
[[__sycl_detail__::__uses_aspects__(aspect::ext_intel_amx)]]
void amx_multiply();
```

This attribute can also be used to decorate class templates where only certain
instantiations correspond to optional features.  See ["Appendix: Adding an
attribute to 8-byte `atomic_ref`"][6] for an illustration of how this attribute
can be used in conjunction with partial specialization to mark only certain
instantiations of `sycl::atomic_ref` as an optional feature.

[6]: <#appendix-adding-an-attribute-to-8-byte-atomic_ref>

Although the examples above show only a single aspect parameter to the
`[[sycl_detail::uses_aspects()]]` attribute, this attribute should support a
list of aspects, similar to the `[[sycl::requires()]]` attribute.  This will
allow us to support future features that depend on a conjunction of aspects
(e.g. a feature that does atomic operations on 64-bit floating point values
might be decorated with
`[[sycl_detail::uses_aspects(aspect::fp64, aspect::atomic64)]]`).

Unfortunately, the fundamental type `double` is also an optional kernel
feature.  Since there is no type alias for `double`, there is no convenient
place to add an attribute.  Instead, the FE device compiler must behave as
though there was an implicit `[[sycl_detail::uses_aspects(aspect::fp64)]]`
attribute for any device code that uses the `double` type.


### New LLVM IR metadata

In order to communicate the information from `[[sycl::requires()]]` and
`[[sycl_detail::uses_aspects()]]` attributes to the DPC++ post-link tool, we
introduce several new LLVM IR metadata.

The named metadata `!intel_types_that_use_aspects` conveys information about
types that are decorated with `[[sycl_detail::uses_aspects()]]`.  This metadata
is not referenced by any instruction in the module, so it must be looked up by
name.  The format looks like this:

```
!intel_types_that_use_aspects = !{!0, !1, !2}
!0 = !{!"class.cl::sycl::detail::half_impl::half", i32 8}
!1 = !{!"class.cl::sycl::amx_type", i32 9}
!2 = !{!"class.cl::sycl::other_type", i32 8, i32 9}
```

The value of the `!intel_types_that_use_aspects` metadata is a list of unnamed
metadata nodes, each of which describes one type that is decorated with
`[[sycl_detail::uses_aspects()]]`.  The value of each unnamed metadata node
starts with a string giving the name of the type which is followed by a list of
`i32` constants where each constant is a value from `enum class aspect` telling
the numerical value of an aspect from the type's
`[[sycl_detail::uses_aspects()]]` attribute.  In the example above, the type
`cl::sycl::detail::half_impl::half` uses an aspect whose numerical value is
`8` and the type `cl::sycl::other_type` uses two aspects `8` and `9`.

**NOTE**: The reason we choose this representation is because LLVM IR does not
allow metadata to be attached directly to types.  This representation works
around that limitation by creating global named metadata that references the
type's name.

We also introduce two metadata that can be attached to a function definition
similar to the existing `!intel_reqd_sub_group_size`.  The
`!intel_declared_aspects` metadata is used for functions that are decorated
with `[[sycl::requires()]]`, and the `!intel_used_aspects` metadata is used to
store the propagated information about all aspects used by a kernel or exported
device function.

In each case, the metadata's parameter is an unnamed metadata node, and the
value of the metadata node is a list of `i32` constants, where each constant is
a value from `enum class aspect`.

For example, the following illustrates the IR that corresponds to a function
`foo` that is decorated with `[[sycl::requires()]]` where the required aspects
have the numerical values `8` and `9`.  In addition, the function uses an
optional feature that corresponds to an aspect with numerical value `8`.

```
define void @foo() !intel_declared_aspects !1 !intel_used_aspects !2 {}
!1 = !{i32 8, i32 9}
!2 = !{i32 8}
```


### Changes to the DPC++ front-end

The front-end of the device compiler is responsible for parsing the
`[[sycl::requires()]]` and `[[sycl_detail::uses_aspects()]]` attributes and
transferring the information to the LLVM IR metadata described above according
to the following rules:

* If the translation unit contains any type definitions that are decorated with
  `[[sycl_detail::uses_aspects()]]`, the front-end creates an
  `!intel_types_that_use_aspects` metadata describing the aspects used by all
  such types.

* If a function is decorated with `[[sycl_detail::uses_aspects()]]`, the
  front-end adds an `!intel_used_aspects` metadata to the function's definition
  listing the aspects from that attribute.

* If a function is decorated with `[[sycl::requires()]]`, the front-end adds
  an `!intel_declared_aspects` metadata to the function's definition listing
  the aspects from that attribute.


### New LLVM IR pass to propagate aspect usage

We add a new IR phase to the device compiler which does the following:

* Creates (or augments) a function's `!intel_used_aspects` metadata with
  aspects that come from references to types in the
  `intel_types_that_use_aspects` list.

* Propagates each function's `!intel_used_aspects` metadata up the static call
  graph so that each function lists the aspects used by that function and by
  any functions it calls.

* Diagnoses a warning if any function that has `!intel_declared_aspects` uses
  an aspect not listed in that declared set.

It is important that this IR phase runs before any other optimization phase
that might eliminate a reference to a type or inline a function call because
such optimizations will cause us to miss information about aspects that are
used.  Therefore, it is recommended that this new phase run first, before all
other IR phases.

Implementing the first bullet point is straightforward.  The implementation can
scan the IR for each function looking for instructions that reference a type.
It can then see if that type is in the `!intel_types_that_use_aspects` set; if
so it adds the type's aspects to the function's `!intel_used_aspects` set.
While doing this, the implementation must have a special case for the `double`
type because the front-end does not include that type in the
`!intel_types_that_use_aspects` set.  If a function references the `double`
type, the implementation implicitly assumes that the function uses
`aspect::fp64` and adds that aspect to the function's `!intel_used_aspects`
set.

**NOTE**: This scan of the IR will require comparing the type referenced by
each IR instruction with the names of the types in the
`!intel_types_that_use_aspects` metadata.  It would be very inefficient if we
did a string comparison each time.  As an optimization, the implementation can
first lookup up each type name in the `!intel_types_that_use_aspects` metadata
set, finding the "type pointer" that corresponds to each type name.  Then the
pass over the IR can compare the type pointer in each IR instruction with the
type pointers from the `!intel_types_that_use_aspects` metadata set.

The second bullet point requires building the static call graph, but the
implementation need not scan the instructions in each function.  Instead, it
need only look at the `!intel_used_aspects` metadata for each function,
propagating the aspects used by each function up to it callers and augmenting
the caller's `!intel_used_aspects` set.

Diagnosing warnings is then straightforward.  The implement looks for functions
that have `!intel_declared_aspects` and compares that set with the
`!intel_used_aspects` set (if any).  If a function uses an aspect that is not
in the declared set, the implementation issues a warning.

One weakness of this design is that the warning message will only be able to
contain the source location of the problem if the compiler was invoked with
`-g` because this is the only time when the front-end propagates source
location information into the IR.  To compensate, the warning message displays
the static call chain that leads to the problem.  For example:

```
warning: function 'foo' uses aspect 'fp64' not listed in 'sycl::requires'
use is from this call chain:
  foo()
  bar()
  boo()
compile with '-g' to get source location
```

Including the call chain in the warning message will require maintaining some
additional information during the traversal of the static call graph described
above.

When the compiler is invoked with `-g`, the implementation uses the
`!DILocation` metadata to improve the warning message with source file, line,
and column information like so:

```
hw.cpp:27:4: warning: function 'foo' uses aspect 'fp64' not listed in 'sycl::requires'
use is from this call chain:
  foo()
  bar() hw.cpp:15:3
  boo() hw.cpp:25:5
```

In the example above, the location `hw.cpp:27:4` gives the source location of
the code that uses the `fp64` aspect, in this case somewhere in the `boo()`
function.  The location `hw.cpp:15:3` tells the location in `foo()` of the call
to `bar()`, etc.

**NOTE**: Issuing this warning message from an IR pass is a compromise.  We
would get better source location if the front-end diagnosed this warning.
However, we feel that the analysis required to diagnose this warning would be
too expensive in the front-end because it requires an additional pass over the
AST.  By contrast, we can diagnose the warning more efficiently in an IR pass
because traversal of the IR is much more efficient than traversal of the AST.
The downside, though, is that the warning message is less informative.


### Assumptions on other phases of clang

The post-link tool (described below) uses the `!intel_used_aspects` and
`!intel_declared_aspects` metadata, so this metadata must be retained by any
other clang passes.  However, post-link only uses this metadata when it
decorates the definition of a kernel function or the definition of an exported
device function, so it does not matter if intervening clang passes discard the
metadata on other device functions.

We think this is a safe assumption for two reasons.  First, the existing design
must already preserve the `!reqd_work_group_size` metadata that decorates
kernel functions.  Second, the kernel functions and exported device functions
always have external linkage, so there is no possibility that a clang phase
will optimize them away.

**NOTE**: Ideally, we would change the llvm-link tool to somehow preserve the
`!intel_declared_aspects` and `!intel_used_aspects` metadata for functions
marked `SYCL_EXTERNAL` so that we could compare the declared aspects (in the
module that imports the function) with the used aspects (in the module the
exports the function).  This would allow us to diagnose errors where the
importing translation unit's declared aspects do not match the aspects actually
used by the function.

We do not propose this change as part of this design, though.  We expect that
this will not be a common error because applications can avoid this problem by
declaring the `SYCL_EXTERNAL` function in a common header that is included by
both the importing and the exporting translation unit.  If the declaration (in
the header) is decorated with `[[sycl::requires()]]`, the shared declaration
will ensure that the definition stays in sync with the declaration.


### Changes to the post-link tool

As noted in the requirements section above, DPC++ currently bundles kernels
together regardless of the optional features they use, and this can lead to
problems resulting from speculative compilation.  To illustrate, consider
kernel *K1* that uses no optional features and kernel *K2* that uses a feature
corresponding to aspect *A*, and consider the case when *K1* and *K2* are
bundled together in the same device image.  Now consider an application that
submits *K1* to a device that does not have aspect *A*.  The application should
expect this to work, but DPC++ currently fails because JIT-compiling *K1*
causes the entire bundle to be compiled, and this fails when trying to compile
*K2* for a device that does not have aspect *A*.

We solve this problem by changing the post-link tool to bundle kernels and
exported device functions according to the aspects that they use.

#### Changes to the device code split algorithm

The algorithm for splitting device functions into images must be changed to
account for the aspects used by each kernel or exported device function.  The
goal is to ensure that two kernels or exported device functions are only
bundled together into the same device image if they use exactly the same set
of aspects.

For the purposes of this analysis, the set of *Used* aspects is computed by
taking the union of the aspects listed in the kernel's (or device function's)
`!intel_used_aspects` and `!intel_declared_aspects` sets.  This is consistent
with the SYCL specification, which says that a kernel decorated with
`[[sycl::requires()]]` may only be submitted to a device that provides the
listed aspects, regardless of whether the kernel actually uses those aspects.

We must also split two kernels into different device images if they have
different `[[sycl::reqd_sub_group_size()]]` or different
`[[sycl::reqd_work_group_size()]]` values.  The reasoning is similar as the
aspect case.  The JIT compiler currently raises an error if it tries to compile
a kernel that has a required sub-group size if the size isn't supported by the
target device.  The behavior for required work-group size is less clear.  The
Intel implementation does not raise a JIT compilation error when compiling a
kernel that uses an unsupported work-group size, but other backends might.
Therefore, it seems safest to split device code based required work-group size
also.

Therefore, two kernels or exported device functions are only bundled together
into the same device image if all of the following are true:

* They share the same set of *Used* aspects,
* They either both have no required work-group size or both have the same
  required work-group size, and
* They either both have the same numeric value for their required sub-group
  size or neither has a numeric value for a required sub-group size.  (Note
  that this implies that kernels decorated with
  `[[intel::named_sub_group_size(automatic)]]` can be bundled together with
  kernels that are decorated with `[[intel::named_sub_group_size(primary)]]`
  and that either of these kernels could be bundled with a kernel that has no
  required sub-group size.)

These criteria are an additional filter applied to the device code split
algorithm after taking into account the `-fsycl-device-code-split` command line
option.  If the user requests `per_kernel` device code split, then each kernel
is already in its own device image, so no further splitting is required.  If
the user requests any other option, device code is first split according to
that option, and then another split is performed to ensure that each device
image contains only kernels or exported device functions that meet the criteria
listed above.

#### Create the "SYCL/device-requirements" property set

The DPC++ runtime needs some way to know about the *Used* aspects, required
sub-group size, and required work-group size of an image.  Therefore, the
post-link tool provides this information in a new property set named
"SYCL/device-requirements".

The following table lists the properties that this set may contain and their
types:

Property Name             | Property Type
-------------             | -------------
"aspect"                  | `PI_PROPERTY_TYPE_BYTE_ARRAY`
"reqd\_sub\_group\_size"  | `PI_PROPERTY_TYPE_BYTE_ARRAY`
"reqd\_work\_group\_size" | `PI_PROPERTY_TYPE_BYTE_ARRAY`

There is an "aspect" property if the image's *Used* set is not empty.  The
value of the property is an array of `uint32` values, where each `uint32` value
is the numerical value of an aspect from `enum class aspect`.  The size of the
property (which is always divisible by `4`) tells the number of aspects in the
array.

There is a "reqd\_sub\_group\_size" property if the image contains any kernels
with a numeric required sub-group size.  (I.e. this excludes kernels where the
required sub-group size is a named value like `automatic` or `primary`.)  The
value of the property is a `uint32` value that tells the required size.

There is a "reqd\_work\_group\_size" property if the image contains any kernels
with a required work-group size.  The value of the property is a `BYTE_ARRAY`
with the following layout:

```
<dim_count (uint32)> <dim0 (uint32)> ...
```

Where `dim_count` is the number of work group dimensions (i.e. 1, 2, or 3), and
`dim0 ...` are the values of the dimensions from the
`[[reqd_work_group_size()]]` attribute, in the same order as they appear in the
attribute.

**NOTE**: One may wonder why the type of the "reqd\_sub\_group\_size" property
is not `PI_PROPERTY_TYPE_UINT32` since its value is always 32-bits.  The
reason is that we may want to expand this property in the future to contain a
list of required sub-group sizes.  Likewise, the "reqd\_work\_group\_size"
property may be expanded in the future to contain a list of required work-group
sizes.


### Changes specific to AOT mode

In AOT mode, for each AOT target specified by the `-fsycl-targets` command
line option, DPC++ normally invokes the AOT compiler for each device IR module
resulting from the sycl-post-link tool.  For example, this is the `ocloc`
command for Intel Gen AOT target and the `opencl-aot` command for the x86 AOT
target with SPIR-V as the input, or other specific tools for the PTX target
with LLVM IR bitcode input.  This causes a problem, though, for IR modules that
use optional features because these commands could fail if they attempt to
compile IR using an optional feature that is not supported by the target
device.  We therefore need some way to avoid calling these commands in these
cases.

The overall design is as follows.  The DPC++ installation includes a
configuration file that has one entry for each device that it supports.  Each
entry contains the set of aspects that the device supports and the set of
sub-group sizes that it supports.  DPC++ then consults this configuration
file to decide whether to invoke a particular AOT compiler on each device IR
module, using the information from the module's "SYCL/device-requirements"
property set.

#### Device configuration file

The configuration file uses a simple YAML format where each top-level key is
a name of a device architecture.  We expect to define a set of device
architecture names that are used consistently in many places (in this
configuration file, in the names of device-specific aspects, as parameters for
the `-fsycl-targets` command line option, etc.)  However, we have not yet
agreed on these architecture names.  There are sub-keys under each device for
the supported aspects, sub-group sizes and AOT compiler ID.  For example:

```
gen11_1:
  aspects: [1, 2, 3]
  sub-group-sizes: [8, 16]
gen_icl:
  aspects: [2, 3]
  sub-group-sizes: [8, 16]
x86_64_avx512:
  aspects: [1, 2, 3, 9, 11]
  sub-group-sizes: [8, 32]
```

The values of the aspects in this configuration file can be the numerical
values from the `enum class aspect` enumeration or the enum identifier itself.

One advantage to encoding this information in a textual configuration file is
that customers can update the file if necessary.  This could be useful, for
example, if a new device is released before there is a new DPC++ release.  In
fact, the DPC++ driver supports a command line option which allows the user
to select an alternate configuration file.

**TODO**: More information will be inserted here when we merge
[this separate PR][7] into this design document.

[7]: <https://github.com/gmlueck/llvm/pull/1>


### Changes to the DPC++ runtime

The DPC++ runtime must be changed to check if a kernel uses any optional
features that the device does not support.  If this happens, the runtime must
raise a synchronous `errc::kernel_not_supported` exception.

When the application submits a kernel to a device, the runtime identifies all
the other device images that export device functions which are needed by the
kernel as described in [Device Code Dynamic Linking][5].  Before the runtime
actually links these images together, it compares each image's
"SYCL/device-requirements" against the features provided by the target
device.  If any of the following checks fail, the runtime throws
`errc::kernel_not_supported`:

* The "aspect" property contains an aspect that is not provided by the device,
  or
* The "reqd\_sub\_group\_size" property contains a sub-group size that the
  device does not support.

There is no way currently for the runtime to query the work-group sizes that a
device supports, so the "reqd\_work\_group\_size" property is not checked.  We
include this property in the set nonetheless for possible future use.

If the runtime throws an exception, it happens even before the runtime tries to
access the contents of the device image.


## Appendix: Adding an attribute to 8-byte `atomic_ref`

As described above under ["Changes to DPC++ headers"][8], we need to decorate
any SYCL type representing an optional device feature with the
`[[sycl_detail::uses_aspects()]]` attribute.  This is somewhat tricky for
`atomic_ref`, though, because it is only an optional feature when specialized
for a 8-byte type.  However, we can accomplish this by using partial
specialization techniques.  The following code snippet demonstrates (best read
from bottom to top):

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
class [[__sycl_detail__::__uses_aspects__(aspect::atomic64)]]
    atomic_ref_impl<T, 8> : public atomic_ref_impl_base<T> {
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
