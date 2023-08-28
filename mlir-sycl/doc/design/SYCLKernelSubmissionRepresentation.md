# Representation of SYCL kernel submission to scheduler

## Introduction

A SYCL command-group function (CGF) submits a kernel function to the runtime's
scheduler to execute over a given range and with a given set of arguments.
Capturing this contextual information is essential for host-device
optimizations, such as constant propagation and simplification of accessors.

In this document, we introduce the `sycl.host.schedule_kernel` operation, and
describe its multi-stage raising process from LLVM-IR involving additional
intermediate operations.

The operations discussed in this document are defined in
`mlir-sycl/include/mlir/Dialect/SYCL/IR/SYCLHostOps.td`. Build the target
`mlir-sycl-doc` in a build directory named `build` to enable the links to the
operations' rendered Markdown documentation.
> TODO: Find a more stable location for the dialect docs to link to.

## Prerequisites / known limitations

- Our approach is currently limited to kernels passed as a lambda function to an
overload of `sycl::handler::parallel_for`.
-  We rely on
`sycl::handler::StoreLambda` being inlined into `sycl::handler::parallel_for`,
and the latter being inlined into the CGF.
- The auto-generated *rounded range kernel* should be deactivated with `-Xclang
  -fsycl-disable-range-rounding`.

## The `sycl.host.schedule_kernel` operation

The
[`schedule_kernel`](../../../build/docs/Dialects/SYCLOps.md#syclhostschedule_kernel-syclsyclhostschedulekernel)
operation ties together the CGF's handler object, a symbol reference to a lambda
function (i.e. the kernel), range information as passed to the `parallel_for`
invocation, and the arguments captured by the lambda object.

```mlir
sycl.host.schedule_kernel %handler ->
  @kernels::@k0[range %1](%2: !sycl_accessor_1_f32_rw_gb, %3)
    : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32) -> ()
```

The operation allows to represent all `parallel_for` variants, i.e. executing
over a range, a range with offset, and an nd-range. The arguments have the same
granularity and types as described in the `kernel_signatures` data structure in
the integration header. In particular, this means that accessors are "passed" as
a pointer to an instance of one of the `sycl::accessor` classes, rather than
being broken down into their parts. Accessor- and local accessor-arguments are
annotated with their respective SYCL dialect type to preserve high-level
information such as the access mode. Other *standard layout* objects such as
generic structs and scalars are captured after reverting any copy optimizations
done by the frontend (see [below](#captured-values)).

At this point, the handler operand is merely an implementation detail which
prevents the operation from being considered trivially dead without interfering
with the dataflow analyses defined in context of the SYCL dialect. Also note
that we do not model events or dependencies between different kernel launches
yet — a future extension of the operation may have a result and operands
representing events.

## Raising approach

Lots of high-level structure and information is lost during the lowering of the
object-oriented SYCL source code to LLVM-IR. In order to recover the information
required for the `schedule_kernel` operation, we employ a multi-stage pattern
matching approach that is aided by source code annotations in the SYCL headers.
The following subsections describe the intermediate steps in detail. The
implementation resides in
`polygeist/lib/Dialect/Polygeist/Transforms/SYCLRaiseHost.cpp`. Note that
example MLIR snippets have been manually edited to use descriptive names for
values and other identifiers.

### Command-group handler

All raising patterns operate on the CGF itself or annotation known to reside in
the CGF after mandatory inlining (see
[above](#prerequisites--known-limitations)). Hence, the handler is the first
argument (after the `this` pointer) of the CGF.

### Host constructors

We raise calls/invokes to specific constructors of SYCL API classes (`buffer`,
`accessor`, `range`, etc.) to the
[`sycl.host.constructor`](../../../build/docs/Dialects/SYCLOps.md#syclhostconstructor-syclsyclhostconstructorop)
op, which ties together arguments for construction and the SYCL dialect type of
the entity being initialized.

For example, the instantiation of an accessor in the CGF results in the
following raised IR:
```mlir
%accessor = llvm.alloca %c1 x !llvm.struct<"class.sycl::_V1::accessor", // ...
...
sycl.host.constructor(%accessor, %buffer, %handler, %propertylist, %codeloc)
  {type = !sycl_accessor_1_21llvm2Evoid_r_gb}
    : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
```

The host-side constructor operation differs from the `sycl.constructor`
operation used in device code in the following aspects:
- `sycl.host.constructor` does *not* include the allocation of the object.
  Hence, a common pattern in the host raising process is looking for a
  `host.constructor` op in the users of a particular `llvm.alloca` operation.
- As SYCL objects are passed around as `!llvm.ptr` values in the host IR
  (instead of `memref`s of SYCL dialect types), the operation carries a type
  attribute to encode additional high-level information (see below) about the
  constructed SYCL entities.

We currently extract the following information and encode it in the type
attribute:
- `sycl::buffer`: dimensions, sub-buffer flag
- `sycl::accessor`: dimensions, access mode, access target. The `!sycl.accessor`
  type's body type list can be either `!llvm.void`, `!sycl.range` or
  `!sycl.range`+`!sycl.id`, and encodes whether the accessor is constructed with
  a range, or range and offset.
- `sycl::range`, `sycl::id`, `sycl::nd_range`: dimensions

As a general rule, we do not attempt to extract the element type of
buffers/accessors from the mangled constructor names, as these types are not
relevant to the host raising process.

Note that the various raising patterns for the constructor operation delete the
original `call` or `invoke` to the constructor. In the latter case, an
unconditional branch to the non-exceptional successor block is introduced.

### Range information

We use the Clang-specific C++ attribute `annotate` to mark up the range-related
arguments in the different overloads of `sycl::handler::parallel_for`, e.g.:

```c++
#define __SYCL_ANNOTATE(str) __attribute__((annotate(#str)))
...
void parallel_for(range<Dims> NumWorkItems __SYCL_ANNOTATE(range),
                  id<Dims> WorkItemOffset __SYCL_ANNOTATE(offset),
                  _KERNELFUNCPARAM(KernelFunc) __SYCL_ANNOTATE(kernel)) { ... }
```

This results in the following MLIR for the CGF:

```mlir
%range_tag = llvm.mlir.addressof @".str.range" : !llvm.ptr   // -> "range"
%offset_tag = llvm.mlir.addressof @".str.offset" : !llvm.ptr // -> "offset"
...
"llvm.intr.var.annotation"(%range, %range_tag, %sourcefilestr, %line_0, %nullptr) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> ()
"llvm.intr.var.annotation"(%offset, %offset_tag, %sourcefilestr, %line_1, %nullptr) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> ()
```

The `RaiseSetNDRange` pattern scans the CGF for these annotations and inserts a
[`sycl.host.handler.set_nd_range`](../../../build/docs/Dialects/SYCLOps.md#syclhosthandlerset_nd_range-syclsyclhosthandlersetndrange)
operation with the marked up pointer values after the last annotation op:

```mlir
sycl.host.handler.set_nd_range %handler -> range %range, offset %offset : (!llvm.ptr, !llvm.ptr, !llvm.ptr)
```

The `parallel_for` variants with just a simple range or an nd-range are raised
analogously.

### Captured values

In the SYCL source code, the variables captured by value in the lambda function
passed to `parallel_for` represent the kernel's arguments. We defined the
[`sycl.host.set_captured`](../../../build/docs/Dialects/SYCLOps.md#syclhostset_captured-syclsyclhostsetcaptured)
op to model the *i*th variable being captured in the CGF.

For example:

```c++
sycl::queue q;
{
  sycl::buffer<float> buf(/*...*/);
  int some_int = /*...*/;
  q.submit([&](sycl::handler &cgh) {
    sycl::accessor acc(buf, cgh, sycl::write_only);
    cgh.parallel_for<KernelName>(sycl::range<1>{N}, [=](sycl::id<1> i) {
      acc[i] = some_int;
    });
  });
}
```

... is represented during the raising process as:

```mlir
sycl.host.set_captured %lambdaObj[0] = %acc_alloca
  : !llvm.ptr, !llvm.ptr (!sycl_accessor_1_21llvm2Evoid_w_gb)
sycl.host.set_captured %lambdaObj[1] = %some_int : !llvm.ptr, i32
```

During the raising from LLVM-IR, instances of SYCL API classes are represented
as opaque LLVM pointers. However, the `set_captured` op optionally stores a type
attribute with the corresponding SYCL dialect type (e.g. `!sycl.accessor<...>`
as shown in the example).

The `RaiseSetCaptured` pattern relies on an annotation in the SYCL headers to
identify the lambda object in the host IR.

```c++
void parallel_for(/* range, offset or nd-range */,
                  _KERNELFUNCPARAM(KernelFunc) __SYCL_ANNOTATE(kernel)) { ... }
```

In the IR, the lambda object is an `!llvm.struct` whose elements correspond to
the types of the captured variables. We then look for assignments to the
top-level elements in the (potentially nested) struct. As this code is generated
by the compiler frontend, it is idiomatic enough to match for stores and
`memcpy`s to addresses computed by GEP instructions into the lambda object.

#### Heuristic adaption of values

Unfortunately, there is not always a 1:1 correspondence between assignments and
captured values. To that end, we apply a set of heuristics to determine the
actual captured object.

- **Accessors:** The frontend generates a series of stores and `memcpy`s when
  assigning an accessors (or local accessor), causing us to match only the first
  member. In order to find a pointer to the entire object, we look through the
  load of the first member, which should point to an allocation of an
  `!llvm.struct` representing the `sycl::accessor` class. If we find the
  corresponding `sycl.host.constructor` op as well, we capture the allocation
  instead and forward the constructor op's type attribute.

- **Coalesced scalars:** We have observed that the frontend sometimes coalesces
  sequences of scalar arguments of the same type into vectors, to assign
  multiple values at once. As this leads to mismatches between captured and
  expected kernel arguments, we reverse this transformation and instead mark the
  individual, extracted elements of the vector as captured.

- **Split arrays:** Similarly, the frontend sometimes chooses to pass arrays as
  a sequence of vectors and scalars, e.g. a 9-element arrays as two 4-element
  vectors and a scalar. From a set of candidate assignments, we define either a
  new vector-typed global if all elements are constant, or else reconstruct a
  vector value on the fly that comprises all previously captured parts.

### Assignment of kernel to handler

We are raising the assignment of `sycl::handler::MKernelName` in the
`sycl::handler::StoreLambda` method (called from `parallel_for`) to the
[`sycl.host.handler.set_kernel`](../../../build/docs/Dialects/SYCLOps.md#syclhosthandlerset_kernel-syclsyclhosthandlersetkernel)
operation. The rationale is that this assignment can be pattern-matched
reliably, links the CGF to the actual kernel function, and marks a point in the
submission process where it is guaranteed that range and argument information is
available.

The `set_kernel`'s symbol reference points to a `gpu.func` with the `kernel`
attribute, e.g.:

```mlir
sycl.host.handler.set_kernel %handler -> @device_functions::@_ZTS10KernelName : !llvm.ptr
```

As an intermediate step to raising `set_kernel`, the SYCL dialect also defines
the
[`sycl.host.get_kernel`](../../../build/docs/Dialects/SYCLOps.md#syclhostget_kernel-syclsyclhostgetkernelop)
operation. It replaces `llvm.mlir.addressof` operations of string constants
containing a kernel function name, and introduces an actual symbol reference to
the corresponding `gpu.func`.

### Raising `schedule_kernel`

With the information associated with the `set_nd_range`, `set_captured` and
`set_kernel` operations, we can finally raise the `schedule_kernel` operation.

The actual submission of the kernel to the scheduler is not part of the CGF,
hence we anchor the `RaiseScheduleKernel` pattern at the previously raised
`set_kernel` operation instead. We scan the CGF for `set_nd_range` and
`set_captured` ops. As we do not support `single_task` launches yet, we expect
to find exactly one `set_nd_range` operation.

To ensure that complete and type-correct set of `set_captured` ops has been
found, we interpret the `kernel_signatures` data structure which is emitted by
the device compiler into the integration header. To find corresponding kernel
parameter descriptors, we annotate additional variables in the handler class'
`StoreLambda` method: 

```c++
// Excerpt from `sycl::handler::StoreLambda`
...
auto numParams __SYCL_ANNOTATE(kernel_num_params) = KI::getNumParams();
auto *paramDesc __SYCL_ANNOTATE(kernel_param_desc) = &KI::getParamDesc(0);
extractArgsAndReqsFromLambda(reinterpret_cast<char *>(KernelPtr),
                             numParams, paramDesc, KI::isESIMD());
MKernelName = KI::getName(); // raised to `sycl.host.handler.set_kernel`
...
```
In the IR, we expect to find the resulting annotation in the predecessor of the
block containing the `set_kernel` op (due to the in between `llvm.invoke` of
`extractArgsAndReqsFromLambda`).

#### What to do when raising fails

- If no `set_kernel` op is present in the IR, check whether not enough or too
  much inlining has happened. The raising approach is brittle and may be
  affected even by minor changes to the IR structure.
- If a `set_kernel` op is present, but has not been raised further to
  `schedule_kernel`, check that there is a unique `set_nd_range` in the same
  function. More likely though, the consistency check regarding the
  `set_captured` ops has failed:
  - Is there a continious sequence of `set_captured` ops present in the same
    function?
  - Are accessor arguments captured with a SYCL-type attribute? If not, raising
    of the `host.constructor` ops may have failed or happened after application
    of the `RaiseSetCaptured` pattern, or the constructor op is in a different
    function.
