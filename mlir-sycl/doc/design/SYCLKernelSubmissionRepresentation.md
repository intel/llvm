# Representation of SYCL kernel submission to scheduler

## Introduction

A SYCL command-group function (CGF) submits a kernel function to the runtime's
scheduler to execute over a given range and with a given set of arguments.
Capturing this contextual information is essential for host-device
optimizations, such as constant propagation and simplification of accessors.

In this document, we introduce the `sycl.host.schedule_kernel` operation, and
describe its multi-stage raising process from LLVM-IR involving additional
intermediate operations.

## Prerequisites / known limitations

- Our approach is currently limited to kernels passed as a lambda function to an
overload of `sycl::handler::parallel_for`.
-  We rely on
`sycl::handler::StoreLambda` being inlined into `sycl::handler::parallel_for`,
and the latter being inlined into the CGF.
- The auto-generated *rounded range kernel* should be deactivated with `-Xclang
  -fsycl-disable-range-rounding`.

## The `sycl.host.schedule_kernel` operation

The [`schedule_kernel`](#syclhostschedule_kernel-syclsyclhostschedulekernel)
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
[`sycl.host.constructor`](#syclhostconstructor-syclsyclhostconstructorop) op,
which ties together arguments for construction and the SYCL dialect type of the
entity being initialized.

For example, the instantiation of an accessor in the CGF results in the
following raised IR:
```mlir
%accessor = llvm.alloca %c1 x !llvm.struct<"class.sycl::_V1::accessor", // ...
...
sycl.host.constructor(%accessor, %buffer, %handler, %propertylist, %codeloc)
  {type = !sycl_accessor_1_21llvm2Evoid_r_gb}
    : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
```

Note that in contrast to the constructor operations
intended for the modeling of device code, the host-side constructor operation
does *not* include the allocation of the object. Hence, a common pattern in the
host raising process is looking for a `host.constructor` op in the users of a
particular `llvm.alloca` operation.

> TODO: Elaborate here, or in a separate document?

### Range information

We use the undocumented Clang attribute `annotate` to mark up the range-related
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
[`sycl.host.handler.set_nd_range`](#syclhosthandlerset_nd_range-syclsyclhosthandlersetndrange)
operation with the marked up pointer values after the last annotation op:

```mlir
sycl.host.handler.set_nd_range %handler -> range %range, offset %offset : (!llvm.ptr, !llvm.ptr, !llvm.ptr)
```

The `parallel_for` variants with just a simple range or an nd-range are raised
analogously.

### Captured values

In the SYCL source code, the variables captured by value in the lambda function
passed to `parallel_for` represent the kernel's arguments. We defined the
[`sycl.host.set_captured`](#syclhostset_captured-syclsyclhostsetcaptured) op to
model the *i*th variable being captured in the CGF.

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

The basic idea in the `RaiseSetCaptured` pattern is to detect the captured
values is to annotate the lambda object in the SYCL headers.

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
[`sycl.host.handler.set_kernel`](#syclhosthandlerset_kernel-syclsyclhosthandlersetkernel)
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
the [`sycl.host.get_kernel`](#syclhostget_kernel-syclsyclhostgetkernelop)
operation, which replaces `llvm.mlir.addressof` operations that reference a
`gpu.func` with the `kernel` attribute.

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
block containing the `set_kernel` op (due to the inbetween `llvm.invoke` of
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

## Appendix: Excerpt from ODS documentation
### `sycl.host.constructor` (sycl::SYCLHostConstructorOp)

_Represents the member initialization of a SYCL object._


Syntax:

```
operation ::= `sycl.host.constructor` `(` operands `)` attr-dict `:` functional-type(operands, results)
```

This operation differs from `sycl.constructor` as it will take a
`llvm.ptr` to any type instead of requiring a memref of a `sycl` type.

This difference is the reason why this operation was introduced in
the first place: this is a short-term solution to represent construction of
a SYCL object in the host side. Using a `sycl.constructor` operation would
imply performing heavy modifications to the host LLVM code (or blurring the
semantics of the `sycl.constructor` operation).

Note that, despite the more relaxed typing, the `type` attribute still needs
to be a type in the `sycl` dialect.

Interfaces: MemoryEffectsOpInterface

#### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
| `type` | ::mlir::TypeAttr | An Attribute containing a Type

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `dst` | LLVM pointer type
| `args` | any type

### `sycl.host.get_kernel` (sycl::SYCLHostGetKernelOp)

_Defines a reference to a SYCL kernel, i.e., a `gpu.func` with `kernel` attribute._


Syntax:

```
operation ::= `sycl.host.get_kernel` $kernel_name attr-dict `:` type($res)
```


Traits: AlwaysSpeculatableImplTrait

Interfaces: ConditionallySpeculatable, NoMemoryEffect (MemoryEffectOpInterface), SymbolUserOpInterface

Effects: MemoryEffects::Effect{}

#### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
| `kernel_name` | ::mlir::SymbolRefAttr | symbol reference attribute

#### Results:

| Result | Description |
| :----: | ----------- |
| `res` | LLVM pointer type

### `sycl.host.handler.set_kernel` (sycl::SYCLHostHandlerSetKernel)

_Assigns a kernel to a `sycl::handler`, thus pairing the handler and the kernel being launched._


Syntax:

```
operation ::= `sycl.host.handler.set_kernel` $handler `->` $kernel_name attr-dict `:` type($handler)
```


Traits: SYCLHostHandlerOp

Interfaces: SymbolUserOpInterface

#### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
| `kernel_name` | ::mlir::SymbolRefAttr | symbol reference attribute

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `handler` | LLVM pointer type

### `sycl.host.handler.set_nd_range` (sycl::SYCLHostHandlerSetNDRange)

_Assigns an nd-range to a `sycl::handler`, setting the nd-range of the kernel to be launched._


Syntax:

```
operation ::= `sycl.host.handler.set_nd_range` $handler `->` (`nd_range` $nd_range^):(`range`)? $range
              (`,` `offset` $offset^)? attr-dict `:` type(operands)
```

The `range` operand expects an `nd_range` pointer or a `range` pointer. In
the latter case, an `id` pointer can be optionally given as the `offset`. In
the former case, the `nd_range` attribute must be set.

Traits: SYCLHostHandlerOp

#### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
| `nd_range` | ::mlir::UnitAttr | unit attribute

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `handler` | LLVM pointer type
| `range` | LLVM pointer type
| `offset` | LLVM pointer type

### `sycl.host.schedule_kernel` (sycl::SYCLHostScheduleKernel)

_Schedules a SYCL kernel launch._


Syntax:

```
operation ::= `sycl.host.schedule_kernel` $handler `->` $kernel_name
              (`[` (`nd_range` $nd_range^):(```range`)? $range^ (`,` `offset` $offset^)? `]`)? ``
              custom<ArgsWithSYCLTypes>($args, $sycl_types)
              attr-dict `:` functional-type(operands, results)
```

This operation represents the scheduling of a launch of the given kernel
function with the specified handler, range and kernel arguments. Its purpose
is to collect the information surfaced by the host-raising process about a 
particular launch, thereby providing an entry-point for host-device analyses
and optimizations.

If the `range` operand is not present, this operation represents a
*single_task* launch, otherwise a *parallel_for* invocation. The `range`
operand expects an `nd_range` pointer or a `range` pointer. In the latter
case, an `id` pointer can be optionally given as the `offset`. In the former
case, the `nd_range` attribute must be set.

Pointer arguments can be annotated with the SYCL type of the entity they
refer to, e.g. an accessor. Internally, this information is stored in the
`sycl_types` type array attribute, sized to match the number of arguments.
If no such type annotation is available for an argument, the `None` type
shall be used as a placeholder.

Note that due to the current focus on host-device optimizations for
individual launches, this operation currently does not model the queue or
the command-group handler associated with the launch, nor does it observe or
yield any events.

Example:
```
sycl.host.schedule_kernel %handler -> @kernels::@k0[range %1]
  (%2: !sycl_accessor_1_f32_rw_gb, %3)
  : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32) -> ()
```

Traits: AttrSizedOperandSegments

Interfaces: MemoryEffectOpInterface, SymbolUserOpInterface

#### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
| `kernel_name` | ::mlir::SymbolRefAttr | symbol reference attribute
| `sycl_types` | ::mlir::ArrayAttr | type array attribute
| `nd_range` | ::mlir::UnitAttr | unit attribute

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `handler` | LLVM pointer type
| `range` | LLVM pointer type
| `offset` | LLVM pointer type
| `args` | any type

### `sycl.host.set_captured` (sycl::SYCLHostSetCaptured)

_Marks a value as captured by a kernel function object._


Syntax:

```
operation ::= `sycl.host.set_captured` $lambda `[` $index `]` `=` $value attr-dict
              `:` type(operands) (` ` `(` $sycl_type^ `)`)?
```

This operation represents that the given `value` was captured in the kernel
function object `lambda` at index `index`. If a special SYCL entity is
captured (e.g. an acccessor), its type is stored in the `sycl_type`
attribute. The op is created during the progressive raising towards the
`sycl.host.schedule_kernel` op.

Example:
```
sycl.host.set_captured %lambda[1] = %scalar : !llvm.ptr, !llvm.ptr, i32
sycl.host.set_captured %lambda[2] = %accessor 
  : !llvm.ptr, !llvm.ptr, !llvm.ptr (!sycl_accessor_1_f32_rw_gb)
```

#### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
| `index` | ::mlir::IntegerAttr | 64-bit signless integer attribute
| `sycl_type` | ::mlir::TypeAttr | any type attribute

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `lambda` | LLVM pointer type
| `value` | any type