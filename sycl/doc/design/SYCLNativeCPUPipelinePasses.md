# SYCL Native CPU Pipeline Passes

The `compiler::utils` module exists under
[compiler_pipeline](https://github.com/intel/llvm/tree/sycl/llvm/lib/SYCLNativeCPUUtils/compiler_passes/compiler_pipeline)
and provides a number of utility functions and LLVM passes inside the
`compiler::utils` namespace, along with `metadata` and Function attributes.
These utility passes are currently only being used by `Native CPU`. These
utilities were originally under the [oneAPI Construction
Kit](https://github.com/uxlfoundation/oneapi-construction-kit/tree/main).

## TransferKernelMetadataPass and EncodeKernelMetadataPass

These passes are responsible for setting up metadata on kernels under
compilation. Many other passes implicitly rely on the metadata and
attributes added by these passes, so it is recommended to run them
first, if possible.

The difference between the two passes concerns the list of kernels it
runs over:

-   The `TransferKernelMetadataPass` runs over multiple kernels in the
    module, using `!opencl.kernels` metadata if present, else all
    functions in the module with the `llvm::CallingConv::SPIR_KERNEL`
    calling convention.
-   The `EncodeKernelMetadataPass` runs on *one* kernel, supplied by
    name to the pass upon construction.

Their job is three-fold:

-   To add [mux-kernel](#mux-kernel-attribute) entry-point function attributes
    to the kernels covered by each pass.

-   To add [!reqd_work_group_size](#metadata) function metadata if not already
    attached. It sets this information based on local work-group size
    information which is:

    > -   (`TransferKernelMetadataPass`) - taken from the kernel's
    >     entry in the `!opencl.kernels` module-level metadata.
    > -   (`EncodeKernelMetadataPass`) - optionally passed to the pass
    >     on construction. The local sizes passed to the pass should
    >     either be empty or correspond 1:1 with the list of kernels
    >     provided.

-   To add [mux-work-item-order](#function-attributes) work-item order function attributes. It uses
    optional data supplied to either pass on construction to encode this
    metadata. If not set, the default `xyz` order is used.

## WorkItemLoopsPass

The `WorkItemLoopsPass` is responsible for adding explicit parallelism
to implicitly parallel SIMT kernels. It does so by wrapping each kernel
up in a triple-nested loop over all work-items in the work-group. Thus,
kernels scheduled by this pass can be invoked once per work-group.

The order in which work-items are executed is fairly flexible, but generally in
ascending order from \[0\] to \[N-1\] through the innermost \[X\] dimension, followed
by the \[Y\] dimension, and lastly the \[Z\] dimension.

Conceptually, the pass transforms `old_kernel` into `new_kernel` in the
example below:

```cpp
void old_kernel(int *in, int *out) {
  size_t id = get_local_linear_id(0);
  out[id] = in[id] * 4;
}

void new_kernel(int *in, int *out) {
  for (size_t z = 0, sizeZ = get_local_size(2); z != sizeZ; z++) {
    for (size_t y = 0, sizeY = get_local_size(1); y != sizeY; y++) {
      for (size_t x = 0, sizeX = get_local_size(0); x != sizeX; x++) {
        size_t id = (z * sizeY * sizeX) + (y * sizeX) + x;
        out[id] = in[id] * 4;
      }
    }
  }
}
```

To satisfy the programming model, the pass must be careful around
control barriers and *barrier-like* functions. The `WorkItemLoopsPass`
splits a kernel into separately executing kernel functions using barrier
calls as boundaries. Each section of the kernel split by these barriers
is known as a *barrier region*.

```cpp
void old_kernel(int *in, int *out) {
  size_t id = get_local_linear_id(0);
  out[id * 2] = in[id];
  // All work-items in the work-group must encounter the barrier before any
  // are allowed to continue execution beyond the barrier.
  work_group_barrier(CLK_GLOBAL_MEM_FENCE);
  out[id * 2 + 1] = in[id] * 4;
}

void new_kernel(int *in, int *out) {
  // Barrier region #0
  for (size_t z = 0, sizeZ = get_local_size(2); z != sizeZ; z++) {
    for (size_t y = 0, sizeY = get_local_size(1); y != sizeY; y++) {
      for (size_t x = 0, sizeX = get_local_size(0); x != sizeX; x++) {
        size_t id = (z * sizeY * sizeX) + (y * sizeX) + x;
        out[id * 2] = in[id];
      }
    }
  }

  // The control aspect of the barrier has been satisfied by the loops, so
  // it has been decomposed to just a memory barrier.
  mem_fence(CLK_GLOBAL_MEM_FENCE);

  // Barrier region #1
  for (size_t z = 0, sizeZ = get_local_size(2); z != sizeZ; z++) {
    for (size_t y = 0, sizeY = get_local_size(1); y != sizeY; y++) {
      for (size_t x = 0, sizeX = get_local_size(0); x != sizeX; x++) {
        size_t id = (z * sizeY * sizeX) + (y * sizeX) + x;
        out[id * 2 + 1] = in[id] * 4;
      }
    }
  }
}
```

To propagate data dependencies between these *barrier regions*, an
analysis is performed to create a struct of live variables which is
passed as an argument to each kernel. Generated kernels then reference
this struct rather than the original values. A simplified example
follows:

```cpp
void old_kernel(int *in, int *out) {
  size_t id = get_local_linear_id(0);
  // X is a barrier-carried dependency: produced in one barrier region and
  // accessed in another.
  int x = in[id] * 4;
  // All work-items in the work-group must encounter the barrier before any
  // are allowed to continue execution beyond the barrier.
  work_group_barrier(CLK_GLOBAL_MEM_FENCE);
  // Use X, produced by the previous barrier region.
  out[id] = x;
}

void new_kernel(int *in, int *out) {
  struct kernel_live_vars {
    int x;
  };
  // Illustrative purposes: this is in reality a stack allocation.
  kernel_live_vars *live_vars =
      malloc(get_local_size(0) * get_local_size(1)
             * get_local_size(2) * sizeof(live_vars));

  for (size_t z = 0, sizeZ = get_local_size(2); z != sizeZ; z++) {
    for (size_t y = 0, sizeY = get_local_size(1); y != sizeY; y++) {
      for (size_t x = 0, sizeX = get_local_size(0); x != sizeX; x++) {
        size_t id = (z * sizeY * sizeX) + (y * sizeX) + x;
        live_vars[id] = in[id] * 4;
      }
    }
  }

  mem_fence(CLK_GLOBAL_MEM_FENCE);

  for (size_t z = 0, sizeZ = get_local_size(2); z != sizeZ; z++) {
    for (size_t y = 0, sizeY = get_local_size(1); y != sizeY; y++) {
      for (size_t x = 0, sizeX = get_local_size(0); x != sizeX; x++) {
        size_t id = (z * sizeY * sizeX) + (y * sizeX) + x;
        out[id] = live_vars[id];
      }
    }
  }
}
```

The loop that reconstructs the kernels in the wrapper function uses the
vectorization dimension as innermost cycle, and it relies on
[mux-work-item-order](#function-attributes) function attributes for the
outermost loops.

Preserving debug info is a problem for the `WorkItemLoopsPass` due to
live variables getting stored in a struct passed as an argument to each
of the generated kernels. As a result the memory locations pointed to by
the debug info are out of date with respect to newly written values. By
specifying the `IsDebug` flag when creating the pass we can resolve this
problem at the expense of performance.

When the `IsDebug` flag is set the pass adds a new `alloca` which
contains a pointer to the live variables struct of the currently
executing work-item, since there is a separate struct for each work-item
in a work-group. A new `store` instruction to this `alloca` is also
inserted before calls to each of the separated kernels with the new
address of the live variables struct for the work-item about to be
executed. These extra writes to the stack have a runtime cost which is
why this transformation is only done when compiling for debug.

The benefit of adding the extra `alloca` is that it forces the address
to be placed on the stack, where we can point to it with
`llvm.dbg.declare()` intrinsics, rather than reading the address from a
register where it won't persist. Not all source variables are classed
as live however if they are not used past the first barrier, so when the
`IsDebug` flag is set we also modify the algorithm for finding live
variables to mark these `alloca` instructions as live. Otherwise their
values won't be updated for the current work item past the first
barrier and the debugger will print incorrect values.

To point to the location in the live variables struct where each source
variable lives we use DWARF expressions, represented in LLVM by a
`DIExpression` metadata node. In our expression we first use a
`DW_OP_deref` DWARF operation to dereference the pointer in our
debugging `alloca` to find the start of the live variables struct. Then
next in the expression we have a `DW_OP_plus` operation taking an
integer operand for the byte offset into the struct for that particular
variable.

In order to establish which values actually cross a barrier, we traverse
the CFG and build inter-barrier regions. We start traversal at the
beginning of the function, and at the barriers, and we end whenever we
encounter another barrier or a return statement. We collect all values
that are defined within one region, which have uses in any other region,
which are called "external uses". We also collect values that are
defined within one region and used in the same region, but where the
definition does not dominate the use. These are "internal uses" and
can occur where a barrier is present in a loop, such that the same
barrier that begins the inter-barrier region can also be hit at the end
of that region. (The definition must have dominated all its uses in the
original function, but a barrier inside a loop can result in the second
part of the loop body preceding the first within the inter-barrier
region.)

We also implement a "Barrier Tidying" optimization that
posts-processes the set of live values to remove certain values where it
is expected that loading and storing these values will incur more
overhead than simply recalculating them from other available values
(including other barrier-stored values and kernel parameters). Values
considered removable are:

> -   NOP casts,
> -   Casts from a narrow type to a wider type,
> -   All other casts where the source operand is already in the
>     barrier,
> -   Vector splats,
> -   Calls to "rematerializable" builtins - see
>     `compiler::utils::eBuiltinPropertyRematerializable`

If the barrier contains scalable vectors, the size of the struct is
dependent on the value of `vscale`, and so is the total number of struct
instances for a given work group size. In this case we create the
barrier memory area as a byte buffer (i.e. an array of `i8`), instead of
an array of barrier structs. The address of the barrier struct for the
subkernel invocations have to be computed knowing the vscale, and
pointer-cast to the barrier struct type. Any scalable vector members of
the barrier struct are put into a flexible array member (of type `i8`)
at the end, so that GEPs to individual members can be constructed by
calculating their byte offsets into this array and the results cast to
pointers of the needed type. The position of individual scalable vector
members is calculated by multiplying their equivalent "fixed width"
offset (i.e. the same as if vscale were equal to 1) by the actual
vscale.

Once we know which values are to be included in the barrier struct, we
can split the kernel proper, creating a new function for each of the
inter-barrier regions, cloning the Basic Blocks of the original function
into it. We apply the barrier in the following order: external uses are
remapped into loads from the barrier struct, then any barrier-resident
values are stored into the barrier, and finally, internal uses are
remapped into the barrier. External and internal uses are dealt with
separately, since external uses can always be safely loaded only once at
the beginning of the new function, where as internal uses may or may not
need to load the loop-updated value. For this reason, stores are always
created immediately after the definitions of the relevant values, rather
than at the barrier at the end of the region. (This may have some scope
for further optimization work.) When tidying has removed a value from
the barrier, we have to also clone those values as well, in order to
re-compute these values from the value actually stored in the barrier
struct. Each subkernel returns an integer ID that maps to the barriers,
corresponding to the barrier that was encountered at the end of the
subkernel. There is a special barrier ID that represents the return
statement of the original kernel, and also one that represents the
kernel entry point.

This pass runs over all functions in the module which have [mux-kernel](#function-attributes) entry-point attributes.

The new wrappers take the name of either the 'tail' or 'main'
kernels - whichever is present - suffixed by
".mux-barrier-wrapper". The wrappers call either the original
kernel(s) if no barriers are present, or the newly-created barrier
regions if barriers are present. The original kernels are left in the
module in either case but are marked as internal so that later passes
can optimize them if they are no longer called once inlined.

Newly-created functions preserve the original calling convention, unless they
are kernels. In that case, the new functions will have `SPIR_FUNC` calling
convention. Newly-created functions steal the [mux-kernel](#function-attributes)
attributes from the original functions.

Once we have all of our subkernels, we apply the 3-dimensional work item
loops individually to each subkernel. The return value of a subkernel is
used to determine which subkernel loop to branch to next, or to exit the
wrapper function, as appropriate.

### Work-group scheduling (vectorized and scalar loops)

The [WorkItemLoopsPass](#workitemloopspass) is responsible for stitching
together multiple kernels to make a single kernel capable of correctly
executing all work-items in the work-group.

In particular, when a kernel has been vectorized with
[Vecz](SYCLNativeCPUVecz.md) it executes multiple work-items at
once. Unless the work-group size in the vectorized dimension is known to
be a multiple of the vectorization factor, there exists the possibility
that some work-items will not be executed by the vectorized loop.

As such, the [WorkItemLoopsPass](#workitemloopspass) is able to stitch
together kernels in several different configurations:

-   Vector + scalar loop
-   Vector loop + vector-predicated tail
-   Vector loop only
-   Scalar loop only

#### Vector + Scalar

The vector + scalar kernel combination is considered the default
behaviour. Most often the work-group size is unknown at compile time and
thus it must be assumed that the vector loop may not execute all
work-items.

This configuration is used if the [WorkItemLoopsPass](#workitemloopspass) is
asked to run on a vectorized function which has
[!codeplay_ca_vecz.derived](#metadata) function metadata linking it back to its
 scalar progenitor. In this case, both the vector and scalar kernel functions
are identified and are used. The vector work-items are executed first, followed
by the scalar work-items.

```cpp
const size_t peel = group_size_x % vec_width;
const size_t peel_limit = group_size_x - peel;

if (group_size_x >= vector_width) {
  for (size_t z = 0; z < group_size_z; ++z) {
    for (size_t y = 0; y < group_size_y; ++y) {
      for (size_t wi = 0; wi < peel_limit; wi += vec_width) {
        // run vectorized kernel if vec_width > 1,
        // otherwise the scalar kernel.
      }
    }
  }
}
if (group_size_x < vector_width || group_size_x % vector_width != 0) {
  for (size_t z = 0; z < group_size_z; ++z) {
    for (size_t y = 0; y < group_size_y; ++y) {
      // peeled loop running remaining work-items (if any) on the scalar
      // kernel
      for (size_t wi = peel_limit; wi < group_size_x; ++wi) {
        // run scalar kernel
      }
    }
  }
}
```

Barriers are supported in this mode by creating a separate barrier
struct for both the vector and scalar versions of the kernel.

There are circumstances in which this mode is skipped in favour of
"vector only" mode:

-   If the local work-group size is known to be a multiple of the
    vectorization factor.

    > -   This is identified through the [!reqd_work_group_size](#metadata) function metadata. This is often automatically
    >     added to functions by compiler frontends if kernels are
    >     supplied with attributes (e.g., `reqd_work_group_size` in
    >     OpenCL). Alternatively, if the work-group size is known at
    >     compile time, use the
    >     [ TransferKernelMetadataPass or EncodeKernelMetadataPass ](#transferkernelmetadatapass-and-encodekernelmetadatapass)
    > to encode functions with this information.

-   If the [WorkItemLoopsPass](#workitemloopspass) has been created with
    the `ForceNoTail` option.

    -   This is a global toggle for *all* kernels in the program.

-   If the kernel has been vectorized with vector predication. In this
    case the vector loop is known to handle scalar iterations itself.

If any of these conditions are true, the "vector only" mode is used.

#### Vector + Vector-predicated

The vector + vector-predicated kernel combination is a special case
optimization of the default behaviour.

If the pass detects both a vector and vector-predicated kernel linked to
the same original kernel with the same vectorization width, the scalar
tail loop is replaced with a straight-line call to the vector-predicated
kernel, which will perform all of the scalar iterations at once.

```cpp
const size_t peel = group_size_x % vec_width;
const size_t peel_limit = group_size_x - peel;

if (group_size_x >= vector_width) {
  for (size_t z = 0; z < group_size_z; ++z) {
    for (size_t y = 0; y < group_size_y; ++y) {
      for (size_t wi = 0; wi < peel_limit; wi += vec_width) {
        // run vectorized kernel if vec_width > 1,
      }
      if (peel) {
        // run vector-predicated kernel
      }
    }
  }
}
```

#### Vector only

If the [WorkItemLoopsPass](#workitemloopspass) is run on a vectorized
kernel for which no [Vecz](SYCLNativeCPUVecz.md) linking metadata is found to
identify the scalar kernel, or if a scalar kernel is found but one of
the conditions listed above hold, then the kernel is emitted using the
vector kernel only. It is assumed that if no scalar kernel is found it
is because targets know that one is not required.

#### Scalar only

If the [WorkItemLoopsPass](#workitemloopspass) is run on a scalar kernel
then only the scalar kernel is used.

### OptimalBuiltinReplacementPass

The `OptimalBuiltinReplacementPass` is an optimization call-graph pass designed
to replace calls to builtin functions with optimal equivalents. This is only
used in the [veczc](SYCLNativeCPUVecz.md#veczc---the-vecz-compiler) tool and
should probably be phased out here.

The `OptimalBuiltinReplacementPass` iterates over the call graph from
kernels inwards to their called functions, and visits all call sites in
the caller functions. If a call is made to a function that the pass is
interested in, the call is deleted and is replaced with a series of
inline IR instructions. Using the call graph guarantees that
replacements are made on a priority basis; outermost functions are
replaced before any functions they themselves call.

Replacements are optionally made according to a specific `BuiltinInfo`
object, which may be passed to this pass. It defaults to `nullptr`. If
this `BuiltinInfo` is present then it is asked whether it recognizes any
builtin functions and is tasked with inlining a suitable sequence of
instructions.

Replacements are also performed on two abacus-internal builtins:
`__abacus_clz` and `__abacus_mul_hi`. Replacing these rather than their
OpenCL user-facing builtins allows replacements in more cases, as the
abacus versions are used to implement several other builtin functions.

The `__abacus_clz` builtin (count leading zeros) can be exchanged
for a hardware intrinsic: `llvm.ctlz`. However, some variants are
skipped: 64-bit scalar and vector variants are skipped, since Arm uses
calls to an external function to help it implement this case.

The `__abacus_mul_hi` builtin (multiplication returning the "high"
part of the product) can be exchanged for a shorter series of LLVM
instructions which perform the multiplication in a wider type before
shifting it down. This is desirable because abacus has a rule that it
never introduces larger types in its calculations. LLVM, however, is
able to match a specific sequence of instructions against a "mul hi"
node, which is canonical, well-optimized, and many targets directly
lower that node to a single instruction. 64-bit versions (scalar and
vector) are skipped since 64-bit "mul hi" and 128-bit integers are not
well supported on all targets.

The `__abacus_fmin` and `__abacus_fmax` builtins can be exchanged for
hardware intrinsics: `llvm.minnum` and `llvm.maxnum`. This is not
performed on ARM targets due to LLVM backend compiler bugs.

### RunVeczPass

The `RunVeczPass` module pass provides a wrapper for using our
[Vecz](SYCLNativeCPUVecz.md) IR vectorizer. This vectorizes the
kernel to a SIMD width specified when the pass is created. In our case
this is typically local size in the first dimension but there are other
factors to consider when picking the width, like being a power of 2.

We only enable the vectorizer in host when the `-cl-wfv={always|auto}`
option is provided, a condition check which is the first thing this pass
does. If this check fails, the pass exits early, otherwise the
vectorizer is invoked through top level API
`vecz::Vectorizer::vectorize`. If the passed option is `-cl-wfv=auto`,
then we first have to check the layout of the input kernel to find out
if it is advantageous to vectorize it, and only do so if it is the case.
If the passed option is `-cl-wfv=always`, then we will try to vectorize
the kernel in any case. If successful, this will return a new vectorized
kernel function created in the LLVM module so that this vectorized
kernel is used instead of our scalar kernel from here on.

#### Cost Model Interface

User cost-modelling in Vecz can be handled by the
`vecz::VeczPassOptionsAnalsis` which takes a user defined query function
on construction. This pass is a required analysis pass for Vecz, so be
sure to add it to your analysis manager.

Vecz queries the result of this analysis before operating on a kernel,
and the user function may fill an array of `VeczPassOptions` which
contain suitably modelled widths, vectorization factors, and scalability
options determined suitable for the target.

The `VeczPassOptionsAnalysis` pass can be default-constructed - in which
case Vecz makes a conservative decision about kernel vectorization - or
be constructed passing in a user callback function. The function takes
as its parameters a reference to the function to be optionally
vectorized, and a reference to a vector of `VeczPassOptions` which it is
expected to fill in.

If it's not interested in seeing the function vectorized, it returns
false; otherwise it fills in the `VeczPassOptions` array with the
choicest vectorization options it can muster for the target. For
example:

```cpp
void InitMyAnalysisManager(llvm::ModuleAnalysisManager &MAM) {
  MyCostModel CM;
  MAM.registerPass([CM] {
  return vecz::VeczPassOptionsAnalysis(
    [CM](llvm::Function &F,
       llvm::SmallVectorImpl<vecz::VeczPassOptions> &Opts) {
       if (CM->getCostWFV(&F) > 0) {
         // Vectorizing will make things worse, so don't
         return false;
       }
       VeczPassOptions O;
       vecz::VectorizationChoices &choices = O.choices;
       if (!MyCostModel->hasDoubles()) {
         choices.enable(eCababilityNoDoubleSupport);
       }
       if (CM->getCostPartialScalarization(&F) < 0) {
         choices.enable(vecz::VectorizationChoices::ePartialScalarization);
       }
       if (CM->getCostBOSCC(&F) < 0) {
         choices.enable(vecz::VectorizationChoices::eLinearizeBOSCC);
       }
      // Our silly target only has 42-wide SIMD units!
      opts.factor = Vectorization::getFixedWidth(42);
      Opts.emplace_back(std::move(O));
      return true;
    });
  });
}
```

To access the `VeczPassOptionsAnalysis` from inside any other pass in
the same pass manager, do the following:

```cpp
auto queryPassOpts = getAnalysis<vecz::VeczPassOptionsAnalysis>();
```

The above returns a pointer to the cost model the wrapper pass was
constructed with, and may return `nullptr` if no cost model was
provided.

The Cost Model header file resides at `utils/cost_model.h`.

### DefineMuxBuiltinsPass

The `DefineMuxBuiltinsPass` performs a scan over all functions in the
module, calling `BuiltinInfo::defineMuxBuiltin` on all mux builtin
function declarations.

If a definition of a mux builtin requires calls to other mux builtins
which themselves need defining, such dependencies can be added to the
end of the module's list of functions so that the
`DefineMuxBuiltinsPass` will visit those in turn. One example of this is
the lowering of `__mux_get_global_id` which calls `__mux_get_local_id`,
among other functions.

### ReplaceLocalModuleScopeVariablesPass

The `ReplaceLocalModuleScopeVariables` pass identifies global variables
in the local address space and places them in a struct called
`localVarTypes`, allocated in a newly created wrapper function. A
pointer to the struct is then passed via a parameter to the original
kernel. The wrapper function takes over function attributes and metadata
from the original kernel.

When creating the struct we need to be aware of the alignment of members
so that they are OpenCL conformant for their type. To do this we
manually pad the struct by keeping track of each elements offset and
adding byte array entries for padding to meet alignment requirements.
Finally the whole struct is aligned to the largest member alignment
found.

Once the struct is created the pass replaces all instructions using each
of the global variables identified in the previous step with
instructions referencing the matching struct member instead. Finally the
identified global variables are removed once all of their uses have been
replaced.

### PrepareBarriersPass

The `PrepareBarriersPass` is useful in order to satisfy the requirements
the [WorkItemLoopsPass](#workitemloopspass) has on kernels containing
barrier-like functions if running in conjunction with the
[RunVeczPass](#runveczpass). If running, it should be run before using
the vectorizer.

It ensures that barriers are synchronized between two or more vectorized
versions of the same kernel. It gives each barrier a unique ID, which
the vectorizer preserves in each vectorized kernel, meaning the
`WorkItemLoopsPass` can correctly schedule the work-item loops for each
barrier region.

### Metadata Utilities

There are several key pieces of metadata used for inter-communication
between the Native CPU passes.

In order to avoid hard-coding assumptions about the metadata's names,
number of operands, types of operands, etc., utility functions
**should** be used to access or manipulate the metadata. The specific
names and/or operands of these metadata is **not** guaranteed to be
stable.

### Attribute Utilities

There are several key attributes used for inter-communication between
the Native CPU passes.

The
`compiler_passes/compiler_pipeline/include/compiler/utils/attributes.h`
header contains all such APIs, several of which are given here by way of
example:

-   `void setIsKernel(llvm::Function &F)`
    -   Adds the `mux-kernel` attribute to function `F`.
-   `void setIsKernelEntryPt(llvm::Function &F)`
    -   Adds `"mux-kernel"="entry-point"` attribute to function `F`
-   `bool isKernel(const llvm::Function &F)`
    -   Returns true if function `F` has a `mux-kernel` attribute
-   `bool isKernelEntryPt(const llvm::Function &F)`
    -   Returns true if function `F` has a `mux-kernel` attribute with
        the value `"entry-point"`.
-   `void dropIsKernel(llvm::Function &F)`
    -   Drops the `mux-kernel` attribute from function `F`, if present.
-   `void takeIsKernel(llvm::Function &ToF, llvm::Function &FromF)`
    -   Transfers `mux-kernel` attributes from function `FromF` to
        function `ToF`, if present on the old function.
        Overwrites any such metadata in the new function.

### Sub-groups

A implementation of SPIR-V sub-group builtins is provided by the
default compiler pipeline.

The SPIR-V sub-group builtins are first translated into the
corresponding Native CPU builtin functions. These functions are
understood by the rest of the compiler and can be identified and
analyzed by the `BuiltinInfo` analysis.

A definition of these mux builtins for where the sub-group size is 1 is
provided by `BIMuxInfoConcept` used by the
[DefineMuxBuiltinsPass](#definemuxbuiltinspass).

Vectorized definitions of the various sub-group builtins are provided by
the Vecz pass, so any target running Vecz (and the above passes) will be
able to support sub-groups of a larger size than 1. Note that Vecz does
not currently interact "on top of" the mux builtins - it replaces them
in the functions it vectorized. This is future work to allow the two to
build on top of each other.

If a target wishes to provide their own sub-group implementation they
should provide a derived `BIMuxInfoConcept` and override
`defineMuxBuiltin` for the sub-group builtins.

### LLVM intermediate representation

#### Mangling

Mangling is used by the vectorizer to declare, define and use internal
overloaded builtin functions. In general, the mangling scheme follows [Appendix
A of the SPIR 1.2
specification](https://www.khronos.org/registry/SPIR/specs/spir_spec-1.2.pdf).
itself an extension of the Itanium C++ mangling scheme.

##### Vector Types

The Itanium specification under-specifies vector types in general, so vendors
are left to establish their own system. In the vectorizer, fixed-length vector
types follow the convention that LLVM, GCC, ICC and others use. The first
component is `Dv` followed by the number of elements in the vector, followed by
an underscore ( `_` ) and then the mangled element type:

``` llvm
   <2 x i32> -> Dv2_i
   <32 x double> -> Dv32_d
```

Scalable-vector IR types do not have an established convention. Certain vendors
such as ARM SVE2 provide scalable vector types at the C/C++ language level, but
those are mangled in a vendor-specific way.

The vectorizer chooses its own mangling scheme using the Itanium
vendor-extended type syntax, which is `u` , followed by the length of the
mangled type, then the mangled type itself.

Scalable-vectors are first mangled with `nx` to indicate the scalable
component. The next part is an integer describing the known multiple of the
scalable component. Lastly, the element type is mangled according to the
established vectorizer mangling scheme (i.e. Itanium).

Example:

``` llvm
   <vscale x 1 x i32>               -> u5nxv1j
   <vscale x 2 x float>             -> u5nxv2f
   <vscale x 16 x double>            -> u6nxv16d
   <vscale x 4 x i32 addrspace(1)*> -> u11nxv4PU3AS1j

   define void @__vecz_b_interleaved_storeV_Dv16_dPU3AS1d(<16 x double> %0, double addrspace(1)* %1, i64 %2) {
   define void @__vecz_b_interleaved_storeV_u6nxv16dPU3AS1d(<vscale x 16 x double> %0, double addrspace(1)* %1, i64 %2) {
```

#### Builtins

The Following intermediate representations are used in the interface to Native CPU. Some of these may not be relevant for Native CPU, and may exist from the time this was part of the `oneAPI Construction Kit`.

* `size_t __mux_get_global_size(i32 %i)` - Returns the number of global
  invocations for the `%i`'th dimension.
* `size_t __mux_get_global_id(i32 %i)` - Returns the unique global
  invocation identifier for the `%i`'th dimension.
* `size_t __mux_get_global_offset(i32 %i)` - Returns the global offset (in
  invocations) for the `%i`'th dimension.
* `size_t __mux_get_local_size(i32 %i)` - Returns the number of local
  invocations within a work-group for the `%i`'th dimension.
* `size_t __mux_get_local_id(i32 %i)` - Returns the unique local invocation
  identifier for the `%i`'th dimension.
* `i32 __mux_get_sub_group_id()` - Returns the sub-group ID.
* `size_t __mux_get_num_groups(i32 %i)` - Returns the number of work-groups
  for the `%i`'th dimension.
* `i32 __mux_get_num_sub_groups()` - Returns the number of sub-groups for
  the current work-group.
* `i32 __mux_get_max_sub_group_size()` - Returns the maximum sub-group size
  in the current kernel.
* `i32 __mux_get_sub_group_size()` - Returns the number of invocations in the
  sub-group.
* `i32 __mux_get_sub_group_local_id()` - Returns the unique invocation ID
  within the current sub-group.
* `size_t __mux_get_group_id(i32 %i)` - Returns the unique work-group
  identifier for the `%i`'th dimension.
* `i32 __mux_get_work_dim()` - Returns the number of dimensions in
  use.
* `__mux_dma_event_t __mux_dma_read_1D(ptr address_space(3) %dst,`
  `ptr address_space(1) %src, size_t %width, __mux_dma_event_t %event)` - DMA
  1D read from `%src` to `%dst` of `%width` bytes. May use `%event`
  from previous DMA call. Returns event used.
* `__mux_dma_event_t __mux_dma_read_2D(ptr address_space(3) %dst,`
  `ptr address_space(1) %src, size_t %width, size_t %dst_stride,`
  `size_t %src_stride, size_t %height __mux_dma_event_t %event)` - DMA 2D
  read from `%src` to `%dst` of `%width` bytes and `%height` rows, with
  `%dst_stride` bytes between dst rows and `%src_stride` bytes between src
  rows. May use `%event` from previous DMA call. Returns event used.
* `__mux_dma_event_t __mux_dma_read_3D(ptr address_space(3) %dst,`
  `ptr address_space(1) %src, size_t %width, size_t %dst_line_stride,`
  `size_t %src_line_stride, size_t %height, size_t %dst_plane_stride,`
  `size_t %src_plane_stride, size_t %depth, __mux_dma_event_t %event)` - DMA
  3D read from `%src` to `%dst` of `%width` bytes, `%height` rows, and
  `%depth` planes, with `%dst_line_stride` bytes between dst rows,
  `%src_line_stride` bytes between src rows, `%dst_plane_stride` bytes
  between dst planes, and `%src_plane_stride` between src planes. May use
  `%event` from previous DMA call. Returns event used.
* `__mux_dma_event_t __mux_dma_write_1D(ptr address_space(1) ptr %dst,`
  `ptr address_space(3) %src, size_t %width, __mux_dma_event_t %event)` - DMA
  1D write from `%src` to `%dst` of `%width` bytes. May use `%event`
  from previous DMA call. Returns event used.
* `__mux_dma_event_t __mux_dma_write_2D(ptr address_space(1) %dst,`
  `ptr address_space(1) %src, size_t %width, size_t %dst_stride,`
  `size_t %src_stride, size_t %height __mux_dma_event_t %event)` - DMA 2D
  write from `%src` to `%dst` of `%width` bytes and `%height` rows,
  with `%dst_stride` bytes between dst rows and `%src_stride` bytes between
  src rows. May use `%event` from previous DMA call. Returns event used.
* `__mux_dma_event_t __mux_dma_write_3D(ptr address_space(3) %dst,`
  `ptr address_space(1) %src, size_t %width, size_t %dst_line_stride,`
  `size_t %src_line_stride, size_t %height, size_t %dst_plane_stride,`
  `size_t %src_plane_stride, size_t %depth,
  `__mux_dma_event_t %event)` - DMA 3D write from `%src` to `%dst` of
  `%width` bytes, `%height` rows, and `%depth` planes, with
  `%dst_line_stride` bytes between dst rows, `%src_line_stride` bytes
  between src rows, `%dst_plane_stride` bytes between dst planes, and
  `src_plane_stride` between src planes. May use `%event` from previous DMA
  call. Returns event used.
* `void __mux_dma_wait(i32 %num_events, __mux_dma_event_t*)` - Wait on
  events initiated by a DMA read or write.
* `size_t __mux_get_global_linear_id()` - Returns a linear ID equivalent
  to `(__mux_get_global_id(2) - __mux_get_global_offset(2)) *`
  `__mux_get_global_size(1) * __mux_get_global_size(0) +`
  `(__mux_get_global_id(1) - __mux_get_global_offset(1)) *`
  `__mux_get_global_size(0) + (__mux_get_global_id(0) -`
  `__mux_get_global_offset(0))`.
* `size_t __mux_get_local_linear_id(void)` - Returns a linear ID equivalent
  to `__mux_get_local_id(2) * __mux_get_local_size(1) *`
  `__mux_get_local_size(0) + __mux_get_local_id(1) * __mux_get_local_size(0)`
  `+ __mux_get_local_id(0)`.
* `size_t __mux_get_enqueued_local_size(i32 i)` - Returns the enqueued
  work-group size in the `i`'th dimension, for uniform work-groups this is
  equivalent to `size_t __mux_get_local_size(i32 %i)`.
* `void __mux_mem_barrier(i32 %scope, i32 %semantics)` - Controls the order
  that memory accesses are observed (serves as a fence instruction). This
  control is only ensured for memory accesses issued by the invocation calling
  the barrier and observed by another invocation executing within the memory
  `%scope`. Additional control over the kind of memory controlled and what
  kind of control to apply is provided by `%semantics`. See [memory and control barriers](#memory-and-control-barriers) for more information.
* `void __mux_work_group_barrier(i32 %id, i32 %scope, i32 %semantics)` and
  `void __mux_sub_group_barrier(i32 %id, i32 %scope, i32 %semantics)` - Wait
  for other invocations of the work-group/sub-group to reach the current point
  of execution (serves as a control barrier). A barrier identifier is provided
  by `%id` (note that implementations **must** ensure uniqueness themselves,
  e.g., by running the `compiler::utils::PrepareBarriersPass`). These
  builtins may also atomically provide a memory barrier with the same semantics
  as `__mux_mem_barrier(i32 %scope, i32 %semantics)`. See [memory and control barriers](#memory-and-control-barriers) for more information.

##### Group operation builtins

Native CPU defines a variety of builtins to handle operations across a
sub-group, work-group, or *vector group*.

The builtin functions are overloadable and are mangled according to the type of
operand they operate on.

Each *work-group* operation takes as its first parameter a 32-bit integer
barrier identifier (`i32 %id`). Note that if barriers are used to implement
these operations, implementations **must** ensure uniqueness of these IDs
themselves, e.g., by running the `compiler::utils::PrepareBarriersPass`. The
barrier identifier parameter is not mangled.

> [!NOTE]
>  The sub-group and work-group builtins are all **uniform**, that is, the
>   behaviour is undefined unless all invocations in the group reach this point
>   of execution.

   Future versions of Native CPU **may** add **non-uniform** versions of these
   builtins.

The groups are defined as:

* `work-group` - a group of invocations running together as part of an ND
  range. These builtins **must** only take scalar values.
* `sub-group` - a subset of invocations in a work-group which can synchronize
  and share data efficiently. Native CPU leaves the choice of sub-group size
  and implementation to the target; Native CPU only defines these builtins with
  a "trivial" sub-group size of 1. These builtins **must** only take scalar
  values.
* `vec-group` - a software level group of invocations processing data in
  parallel *on a single invocation*. This allows the compiler to simulate a
  sub-group without any hardware sub-group support (e.g., through
  vectorization). These builtins **may** take scalar *or vector* values. The
  scalar versions of these builtins are essentially identical to the
  corresponding `sub-group` builtins with a sub-group size of 1.


##### `any`/`all` builtins

The `any` and `all` builtins return `true` if any/all of their operands
are `true` and `false` otherwise.

```llvm
   i1 @__mux_sub_group_any_i1(i1 %x)
   i1 @__mux_work_group_any_i1(i32 %id, i1 %x)
   i1 @__mux_vec_group_any_v4i1(<4 x i1> %x)
```

##### `broadcast` builtins

The `broadcast` builtins broadcast the value corresponding to the local ID to
the result of all invocations in the group. The sub-group version of this
builtin takes an `i32` sub-group linear ID to identify the invocation to
broadcast, and the work-group version take three `size_t` indices to locate
the value to broadcast. Unused indices (e.g., in lower-dimension kernels)
**must** be set to zero - this is the same value returned by
`__mux_get_global_id` for out-of-range dimensions.

```llvm
   i64 @__mux_sub_group_broadcast_i64(i64 %val, i32 %sg_lid)
   i32 @__mux_work_group_broadcast_i32(i32 %id, i32 %val, i64 %lidx, i64 %lidy, i64 %lidz)
   i64 @__mux_vec_group_broadcast_v2i64(<2 x i64> %val, i32 %vec_id)
```

##### `reduce` and `scan` builtins

The `reduce` and `scan` builtins return the result of the group operation
for all values of their parameters specified by invocations in the group.

Scans may be either `inclusive` or `exclusive`. Inclusive scans perform the
operation over all invocations in the group. Exclusive scans perform the
operation over the operation's identity value and all but the final invocation
in the group.

The group operation may be specified as one of:

* `add`/`fadd` - integer/floating-point addition.
* `mul`/`fmul` - integer/floating-point multiplication.
* `smin`/`umin`/`fmin` - signed integer/unsigned integer/floating-point minimum.
* `smax`/`umax`/`fmax` - signed integer/unsigned integer/floating-point maximum.
* `and`/`or`/`xor` - bitwise `and`/`or`/`xor`.
* `logical_and`/`logical_or`/`logical_xor` - logical `and`/`or`/`xor`.

Examples:

```llvm
   i32 @__mux_sub_group_reduce_add_i32(i32 %val)
   i32 @__mux_work_group_reduce_add_i32(i32 %id, i32 %val)
   float @__mux_work_group_reduce_fadd_f32(i32 %id, float %val)

   i32 @__mux_sub_group_scan_inclusive_mul_i32(i32 %val)
   i32 @__mux_work_group_scan_inclusive_mul_i32(i32 %id, i32 %val)
   float @__mux_work_group_scan_inclusive_fmul_f32(i32 %id, float %val)

   i64 @__mux_sub_group_scan_exclusive_mul_i64(i64 %val)
   i64 @__mux_work_group_scan_exclusive_mul_i64(i32 %id, i64 %val)
   double @__mux_work_group_scan_exclusive_fmul_f64(i32 %id, double %val)

   i64 @__mux_vec_group_scan_exclusive_mul_nxv1i64(<vscale x 1 x i64> %val)
```


##### Sub-group `shuffle` builtin

The `sub_group_shuffle` builtin allows data to be arbitrarily transferred
between invocations in a sub-group. The data that is returned for this
invocation is the value of `%val` for the invocation identified by `%lid`.

`%lid` need not be the same value for all invocations in the sub-group.

```llvm
   i32 @__mux_sub_group_shuffle_i32(i32 %val, i32 %lid)
```

##### Sub-group `shuffle_up` builtin

The `sub_group_shuffle_up` builtin allows data to be transferred from an
invocation in the sub-group with a lower sub-group local invocation ID up to an
invocation in the sub-group with a higher sub-group local invocation ID.

The builtin has two operands: `%prev` and `%curr`. To determine the result
of this builtin, first let `SubgroupLocalInvocationId` be equal to
`__mux_get_sub_group_local_id()`, let the signed shuffle index be equivalent
to this invocation’s `SubgroupLocalInvocationId` minus the specified
`%delta`, and `MaxSubgroupSize` be equal to
`__mux_get_max_sub_group_size()` for the current kernel.

* If the shuffle index is greater than or equal to zero and less than the
  `MaxSubgroupSize`, the result of this builtin is the value of the `%curr`
  operand for the invocation with `SubgroupLocalInvocationId` equal to the
  shuffle index.

* If the shuffle index is less than zero but greater than or equal to the
  negative `MaxSubgroupSize`, the result of this builtin is the value of the
  `%prev` operand for the invocation with `SubgroupLocalInvocationId` equal
  to the shuffle index plus the `MaxSubgroupSize`.

All other values of the shuffle index are considered to be out-of-range.

`%delta` need not be the same value for all invocations in the sub-group.

```llvm

   i8 @__mux_sub_group_shuffle_up_i8(i8 %prev, i8 %curr, i32 %delta)
```

##### Sub-group `shuffle_down` builtin

The `sub_group_shuffle_down` builtin allows data to be transferred from an
invocation in the sub-group with a higher sub-group local invocation ID down to
a invocation in the sub-group with a lower sub-group local invocation ID.

The builtin has two operands: `%curr` and `%next`. To determine the result
of this builtin , first let `SubgroupLocalInvocationId` be equal to
`__mux_get_sub_group_local_id()`, the unsigned shuffle index be equivalent to
the sum of this invocation’s `SubgroupLocalInvocationId` plus the specified
`%delta`, and `MaxSubgroupSize` be equal to
`__mux_get_max_sub_group_size()` for the current kernel.

* If the shuffle index is less than the `MaxSubgroupSize`, the result of this
  builtin is the value of the `%curr` operand for the invocation with
  `SubgroupLocalInvocationId` equal to the shuffle index.

* If the shuffle index is greater than or equal to the `MaxSubgroupSize` but
  less than twice the `MaxSubgroupSize`, the result of this builtin is the
  value of the `%next` operand for the invocation with
  `SubgroupLocalInvocationId` equal to the shuffle index minus the
  `MaxSubgroupSize`. All other values of the shuffle index are considered to
  be out-of-range.

All other values of the shuffle index are considered to be out-of-range.

`%delta` need not be the same value for all invocations in the sub-group.

```llvm
   float @__mux_sub_group_shuffle_down_f32(float %curr, float %next, i32 %delta)
```

##### Sub-group `shuffle_xor` builtin

These `sub_group_shuffle_xor` builtin allows for efficient sharing of data
between items within a sub-group.

The data that is returned for this invocation is the value of `%val` for the
invocation with sub-group local ID equal to this invocation’s sub-group local
ID XOR’d with the specified `%xor_val`. If the result of the XOR is greater
than the current kernel's maximum sub-group size, then it is considered
out-of-range.

```llvm
   double @__mux_sub_group_shuffle_xor_f64(double %val, i32 %xor_val)
```

##### Memory and Control Barriers

The mux barrier builtins synchronize both memory and execution flow.

The specific semantics with which they synchronize are defined using the
following enums.

The `%scope` parameter defines which other invocations observe the memory
ordering provided by the barrier. Only one of the values may be chosen
simultaneously.

```cpp
  enum MemScope : uint32_t {
    MemScopeCrossDevice = 0,
    MemScopeDevice = 1,
    MemScopeWorkGroup = 2,
    MemScopeSubGroup = 3,
    MemScopeWorkItem = 4,
  };
```

The `%semantics` parameter defines the kind of memory affected by the
barrier, as well as the ordering constraints. Only one of the possible
**ordering**s may be chosen simultaneously. The **memory** field is a
bitfield.

```cpp
  enum MemSemantics : uint32_t {
    // The 'ordering' to apply to a barrier. A barrier may only set one of the
    // following at a time:
    MemSemanticsRelaxed = 0x0,
    MemSemanticsAcquire = 0x2,
    MemSemanticsRelease = 0x4,
    MemSemanticsAcquireRelease = 0x8,
    MemSemanticsSequentiallyConsistent = 0x10,
    MemSemanticsMask = 0x1F,
    // What kind of 'memory' is controlled by a barrier. Acts as a bitfield, so
    // a barrier may, e.g., synchronize both sub-group, work-group and cross
    // work-group memory simultaneously.
    MemSemanticsSubGroupMemory = 0x80,
    MemSemanticsWorkGroupMemory = 0x100,
    MemSemanticsCrossWorkGroupMemory = 0x200,
  };
```

##### Atomics and Fences

The LLVM intermediate representation stored in
`compiler::BaseModule::finalized_llvm_module` **may** contain any of the
following atomic instructions:

* [`cmpxchg`](https://llvm.org/docs/LangRef.html#cmpxchg-instruction) for the `monotonic ordering`_ with *strong* semantics only
* [`atomicrmw`](https://llvm.org/docs/LangRef.html#atomicrmw-instruction) for the following opcodes: `add`, `and`, `sub`, `min`,
  `max`, `umin`, `umax`, `or`, `xchg`, `xor` for the `monotonic
  ordering`_ only

A compiler **shall** correctly legalize or select these instructions to ISA
specific operations.

The LLVM intermediate representation stored in
`compiler::BaseModule::finalized_llvm_module` **may** also contain any of the
following atomic instructions:
https://llvm.org/docs/LangRef.html#atomicrmw-instruction
* [cmpxchg](https://llvm.org/docs/LangRef.html#cmpxchg-instruction) for the [monotonic ordering](https://llvm.org/docs/LangRef.html#ordering) with *weak* semantics
* [load](https://llvm.org/docs/LangRef.html#load-instruction) with the instruction marked as *atomic* for the [monotonic ordering](https://llvm.org/docs/LangRef.html#ordering)
  only
* [store](https://llvm.org/docs/LangRef.html#store-instruction) with the instruction marked as *atomic* for the [monotonic ordering](https://llvm.org/docs/LangRef.html#ordering)
  only
* [fence](https://llvm.org/docs/LangRef.html#fence-instruction) for the [acquire
  ordering](https://llvm.org/docs/LangRef.html#ordering), [release
  ordering](https://llvm.org/docs/LangRef.html#ordering) and [acq_rel
  ordering](https://llvm.org/docs/LangRef.html#ordering) only.

The atomic instructions listed above **shall not** have a
[syncscope](https://llvm.org/docs/LangRef.html#syncscope) argument.

No lock free requirements are made on the above atomic instructions. A target
**may** choose to provide a software implementation of the atomic instructions
via some other mechanism such as a hardware mutex.

### Metadata

The following table describes metadata which can be introduced at different stages of the
pipeline:

   | Name | Fields | Description |
   |------|--------|-------------|
   |`!reqd_work_group_size`|i32, i32, i32|Required work-group size encoded as *X*, *Y*, *Z*. If not present, no required size is assumed.|
   |`!max_work_dim`| i32 | Maximum dimension used for work-items. If not present, `3` is assumed.|
   |`!codeplay_ca_wrapper`|various (incl. *vectorization options*)|Information about a *kernel entry point* regarding its work-item iteration over *sub-kernels* as stitched together by the `WorkItemLoopsPass` pass in the `compiler::utils` module. Typically this involves the loop structure, the vectorization width and options of each loop.|
   |`!codeplay_ca_vecz.base`|*vectorization options*, `Function*`| Links one function to another, indicating that the function acts as the *base* - or *source* - of vectorization with the given vectorization options, and the linked function is the result of a *successful* vectorization. A function may have *many* such pieces of metadata, if it was vectorized multiple times.|
   |`!codeplay_ca_vecz.derived`|*vectorization options*, `Function*`| Links one function to another, indicating that the function is the result of a *successful* vectorization with the given vectorization options, using the linked function as the *base* - or *source* - of vectorization. A function may only have **one** such piece of metadata.|
   |`!codeplay_ca_vecz.base.fail`|*vectorization options*| Metadata indicating a *failure* to vectorize with the provided vectorization options.|
   |`!mux_scheduled_fn`|i32, i32(, i32, i32)?| Metadata indicating the function parameter indices of the pointers to MuxWorkItemInfo and MuxWorkGroupInfo structures, respectively. A negative value (canonicalized as -1) indicates the function has no such parameter. Up to two additional custom parameter indices can be used by targets.|
   |`!intel_reqd_sub_group_size`|i32|Required sub-group size encoded as a 32-bit integer. If not present, no required sub-group size is assumed.|

Users **should not** rely on the name, format, or operands of these metadata.
Instead, utility functions are provided by the `utils` module to work with
accessing, setting, or updating each piece of metadata.

> [!NOTE]
>  The metadata above which refer to *vectorization options* have no concise
  metadata form as defined by the specification and **are not** guaranteed to
  be backwards compatible. See the C++ utility APIs in the `utils` module as
  described above for the specific information encoded/decoded by
  vectorization.

   | Name | Fields | Description |
   |------|--------|-------------|
   |`!mux-scheduling-params`|string, string, ...| A list of scheduling parameter names used by this target. Emitted into       the module at the time scheduling parameters are added to functions that requires them. The indices found in `!mux_scheduled_fn` function metadata are indices into this list.

### Function Attributes

The following table describes function attributes which can be introduced at
different stages of the pipeline:


   | Attribute        | Description |
   |------------------|-------------|
   |`"mux-kernel"/"mux-kernel"="x"`| Denotes a *"kernel"* function. Additionally denotes a       *"kernel entry point"* if the value is `"entry-point"`. `See below [mux-kernel](#mux-kernel-attribute) for more details. |
   |`"mux-orig-fn"="val"`| Denotes the name of the *"original function"* of a function. This original function may or may not exist in the module. The original function name is propagated through the compiler pipeline each time Native CPU creates a new function to wrap or replace a function. |
   |`"mux-base-fn-name"="val"`| Denotes the *"base name component"* of a function. Used by several passes when creating new versions of a kernel, rather than appending suffix upon suffix.|

  For example, a pass that suffixes newly-created functions with
  `".pass2"` will generate `@foo.pass1.pass2` when given function
  `@foo.pass1`, but will generate simply `@foo.pass2` if the same
  function has `"mux-base-name"="foo"`.

   | Attribute | Description |
   |-----------|-------------|
   |`"mux-local-mem-usage"="val"`| Estimated local-memory usage for the function. Value must be a positive integer. |
   |`"mux-work-item-order"="val"`| Work-item order (the dimensions over which work-items are executed from innermost to outermost) as defined by the `utils_work_item_order_e` enum. If not present, `"xyz"` may be assumed. |
   | `"mux-barrier-schedule"="val"`| Typically found on call sites. Determines the ordering of work-item execution after a berrier. See the `BarrierSchedule` enum. |
   | `"mux-no-subgroups"`| Marks the function as not explicitly using sub-groups (e.g., identified by the use of known mux sub-group builtins). If a pass introduces the explicit use of sub-groups to a function, it should remove this  attribute. |

#### mux-kernel attribute

SYCL programs generally consist of a number of *kernel functions*, which
have a certain programming model and may be a subset of all functions in the
*module*.

Native CPU compiler passes often need to identity kernel functions amongst
other functions in the module. Further to this, a Native CPU implementation may
know that an even smaller subset of kernels are in fact considered *kernels
under compilation*. In the interests of compile-time it is not desirable to
optimize kernels that are known to never run.

Under this scheme, it is further possible to distinguish between kernels that
are *entry points* and those that aren't. Entry points are kernels which may be
invoked from the runtime. Other kernels in the module may only be run when
invoked indirectly: called from kernel entry points.

The `mux-kernel` function attribute is used to
communicate *kernels under compilation* and *kernel entry points* (a subset of
those) between passes. This approach has a myriad of advantages. It provides a
stable, consistent, kernel identification method which other data do not: names
cannot easily account for new kernels introduced by optimizations like
vectorization; calling conventions are often made target-specific at some point
in the pipeline; pointers to functions are unstable when kernels are
replaced/removed.

Passes provided by Native CPU ensure this attribute is updated when adding,
removing, or replacing kernel functions. Each Native CPU pass in its
documentation lists whether it operates on *kernels* or *kernel entry points*,
if applicable.
