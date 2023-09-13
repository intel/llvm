# Vecz Documentation

Codeplay's Vecz is a library based on LLVM that allows vectorization of SPMD
programs such as OpenCL kernels.

Vecz is automatically built during the oneAPI Construction Kit build process
(only if a runtime compiler is found) but needs to be manually enabled to be
used during the kernel compilation process. This is done by providing the
`-cl-wfv={always|auto}` option before running any OpenCL program.

Vecz ships with a standalone tool called `veczc`. This tool will consume LLVM IR
- in bitcode or textual format, producing vectorized output.

## Design ideas

Vecz's design is based on the automatic whole function vectorization research
by [Ralf Karrenberg][1], titled "Automatic SIMD Vectorization of SSA-based
Control Flow Graphs", and a combination of other papers referenced at various
places in this document. While the process followed by Vecz is not exactly the
same, understanding the research would help to understand Vecz better.

## Supporting infrastructure

Vecz relies on certain other classes and functions provided by
`compiler::utils`.  The `BuiltinInfo` interface is used to ascertain certain
properties of OpenCL builtin functions such as whether a particular function
has a vector equivalent, or whether it is a work item ID query. Specific
implementations of the interface are provided for OpenCL.

Before running vecz, it is recommended to run the
`compiler::utils::OptimalBuiltinReplacementPass`. This replaces certain builtin
calls with LLVM instructions or intrinsics that perform the equivalent
operation, which enables later optimizations to work with them (which applies
both to LLVM's own optimization passes that vecz runs, and to some of vecz's
own transform passes). Furthermore, it allows these builtins to be widened
arbitrarily, without being limited to the widths available as OpenCL builtins.

If a target intends to use the `compiler::utils::WorkItemLoopsPass` after Vecz,
it is important to ensure that, **before vecz**,  all calls to barrier-like
functions in the full nested kernel call-graph are given unique barrier IDs.
Note that this effectively mandates full inlining of all functions containing
barrier-like calls.

This is necessary because vectorization can considerably affect control flow,
so the ordering of the barriers in the function may change. If the
`WorkItemLoopsPass` needs to combine two different versions of the same kernel
into a single scheduled kernel, it is vital that direct correspondence of the
barrier calls is maintained.

Users can run the `compiler::utils::PrepareBarriersPass`, which satisfies these
requirements.

The `vecz::RunVeczPass` does not delete the original scalar kernel after
vectorization, nor does it transfer the scalar kernel name to the vectorized
function.

## Target specialization

Vecz provides an interface, `vecz::TargetInfo`, that allows the vectorizer to make
target-dependent decisions. This is retrieved via an analysis pass:
`vecz::TargetInfoAnalysis`. Targets may override the `vecz::TargetInfo`
constructed by this Analysis. The interface has a default implementation, which
may be overridden.

Targets can override:

* Builder functions for all of the various forms of memory
  operation that vecz can output: loads and stores in masked or unmasked forms;
  contiguous access; interleaved access; scatter/gather access. Targets may want
  to provide their own intrinsics for these operations, if they exist.
* Builder functions for special forms of vector shuffles on scalable vectors,
  namely "inner" and "outer" broadcasts (meaning the duplication of each vector
  element `n` times, and the duplication of the entire vector `n` times), as
  well as vectorized scalable extract and insert instructions (which vectorize
  to a pick or replacement of every nth element to/from a narrower vector).
  Since there are no LLVM instructions to efficiently perform these operations
  on scalable vectors, the default implementation involves writing to memory and
  reading it back, which is likely to be suboptimal.
* Builder function to construct the vector length argument of predicated vector
  instructions, on targets that support this feature.
* Functions to support the Interleaved Groups Combining Pass.
* A function to support the SIMD Width Analysis that returns the widest
  vectorization factor possible for a given set of values.
* A function to compute the preferred vector width of a given scalar type. The
  packetizer can use this to emit multiple vectors per scalar-kernel value,
  instead of a single wider vector. The default implementation is based on the
  bit width of a single vector register (from `llvm::TargetTransformInfo`).

For full details, see the documentation of the `vecz/target_info.h` header file.

## Vectorization process

* Clone the kernel function
* Analyze the kernel function to determine if it has a pattern not prone to
  being vectorized
* Run preparation passes
* Perform control-flow to data-flow conversion (to handle divergent CFGs)
* Scalarize the kernel if desired/needed
* Run pre-packetization "middle optimization" passes
* Determine the optimal SIMD width for the kernel
* Packetize the kernel
* Perform optimizations and cleanup
* Define internal builtins

## Uniform Analysis

Instructions in a kernel function may be uniform (i.e. they evaluate to the
same value or have the same side-effects on all lanes) or varying. Varying
instructions usually have a data dependency on either the global ID or the local
ID of the work-item executing the kernel. As an example, in the following kernel
the store to the global memory, as well as the calculation of the address for
the store, are varying instructions depending on the global ID of the work-item.
The multiplication of `in` by 2 on the other hand is uniform, since it is the
same across all the work-items.

```c
kernel void fn(int in, global int *out) {
  size_t tid = get_global_id(0);
  in = in * 2;
  out[tid] = in;
}
```

Assuming to vectorize on the x dimension, after packetization, functions calls
like `get_global_id(0)` or `get_local_id(0)` will return vectors of consecutive
indices, which will allow us to packetize the store into a vector store.

On architectures that support both scalar and vector instructions we do not want
to packetize uniform instructions, as each vector lane will perform the same
operation on the same data and return the same result. Instead, we want to keep
uniform instructions scalar (i.e. keep the original instructions untouched), and
broadcast their value into a vector when necessary.

In order to differentiate between the varying and the uniform instructions,
we have implemented an analysis pass called the `Uniform Analysis`. This
analysis starts by finding "vector roots" in the function. By "roots" we mean
instructions (usually function calls) that we know to be varying. For example,
work-item ID related functions, or packetized arguments in non-kernel functions
are some common vector roots. Each root and its users are recursively marked as
varying. Marking a value happens before marking its users, so that use cycles
(e.g. phi nodes) do not cause infinite recursion. The instructions remaining
after this process are considered to be uniform.

In the previous example kernel, the vector root used is the call to the
`get_global_id(0)` function. Starting from that point and then recursively
going through its users and their users etc. We recursively mark the address
calculation (`getelementptr`) for the output and the store to the output as
varying too. The `in` value and its multiplication are not marked as varying
since they are not using any varying values, they are being used by one. We
should note here that under special cases, such as an `alloca` instruction that
is stored into by a varying store, we might mark some instructions as varying
just because they are used by a varying instruction but in the general case we
do not.

> The relevant classes and functions can be found in
> `source/include/analysis/uniform_value_analysis.h` and
> `source/analysis/uniform_value_analysis.cpp`.

## Stride Analysis

Memory operations can access their data in several different patterns, as a
function of the work item ID, categorized as:

* Uniform: data is accessed from the same address for all work items;
* Contiguous: data is accessed sequentially with no gaps;
* Strided: data is accessed linearly but spaced apart with a constant or
  uniform stride;
* Divergent: data is accessed with no discernible pattern.

The stride analysis traverses address computation expressions, to ascertain
which kind of memory access is required, computing any constant strides
encountered. Uniform but variable strides are not computed during the analysis,
because doing so would require creating new instructions in the function, which
is at odds with the idea of an analysis pass. However, it is usually sufficient
to know that access is linear, without needing to know its actual value. When a
transform pass wishes to make use of the actual value, it can call the
`manifest()` function of the analysis, which will traverse its internal dataset
and create any instructions required. Note that pointers to these instructions
will survive until the analysis is invalidated.

This analysis uses the result of [Uniform Analysis](#uniform-analysis).

> The relevant classes and functions can be found in
> `source/include/analysis/stride_analysis.h` and
> `source/analysis/stride_analysis.cpp`.

## Packetization Analysis

The packetizer needs to know which instructions require packetization in
advance, for optimal functioning. An instruction that has been marked as
varying by the [Uniform Analysis](#uniform-analysis) may or may not require
packetization, since some varying values will form an expression computing
the address of a contiguous or strided memory operation. Therefore, this
analysis starts at the function's vector leaves, and works backwards through
operands, recursively marking values for packetization. When a contiguous or
strided memory operation is encountered, its address operand is not
followed. This allows a more accurate estimation of packetization requirements
prior to actual packetization, which is useful for the
[SIMD Width Analysis](#simd-width-analysis).

This analysis uses the result of [Stride Analysis](#stride-analysis).

> The relevant classes and functions can be found in
> `source/include/analysis/packetization_analysis.h` and
> `source/analysis/packetization_analysis.cpp`.

## Control Flow Graph Analysis

Another useful analysis we employ is the Control Flow Graph (CFG) Analysis. As
the name suggests, it analyzes the control flow graph to store useful
information about blocks and loops such as what loop does a block live in and
what are the lcssa values of a loop (values that are live through a loop).

The analysis works by iterating over the basic blocks, create a tag for each
block, and if the block belongs in a loop, then create a tag for that loop (if
one does not yet exist) and mark the loop as owning the block.

If in the process of visiting the blocks we encounter a divergent branch, then
we say that the CFG needs to be converted into a Data-Flow Graph (see the
[Control Flow Conversion](#control-flow-conversion-pass) section).

> The relevant classes and functions can be found in
> `source/include/analysis/control_flow_analysis.h` and
> `source/analysis/control_flow_analysis.cpp`.

## Divergence Analysis

> Formally known as `Rewire Target Analysis`.

This analysis is used to find all the information regarding divergence in a CFG.
It uses as a pre-requisite the [Uniform Analysis](#uniform-analysis) to know
which instructions are varying when we first evaluate the CFG. Those
instructions are the basis to find divergent branches, which are branches whose
operands are varying. We name blocks having such instructions `div_causing`.
The analysis works by iterating over all the branches in the CFG until no more
new `div_causing` blocks are found.
When we find a `div_causing` block, we first compute the divergent path that
this block creates. This divergent path contains all the blocks from the
`div_causing` block to the post dominator of the latter. All the blocks that
belong in a divergent path may need to have their instructions marked varying,
in the case where they might be used outside the divergent path and thus need
to be packetized.
We then find all the join points of the `div_causing` block. Such blocks have
disjoint paths from the `div_causing` block. These blocks are called `blend`
blocks and will need to have their PHI nodes transformed into select
instructions because their path will be linearized. These blocks also need to
have their PHI nodes marked as varying.
After we have processed the `div_causing` block, we must find if this block
makes a loop divergent. A loop is divergent if it is possible that some work
items leave the loop at an exit, while others keep iterating. In essence, this
means that a divergent branch has no post-dominator in the loop. We also mark
all the exits of the loop where *some* work items may leave as divergent because
they will need to be linearized.
Finally, we compute `by_all` blocks which are blocks that need not be predicated
because no divergence is present when they are executed.

> The relevant functions can be found in
> `source/include/analysis/divergence_analysis.h` and
> `source/analysis/divergence_analysis.cpp`.

## Liveness Analysis

This analysis is used to determine the set of live values at any point in the
program. A value becomes "live" at its definition and remains live until all of
its uses have been encountered. The result of this analysis provides an info
object for every basic block in the program, that contains the "Live In" set
(i.e. the values that are live at the start of the Basic Block, including all
of its PHI nodes), and the "Live Out" set (i.e. the values that are still live
at the end of the block). By iterating backwards over a Basic Block, starting
from the Live Outs, one can determine the set of values that are live at any
point in the program.

The implementation is based on Section 5.2 of the paper "Computing Liveness Sets
for SSA-Form Programs." by Florian Brandner, Benoit Boissinot, Alain Darte,
Benoît Dupont de Dinechin, Fabrice Rastello.

This analysis is used by [BOSCC](#branch-on-superword-condition-code), and by
[SIMD Width Analysis](#simd-width-analysis).

> The relevant classes and functions can be found in
> `source/include/analysis/liveness_analysis.h` and
> `source/analysis/liveness_analysis.cpp`

## SIMD Width Analysis

This analysis is used to estimate the optimal vectorization width, depending on
the contents of the kernel. The current strategy is a two-stage process: first
we find the widest used varying value type, disregarding types that make up less
than a small proportion of the program according to some tolerance threshold.
The SIMD width computed is the number of elements of this type that will fit
into a single vector register. Then we analyze the program using the
[Liveness Analysis](#liveness-analysis) to estimate the maximum SIMD width that
will fit into vector registers, if it is wider than the previously computed
result. This allows vectorization to produce values that will not necessarily
fit into single vector registers, but will fit across multiple registers after
legalization.

SIMD Width Analysis is performed only when vectorization is set to automatic
(either by using the `-cl-wfv=auto` option or by passing `bool Auto=true` to
`Vectorizer::vectorize()`). The analysis is performed after control flow
conversion, so that any changes made by this or prior passes will be taken into
account.

This analysis uses the result of
[Packetization Analysis](#packetization-analysis).

> The relevant classes and functions can be found in
> `source/include/analysis/simd_width_analysis.h` and
> `source/analysis/simd_width_analysis.cpp`

## Vectorizability Analysis

This is not an analysis in the traditional sense but more of a filter for
kernels that we know we will not be able to vectorize. There are a number of
cases that we cannot handle in Vecz, such as kernels containing specific atomic
instructions, functions returning vector results, or specific OpenCL builtins.
By checking for these conditions early in the vectorization process we can save
on compile time and also avoid accidentally generating an incorrectly vectorized
kernel.

> The relevant functions can be found in
> `source/include/vectorization_context.h` and
> `source/vectorization_context.cpp`.

## Reachability Analysis

This is a utility class created to speed up CFG reachability queries required by
[BOSCC](#branch-on-superword-condition-code). It is not an analysis pass managed
by LLVM, but must be created manually where required. The algorithm is based on
an Open Proceedings paper entitled "Reachability Queries in Very Large Graphs: A
Fast Refined Online Search Approach" by Renê R. Veloso, Loïc Cerf, Wagner Meira
Jr, Mohammed J. Zaki. In addition to that approach, dominator and post-dominator
trees are used to further accelerate the process.

The reachability data structure must be constructed from a directed acyclic
graph. Backedge traversal is a case not yet handled.

> The relevant functions can be found in `source/include/reachability.h` and
> `source/reachability.cpp`.

## Preparation Passes

We employ a set of preparation passes that includes both optimization passes,
such as the Mem2Reg pass discussed later on, and passes necessary to generate IR
that Vecz can handle:

* Dominator Tree Wrapper Pass (from LLVM)
* Loop Info Wrapper Pass (from LLVM)
* Switch Lowering Pass (from LLVM)
* Function Exit Nodes Unification Pass (from LLVM)
* Builtin Inlining Pass (from Vecz, described below)
* Promote Memory To Register Pass (from LLVM)
* Basic Mem2Reg Pass (from Vecz, described below)
* Instruction Combining Pass (from LLVM)
* Dead Code Elimination Pass (from LLVM)
* Pre-linearization Pass (from Vecz, described below)
* Instruction Combining Pass (again, to deal with Pre-linearization changes)
* CFG Simplification Pass (from LLVM)
* Unify Function Exit Nodes Pass (from LLVM)
* Loop Simplification Pass (from LLVM)
* Loop Rotate Pass (from LLVM)
* Simplify Infinite Loop Pass (from Vecz, described below)
* Induction Variable Simplification Pass (from LLVM)
* Early CSE Pass (from LLVM)
* LCSSA Pass (from LLVM - restores LCSSA form if broken by Early CSE)

> The relevant classes and functions can be found in `vectorizer.cpp`, as
> well as `builtin_inlining_pass.h`, `builtin_inlining_pass.cpp`,
> `basic_mem2reg_pass.h`, and `basic_mem2reg_pass.cpp` for the Vecz
> passes.

### Builtin Inlining Pass

The Builtin Inlining pass replaces calls to OpenCL builtins with an inline
version of the builtin. This is done because the generic approach that we follow
for getting the scalar or vector equivalent of a builtin (described later in the
[Packetizing OpenCL Builtins](#packetizing-opencl-builtins) section) does not
work for all of them, so instead we bring the implementation in the same module
as the kernel and let the vectorizer vectorize it. More details can be found in
the [Vectorizing Builtin Functions](#vectorizing-builtin-functions) section.

> The relevant class can be found in
> `source/include/transform/builtin_inlining_pass.h` and
> `source/transform/builtin_inlining_pass.cpp`.

### Basic Mem2Reg Pass

The Basic Mem2Reg pass performs `alloca` promotions, similar to how the LLVM
Mem2Reg pass operates. Of course, there are a number of requirements that an
`alloca` and its users need to fulfill before it is possible to perform such an
optimization but the general idea is this: we first check if it is possible
to determine the value (as an LLVM Value, not necessarily as a compile time
constant value) stored in the `alloca`, and if it is the case, then users of
the `alloca` are updated to use the stored value directly, instead of loading it
from the `alloca`.

The Basic Mem2Reg is somewhat simpler than LLVM's own Promote Memory To Register
Pass, and as a result more strict in what it will promote. However, it is able
to promote some `alloca`s that LLVM's own pass cannot, for instance where there
are bitcasts involved.

> The pass can be found in `basic_mem2reg_pass.h` and `basic_mem2reg_pass.cpp`.

### Pre-linearization Pass

This pass transforms simple `if-then` and `if-then-else` constructs by hoisting
their contents out into the parent scope, when it determines that the cost of
executing the instructions is less than the cost of the branches. This also takes
into account of the extra branching logic that will be inserted by
[BOSCC](#branch-on-superword-condition-code) during Control Flow Conversion.

The CFG itself is not modified, instead being left for LLVM's CFG Simplification
Pass to tidy up.

> The pass can be found in `source/transform/pre_linearize_pass.cpp`.

### Simplify Infinite Loop Pass

The Simplify Infinite Loop pass checks the CFG for infinite loops that may still
be present after the LLVM loop simplifications passes we call. This pass is
necessary for VECZ to make all loops have the same layout, i.e. in this case at
least one exit block, as handling infinite loops within the
[Control Flow Conversion Pass](#control-flow-conversion-pass) would add too much
overhead.

This pass is a loop pass and will check, for each loop, if exit blocks are
present. If not, it means the loop cannot terminate (as it cannot be exited) so
we have to mutate it. It then tries to find the unique return block of the
function (it should only have one as we call `UnifyFunctionExitNodesPass` prior
to this pass to make sure we do have only one return block). This return block
will be the exit block of the infinite loop after mutation. After finding the
latter, we add a conditional branch to the latch that will either branch to the
header or to the return block. The condition of that conditional branch will
actually always be true such that it will still always branch to the loop
header to respect the semantic of the original program. Finally, the pass
updates the PHI nodes in the return block by adding an incoming block to them,
which is the latch of the infinite loop. It also adds new PHI nodes for uses in
the return block that may be defined after the infinite loop, for which adding
the edge from the infinite loop to the return block may break the SSA form.

> The pass can be found in `source/transform/simplify_infinite_loop_pass.cpp`.

## Remove Intptr Pass

This pass scans for `PtrToInt` casts that can be eliminated and converted into
bitcasts or GEP instructions. It is able to eliminate a `PtrToInt` in the
following cases:

* A PtrToInt followed by an IntToPtr, which is replaced by a pointer cast;
* A PtrToInt used by a PHI node, in which case the PHI node is replaced
  by one of the pointer type;
* A PtrToInt where the pointer type is `i8*`, followed by an integer add or
  subtract, in which case it is replaced by a GEP.

Removing intptr casts makes it possible for uniform pointer strides to be
identified.

> The pass can be found in `source/transform/remove_intptr_pass.cpp`

## Squash Small Vectors Pass

This pass looks for loads and stores of vector types that fit into a legal
integer, where packetization would result in non-contiguous access, and replaces
them with loads or stores of an integer scalar of the same size (where alignment
requirements permit). This allows more efficient generation of scatter/gather or
interleaved memory operations on these types.

> The pass can be found in `source/transform/squash_small_vectors_pass.cpp`

## Scalarization Pass

This pass converts code that is already in a vector form into scalar code, or
retains a partially scalarized code, so that the packetizer can produce vector
IR at a vectorization width optimal for the target hardware.

The scalarization pass is divided into two stages: the analysis and the actual
transformation stage.

In the analysis we mark the values that need scalarization, which includes
vector leaves and non-vector instructions using vector operands.
The non-vector instructions using vector operands are either `ExtractElement`s
with vector operands or `BitCastInst`s from vector to non-vector. Note that the
analysis is not performed in an analysis pass, but a utility class that runs
locally within the transform pass, since this information is not needed by any
other pass.

If the vector leaf instruction or any of its arguments are of vector type and
the primitive size is greater than the partial scalarization factor (called
primitive size), the instruction is marked for needing scalarization. This marks
the end of analysis.

Once we have the values that were marked for transformation by the analysis
stage, the vector operands are first scalarized and then followed by the vector
leaf instructions that need scalarization.

> The utility classes can be found in
> `source/include/transform/scalarizer.h` and
> `source/transform/scalarizer.cpp`.
> The transform pass can be found in
> `source/include/transform/scalarization_pass.h` and
> `source/transform/scalarization_pass.cpp`.

## Control Flow Conversion Pass

The control flow conversion linearizes the divergent control flow executing both
if and else condition blocks. In order to preserve safe access, the blocks are
predicated with masks, to allow only legal access for calls and memory accesses
that have side-effects.

Control flow conversion is the actual control-flow to data-flow conversion pass
that uses information from the control flow analysis and divergence analysis.
Conversion starts with generating masks, applying masks and generating selects,
followed by linearizing the control flow and finally repair the SSA form that
the linearization may have broken.

The mask for every basic block is generated starting with the entry mask, which
is followed by a branch mask for cases where the entry mask is a phi node of its
predecessors. Then special masks in the loop are generated to handle run time
control flow divergence, namely; the loop active mask, combined exit mask. Next,
the loop live values need to reflect the right values for early exited lanes.
The masks are then applied to prevent side-effects for the inactive instances.
In case of a call to memory operation, it is replaced with a corresponding masked
internal builtin call [Defining Internal Builtins](#defining-internal-builtins).
The phi nodes are then transformed into selects, to enable control-flow to
data-flow conversion.

The CFG is then linearized, where necessary. It is actually partially linearized
to retain uniform branches that we know need not be linearized. We apply the
partial linearization by identifying every divergent blocks thanks to the
divergence analysis to know which blocks should be linearized, and which blocks
may remain untouched. A divergent block is called a `div causing block`. To
linearize the CFG, we keep a deferral list that represents all the blocks that
lost their predecessor's edge because of divergence. When we reach a block, if
the block is a div causing block, then it can only have one successor, otherwise
the block can keep the same amount of successors it has. To know which block
should be the successor of another block, we choose between the current
successors, and the deferral list available for that block. The choice is then
made based on the Dominance-Compact Block Indexing, which assigns each block a
unique index. Dominance compactness means that for any block, all other blocks
dominated by that block follow on in a contiguous sequence. This is constructed
by a depth-first traversal of the dominator tree, visiting children in CFG
reverse post-order. (In actual fact, Loop-Compactness takes precedence over
Dominance-Compactness; the latter usually implies the former, but certain loops
with multiple exits can break this, so special care has to be taken.) Using that
index to choose the successor guarantees that if an edge `A to B` existed in the
original graph, an edge `A to X to B` will exist in the linearized graph, thus
conserving dominance.

Once the CFG is linearized, we may have introduced new edges that were not there
previously, which may have broken the SSA form. Therefore, we must repair the
SSA form by introducing blend instructions (in the form of phi nodes) at the new
converging points.

The partial linearization implementation was inspired from the paper
`Automatic SIMD Vectorization of SSA-based Control Flow Graphs` by
Ralf Karrenberg and `Partial Control-Flow Linearization` by
Simon Moll & Sebastian Hack.

> The pass can be found in
> `source/include/transform/control_flow_conversion_pass.h` and
> `source/transform/control_flow_conversion_pass.cpp`.

### Branch On Superword Condition Code

Various optimizations directly linked to the partial linearization can be
applied. One of those optimizations is BOSCC (Branch On Superword Condition
Code), whose purpose is to duplicate predicated code into their uniform,
original, form so that this duplicated code can be executed when all lanes of
the SIMD group are either all true or all false. In fact, when this is the case,
there is no point to execute predicated instructions as all lanes will be
executed, or none.

The first step of this optimization is to duplicate all the code paths that may
diverge so that we can execute that code when all lanes are true/false. We thus
have one part of the CFG that diverges and one that stays uniform, throughout
the execution of the code. However, just after duplicating the code, the latter
is separated from the original CFG and the rewiring will be done later, once the
linearization is done. In order to identify which blocks need to be duplicated,
we need to identify Single-Entry, Single-Exit (SESE) regions that contain
divergence-causing branches. We leverage the Dominance-Compact Block Indexing to
do this, since any SESE region is necessarily dominance compact. In the simple
case, a divergence-causing branch will be from the entry block of a SESE region.
However, this is not strictly necessarily the case in more complex CFGs, where
the SESE entry block might not be a divergent branch, but multiple divergent
branches may exist within the region. Therefore we deal with Multiple-Entry,
Single-Exit predicated subregions of the SESE that can potentially overlap each
other (although we only ever duplicate each predicated block once, regardless of
how many different regions it appears in), each beginning with a single
divergence-causing branch.

Once the linearization is done, and we start repairing the CFG from all the
changes we made, we can start rewiring the duplicated (i.e. uniform) parts of
the CFG into the divergent ones. The first thing we do is to make the outermost
loop preheaders of duplicated loops always target the uniform loop because the
first time we enter the loop, all our lanes are activated/deactivated so there
is no need to execute the divergent loop. Then, for each divergent branch, we
add a run time checker that checks if all lanes are activated, in which case we
hit the all-lanes-activated uniform path. Otherwise, we check if none of the
lanes are activated, in which case we hit the no-lanes-activated uniform path.
Finally, if none of those two checks were true, then that means some condition
diverges: some lanes evaluate to true, and some evaluate to false; we thus have
to go into the divergent part of the CFG. As soon as we go into the divergent
part of the CFG (the one which contains predicated instructions), it is not
possible to go back into the uniform part of the CFG (the one that contains no
predicated instructions), until we reach a blend block, that is, a block where
all the previous divergent branches meet.

In order to allow fast reachability queries of the CFG, all of the blend points
are computed and stored during modification of the CFG, which allows us to
construct a data structure to speed up the required reachability queries at the
point the PHI nodes are actually created, since if we were modifying the CFG
during this process, the reachability data structre would be continuously
invalidated. It also means that the PHI nodes can be created with all
predecessors known, and avoids cases where a reachable PHI node would be falsely
classified as unreachable simply because it hasn't been connected up yet.

Reachability queries are handled by the
[Reachability Analysis](#reachability-analysis) class described earlier in this
document, except in some remaining cases outside of BOSCC, and in one case
inside of BOSCC where reachability needs to traverse backedges, which is not
handled by the aforementioned data structure.

The BOSCC implementation was inspired from the paper
`Predicate Vectors if you must` by Shahar Timnat, Ohad Shacham and Ayal Zaks.

> The class can be found in `source/control_flow_boscc.h` and
> `source/control_flow_boscc.cpp`.

### Return on Superword Condition Code

ROSCC is a simpler alternative to BOSCC that doesn't require any code
duplication. It handles only "early exit branches", i.e. code of the form:

```c
if (some_condition) {
  return;
}
```

Where `some_condition` is a varying expression, ROSCC will insert an additional
uniform branch directly to the exit block.

ROSCC is applied only when BOSCC is turned off, since BOSCC will handle this
special case in a more general way.

> The class can be found in `source/control_flow_roscc.h` and
> `source/control_flow_roscc.cpp`.

### Instantiating functions with side effects

Much like the memory operations, functions that may produce side-effects also
need to be masked. Call instructions are examined, and if it is determined that
we will not be able to handle the call in any other way, the call is replaced
with a call to a masked version of the function.

The masked version is nothing more than a wrapper around the original call. The
wrapper function accepts the same arguments as the unmasked version and an
additional boolean (`i8`) argument for the mask. If the mask is true, the
wrapped function is executed and its result is returned, otherwise  `undef` is
returned, without executing the wrapped function.

After replacing the call with the masked call, we mark the call for
instantiation, as we cannot packetize it into a vector instruction.

### Division by zero exceptions

On some hardware, a divide by zero operation and/or a numerical overflow will
result in a CPU exception. Since inactive vector lanes should never trigger such
an exception, masks may also need to be applied using `select` instructions on
the divisor, that result in a divisor of `1` for inactive lanes. There is no way
for Vecz to get the information about this requirement from the target, and since
most GPU hardware silently ignores division by zero, by default this behaviour is
disabled. It can be enabled explicitly by using the `DivisionExceptions`
[Vecz Choice](#vecz-choices).

Note that the mask applied to the divisor is derived purely from the CFG. The
behaviour of any division by zero on an active vector lane will be unaffected.


## Packetization

During packetization, instructions that define varying values or produce varying
side-effects are turned into instructions that define the same value or produce
the same effects for each SIMD instance. For example, an `add` instruction
that takes two `i32` operands is turned into another `add` instruction that
takes two `<N x i32>` operands (where `N` is the SIMD width). This is done
recursively in three steps; first, we packetize any branch condition that
requires packetization, and then, starting at the "vector leaves" of the
function, the rest of the instructions. Vector leaves are instructions that
allow varying values to "escape" from the function. Some examples of leaves
include:

* Store instructions, when the value to store and/or the pointer is varying
* Call instructions, when varying operands are present or when the call has no
  use
* Return instructions

After those two steps, then we proceed to packetize any remaining phi nodes,
explained in more details in a following [subsection](#packetizing-phi-nodes).

During the packetization process, we might run into cases where we cannot
packetize a varying instruction but instead we have to instantiate it. By
instantiation we mean repeating an instruction `N` times, one for each SIMD
lane. A common example would be calls to the `printf` function, as in the
following kernel:

```c
kernel void fn(global int *in, global int *out) {
  size_t tid = get_global_id(0);
  int load_in = in[tid];
  int result = load_in * tid;
  printf("in[%d] = %d\n", tid, result);
  out[tid] = result;
}
```

In this kernel, the call to the `printf` function will be repeated `N` times,
with its arguments adjusted to use the correct global ID for each lane. On the
other hand, the load and store instructions, as well as the multiplication, will
be packetized into vector instructions of width `N`. More details on
instantiation can be found in the [relevant section](#instantiation).

As we have already mentioned, the packetization (or instantiation) starts from
the vector leaves and recursively continues into their operands. As a matter of
fact, the operands are packetized before the instruction itself; in order to
generate the correct instruction, we first need to have the correct operands.
This process stops when we reach either:

1. An operand that is uniform, such as constants or kernel arguments.
2. A vector root such as `get_global_id(0)`.
3. A pointer that we can handle in its scalar form.

In the first case, we create a packetized version of the operand by simply
broadcasting its value into a vector, so that each element of the vector
contains the same value. Since the value is uniform, we do not need to proceed
and packetize its operands. The second case is handled specially by using the
scalar value to create a vector of sequentially increasing values.

The third case is also special, because it depends on the access pattern
of the pointer. If the pointer is varying and we can determine a stride
to the access pattern then we do not need to packetize the pointer.
Instead, we can use the same pointer value as the base for a vector memory
instruction. More details on this can be found in the [Packetizing Memory
Operations](#packetizing-memory-operations) subsection.

Given all of these, we can now see how the example kernel given above will be
vectorized, with a vector width of 4. Of course, Vecz is not a source-to-source
transformation pass but the following kernel captures the equivalent IR changes
that will be performed in an easier to read format:

```c
kernel void fn(global int *in, global int *out) {
  size_t tid = get_global_id(0);
  size_t4 tid4 = {tid, tid, tid, tid} + {0, 1, 2, 3};
  int4 load_in4 = in[tid];
  int4 result4 = load_in4 * tid4;
  printf("in[%d] = %d\n", tid4.s0, result4.s0);
  printf("in[%d] = %d\n", tid4.s1, result4.s1);
  printf("in[%d] = %d\n", tid4.s2, result4.s2);
  printf("in[%d] = %d\n", tid4.s3, result4.s3);
  out[tid] = result4;
}
```

Notice how the address for the vector load and store are still calculated using
the scalar ID variable (`tid`), since the kernel accesses the elements of the
array consecutively and thus we can use a vector load and a vector store with
the same base address.

Regardless of how a varying instruction has been handled, after we have
packetized or instantiated it, we mark it for deletion. After we have gone
through all the vector leaves, we proceed to delete all the instructions that we
marked for deletion, as long as they have no remaining users.

> The packetization pass can be found in
> `source/include/transform/packetization_pass.h` and
> `source/transform/packetization_pass.cpp`.

This is the general approach taken to packetize instructions but some cases
need to be handled specially. We will now explain in more depth some special
packetization cases.

### Packetizing Memory Operations

Memory operations are special as their access pattern determines how they are
packetized. Specifically, a memory operation will have one of these mutually
exclusive access patterns:

1. No recognizable stride
2. A stride of 0 elements (i.e. being uniform)
3. A stride of 1 element (i.e. being contiguous)
4. A stride of `X` elements

In the first case we will packetize the memory operation using a scatter/gather
internal builtin. This means that we will generate an address for each SIMD
lane and the scatter/gather internal builtin will handle storing or loading the
elements to and from vectors.

In the second case, we will choose between two different approaches depending
on the need for masking. If masking is necessary, we will use masked
scatter/gather, with all the lanes getting the same address. If, on the other
hand, masking is not required, we will keep the scalar instruction and if it is
a load instruction, use a vector splat for the loaded value.

In the third case we could generate the addresses of the sequential elements and
use the same approach as in the first two but there is a much better solution.
All we need to do is perform a vector memory operation of width `N` with the
same pointer as the scalar version of the instruction. This will efficiently
load `N` elements from the memory into a vector. This is usually the most
optimal way to load and store vectors.

Finally, for the fourth case, we will use an interleaved memory operation
internal builtin. This builtin takes the base pointer and the stride of the
memory operation, so calculating each individual address is (in theory, see next
paragraph) not required.

How the internal builtins are implemented in practice differs based on the
target hardware but a description of the generic version emitted by Vecz can be
found in the [Defining Internal Builtins](#defining-internal-builtins) section.

> The relevant functions and classes can be found in
> `source/include/transform/memory_operations.h` and
> `source/transform/memory_operations.cpp`.

### Packetizing Phi Nodes

Phi nodes are a bit more tricky to packetize as they introduce cycles in the
def/use chain. To avoid this, an empty vector phi node (i.e. with no incoming
value) is created at first, when packetizing from the leaves. Once all the
leaves have been packetized, the incoming values of each empty phi is packetized
in turn. Since packetizing an incoming value may involve packetizing a new phi,
this process needs to be repeated until all phi nodes have been handled.

### Packetizing OpenCL Builtins

Since all the OpenCL builtins are known, we can use a special technique
for easily and efficiently packetizing many of them. Many of the builtins
already have vector equivalents, so we can just use them instead of
vectorizing the scalar version (for that approach see [Vectorizing Builtin
Functions](#vectorizing-builtin-functions) instead). This is done by first
determining if it is safe to use the vector equivalent. For example, the vector
version of the `length` builtin does not operate element-wise, so we cannot use
it.

After we make sure that there are no known issues with the vector version of
the builtin, we construct the expected function signature based on the scalar
and the vector types that we have. We then search for the function matching
that signature in the current module, and also in the builtins module (if it is
available). If the function is found then we can use it, otherwise we report
that vectorizing the builtin failed. This means that this step can detect
builtins that have no vector versions but have been vectorized from the scalar
version using Vecz.

> The builtin information code can be found in
> `include/vecz/vecz_builtin_info.h`,
> `source/include/cl_builtin_info.h`, and
> `source/cl_builtin_info.cpp`.

### Instantiation

Instantiating a value or instruction means evaluating it in the context of each
SIMD lane and creating a separate copy for each lane. Instantiating a call to
`get_global_id(0)` or `get_local_id(0)` results in the SIMD lane's global or
local ID. If the instruction has varying operands, they need to be instantiated
(and so on recursively) too. Even though looking at the source code it looks as
if the instantiation is handled by a different pass, it is in the packetization
pass that calls for instructions to be instantiated when necessary. In turn, the
instantiator will call back in the packetization pass when it determines that it
shouldn't instantiate an instruction.

When an instruction should be instantiated or not is determined by the
Instantiation Analysis. This analysis goes through all the instructions in the
function and looks for instructions that we know we shouldn't try to packetize,
such as `printf` calls, instructions that have types that we cannot create a
vector with, or the masked user functions we talked about in the [Control Flow
Conversion](#control-flow-conversion-pass) section.

> The analysis pass can be found in
> `source/include/analysis/instantiation_analysis.h` and
> `source/analysis/instantiation_analysis.cpp`.
> The transform pass can be found in
> `source/include/transform/instantiation_pass.h` and
> `source/transform/instantiation_pass.cpp`.

As an example of what instantiation looks like in the actual IR, let's say that
we have the following code snippet that loads a variable:

```c
... = in[get_global_id(0)]
```

The IR for the snippet looks like this:

```
%call = call spir_func i64 @_Z13get_global_idj(i32 0)
%arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %in, i64 %call
%0 = load i32, i32 addrspace(1)* %arrayidx, align 4
```

If this code was to be instantiated, it would look like this:

```
%call = call spir_func i64 @_Z13get_global_idj(i32 0) #2
%arrayidx0 = getelementptr inbounds i32, i32 addrspace(1)* %in1, i64 %call, i64 0
%arrayidx1 = getelementptr inbounds i32, i32 addrspace(1)* %in1, i64 %call, i64 1
%arrayidx2 = getelementptr inbounds i32, i32 addrspace(1)* %in1, i64 %call, i64 2
%arrayidx3 = getelementptr inbounds i32, i32 addrspace(1)* %in1, i64 %call, i64 3
%0 = load i32, i32 addrspace(1)* %arrayidx0, align 4
%1 = load i32, i32 addrspace(1)* %arrayidx1, align 4
%2 = load i32, i32 addrspace(1)* %arrayidx2, align 4
%3 = load i32, i32 addrspace(1)* %arrayidx3, align 4
```

Note how each load takes a different memory address which depends on the SIMD
lane index and the current global ID.

Instantiation is usually done when a varying instruction cannot be packetized,
e.g. calls to functions like `printf` which have no SIMD equivalent. The above
example does not require instantiation as the scalar load can simply be turned
into a vector load. The example was simply given for demonstration purposes, an
actual example can be found in the [Packetization](#packetization) section.

### Vectorizing Builtin Functions

Sometimes vectorizing kernels is not enough and the OpenCL builtin functions
called by the kernel have to be vectorized too. One example is `isequal` which
returns `1` if the two arguments have the same value or `0` if they don't or if
one of the argument is NaN. The IR implementation is simple:

```
define spir_func i32 @_Z7isequalff(float %x, float %y) {
entry:
  %cmp.i = fcmp oeq float %x, %y
  %conv.i = zext i1 %cmp.i to i32
  ret i32 %conv.i
}
```

The first step to vectorize this builtin function is declare the vectorized
builtin:

```
declare spir_func <4 x i32> @__vecz_v4__Z7isequalff(<4 x float> %x, <4 x float> %y)
```

The second step is to copy the instructions from the original function to the
vectorized function. One issue is that the function arguments now have type
`<4 x float>` instead of `float` which prevents copying instructions that
refer to the original arguments. One way to work around this is to create
`extractelement` instructions to act as placeholders for the arguments.
Instructions that referred to the old arguments are changed to refer to the
relevant placeholder instead:

```
define spir_func <4 x i32> @__vecz_v4__Z7isequalff(<4 x float> %x, <4 x float> %y) {
entry:
  %placeholder_x = extractelement <4 x float> %x, i32 0
  %placeholder_y = extractelement <4 x float> %y, i32 0
  %cmp.i = fcmp oeq float %placeholder_x, %placeholder_y
  %conv.i = zext i1 %cmp.i to i32
  ret i32 %conv.i
}
```

Placeholders instructions are marked as such so that they are not mistaken with
regular instructions. When the placeholder needs to be packetized, it is
replaced with the actual argument:

```
define spir_func <4 x i32> @__vecz_v4__Z7isequalff(<4 x float> %x, <4 x float> %y) {
entry:
  %cmp.i1 = fcmp oeq <4 x float> %x, %y
  %conv.i2 = zext <4 x i1> %cmp.i1 to <4 x i32>
  ret <4 x i32> %conv.i2
}
```

## Post-Vectorization Optimizations and Cleanup

After the vectorization process is completed, we run some additional passes to
further optimize and cleanup the code:

* Inline Post Vectorization Pass (from Vecz)
* CFG Simplification Pass (from LLVM)
* Global Value Numbering Pass (from LLVM)
* Dead Code Elimination Pass (from LLVM)
* Interleaved Group Combining pass (from Vecz)
* Instruction Combining pass (from LLVM)
* Masked Memory Operations Simplification pass (from Vecz)
* Internal Builtin Definition pass (from Vecz)

> The passes can be found in the files
> `source/include/transform/passes.h` and
> `source/transform/passes.cpp`.

### Inline Post Vectorization Pass

The Inline Post Vectorization Pass is responsible for inlining builtins that
have no vector/scalar equivalent or called functions that don't have the
`NoInline` attribute.

### Interleaved Group Combining Pass

The Interleaved Group Combining pass is responsible for lowering groups of
interleaved memory operations into vector memory operations. Specifically,
if there is a group of `K` interleaved operations with stride `K`, each
accessing the elements in between the others, they will be transformed into
`K` consecutive vector operations. For example, if we have the interleaved
operations *A*, *B*, *C*, and *D* (with a number next to the letter signifying
the element index),

```
-------------------------------------------------
|A1|B1|C1|D1|A2|B2|C2|D2|A3|B3|C3|D3|A4|B4|C4|D4|
-------------------------------------------------
```

They will be optimized into the vector operations *a*, *b*, *c*, and *d*.

```
-------------------------------------------------
|a1|a2|a3|a4|b1|b2|b3|b4|c1|c2|c3|c4|d1|d2|d3|d4|
-------------------------------------------------
```

The first pattern commonly appears after scalarizing vector memory operations in
a kernel and then revectorizing each one of them into a vector instructions.

> The pass can be found in
> `source/include/transform/interleaved_group_combine_pass.h` and
> `source/transform/interleaved_group_combine_pass.cpp` while some of
> the optimization code can be found in `include/vecz_target_info.h`,
> `source/vector_target_info.cpp` and
> `source/vector_target_info_arm.cpp`.

### Masked Memory Operations Simplification Pass

This pass is responsible for lowering masked operations into unmasked or nop
operations, assuming that we can determine the mask values at compile time.
If all the lanes in the mask are set to `true` then the mask is unnecessary
and the operation can be lowered to the equivalent unmasked operation. If,
on the other hand, all the mask lanes are set to `false`, the operation will
not be executed at all and it can thus be replaced by a nop. Note that such
optimizations are only possible if the mask values are known at compile time, as
runtime optimizations need to be handled separately, specifically when the code
for the internal builtins is generated.

> The pass can be found in `vectorizer.h` and `vectorizer.cpp`.

### Defining Internal Builtins

We have already mentioned how the internal builtins are used in
the [Control Flow Conversion](#control-flow-conversion-pass) and
[Packetization](#packetization) sections. We have the following internal
builtins:

* masked load/store
* interleaved load/store
* masked interleaved load/store
* scatter store / gather load
* masked scatter store / gather load

The masked versions perform the same operation as their unmasked counterparts,
with the exception that the operation is only performed for the lanes for which
the mask is `true`.

For the masked loads and stores, as well as the masked scatter stores and gather
loads, LLVM provides intrinsics that perform these operations. How the
intrinsics are implemented obviously depends on the backend.

However, LLVM does not provide intrinsics for the remaining operations,
interleaved loads and stores, masked interleaved loads and stores, and unmasked
scatter stores and gather loads. Assuming that the masked scatter/gather
intrinsics that LLVM provides are at least as efficient as manually performing
each memory operation separately and then collecting them into a vector, we use
those LLVM intrinsics for these operations as well. For the interleaved
operations, we first need to generate all the pointers, using the pointer base
and the stride, and then call the LLVM intrinsic.

In case that intrinsic generation fails, we define the function by emulating the
vector with appropriate masking when required, which of course is suboptimal.

This design is the default used in Vecz but since it is modular, it is possible
to change the implementation for any target that Vecz is ported to.
Specifically, in the `vector_target_info.cpp` file exists a number of
`createX` functions (where `X` is the internal builtin name, e.g.
"`MaskedLoad`") where the actual code for the builtins is generated. The
functions are very generic; they take an `IRBuilder` and the required pointers
and values, so it is easy to modify them without having to modify any other part
of the vectorizer. Their code can be replaced with more optimal and target
specific code. It can also be modified to solve any issues the target might have
with the current internal builtins implementation.

> Note: The current implementation for scatter/gather uses an `addrspacecast`
> instruction in order to use the LLVM intrinsics with pointers having an
> address space other than 0. This works for the x86 implementation but
> it might not work on other architectures.
>
> Note: The interleaved memory operations use the same fallback as the masked
> interleaved ones.
>
> The pass and the relevant materialization code can be found in
> `source/include/vectorization_context.h`,
> `source/vectorization_context.cpp`, `include/vecz/vecz_target_info.h`,
> and `source/vector_target_info.cpp`.

### Cloning of the OpenCL Metadata

After the vectorization process has been completed, and only if it has been
successful, we update the OpenCL metadata in the module to include the
vectorized kernel. This isn't done by a pass, it's just a function call
at the end of the main vectorizer pass. Since some of the metadata requires
information known only by the frontend compiler (clang), we use the existing
metadata by cloning the relevant nodes and then replacing the function pointer.

> Note: When copying the metadata, we do not adjust the workgroup size, even
> though we are now executing fewer work items. Since each invocation of the
> kernel is now performing the work of `N` scalar kernels, where `N` the vector
> width, we only need to execute `1/N` work items for each workgroup.

## Miscellaneous

This section covers various necessary utilities that the main vectorizer passes
are using.

### Builtins Information

Vecz handles OpenCL builtins specially, so we need to be able to identify them,
query for various characteristics, and of course pull their definition or
implementation into the current module. This is all handled by the builtins info
code found in `vecz_builtins_info.h`, `cl_builtin_info`, and
`cl_builtin_info.cpp`. The `BuiltinInfo` class allows us to:

* Identify builtins based on their name.
* Identify various characteristics of a builtin, such as if it safe to vectorize
  it, or if it is a builtin we need to handle specially.
* Determine if a builtin is uniform.
* Determine if a builtin has side-effects.
* Get the vector or scalar equivalent of a builtin.
* Materialize a builtin from the builtins module.
* Emit a custom inline implementation for specific builtins.

While the code is mostly OpenCL centered, it can also handle LLVM intrinsics,
and it can be expanded to handle any builtin functions necessary.

As far as the identification of a builtin and its properties, parts of it are
done in a generic way that works with all the builtins, and parts are hardcoded
by the developers. For example, demangling a builtin and getting its name can be
easily done automatically, while determining if a builtin returns a global or
local ID needs to be hardcoded by the developers. For this reason, we have a
large list of known builtins that have a set of special characteristics, while
any builtin omitted from this list is assumed to conform to some default set of
characteristics.

### Function and Type Mangler and Demangler

The SPIR standard mandates that all OpenCL builtins needs to be mangled
according to the Itanium name mangling convention, with some additional
extensions. Normally, the mangling of function names is handled by the frontend
of the compiler (clang), since it depends on language specific types but we can
correctly identify and mangle/demangle all the OpenCL primitives, vectors, and
image types at the IR level as well.

Mangling is used by the builtins module to determine the name of a newly
generated builtin (for example when creating the vector equivalent of a
builtin), while demangling is used to identify the builtins. Furthermore, other
parts of the vectorizer use the mangler/demangler for their own purposes.

Currently, we only support a subset of the Itanium mangling rules but this is
enough for most OpenCL kernels. For example, we cannot mangle `struct` types, as
we cannot easily map between the C type name and the LLVM one.

> The relevant files are `include/vecz/vecz_mangling.h` and
> `source/mangling.cpp`.

### Stride and Offset Information

As we explained in previous sections ([Packetization](#packetization), [Defining
Internal Builtins](#defining-internal-builtins)), it is necessary for Vecz to be
able to determine the access pattern of a memory operation. This essentially
involves three attributes: the base pointer, the offset, and the stride.
These can be determined with the help of the `OffsetInfo` class found in the
`offset_info.h` and `offset_info.cpp` files.

The class accepts a pointer and tries to determine the offset and the stride of
the pointer. This is done by tracing the base of the pointer and keeping track
of the operations performed on the way. As an example

```c
kernel void fn(global int *in) {
  size_t tid = get_global_id(0);
  global int* ptr = (in + tid) * 2;
  ...
}
```

The `ptr` pointer value in this kernel is calculated by adding the global ID to
it and then multiplying it by 2. Its base pointer is the `in` kernel argument,
which is uniform. `OffsetInfo` can determine that this pointer had an offset
depending on the global ID and that it has a stride of 2 elements (this is
similar to the scalar evolution analysis used for loop optimizations). Having
this information, we now know that any memory accesses using `ptr` needs to be
packetized as interleaved memory operations with a stride of 2.

### Vectorization Dimension

User may decide to vectorize the code on whichever of the possible three
dimension the workgroup is composed of. Vecz refers to them as dimension `0`
(x), dimension `1` (y), and dimension `2` (z). Vecz supports this configuration
via an additional parameter. This parameter directly affects the [Uniform
Analysis](#uniform-analysis), the [Packetization](#packetization), and the
[Stride and Offset Information](#stride-and-offset-information). If no
parameter is specified, vectorization on the x dimension is assumed.

### Vecz Choices

"Choices" are options that the programmer can select regarding various aspects
of the vectorization process. For example, it is possible to set Vecz to always
vectorize uniform instructions. This is handled by the `VectorizationChoices`
class. The choices can be set in three different ways:

1. By modifying the code to explicitly enable or disable a choice. This is meant
   to be used by developers when optimizing Vecz for a custom target.
2. By passing the `-vecz-choice=<Choice>` flag to `opt`. This is meant to be
   used for testing and debugging purposes. More details for this option can be
   found through `opt`'s `help` function.
3. By setting the `CODEPLAY_VECZ_CHOICES` environment variable.
4. By calling the `addChoice` method from the `Vectorizer` class.

The `CODEPLAY_VECZ_CHOICES` variable accepts a string of Choices separated by
a semicolon (`;`) character. More details, as well as the choices available, can
be found in the `vecz_choices.h` file.

> The `VectorizationChoices` class can be found in
> `include/vecz/vecz_choices.h` and
> `source/vectorization_choices.cpp`

## Obtaining Vectorization Statistics

LLVM has support for collecting counters (called statistics in LLVM) from the
passes. Vecz is among the passes that can produce such information. This can be
done in two ways.

First, the official way is to use `opt` with the `-stats` option. This will
print the statistics from all the passes that have any.

The second way is to pass the `-cl-llvm-stats` option to the oneAPI
Construction Kit. This will do pretty much the same work that the `-stats`
option does, but it can be used in cases where it is not possible to use `-stats`.

## Optimization Remarks

Vecz utilizes the Remarks system available in LLVM, mostly to warn about
vectorization failures. The remarks can be enabled by passing the
`-pass-remarks=vecz` and `-pass-remarks-missed=vecz` command line option to
 `opt`, or the `-v` flag to `oclc`.

## veczc - the VECZ Compiler

The command line tool veczc is a standalone compiler that is used to vectorize
LLVM bitcode binary files. Its main use is in our vecz LIT-based testing (see
modules/compiler/vecz/test).

It has the following arguments:

* -o `file` output bitcode file
* -w `width` the width to vectorize the code to
* -d `dimension` the dimension index to vectorize the code on

* -k `name` the function names to select for vectorization. It can appear
  multiple times, in one of several forms. In the standard form, simply passing
  the names of kernels will ensure those kernels are vectorized by the globally
  selected parameters
  e.g.
    `veczc -k foo -k bar ...`
  selects both the `foo` and `bar` kernels for
    vectorization.
  The more complex form allows specifying the vectorization parameters
  *per*-kernel in multiplicate:
  e.g.
    `veczc -k foo:4,8,16 ...`
  will generate multiple vectorized versions of `foo` at vectorization factors
  of 4, 8, and 16. All other paremeters will be inherited from the global
  configuration.
  The complete syntax for the kernel specification switch (`k`) value is as follows:

   ```bnf
   <kernel_spec> ::= <kernel_name> ':' <spec>
   <kernel_spec> ::= <kernel_name>
   <spec> ::= <vf><dimension>(opt)<width>(opt)<scalable_spec>(opt)<auto>(opt)
   <spec> ::= <spec> ',' <spec> // multiple specs are comma-separated
   <number> ::= [0-9]+ // a decimal integer
   <kernel_name> ::= [a-zA-Z_][a-zA-Z_0-9]+ // As in the simple form - the name of the kernel to vectorize
   <dim> ::= '.' [123] // Vectorize only the given dimension
   <vf> ::= <number> // vectorize by the given factor
   <vf> ::= 'a' // automatic vectorization factor
   <simd_width> ::= '@' <number> // Assume local size (SIMD width) is the given number
   <scalable_spec> ::= 's' // Turn on scalable vector support
   ```
  n.b. (There should be no whitespace as this interface is designed for easy
  nonquoted use in common shells)

It supports bitcode files with the following target triples:

* `spir-unknown-unknown` 32-bit SPIR binaries
* `spir64-unknown-unknown` 64-bit SPIR binaries

Because veczc doesn't load all of the builtins prior to vectorization,
declarations of scalar or vector versions of any builtins used in the input file
must be present, otherwise scalarization or packetization will not be able to
materialize the scalarized/vectorized builtin calls and veczc will fail with an
error message.

## References

[1]: http://dblp.uni-trier.de/pers/hd/k/Karrenberg:Ralf
