# Implementation design for parallel loop annotations and `ivdep`

This document describes the LLVM and SPIR-V implementation for representing
loop-carried memory dependence distance (or lack-there off) for kernel loops
with parallelism annotations like ivdep.

## SYCL Attributes

### Attributes for specifying no loop-carried dependences: `[[intel::ivdep]]` and `[[intel::ivdep(array)]]`

When placed on a kernel loop, these attributes indicate that there are no
loop-carried memory (data) dependences on all pointer accesses (`ivdep`) or on the specified
arrays (`ivdep(array)`). This annotation allows the compiler to avoid making
conservative assumptions when it cannot infer the loop-carried dependences on a
specific array. For example consider the following kernel code:

```
q.single_task([=] {
    for (int i = 0; i < N; ++i) {
        B[i] += B[idx[i]];
    }
}).wait();
```

Since the access `B[idx[i]]` is unknown at compile time, the compiler will have
to make the conservative assumption that there is a loop carried dependence of
distance 1 between the write to B and the read from B (in both directions) when
it tries to parallelize the loop. If however `ivdep` is used, then the
annotation tells the compiler that there are no loop-carried dependences on `B`,
and the compiler can make parallelism decisions without this assumption.

```
q.single_task([=] {
    [[intel::ivdep]]
    for (int i = 0; i < N; ++i) {
        B[i] += B[idx[i]];
    }
}).wait();
```

### Attributes for specifying loop-carried dependence distance: `[[intel::ivdep(safelen)]]` and `[[intel::ivdep(array,safelen)]]`

When placed on a kernel loop, these attributes indicate that any loop-carried
dependences on all pointer accesses (`ivdep(safelen)`) or on the specified
arrays/pointers (`ivdep(array,safelen)`) have a distance of at least `safelen`.
This can be used to disambiguate any aliasing when the compiler cannot
automatically infer the dependence distance and instead has to make the
conservative assumption that the distance is 1.

### Marking parallel loops

We are considering adding a mechanism for users to indicate when a kernel loop
is parallel. THe difference between this and `ivdep` is that a parallel loop
implies that memory-ordering dependences can be ignored, in addition to memory
data dependences. Marking parallel loops can either be done using new annotation
`[[intel::parallel]]`, a new lambda-style `parallel_for` API, or using SYCL
nested parallelism.

## LLVM Metadata changes

We are proposing the following new LLVM metadata:

1. A new metadata, `llvm.loop.no_depends`, should be generated on loops that are
   marked with the versions of `ivdep` that don't take a `safelen` (i.e.
   `intel::ivdep` and `intel::ivdep(array)`). It will replace the
   previously-generated `llvm.loop.parallel_access_indices` due to the mismatch
   in the semantics of `llvm.loop.paralllel_accesses` and `ivdep`. The semantics
   are that all accesses with access groups listed in this metadata have no
   loop-carried memory data dependences.
2. A new metadata, `llvm.loop.no_depends_safelen` should be generated on loops
   that are marked with the versions of ivdep that take a safelen (i.e.,
   `intel::ivdep(safelen)` and `intel::ivdep(array, safelen)`). It will replace our
   `llvm.loop.parallel_access_indices_safelen` for the same reason described
   above. As with `llvm.loop.no_depends`, the semantics indicate that all accesses
   with access groups listed in this metadata have dependence distances of at
   least safelen.
3. The built-in `llvm.loop.parallel_accesses` metadata should be generated on
   loops that are marked as parallel, or in cases where the compiler knows that
   the loop is parallel. This metadata allows the compiler to not only ignore
   loop-carried data dependences, but also loop-carried memory-ordering dependences.
4. The built-in `llvm.access.group` metadata should be generated for all memory
   operations (loads, stores, and function calls) in the body of a loop that has
   either an ivdep attribute, or is marked as parallel (by the user or the
   compiler). The access group identifier will be passed to the corresponding
   loop metadata described above.

## SPIR-V Extension

The SPIR-V extension for this design doc is presented in [this sCART
Review](https://github.com/intel-innersource/documentation.xpu.architecture.spirv-extension-drafts/pull/162)
(will update link to Khronos repo when extension review is complete). This list
summarizes the additions:

1. Access groups defined using an LLVM `llvm.access.group` metadata are
represented as a literal in SPIR-V.
2. A new instruction `OpAGListINTEL` is used to aggregate multiple access groups
   into a single list.
3. A new Decoration `AccessGroupINTEL` is used to specify which access groups a
   function call or atomic instruction belongs to.
4. A new Memory Operand `AccessGroupINTELMask` is used to specify which access
   groups a memory operation belongs to.
5. Two new Loop Controls are added to represent `llvm.loop.parallel_accesses`
   and `llvm.loop.no_depends/no_depends_safelen`. They are called
   `ParallelAccessesINTEL` and `DependencyAccessesINTEL` accordingly.
   1. `ParallelAccessesINTEL` takes one or more `OpAGListINTEL` access group
      lists and specifies that for each list, all the accesses that belong to
      the specified access groups do not have loop-carried dependences.
   2. `DependencyAccessesINTEL` takes one or more pairs. The first element in
      each pair is a list of access groups from `OpAGListINTEL`, and the second
      element is an integer `S` that corresponds to the safelen. 

## Translation Rules

### SYCL to LLVM Metadata Translation

   1. For the different versions of ivdep:
      1. `[[intel::ivdep]]` on a loop should be translated to
         `llvm.access.group` metadata on all memory instructions and function
         calls in the loop body and the `llvm.loop.no_depends` metadata should
         be placed on the loop and passed the same access group(s). The front
         end is free to assign access groups as it sees fit as long as they all
         appear in the `llvm.loop.no_depends` metadata.
      2. `[[intel::ivdep(array)]]` on a loop should be translated to
         `llvm.access.group` metadata on all memory instructions that use the
         specified array, and `llvm.loop.no_depends` metadata should be placed
         on the loop and passed these access group(s). The front end is free to
         assign access groups as it sees fit as long as all the access groups
         for the accesses to the corresponding array appear in the
         `llvm.loop.no_depends metadata`. Multiple instances of `ivdep(array)`
         should generate a separate `llvm.loop.no_depends` metadata for each
         array.
      3. `[[intel::ivdep(safelen)]]` on a loop should be translated to
         `llvm.access.group` metadata on all memory instructions and function
         calls in the loop body and `llvm.loop.no_depends_safelen` metadata
         should be placed on the loop and passed the same access group(s) along
         with the safelen. The front end is free to assign access groups as it
         sees fit as long as they all appear in the
         `llvm.loop.no_depends_safelen` metadata. 
      4. `[[intel::ivdep(array, safelen)]]` should be translated to
        `llvm.access.group` metadata on all memory instructions that use the
        specified array, and `llvm.loop.no_depends_safelen` metadata should be
        placed on the loop and passed these access group(s). The front end is
        free to assign access groups as it sees fit as long as all the access
        groups for the accesses to the corresponding array appear in the
        `llvm.loop.no_depends_safelen` metadata. Multiple instances of
        ivdep(array, safelen) should generate a separate
        `llvm.loop.no_depends_safelen` metadata for each array.
      5. A few considerations:
         1. The front end is free to generate any access group and loop metadata
         it sees fit as long as it maintains the semantics that all the accesses
         that are independent (as indicated by the corresponding version of
         ivdep) have their access groups listed in the same loop metadata.  
         2. If overlapping attributes are detected (e.g. i`ntel::ivdep(a)` and
            `intel::ivdep(a,5)`) the front end is free to generate any or all
            metadata as it sees fit, and it must issue a warning to the user.
         3. In nested loops, outer loop metadata will have to take the access
           groups of both outer loop and inner loop accesses, but inner loop
           metadata will only take the inner loop accesses. This is to say that
           the access groups for inner loop accesses should be distinct from the
           access group for outer loop accesses that are not also in the inner
           loop.
   2. For the new parallel loop annotation, the front end should generate
   `llvm.access.group` metadata on all the memory accesses and function calls in
   the loop body, and a `llvm.loop.parallel_accesses` on the loop with that
   access group(s) as argument. As with ivdep, in nested loops the outer loop
   will have the access groups of the accesses in the outer loop as well as the
   accesses in the inner loop, whereas the inner loop will only have the access
   groups of the inner loop accesses. Additionally, the frontend is free to
   generate the access groups as it sees fit, as long as all the access groups
   are provided to the same `llvm.loop.parallel_accesses` metadata.

### LLVM Metadata to SPIR-V Translation

1. A new access group literal should be created for each `llvm.access.group`
   distinct metadata.
2. Memory operations that have an assigned access group should get the
   `AccessGroupINTELMask` memory operand.
3. Atomic operations with assigned access group should get decorated using the
   `AccessGroupINTEL` decoration.
4. For each instance of `llvm.loop.no_depends`, `llvm.loop.no_depends_safelen`,
   or `llvm.loop.parallel_accesses` on a given loop, a new `OpAGListINTEL`
   should be created to aggregate the access groups defined in the corresponding
   metadata.
5. All `llvm.loop.no_depends` and `llvm.loop.no_depends_safelen` metadata on a
   single loop should get translated to one `DependencyAccessesINTEL` loop
   control. The first operand of the loop control will be total number metadata
   instances. Then, for each metadata instances, a pair should be constructed
   and provided to the loop control. This pair consists of the aggregated access
   group list for that metadata, and either 0 (if the metadata is
   `llvm.loop.no_depends`) or the corresponding safelen (if the metadata is
   `llvm.loop.no_depends_safelen`).
6. All `llvm.loop.parallel_accesses` metadata on a single loop should get
   translated to one `ParallelAccessINTEL` loop control.  The first operand of the loop control will be total number metadata
   instances. Then, for each metadata instance, the corresponding aggregated
   access groups is provided to the loop control.
