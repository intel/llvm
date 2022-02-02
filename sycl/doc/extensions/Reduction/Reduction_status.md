# Reductions: current implementation and known problems/TODOs

**NOTE**: This document is a quick draft. It is written to help developers of SYCL headers/library to understand the current status, currently used algorithms and known problems.



# Reduction specifications

There are 2 specifications of the reduction feature and both are still actual:

* `sycl::ext::oneapi::reduction` is described in [this document](../deprecated/SYCL_EXT_ONEAPI_ND_RANGE_REDUCTIONS.md). This extension is deprecated, and was created as part of a pathfinding/prototyping work before it was added to SYCL 2020 standard.

* `sycl::reduction` is described in [SYCL 2020 standard](https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html#sec:reduction).

These two specifications for reduction are pretty similar. The implementation of `sycl::reduction` is based on (basically re-uses) the implementation of `sycl::ext::oneapi::reduction`. 

There are non-critical differences in API to create the reduction object. `sycl::reduction` accepts either `sycl::buffer` or `usm memory` and optional property `property::reduction::initialize_to_identity` as parameter to create a reduction, while `sycl::ext::oneapi::reduction` accepts `sycl::accessor` that has `access::mode` equal to either `read_write` (which corresponds to SYCL 2020 reduction initialized without `property::reduction::initialize_to_identity`) or `discard_write`(corresponds to case when `property::reduction::initialize_to_identity` is used).

---
---
# Implementation details: `reduction` in `parallel_for()` accepting `nd_range`

## Reduction inside 1 work-group - the main building block for reduction

The reduction algorithm for 1 work-group depends currently on combination/availability of 2 features:
- `fast atomics` (i.e. operations available on the target device), such as fetch_add(), fetch_min(), etc.
- `fast reduce` operation for work-group, i.e. `reduce_over_group()`

So, if the reduction operation/type has both `fast atomics` and `fast reduce`, then the reduction on work-group with 1 reduction variable does the following: the elements inside the work-group are reduced using `ext::oneapi::reduce` and the final result is atomically added to the final/global reduction variable.

```c++
    // Call user's lambda function or functor. Reducer.MValue gets initialized there.
    typename Reduction::reducer_type Reducer;
    KernelFunc(NDIt, Reducer);

    typename Reduction::binary_operation BOp;
    // Compute the partial sum for the work-group.
    // Then the 0-th work-iteam stores it to the final/global reduction variable.
    Reducer.MValue = ext::oneapi::reduce(NDIt.get_group(), Reducer.MValue, BOp);
    if (NDIt.get_local_linear_id() == 0)
      Reducer.atomic_combine(Reduction::getOutPointer(Out));
```

The most general case is when the `fast atomics` and `fast reduce` are not available.
It computes the partial sum using tree-reduction loop and stores the partial sum for the work-group to a global array of partial sums, which later must be also reduced (by using additional kernel(s)). This general case algorithm also requires allocation of a local memory for the tree-reduction loop:
```c++
    // Call user's functions. Reducer.MValue gets initialized there.
    typename Reduction::reducer_type Reducer(ReduIdentity, BOp);
    KernelFunc(NDIt, Reducer);

    size_t WGSize = NDIt.get_local_range().size();
    size_t LID = NDIt.get_local_linear_id();
    // Copy the element to local memory to prepare it for tree-reduction.
    LocalReds[LID] = Reducer.MValue;
    if (!IsPow2WG)
      LocalReds[WGSize] = ReduIdentity;
    NDIt.barrier();

    // Tree-reduction: reduce the local array LocalReds[:] to LocalReds[0]
    // LocalReds[WGSize] accumulates last/odd elements when the step
    // of tree-reduction loop is not even.
    size_t PrevStep = WGSize;
    for (size_t CurStep = PrevStep >> 1; CurStep > 0; CurStep >>= 1) {
      if (LID < CurStep)
        LocalReds[LID] = BOp(LocalReds[LID], LocalReds[LID + CurStep]);
      else if (!IsPow2WG && LID == CurStep && (PrevStep & 0x1))
        LocalReds[WGSize] = BOp(LocalReds[WGSize], LocalReds[PrevStep - 1]);
      NDIt.barrier();
      PrevStep = CurStep;
    }

    // Compute the partial sum/reduction for the work-group and store it to the global array of partial sums.
    if (LID == 0) {
      size_t GrID = NDIt.get_group_linear_id();
      typename Reduction::result_type PSum =
          IsPow2WG ? LocalReds[0] : BOp(LocalReds[0], LocalReds[WGSize]);
      if (IsUpdateOfUserVar) // depends on accessor read/write mode or on property::reduction::initialize_to_identity
        PSum = BOp(*(Reduction::getOutPointer(Out)), PSum);
      Reduction::getOutPointer(Out)[GrID] = PSum;
    }
```

The *additional kernel(s)* that reduce the partial sums look very similar to the *main* kernel, the only difference is that they do not call user's function.

---
TODO #1 (Performance): After the reduction accepting 'range' was implemented it became possible to simply re-use that implementation for the *additional kernel* runs. Instead of running the *additional kernel* several times (if the number of partial sums is still big), it is better to run the parallel_for accepting 'range' once.

---
TODO #2 (Performance): There are 4 implementations for `parallel_for()` accepting nd_range and 1 reduction variable now:
*  `fast atomics` and `fast reduce` (A)
*  `fast atomics` and `no fast reduce` (B)
*  `no fast atomics` and `fast reduce` (C)
*  `no fast atomics` and `no fast reduce` (D)

Accordingly to the latest performance results/experiments it may be more efficient to replace (A) and (B) with one implementation using `fast atomics` for 1) reducing inside work group - use atomic ops to 1 scalar allocated in group local memory, 2) use atomics to store the partial sum for work-group to final/global reduction variable.

---
TODO #3 (Performance): Currently, there is more or less efficient implementation for 1 reduction variable. Using `fast atomics` and `fast reduce` does give better results. The implementation for any number of reduction variables used in same `parallel_for()` currently uses only the most basic implementation using `tree-reduction` algorithm for all reduction operations/types even those that do have `fast atomics` or `fast reduce` algorithm.

---
---
# Implementation details: `reduction` in `parallel_for()` accepting `range`

For `parallel_for()` accepting 1 reduction variable the implementation chooses one of 3 algorithms depending on existence of `fast atomics` and `fast reduce` features in the target device, which gives 3 variants:
*  `fast atomics` (A)
*  `no fast atomics` and `fast reduce` (B)
*  `no fast atomics` and `no fast reduce` (C)

The implementation of reduction accepting `range` has more freedom comparing to `nd_range` case as the order of reducing elements in each work-items is not specificed and the work-group sizes not specified too.

The implementation queries the target device on the number of execution units and max work-group-size, after which it chooses number and size of work-groups, then calls `parallel_for()` accepting `nd_range` inside the original/user's `parallel_for()` accepting `range`. Each of work-items in that nd-range space gets 1 or more indices from the global index space for which it calls user's function.

For the case (A) `fast atomics` the implementation looks this way:

```c++
  // Note that this parallel_for() is not user's parallel_for(). It is called inside user's one.
  CGH.parallel_for<Name>(NDRange, [=](nd_item<1> NDId) {
    // Call user's functions. Reducer.MValue gets initialized there.
    typename Reduction::reducer_type Reducer;
    reductionLoop(Range, Reducer, NDId, KernelFunc); // Each work-item handles 1 or many indices
                                                     // from the original/user's global index space.

    // Store the accumulated partial sum for the work-item to local var holding the partial sum for the work-group.
    auto LID = NDId.get_local_id(0);
    if (LID == 0)
      GroupSum[0] = Reducer.getIdentity();
    sycl::detail::workGroupBarrier();
    Reducer.template atomic_combine<access::address_space::local_space>(
        &GroupSum[0]);

    // Store the partial sum for the work-group to the final/global reduction variable 
    sycl::detail::workGroupBarrier();
    if (LID == 0) {
      Reducer.MValue = GroupSum[0];
      Reducer.template atomic_combine(Reduction::getOutPointer(Out));
    }
  });
```
Variants (B) and (C) use the same approach. The only difference is how the partial sums accumulated per work-items are combined into a partial sum for the work-group. Instead of fast atomic to a local scalar either the `fast reduce` or `tree-reduce-loop` used there. 

---

TODO #4 (Performance): The `reductionLoop()` has some order in which it choses indexes from the global index space. Currently it has huge stride to help vectorizer and get more vector insturction for the device code, which though may cause competition among devices for the memory due to pretty bad memory locality. On two-socket server CPUs using smaller stride to prioritize better memory locality gives additional perf improvement. 

---
TODO #5 (Performance): Some devices may provide unique-thread-id where the number of worker threads running simultaneously is limited. Such feature opens way for more efficient implementations (up to 2x faster, especially on many stacks/tiles devices). See this extension for reference: https://github.com/intel/llvm/pull/4747

---
---

# Missing `reduction` functional features to be implemented soon:

### 1) Support `sycl::queue::parallel_for()` shortcuts accepting reduction variables.
Currently, only `sycl::queue::parallel_for()` accepting `nd_range` and 1 reduction variable is supported.
The rest of this work is temporarily blocked by XPTI instrumentation that need the last argument to be hidden from user and having a default value, which is not possible if `parallel_for` starts accepting parameter packs like this one:
```c++
 // Parameter pack acts as-if: Reductions&&... reductions, const KernelType &kernelFunc
 template <typename KernelName, int Dims, typename... Rest>
 event parallel_for(range<Dims> numWorkItems, Rest&&... rest);
```

The problem is known, the fix in SYCL headers is implemented: https://github.com/intel/llvm/pull/4352 and is waiting for some re-work in XPTI component that must be done before the fix merge.

---
### 2) Support `parallel_for` accepting `range` and having `item` as the parameter of the kernel function. 
Currently only kernels accepting `id` are supported.

---
### 3) Support `parallel_for` accepting `range` and 2 or more reduction variables.
Currently `parallel_for()` accepting `range` may handle only 1 reduction variable. It does not support 2 or more. 

The temporary work-around for that is to use some container multiple reduction variables, i.e. std::pair, std::tuple or a custom struct/class containing 2 or more reduction variables, and also define a custom operator that would be passed to `reduction` constructor.
Another work-around is to provide `nd_range`.

---
### 4) Support `parallel_for` accepting `reduction` constructed with `span`:
```c++
template <typename T, typename Extent, typename BinaryOperation>
__unspecified__ reduction(span<T, Extent> vars, const T& identity, BinaryOperation combiner);
```

---
### 5) Support identity-less reductions even when the reduction cannot be determinted automatically.

Currently identity-less reductions are supported, but only in cases when sycl::has_known_identity<BinaryOperation, ElementType> returns true.
When sycl::has_known_identity returns false, the implementation of the reduction may be less efficient, but still be functional.

---
## Note the document is not finished. It should be updated with more details soon.
