# Parallel For Range Rounding

Kernels to be executed using a `sycl::range`, and not a `sycl::nd_range`,
may have their execution space reconfigured by the SYCL runtime. This is done
since oddly shaped execution dimensions can hinder performance, especially when
executing kernels on GPUs. It is worth noting that although the
`sycl::parallel_for` using a `sycl::range` does not expose the concept of a
`work_group` to the user, behind the scenes all GPU APIs require a work group
configuration when dispatching kernels. In this case the work group
configuration is provided by the implementation and not the user.

As an example, imagine a SYCL kernel is dispatched with 1d range `{7727}`. Since
7727 is a prime number, there is no way to divide this kernel up into workgroups
of any size other than 1. Therefore 7727 workgroups are dispatched, each with
size 1. Because of the parallel nature of execution on modern GPUs, this
results in low occupancy, since we are not using all of the available work items
that execute in lockstep in each (implicit) subgroup. This can hinder
performance.

To mitigate the performance hit of choosing an awkward implicit workgroup size,
for each kernel using a `sycl::range`, the SYCL runtime will generate two
kernels:

1. The original kernel without any modifications.
2. The "Range rounded" kernel, which checks the global index of each work item
   at the beginning of execution, exiting early for a work item if the global
   index exceeds the user provided execution range. If the original kernel has
   the signature `foo`, then this kernel will have a signature akin to
   `_ZTSN4sycl3_V16detail19__pf_kernel_wrapperI3fooEE`.

In this way, if a range rounded kernel is executed at runtime, a kernel
dispatched with the range `{7727}` may be executed by `{7808}` work items,
where work items `{7727..7807}` all exit early before doing any work. This would
give much better performance on a GPU platform since the implementation can use
the implicit `nd_range` `{7808, 32}`, which corresponds to a workgroup size of
32, instead of `{7727, 1}`, which corresponds to a workgroup size of 1.

The parallel for range rounding will only be used in the X (outermost)
dimension of a `sycl::range`, since if the inner dimensions are changed by the
SYCL runtime this can change the stride offset of different dimensions. Range
rounding will only be used if the SYCL runtime X dimension exceeds some minimum
value, which can be configured using the
`SYCL_PARALLEL_FOR_RANGE_ROUNDING_PARAMS` environment variable.

Generation of range rounded kernels can be disabled by using the compiler flag
`-fsycl-disable-range-rounding`.
