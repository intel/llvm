# Test plan for [`sycl_ext_oneapi_launch_queries`][spec-link] extension

## Testing scope

### Device coverage

The unit tests should be launched on every supported device configuration we
have.

### Type coverage

Each query is templated on a single template type argument `Param`
with some queries being template overloads of each other.
Each query has a restriction on the value of `Param` defined
in the relevant section of the spec. Param can only be one of
`max_work_item_sizes<1>`, `max_work_item_sizes<2>`, `max_work_item_sizes<3>`,
`max_work_group_size`, `max_num_work_groups`, `max_sub_group_size`, and
`num_sub_groups` which are defined in the namespace 
`ext::oneapi::experimental::info::kernel`. 
The tests should cover all of these types.

## Tests

### Unit tests

#### Interface tests

These tests are intended to check that all classes and methods defined by the
extension have correct implementation, i.e.: right signatures, right return
types, all necessary constraints are checked/enforced, etc.

These tests should check the following:

- that each query is not available when the template argument `Param` has
  a value different than the one in the spec.
- that each query can be called with the appropriate value for `Param` and the
  appropriate argument types as defined by its signature.
- the return types of all queries match the spec.

Tests in this category may not perform some useful actions to exercise the
extension functionality in full, but instead they are focused on making sure
that all APIs are consistent with respect to other APIs.

#### Consistency of `max_subgroup_size` queries

The test needs to check that all three `ext_oneapi_get_info` queries for
which `Param == max_subgroup_size` return the same value as
`sub_group::get_max_local_range()` inside the kernel.

#### Check behavior in the case of invalid arguments to queries

For all queries for which `Param == max_num_work_groups` or
`Param == max_sub_group_size` or `Param == num_sub_groups`, verify that
a synchronous `exception` with the error code `errc::invalid` is thrown
if the work-group size `r` is 0.

#### Check return value of queries depending on queue submission status

Verify that if kernel submission to a queue does not throw then
the return value of each query on the queue with the given kernel 
is strictly greater than 0.

[spec-link]: https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/proposed/sycl_ext_oneapi_launch_queries.asciidoc
