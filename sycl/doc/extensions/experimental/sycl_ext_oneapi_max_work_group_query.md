# sycl_ext_oneapi_max_work_group_query

## Notice

This document describes an **experimental** API that applications can use to try
out a new feature.  Future versions of this API may change in ways that are
incompatible with this experimental version.  


## Introduction

This extension adds functionally two new device information descriptors. They provide the ability to query a device for the maximum numbers of work-groups that can be submitted in each dimension as well as globally (across all dimensions).

OpenCL never offered such query - which is probably why it is absent from SYCL. Now that SYCL supports back-ends where the maximum number of work-groups in each dimension can be different, having the ability to query that limit is crucial in writing safe and portable code.

## Feature test macro

As encouraged by the SYCL specification, a feature-test macro, `SYCL_EXT_ONEAPI_MAX_WORK_GROUP_QUERY`, is provided to determine whether this extension is implemented.

## New device descriptors

| Device descriptors                                     | Return type | Description                                                                                                                                                                                                             |
| ------------------------------------------------------ | ----------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ext::oneapi::experimental::info::device::max_work_groups<1>     |  id<1>      | Returns the maximum number of work-groups that can be submitted in each dimension of the `globalSize` of a `nd_range<1>`. The minimum value is `(1)` if the device is different than `info::device_type::custom`.       |
| ext::oneapi::experimental::info::device::max_work_groups<2>     |  id<2>      | Returns the maximum number of work-groups that can be submitted in each dimension of the `globalSize` of a `nd_range<2>`. The minimum value is `(1, 1)` if the device is different than `info::device_type::custom`.    |
| ext::oneapi::experimental::info::device::max_work_groups<3>     |  id<3>      | Returns the maximum number of work-groups that can be submitted in each dimension of the `globalSize` of a `nd_range<3>`. The minimum value is `(1, 1, 1)` if the device is different than `info::device_type::custom`. |
| ext::oneapi::experimental::info::device::max_global_work_groups |  size_t     | Returns the maximum number of work-groups that can be submitted across all the dimensions. The minimum value is `1`.                                                                                                    |

### Note

- The returned values have the same ordering as the `nd_range` arguments.
- The implementation does not guarantee that the user could select all the maximum numbers returned by `max_work_groups` at the same time. Thus the user should also check that the selected number of work-groups across all dimensions is smaller than the maximum global number returned by `max_global_work_groups`.

## Examples

```c++
sycl::device gpu = sycl::device{sycl::gpu_selector_v};
std::cout << gpu.get_info<sycl::info::device::name>() << '\n';

#ifdef SYCL_EXT_ONEAPI_MAX_WORK_GROUP_QUERY
sycl::id<3> groups = gpu.get_info<sycl::ext::oneapi::experimental::info::device::max_work_groups<3>>();
size_t global_groups = gpu.get_info<sycl::ext::oneapi::experimental::info::device::max_global_work_groups>();
std::cout << "Max number groups: x_max: " << groups[2] << " y_max: " << groups[1] << " z_max: " << groups[0] << '\n';
std::cout << "Max global number groups: " << global_groups << '\n';
#endif
```

Ouputs to the console:

```
NVIDIA ...
Max number groups: x_max: 2147483647 y_max: 65535 z_max: 65535
Max global number groups: 2147483647
```

See: [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities)

Then the following assertions should be satisfied at kernel submission:

```C++
sycl::nd_range<3> work_range(global_size, local_size);

assert(global_size[2] <= groups[2]
    && global_size[1] <= groups[1]
    && global_size[0] <= groups[0]);

assert(global_size[2] * global_size[1] * global_size[0] <= global_groups); //Make sure not to exceed integer representation size in the multiplication.

gpu_queue.submit(work_range, ...);
```
## Deprecated queries

The table below lists the soon to be removed deprecated descriptors and their replacements:

|Deprecated Descriptors| Replacement Decriptors|
| -------------------- | --------------------  |
| sycl::info::ext_oneapi_max_global_work_groups |sycl::ext::oneapi::experimental::info::max_global_work_groups |
| sycl::info::ext_oneapi_max_work_groups_*N*d | sycl::ext::oneapi::experimental::info::max_work_groups\<*N*\> |

* Note *N* can take the value 1,2, or 3


## Implementation

### Consistency with existing checks

The implementation already checks when enqueuing a kernel that the global and per dimension work-group number is smaller than `std::numeric_limits<int>::max`. This check is implemented in `sycl/include/sycl/handler.hpp`. For consistency, values returned by the two device descriptors are bound by this limit.

### Example of returned values

- If the device is the host or has an OpenCL back-end, the values returned - as they are not applicable - are the maximum values accepted at kernel submission (see `sycl/include/sycl/handler.hpp`) which are currently `std::numeric_limits<int>::max`.
- CUDA: Back-end query using `CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_[X,Y,Z]`.
