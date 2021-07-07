# Device information descriptor: ext_oneapi_max_number_work_groups

This extension adds a new device information descriptor which returns a `sycl::id<3>` containing the maximum number of work-groups that can be submitted on a device, per dimension. 

OpenCL never offered such query, but now that SYCL supports GPU back-ends where these sizes are constant anymore, this query becomes mandatory. 

Examples:
```c++
#ifdef SYCL_EXT_ONEAPI_MAX_NUMBER_WORK_GROUPS
sycl::device cpu = sycl::device{sycl::cpu_selector{}};
sycl::id<3> cpu_sizes = cpu.get_info<sycl::info::device::ext_oneapi_max_number_work_groups>();
std::cout << cpu.get_info<sycl::info::device::name>() << '\n';
std::cout <<  "Max sizes: x_max: " << cpu_sizes[2] << " y_max: " << cpu_sizes[1] << " z_max: " << cpu_sizes[0] << '\n';
#endif
```
Returns:
```
Intel(R) Xeon(R) CPU ...
Max sizes: x_max: 2147483647 y_max: 2147483647 z_max: 2147483647
```


Whereas on a GPU:
```c++
sycl::device gpu = sycl::device{sycl::gpu_selector{}};
sycl::id<3> gpu_sizes = gpu.get_info<sycl::info::device::ext_oneapi_max_number_work_groups>();
std::cout << gpu.get_info<sycl::info::device::name>() << '\n';
std::cout << " Max sizes: x_max: " << gpu_sizes[2] << " y_max: " << gpu_sizes[1] << " z_max: " << gpu_sizes[0] << '\n';
```
Returns:
```
NVIDIA ...
Max sizes: x_max: 2147483647 y_max: 65535 z_max: 65535
```
See: [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities)

## Implementation
- For the host and openCL the values returned - as they are not applicable - are the maximum values accepted at kerrnel submission (see `sycl/include/CL/sycl/handler.hpp`) which are currently `std::numeric_limits<int>::max`. 

- CUDA: Backend query using `CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_[X,Y,Z]`.

## Caveat
There's no guarantee one could submit a kernel with all global sizes maxed out.
