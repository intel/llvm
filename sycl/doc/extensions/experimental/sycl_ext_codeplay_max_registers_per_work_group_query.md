# sycl_ext_codeplay_max_registers_per_work_group_query

## Notice

This document describes an **experimental** API that applications can use to try
out a new feature. Future versions of this API may change in ways that are
incompatible with this experimental version.

## Introduction

This extension adds a new device information descriptor that provides the
ability to query a device for the maximum number of registers available per
work-group.

OpenCL never offered such query due to the nature of being a very platform
specific one - which is why it is also absent from SYCL. Now that SYCL supports
back-ends where the register usage is a limiting resource factor of the possible
maximum work-group size for a kernel, having the ability to query that limit is
important for writing safe and portable code.

## Feature test macro

As encouraged by the SYCL specification, a feature-test macro,
`SYCL_EXT_CODEPLAY_MAX_REGISTERS_PER_WORK_GROUP_QUERY`, is provided to determine
whether this extension is implemented.

## New device descriptor

| Device descriptor | Return type | Description |
| ----------------- | ----------- | ----------- |
| ext::codeplay::experimental::info::device::max_registers_per_work_group | unsigned int | Returns the maximum number of registers available for use per work-group based on the capability of the device. |

### Note

## Examples

```c++
sycl::device gpu = sycl::device{sycl::gpu_selector_v};
std::cout << gpu.get_info<sycl::info::device::name>() << '\n';

#ifdef SYCL_EXT_CODEPLAY_MAX_REGISTERS_PER_WORK_GROUP_QUERY
unsigned int registers_per_group = gpu.get_info<sycl::ext::codeplay::experimental::info::device::max_registers_per_work_group>();
std::cout << "Max registers per work-group: " << registers_per_group << '\n';
#endif
```

Ouputs to the console:

Executed using the CUDA back-end on NVIDIA.

```
NVIDIA ...
Max registers per work-group: 65536
```

- See: [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities)
