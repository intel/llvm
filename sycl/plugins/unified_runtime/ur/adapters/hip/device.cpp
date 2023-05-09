//===--------- device.cpp - HIP Adapter -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//

#include "device.hpp"
#include "context.hpp"

#include <sstream>

int getAttribute(ur_device_handle_t device, hipDeviceAttribute_t attribute) {
  int value;
  sycl::detail::ur::assertion(
      hipDeviceGetAttribute(&value, attribute, device->get()) == hipSuccess);
  return value;
}

UR_APIEXPORT ur_result_t UR_APICALL urDeviceGetInfo(ur_device_handle_t device,
                                                    ur_device_info_t infoType,
                                                    size_t propSize,
                                                    void *pDeviceInfo,
                                                    size_t *pPropSizeRet) {
  UR_ASSERT(device, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UrReturnHelper ReturnValue(propSize, pDeviceInfo, pPropSizeRet);

  static constexpr uint32_t max_work_item_dimensions = 3u;

  switch ((uint32_t)infoType) {
  case UR_DEVICE_INFO_TYPE: {
    return ReturnValue(UR_DEVICE_TYPE_GPU);
  }
  case UR_DEVICE_INFO_VENDOR_ID: {
#if defined(__HIP_PLATFORM_AMD__)
    uint32_t vendor_id = 4098u;
#elif defined(__HIP_PLATFORM_NVIDIA__)
    uint32_t vendor_id = 4318u;
#else
    uint32_t vendor_id = 0u;
#endif
    return ReturnValue(vendor_id);
  }
  case UR_DEVICE_INFO_MAX_COMPUTE_UNITS: {
    int compute_units = 0;
    sycl::detail::ur::assertion(
        hipDeviceGetAttribute(&compute_units,
                              hipDeviceAttributeMultiprocessorCount,
                              device->get()) == hipSuccess);
    sycl::detail::ur::assertion(compute_units >= 0);
    return ReturnValue(static_cast<uint32_t>(compute_units));
  }
  case UR_DEVICE_INFO_MAX_WORK_ITEM_DIMENSIONS: {
    return ReturnValue(max_work_item_dimensions);
  }
  case UR_DEVICE_INFO_MAX_WORK_ITEM_SIZES: {
    struct {
      size_t sizes[max_work_item_dimensions];
    } return_sizes;

    int max_x = 0, max_y = 0, max_z = 0;
    sycl::detail::ur::assertion(
        hipDeviceGetAttribute(&max_x, hipDeviceAttributeMaxBlockDimX,
                              device->get()) == hipSuccess);
    sycl::detail::ur::assertion(max_x >= 0);

    sycl::detail::ur::assertion(
        hipDeviceGetAttribute(&max_y, hipDeviceAttributeMaxBlockDimY,
                              device->get()) == hipSuccess);
    sycl::detail::ur::assertion(max_y >= 0);

    sycl::detail::ur::assertion(
        hipDeviceGetAttribute(&max_z, hipDeviceAttributeMaxBlockDimZ,
                              device->get()) == hipSuccess);
    sycl::detail::ur::assertion(max_z >= 0);

    return_sizes.sizes[0] = size_t(max_x);
    return_sizes.sizes[1] = size_t(max_y);
    return_sizes.sizes[2] = size_t(max_z);
    return ReturnValue(return_sizes);
  }

  case UR_DEVICE_INFO_MAX_WORK_GROUPS_3D: {
    struct {
      size_t sizes[max_work_item_dimensions];
    } return_sizes;

    int max_x = 0, max_y = 0, max_z = 0;
    sycl::detail::ur::assertion(
        hipDeviceGetAttribute(&max_x, hipDeviceAttributeMaxGridDimX,
                              device->get()) == hipSuccess);
    sycl::detail::ur::assertion(max_x >= 0);

    sycl::detail::ur::assertion(
        hipDeviceGetAttribute(&max_y, hipDeviceAttributeMaxGridDimY,
                              device->get()) == hipSuccess);
    sycl::detail::ur::assertion(max_y >= 0);

    sycl::detail::ur::assertion(
        hipDeviceGetAttribute(&max_z, hipDeviceAttributeMaxGridDimZ,
                              device->get()) == hipSuccess);
    sycl::detail::ur::assertion(max_z >= 0);

    return_sizes.sizes[0] = size_t(max_x);
    return_sizes.sizes[1] = size_t(max_y);
    return_sizes.sizes[2] = size_t(max_z);
    return ReturnValue(return_sizes);
  }

  case UR_DEVICE_INFO_MAX_WORK_GROUP_SIZE: {
    int max_work_group_size = 0;
    sycl::detail::ur::assertion(
        hipDeviceGetAttribute(&max_work_group_size,
                              hipDeviceAttributeMaxThreadsPerBlock,
                              device->get()) == hipSuccess);

    sycl::detail::ur::assertion(max_work_group_size >= 0);

    return ReturnValue(size_t(max_work_group_size));
  }
  case UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_CHAR: {
    return ReturnValue(1u);
  }
  case UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_SHORT: {
    return ReturnValue(1u);
  }
  case UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_INT: {
    return ReturnValue(1u);
  }
  case UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_LONG: {
    return ReturnValue(1u);
  }
  case UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_FLOAT: {
    return ReturnValue(1u);
  }
  case UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_DOUBLE: {
    return ReturnValue(1u);
  }
  case UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_HALF: {
    return ReturnValue(0u);
  }
  case UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_CHAR: {
    return ReturnValue(1u);
  }
  case UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_SHORT: {
    return ReturnValue(1u);
  }
  case UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_INT: {
    return ReturnValue(1u);
  }
  case UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_LONG: {
    return ReturnValue(1u);
  }
  case UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_FLOAT: {
    return ReturnValue(1u);
  }
  case UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_DOUBLE: {
    return ReturnValue(1u);
  }
  case UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_HALF: {
    return ReturnValue(0u);
  }
  case UR_DEVICE_INFO_MAX_NUM_SUB_GROUPS: {
    // Number of sub-groups = max block size / warp size + possible remainder
    int max_threads = 0;
    sycl::detail::ur::assertion(
        hipDeviceGetAttribute(&max_threads,
                              hipDeviceAttributeMaxThreadsPerBlock,
                              device->get()) == hipSuccess);
    int warpSize = 0;
    sycl::detail::ur::assertion(
        hipDeviceGetAttribute(&warpSize, hipDeviceAttributeWarpSize,
                              device->get()) == hipSuccess);
    int maxWarps = (max_threads + warpSize - 1) / warpSize;
    return ReturnValue(maxWarps);
  }
  case UR_DEVICE_INFO_SUB_GROUP_INDEPENDENT_FORWARD_PROGRESS: {
    // Volta provides independent thread scheduling
    // TODO: Revisit for previous generation GPUs
    int major = 0;
    sycl::detail::ur::assertion(
        hipDeviceGetAttribute(&major, hipDeviceAttributeComputeCapabilityMajor,
                              device->get()) == hipSuccess);
    bool ifp = (major >= 7);
    return ReturnValue(ifp);
  }
  case UR_DEVICE_INFO_SUB_GROUP_SIZES_INTEL: {
    int warpSize = 0;
    sycl::detail::ur::assertion(
        hipDeviceGetAttribute(&warpSize, hipDeviceAttributeWarpSize,
                              device->get()) == hipSuccess);
    size_t sizes[1] = {static_cast<size_t>(warpSize)};
    return ReturnValue(sizes, 1);
  }
  case UR_DEVICE_INFO_MAX_CLOCK_FREQUENCY: {
    int clock_freq = 0;
    sycl::detail::ur::assertion(
        hipDeviceGetAttribute(&clock_freq, hipDeviceAttributeClockRate,
                              device->get()) == hipSuccess);
    sycl::detail::ur::assertion(clock_freq >= 0);
    return ReturnValue(static_cast<uint32_t>(clock_freq) / 1000u);
  }
  case UR_DEVICE_INFO_ADDRESS_BITS: {
    auto bits = uint32_t{std::numeric_limits<uintptr_t>::digits};
    return ReturnValue(bits);
  }
  case UR_DEVICE_INFO_MAX_MEM_ALLOC_SIZE: {
    // Max size of memory object allocation in bytes.
    // The minimum value is max(min(1024 × 1024 ×
    // 1024, 1/4th of CL_DEVICE_GLOBAL_MEM_SIZE),
    // 32 × 1024 × 1024) for devices that are not of type
    // CL_DEVICE_TYPE_CUSTOM.

    size_t global = 0;
    sycl::detail::ur::assertion(hipDeviceTotalMem(&global, device->get()) ==
                                hipSuccess);

    auto quarter_global = static_cast<uint32_t>(global / 4u);

    auto max_alloc = std::max(std::min(1024u * 1024u * 1024u, quarter_global),
                              32u * 1024u * 1024u);

    return ReturnValue(uint64_t{max_alloc});
  }
  case UR_DEVICE_INFO_IMAGE_SUPPORTED: {
    return ReturnValue(uint32_t{true});
  }
  case UR_DEVICE_INFO_MAX_READ_IMAGE_ARGS: {
    // This call doesn't match to HIP as it doesn't have images, but instead
    // surfaces and textures. No clear call in the HIP API to determine this,
    // but some searching found as of SM 2.x 128 are supported.
    return ReturnValue(128u);
  }
  case UR_DEVICE_INFO_MAX_WRITE_IMAGE_ARGS: {
    // This call doesn't match to HIP as it doesn't have images, but instead
    // surfaces and textures. No clear call in the HIP API to determine this,
    // but some searching found as of SM 2.x 128 are supported.
    return ReturnValue(128u);
  }
  case UR_DEVICE_INFO_IMAGE2D_MAX_HEIGHT: {
    // Take the smaller of maximum surface and maximum texture height.
    int tex_height = 0;
    sycl::detail::ur::assertion(
        hipDeviceGetAttribute(&tex_height, hipDeviceAttributeMaxTexture2DHeight,
                              device->get()) == hipSuccess);
    sycl::detail::ur::assertion(tex_height >= 0);
    int surf_height = 0;
    sycl::detail::ur::assertion(
        hipDeviceGetAttribute(&surf_height,
                              hipDeviceAttributeMaxTexture2DHeight,
                              device->get()) == hipSuccess);
    sycl::detail::ur::assertion(surf_height >= 0);

    int min = std::min(tex_height, surf_height);

    return ReturnValue(static_cast<size_t>(min));
  }
  case UR_DEVICE_INFO_IMAGE2D_MAX_WIDTH: {
    // Take the smaller of maximum surface and maximum texture width.
    int tex_width = 0;
    sycl::detail::ur::assertion(
        hipDeviceGetAttribute(&tex_width, hipDeviceAttributeMaxTexture2DWidth,
                              device->get()) == hipSuccess);
    sycl::detail::ur::assertion(tex_width >= 0);
    int surf_width = 0;
    sycl::detail::ur::assertion(
        hipDeviceGetAttribute(&surf_width, hipDeviceAttributeMaxTexture2DWidth,
                              device->get()) == hipSuccess);
    sycl::detail::ur::assertion(surf_width >= 0);

    int min = std::min(tex_width, surf_width);

    return ReturnValue(static_cast<size_t>(min));
  }
  case UR_DEVICE_INFO_IMAGE3D_MAX_HEIGHT: {
    // Take the smaller of maximum surface and maximum texture height.
    int tex_height = 0;
    sycl::detail::ur::assertion(
        hipDeviceGetAttribute(&tex_height, hipDeviceAttributeMaxTexture3DHeight,
                              device->get()) == hipSuccess);
    sycl::detail::ur::assertion(tex_height >= 0);
    int surf_height = 0;
    sycl::detail::ur::assertion(
        hipDeviceGetAttribute(&surf_height,
                              hipDeviceAttributeMaxTexture3DHeight,
                              device->get()) == hipSuccess);
    sycl::detail::ur::assertion(surf_height >= 0);

    int min = std::min(tex_height, surf_height);

    return ReturnValue(static_cast<size_t>(min));
  }
  case UR_DEVICE_INFO_IMAGE3D_MAX_WIDTH: {
    // Take the smaller of maximum surface and maximum texture width.
    int tex_width = 0;
    sycl::detail::ur::assertion(
        hipDeviceGetAttribute(&tex_width, hipDeviceAttributeMaxTexture3DWidth,
                              device->get()) == hipSuccess);
    sycl::detail::ur::assertion(tex_width >= 0);
    int surf_width = 0;
    sycl::detail::ur::assertion(
        hipDeviceGetAttribute(&surf_width, hipDeviceAttributeMaxTexture3DWidth,
                              device->get()) == hipSuccess);
    sycl::detail::ur::assertion(surf_width >= 0);

    int min = std::min(tex_width, surf_width);

    return ReturnValue(static_cast<size_t>(min));
  }
  case UR_DEVICE_INFO_IMAGE3D_MAX_DEPTH: {
    // Take the smaller of maximum surface and maximum texture depth.
    int tex_depth = 0;
    sycl::detail::ur::assertion(
        hipDeviceGetAttribute(&tex_depth, hipDeviceAttributeMaxTexture3DDepth,
                              device->get()) == hipSuccess);
    sycl::detail::ur::assertion(tex_depth >= 0);
    int surf_depth = 0;
    sycl::detail::ur::assertion(
        hipDeviceGetAttribute(&surf_depth, hipDeviceAttributeMaxTexture3DDepth,
                              device->get()) == hipSuccess);
    sycl::detail::ur::assertion(surf_depth >= 0);

    int min = std::min(tex_depth, surf_depth);

    return ReturnValue(static_cast<size_t>(min));
  }
  case UR_DEVICE_INFO_IMAGE_MAX_BUFFER_SIZE: {
    // Take the smaller of maximum surface and maximum texture width.
    int tex_width = 0;
    sycl::detail::ur::assertion(
        hipDeviceGetAttribute(&tex_width, hipDeviceAttributeMaxTexture1DWidth,
                              device->get()) == hipSuccess);
    sycl::detail::ur::assertion(tex_width >= 0);
    int surf_width = 0;
    sycl::detail::ur::assertion(
        hipDeviceGetAttribute(&surf_width, hipDeviceAttributeMaxTexture1DWidth,
                              device->get()) == hipSuccess);
    sycl::detail::ur::assertion(surf_width >= 0);

    int min = std::min(tex_width, surf_width);

    return ReturnValue(static_cast<size_t>(min));
  }
  case UR_DEVICE_INFO_IMAGE_MAX_ARRAY_SIZE: {
    return ReturnValue(0lu);
  }
  case UR_DEVICE_INFO_MAX_SAMPLERS: {
    // This call is kind of meaningless for HIP, as samplers don't exist.
    // Closest thing is textures, which is 128.
    return ReturnValue(128u);
  }
  case UR_DEVICE_INFO_MAX_PARAMETER_SIZE: {
    // __global__ function parameters are passed to the device via constant
    // memory and are limited to 4 KB.
    return ReturnValue(4000lu);
  }
  case UR_DEVICE_INFO_MEM_BASE_ADDR_ALIGN: {
    int mem_base_addr_align = 0;
    sycl::detail::ur::assertion(
        hipDeviceGetAttribute(&mem_base_addr_align,
                              hipDeviceAttributeTextureAlignment,
                              device->get()) == hipSuccess);
    // Multiply by 8 as clGetDeviceInfo returns this value in bits
    mem_base_addr_align *= 8;
    return ReturnValue(mem_base_addr_align);
  }
  case UR_DEVICE_INFO_HALF_FP_CONFIG: {
    return ReturnValue(0u);
  }
  case UR_DEVICE_INFO_SINGLE_FP_CONFIG: {
    uint64_t config =
        UR_DEVICE_FP_CAPABILITY_FLAG_DENORM |
        UR_DEVICE_FP_CAPABILITY_FLAG_INF_NAN |
        UR_DEVICE_FP_CAPABILITY_FLAG_ROUND_TO_NEAREST |
        UR_DEVICE_FP_CAPABILITY_FLAG_ROUND_TO_ZERO |
        UR_DEVICE_FP_CAPABILITY_FLAG_ROUND_TO_INF |
        UR_DEVICE_FP_CAPABILITY_FLAG_FMA |
        UR_DEVICE_FP_CAPABILITY_FLAG_CORRECTLY_ROUNDED_DIVIDE_SQRT;
    return ReturnValue(config);
  }
  case UR_DEVICE_INFO_DOUBLE_FP_CONFIG: {
    uint64_t config = UR_DEVICE_FP_CAPABILITY_FLAG_DENORM |
                      UR_DEVICE_FP_CAPABILITY_FLAG_INF_NAN |
                      UR_DEVICE_FP_CAPABILITY_FLAG_ROUND_TO_NEAREST |
                      UR_DEVICE_FP_CAPABILITY_FLAG_ROUND_TO_ZERO |
                      UR_DEVICE_FP_CAPABILITY_FLAG_ROUND_TO_INF |
                      UR_DEVICE_FP_CAPABILITY_FLAG_FMA;
    return ReturnValue(config);
  }
  case UR_DEVICE_INFO_GLOBAL_MEM_CACHE_TYPE: {
    return ReturnValue(UR_DEVICE_MEM_CACHE_TYPE_READ_WRITE_CACHE);
  }
  case UR_DEVICE_INFO_GLOBAL_MEM_CACHELINE_SIZE: {
    // The value is dohipmented for all existing GPUs in the HIP programming
    // guidelines, section "H.3.2. Global Memory".
    return ReturnValue(128u);
  }
  case UR_DEVICE_INFO_GLOBAL_MEM_CACHE_SIZE: {
    int cache_size = 0;
    sycl::detail::ur::assertion(
        hipDeviceGetAttribute(&cache_size, hipDeviceAttributeL2CacheSize,
                              device->get()) == hipSuccess);
    sycl::detail::ur::assertion(cache_size >= 0);
    // The L2 cache is global to the GPU.
    return ReturnValue(static_cast<uint64_t>(cache_size));
  }
  case UR_DEVICE_INFO_GLOBAL_MEM_SIZE: {
    size_t bytes = 0;
    // Runtime API has easy access to this value, driver API info is scarse.
    sycl::detail::ur::assertion(hipDeviceTotalMem(&bytes, device->get()) ==
                                hipSuccess);
    return ReturnValue(uint64_t{bytes});
  }
  case UR_DEVICE_INFO_MAX_CONSTANT_BUFFER_SIZE: {
    int constant_memory = 0;

    // hipDeviceGetAttribute takes a int*, however the size of the constant
    // memory on AMD GPU may be larger than what can fit in the positive part
    // of a signed integer, so use an unsigned integer and cast the pointer to
    // int*.
    sycl::detail::ur::assertion(
        hipDeviceGetAttribute(&constant_memory,
                              hipDeviceAttributeTotalConstantMemory,
                              device->get()) == hipSuccess);
    sycl::detail::ur::assertion(constant_memory >= 0);

    return ReturnValue(static_cast<uint64_t>(constant_memory));
  }
  case UR_DEVICE_INFO_MAX_CONSTANT_ARGS: {
    // TODO: is there a way to retrieve this from HIP driver API?
    // Hard coded to value returned by clinfo for OpenCL 1.2 HIP | GeForce GTX
    // 1060 3GB
    return ReturnValue(9u);
  }
  case UR_DEVICE_INFO_LOCAL_MEM_TYPE: {
    return ReturnValue(UR_DEVICE_LOCAL_MEM_TYPE_LOCAL);
  }
  case UR_DEVICE_INFO_LOCAL_MEM_SIZE: {
    // OpenCL's "local memory" maps most closely to HIP's "shared memory".
    // HIP has its own definition of "local memory", which maps to OpenCL's
    // "private memory".
    int local_mem_size = 0;
    sycl::detail::ur::assertion(
        hipDeviceGetAttribute(&local_mem_size,
                              hipDeviceAttributeMaxSharedMemoryPerBlock,
                              device->get()) == hipSuccess);
    sycl::detail::ur::assertion(local_mem_size >= 0);
    return ReturnValue(static_cast<uint64_t>(local_mem_size));
  }
  case UR_DEVICE_INFO_ERROR_CORRECTION_SUPPORT: {
    int ecc_enabled = 0;
    sycl::detail::ur::assertion(
        hipDeviceGetAttribute(&ecc_enabled, hipDeviceAttributeEccEnabled,
                              device->get()) == hipSuccess);

    sycl::detail::ur::assertion((ecc_enabled == 0) | (ecc_enabled == 1));
    auto result = static_cast<bool>(ecc_enabled);
    return ReturnValue(result);
  }
  case UR_DEVICE_INFO_HOST_UNIFIED_MEMORY: {
    int is_integrated = 0;
    sycl::detail::ur::assertion(
        hipDeviceGetAttribute(&is_integrated, hipDeviceAttributeIntegrated,
                              device->get()) == hipSuccess);

    sycl::detail::ur::assertion((is_integrated == 0) | (is_integrated == 1));
    auto result = static_cast<bool>(is_integrated);
    return ReturnValue(result);
  }
  case UR_DEVICE_INFO_PROFILING_TIMER_RESOLUTION: {
    // Hard coded to value returned by clinfo for OpenCL 1.2 HIP | GeForce GTX
    // 1060 3GB
    return ReturnValue(1000lu);
  }
  case UR_DEVICE_INFO_ENDIAN_LITTLE: {
    return ReturnValue(true);
  }
  case UR_DEVICE_INFO_AVAILABLE: {
    return ReturnValue(true);
  }
  case UR_DEVICE_INFO_BUILD_ON_SUBDEVICE: {
    return ReturnValue(true);
  }
  case UR_DEVICE_INFO_COMPILER_AVAILABLE: {
    return ReturnValue(true);
  }
  case UR_DEVICE_INFO_LINKER_AVAILABLE: {
    return ReturnValue(true);
  }
  case UR_DEVICE_INFO_EXECUTION_CAPABILITIES: {
    auto capability = ur_device_exec_capability_flags_t{
        UR_DEVICE_EXEC_CAPABILITY_FLAG_KERNEL};
    return ReturnValue(capability);
  }
  case UR_DEVICE_INFO_QUEUE_ON_DEVICE_PROPERTIES: {
    // The mandated minimum capability:
    uint64_t capability = UR_QUEUE_FLAG_PROFILING_ENABLE |
                          UR_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE;
    return ReturnValue(capability);
  }
  case UR_DEVICE_INFO_QUEUE_ON_HOST_PROPERTIES:
  case UR_DEVICE_INFO_QUEUE_PROPERTIES: {
    // The mandated minimum capability:
    uint64_t capability = UR_QUEUE_FLAG_PROFILING_ENABLE;
    return ReturnValue(capability);
  }
  case UR_DEVICE_INFO_BUILT_IN_KERNELS: {
    // An empty string is returned if no built-in kernels are supported by the
    // device.
    return ReturnValue("");
  }
  case UR_DEVICE_INFO_PLATFORM: {
    return ReturnValue(device->get_platform());
  }
  case UR_DEVICE_INFO_NAME: {
    static constexpr size_t MAX_DEVICE_NAME_LENGTH = 256u;
    char name[MAX_DEVICE_NAME_LENGTH];
    sycl::detail::ur::assertion(hipDeviceGetName(name, MAX_DEVICE_NAME_LENGTH,
                                                 device->get()) == hipSuccess);
    // On AMD GPUs hipDeviceGetName returns an empty string, so return the arch
    // name instead, this is also what AMD OpenCL devices return.
    if (strlen(name) == 0) {
      hipDeviceProp_t props;
      sycl::detail::ur::assertion(
          hipGetDeviceProperties(&props, device->get()) == hipSuccess);

      return ReturnValue(props.gcnArchName, strlen(props.gcnArchName) + 1);
    }
    return ReturnValue(name, strlen(name) + 1);
  }
  case UR_DEVICE_INFO_VENDOR: {
    return ReturnValue("AMD Corporation");
  }
  case UR_DEVICE_INFO_DRIVER_VERSION: {
    auto version = getHipVersionString();
    return ReturnValue(version.c_str());
  }
  case UR_DEVICE_INFO_PROFILE: {
    return ReturnValue("HIP");
  }
  case UR_DEVICE_INFO_REFERENCE_COUNT: {
    return ReturnValue(device->get_reference_count());
  }
  case UR_DEVICE_INFO_VERSION: {
    std::stringstream s;

    hipDeviceProp_t props;
    sycl::detail::ur::assertion(hipGetDeviceProperties(&props, device->get()) ==
                                hipSuccess);
#if defined(__HIP_PLATFORM_NVIDIA__)
    s << props.major << "." << props.minor;
#elif defined(__HIP_PLATFORM_AMD__)
    s << props.gcnArchName;
#else
#error("Must define exactly one of __HIP_PLATFORM_AMD__ or __HIP_PLATFORM_NVIDIA__");
#endif
    return ReturnValue(s.str().c_str());
  }
  case UR_EXT_DEVICE_INFO_OPENCL_C_VERSION: {
    return ReturnValue("");
  }
  case UR_DEVICE_INFO_EXTENSIONS: {
    // TODO: Remove comment when HIP support native asserts.
    // DEVICELIB_ASSERT extension is set so fallback assert
    // postprocessing is NOP. HIP 4.3 docs indicate support for
    // native asserts are in progress
    std::string SupportedExtensions = "";
    SupportedExtensions += "pi_ext_intel_devicelib_assert ";
    SupportedExtensions += " ";

    hipDeviceProp_t props;
    sycl::detail::ur::assertion(hipGetDeviceProperties(&props, device->get()) ==
                                hipSuccess);

    if (props.arch.hasDoubles) {
      SupportedExtensions += "cl_khr_fp64 ";
    }

    return ReturnValue(SupportedExtensions.c_str());
  }
  case UR_DEVICE_INFO_PRINTF_BUFFER_SIZE: {
    // The minimum value for the FULL profile is 1 MB.
    return ReturnValue(1024lu);
  }
  case UR_DEVICE_INFO_PREFERRED_INTEROP_USER_SYNC: {
    return ReturnValue(true);
  }
  case UR_DEVICE_INFO_PARENT_DEVICE: {
    return ReturnValue(nullptr);
  }
  case UR_DEVICE_INFO_PARTITION_MAX_SUB_DEVICES: {
    return ReturnValue(0u);
  }
  case UR_DEVICE_INFO_PARTITION_PROPERTIES: {
    return ReturnValue(static_cast<ur_device_partition_t>(0u));
  }
  case UR_DEVICE_INFO_PARTITION_AFFINITY_DOMAIN: {
    return ReturnValue(0u);
  }
  case UR_DEVICE_INFO_PARTITION_TYPE: {
    return ReturnValue(static_cast<ur_device_partition_t>(0u));
  }

  // Intel USM extensions
  case UR_DEVICE_INFO_USM_HOST_SUPPORT: {
    // from cl_intel_unified_shared_memory: "The host memory access capabilities
    // apply to any host allocation."
    //
    // query if/how the device can access page-locked host memory, possibly
    // through PCIe, using the same pointer as the host
    uint64_t value = {};
    // if (getAttribute(device, HIP_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING)) {
    // the device shares a unified address space with the host
    if (getAttribute(device, hipDeviceAttributeComputeCapabilityMajor) >= 6) {
      // compute capability 6.x introduces operations that are atomic with
      // respect to other CPUs and GPUs in the system
      value = UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_ACCESS |
              UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_ATOMIC_ACCESS |
              UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_CONCURRENT_ACCESS |
              UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_ATOMIC_CONCURRENT_ACCESS;
    } else {
      // on GPU architectures with compute capability lower than 6.x, atomic
      // operations from the GPU to CPU memory will not be atomic with respect
      // to CPU initiated atomic operations
      value = UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_ACCESS |
              UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_CONCURRENT_ACCESS;
    }
    return ReturnValue(value);
  }
  case UR_DEVICE_INFO_USM_DEVICE_SUPPORT: {
    // from cl_intel_unified_shared_memory:
    // "The device memory access capabilities apply to any device allocation
    // associated with this device."
    //
    // query how the device can access memory allocated on the device itself (?)
    uint64_t value =
        UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_ACCESS |
        UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_ATOMIC_ACCESS |
        UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_CONCURRENT_ACCESS |
        UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_ATOMIC_CONCURRENT_ACCESS;
    return ReturnValue(value);
  }
  case UR_DEVICE_INFO_USM_SINGLE_SHARED_SUPPORT: {
    // from cl_intel_unified_shared_memory:
    // "The single device shared memory access capabilities apply to any shared
    // allocation associated with this device."
    //
    // query if/how the device can access managed memory associated to it
    uint64_t value = {};
    if (getAttribute(device, hipDeviceAttributeManagedMemory)) {
      // the device can allocate managed memory on this system
      value = UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_ACCESS |
              UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_ATOMIC_ACCESS;
    }
    if (getAttribute(device, hipDeviceAttributeConcurrentManagedAccess)) {
      // the device can coherently access managed memory concurrently with the
      // CPU
      value |= UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_CONCURRENT_ACCESS;
      if (getAttribute(device, hipDeviceAttributeComputeCapabilityMajor) >= 6) {
        // compute capability 6.x introduces operations that are atomic with
        // respect to other CPUs and GPUs in the system
        value |= UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_ATOMIC_CONCURRENT_ACCESS;
      }
    }
    return ReturnValue(value);
  }
  case UR_DEVICE_INFO_USM_CROSS_SHARED_SUPPORT: {
    // from cl_intel_unified_shared_memory:
    // "The cross-device shared memory access capabilities apply to any shared
    // allocation associated with this device, or to any shared memory
    // allocation on another device that also supports the same cross-device
    // shared memory access capability."
    //
    // query if/how the device can access managed memory associated to other
    // devices
    uint64_t value = {};
    if (getAttribute(device, hipDeviceAttributeManagedMemory)) {
      // the device can allocate managed memory on this system
      value |= UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_ACCESS;
    }
    if (getAttribute(device, hipDeviceAttributeConcurrentManagedAccess)) {
      // all devices with the CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS
      // attribute can coherently access managed memory concurrently with the
      // CPU
      value |= UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_CONCURRENT_ACCESS;
    }
    if (getAttribute(device, hipDeviceAttributeComputeCapabilityMajor) >= 6) {
      // compute capability 6.x introduces operations that are atomic with
      // respect to other CPUs and GPUs in the system
      if (value & UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_ACCESS)
        value |= UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_ATOMIC_ACCESS;
      if (value & UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_CONCURRENT_ACCESS)
        value |= UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_ATOMIC_CONCURRENT_ACCESS;
    }
    return ReturnValue(value);
  }
  case UR_DEVICE_INFO_USM_SYSTEM_SHARED_SUPPORT: {
    // from cl_intel_unified_shared_memory:
    // "The shared system memory access capabilities apply to any allocations
    // made by a system allocator, such as malloc or new."
    //
    // query if/how the device can access pageable host memory allocated by the
    // system allocator
    uint64_t value = {};
    if (getAttribute(device, hipDeviceAttributePageableMemoryAccess)) {
      // the link between the device and the host does not support native
      // atomic operations
      value = UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_ACCESS |
              UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_CONCURRENT_ACCESS;
    }
    return ReturnValue(value);
  }

  case UR_DEVICE_INFO_BACKEND_RUNTIME_VERSION: {
    int major = 0, minor = 0;
    sycl::detail::ur::assertion(
        hipDeviceComputeCapability(&major, &minor, device->get()) ==
        hipSuccess);
    std::string result = std::to_string(major) + "." + std::to_string(minor);
    return ReturnValue(result.c_str());
  }

  case UR_DEVICE_INFO_ATOMIC_64: {
    // TODO: Reconsider it when AMD supports SYCL_USE_NATIVE_FP_ATOMICS.
    hipDeviceProp_t props;
    sycl::detail::ur::assertion(hipGetDeviceProperties(&props, device->get()) ==
                                hipSuccess);
    return ReturnValue(props.arch.hasGlobalInt64Atomics &&
                       props.arch.hasSharedInt64Atomics);
  }

  case UR_DEVICE_INFO_GLOBAL_MEM_FREE: {
    size_t FreeMemory = 0;
    size_t TotalMemory = 0;
    sycl::detail::ur::assertion(hipMemGetInfo(&FreeMemory, &TotalMemory) ==
                                    hipSuccess,
                                "failed hipMemGetInfo() API.");
    return ReturnValue(FreeMemory);
  }

  case UR_DEVICE_INFO_MEMORY_CLOCK_RATE: {
    int value = 0;
    sycl::detail::ur::assertion(
        hipDeviceGetAttribute(&value, hipDeviceAttributeMemoryClockRate,
                              device->get()) == hipSuccess);
    sycl::detail::ur::assertion(value >= 0);
    // Convert kilohertz to megahertz when returning.
    return ReturnValue(value / 1000);
  }

  case UR_DEVICE_INFO_MEMORY_BUS_WIDTH: {
    int value = 0;
    sycl::detail::ur::assertion(
        hipDeviceGetAttribute(&value, hipDeviceAttributeMemoryBusWidth,
                              device->get()) == hipSuccess);
    sycl::detail::ur::assertion(value >= 0);
    return ReturnValue(value);
  }
  case UR_DEVICE_INFO_MAX_COMPUTE_QUEUE_INDICES: {
    return ReturnValue(int32_t{1});
  }

  case UR_DEVICE_INFO_ATOMIC_MEMORY_ORDER_CAPABILITIES: {
    uint64_t capabilities = UR_MEMORY_ORDER_CAPABILITY_FLAG_RELAXED |
                            UR_MEMORY_ORDER_CAPABILITY_FLAG_ACQUIRE |
                            UR_MEMORY_ORDER_CAPABILITY_FLAG_RELEASE;
    return ReturnValue(capabilities);
  }
  case UR_DEVICE_INFO_ATOMIC_MEMORY_SCOPE_CAPABILITIES:
  case UR_DEVICE_INFO_ATOMIC_FENCE_SCOPE_CAPABILITIES: {
    // SYCL2020 4.6.4.2 minimum mandated capabilities for
    // atomic_fence/memory_scope_capabilities.
    // Because scopes are hierarchical, wider scopes support all narrower
    // scopes. At a minimum, each device must support WORK_ITEM, SUB_GROUP and
    // WORK_GROUP. (https://github.com/KhronosGroup/SYCL-Docs/pull/382)
    uint64_t capabilities = UR_MEMORY_SCOPE_CAPABILITY_FLAG_WORK_ITEM |
                            UR_MEMORY_SCOPE_CAPABILITY_FLAG_SUB_GROUP |
                            UR_MEMORY_SCOPE_CAPABILITY_FLAG_WORK_GROUP;
    return ReturnValue(capabilities);
  }
  case UR_DEVICE_INFO_ATOMIC_FENCE_ORDER_CAPABILITIES: {
    // SYCL2020 4.6.4.2 minimum mandated capabilities for
    // atomic_fence_order_capabilities.
    ur_memory_order_capability_flags_t capabilities =
        UR_MEMORY_ORDER_CAPABILITY_FLAG_RELAXED |
        UR_MEMORY_ORDER_CAPABILITY_FLAG_ACQUIRE |
        UR_MEMORY_ORDER_CAPABILITY_FLAG_RELEASE |
        UR_MEMORY_ORDER_CAPABILITY_FLAG_ACQ_REL;
    return ReturnValue(capabilities);
  }
  case UR_DEVICE_INFO_DEVICE_ID: {
    int value = 0;
    sycl::detail::ur::assertion(
        hipDeviceGetAttribute(&value, hipDeviceAttributePciDeviceId,
                              device->get()) == hipSuccess);
    sycl::detail::ur::assertion(value >= 0);
    return ReturnValue(value);
  }
  case UR_DEVICE_INFO_UUID: {
#if ((HIP_VERSION_MAJOR == 5 && HIP_VERSION_MINOR >= 2) ||                     \
     HIP_VERSION_MAJOR > 5)
    hipUUID uuid = {};
    // Supported since 5.2+
    sycl::detail::ur::assertion(hipDeviceGetUuid(&uuid, device->get()) ==
                                hipSuccess);
    std::array<unsigned char, 16> name;
    std::copy(uuid.bytes, uuid.bytes + 16, name.begin());
    return ReturnValue(name.data(), 16);
#endif
    return UR_RESULT_ERROR_INVALID_VALUE;
  }

  case UR_DEVICE_INFO_MEM_CHANNEL_SUPPORT:
    return ReturnValue(false);
  case UR_DEVICE_INFO_IMAGE_SRGB:
    return ReturnValue(false);

  // TODO: Investigate if this information is available on HIP.
  case UR_DEVICE_INFO_PCI_ADDRESS:
  case UR_DEVICE_INFO_GPU_EU_COUNT:
  case UR_DEVICE_INFO_GPU_EU_SIMD_WIDTH:
  case UR_DEVICE_INFO_GPU_EU_SLICES:
  case UR_DEVICE_INFO_GPU_SUBSLICES_PER_SLICE:
  case UR_DEVICE_INFO_GPU_EU_COUNT_PER_SUBSLICE:
  case UR_DEVICE_INFO_GPU_HW_THREADS_PER_EU:
  case UR_DEVICE_INFO_MAX_MEMORY_BANDWIDTH:
  case UR_DEVICE_INFO_BFLOAT16:
    return UR_RESULT_ERROR_INVALID_ENUMERATION;

  default:
    break;
  }
  return UR_RESULT_ERROR_INVALID_ENUMERATION;
}

/// \return UR_RESULT_SUCCESS if the function is executed successfully
/// HIP devices are always root devices so retain always returns success.
UR_APIEXPORT ur_result_t UR_APICALL urDeviceRetain(ur_device_handle_t device) {
  UR_ASSERT(device, UR_RESULT_ERROR_INVALID_NULL_HANDLE);

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urDevicePartition(ur_device_handle_t, const ur_device_partition_property_t *,
                  uint32_t, ur_device_handle_t *, uint32_t *) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

/// \return UR_RESULT_SUCCESS always since HIP devices are always root
/// devices.
UR_DLLEXPORT ur_result_t UR_APICALL urDeviceRelease(ur_device_handle_t device) {
  UR_ASSERT(device, UR_RESULT_ERROR_INVALID_NULL_HANDLE);

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urDeviceGet(ur_platform_handle_t hPlatform,
                                                ur_device_type_t DeviceType,
                                                uint32_t NumEntries,
                                                ur_device_handle_t *phDevices,
                                                uint32_t *pNumDevices) {
  ur_result_t err = UR_RESULT_SUCCESS;
  const bool askingForDefault = DeviceType == UR_DEVICE_TYPE_DEFAULT;
  const bool askingForGPU = DeviceType == UR_DEVICE_TYPE_GPU;
  const bool askingForAll = DeviceType == UR_DEVICE_TYPE_ALL;
  const bool returnDevices = askingForDefault || askingForGPU || askingForAll;

  UR_ASSERT(hPlatform, UR_RESULT_ERROR_INVALID_NULL_HANDLE);

  size_t numDevices = returnDevices ? hPlatform->devices_.size() : 0;

  try {
    UR_ASSERT(pNumDevices || phDevices, UR_RESULT_ERROR_INVALID_VALUE);

    if (pNumDevices) {
      *pNumDevices = numDevices;
    }

    if (returnDevices && phDevices) {
      for (size_t i = 0; i < std::min(size_t(NumEntries), numDevices); ++i) {
        phDevices[i] = hPlatform->devices_[i].get();
      }
    }

    return err;
  } catch (ur_result_t err) {
    return err;
  } catch (...) {
    return UR_RESULT_ERROR_OUT_OF_RESOURCES;
  }
}

/// Gets the native HIP handle of a UR device object
///
/// \param[in] device The UR device to get the native HIP object of.
/// \param[out] nativeHandle Set to the native handle of the UR device object.
///
/// \return UR_RESULT_SUCCESS

UR_APIEXPORT ur_result_t UR_APICALL urDeviceGetNativeHandle(
    ur_device_handle_t hDevice, ur_native_handle_t *phNativeHandle) {
  UR_ASSERT(hDevice, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(phNativeHandle, UR_RESULT_ERROR_INVALID_NULL_POINTER);

  *phNativeHandle = reinterpret_cast<ur_native_handle_t>(hDevice->get());
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urDeviceCreateWithNativeHandle(
    ur_native_handle_t hNativeDevice, ur_platform_handle_t hPlatform,
    ur_device_handle_t *phDevice) {
  UR_ASSERT(hNativeDevice, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(phDevice, UR_RESULT_ERROR_INVALID_NULL_POINTER);

  return UR_RESULT_ERROR_INVALID_OPERATION;
}
