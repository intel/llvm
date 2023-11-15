//===--------- device.cpp - HIP Adapter -----------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "device.hpp"
#include "context.hpp"
#include "event.hpp"

#include <sstream>

int getAttribute(ur_device_handle_t Device, hipDeviceAttribute_t Attribute) {
  int Value;
  UR_CHECK_ERROR(hipDeviceGetAttribute(&Value, Attribute, Device->get()));
  return Value;
}

UR_APIEXPORT ur_result_t UR_APICALL urDeviceGetInfo(ur_device_handle_t hDevice,
                                                    ur_device_info_t propName,
                                                    size_t propSize,
                                                    void *pPropValue,
                                                    size_t *pPropSizeRet) {
  UrReturnHelper ReturnValue(propSize, pPropValue, pPropSizeRet);

  static constexpr uint32_t MaxWorkItemDimensions = 3u;

  switch ((uint32_t)propName) {
  case UR_DEVICE_INFO_TYPE: {
    return ReturnValue(UR_DEVICE_TYPE_GPU);
  }
  case UR_DEVICE_INFO_VENDOR_ID: {
#if defined(__HIP_PLATFORM_AMD__)
    uint32_t VendorId = 4098u;
#elif defined(__HIP_PLATFORM_NVIDIA__)
    uint32_t VendorId = 4318u;
#else
    uint32_t VendorId = 0u;
#endif
    return ReturnValue(VendorId);
  }
  case UR_DEVICE_INFO_MAX_COMPUTE_UNITS: {
    int ComputeUnits = 0;
    UR_CHECK_ERROR(hipDeviceGetAttribute(
        &ComputeUnits, hipDeviceAttributeMultiprocessorCount, hDevice->get()));
    detail::ur::assertion(ComputeUnits >= 0);
    return ReturnValue(static_cast<uint32_t>(ComputeUnits));
  }
  case UR_DEVICE_INFO_MAX_WORK_ITEM_DIMENSIONS: {
    return ReturnValue(MaxWorkItemDimensions);
  }
  case UR_DEVICE_INFO_MAX_WORK_ITEM_SIZES: {
    struct {
      size_t sizes[MaxWorkItemDimensions];
    } return_sizes;

    int MaxX = 0, MaxY = 0, MaxZ = 0;
    UR_CHECK_ERROR(hipDeviceGetAttribute(&MaxX, hipDeviceAttributeMaxBlockDimX,
                                         hDevice->get()));
    detail::ur::assertion(MaxX >= 0);

    UR_CHECK_ERROR(hipDeviceGetAttribute(&MaxY, hipDeviceAttributeMaxBlockDimY,
                                         hDevice->get()));
    detail::ur::assertion(MaxY >= 0);

    UR_CHECK_ERROR(hipDeviceGetAttribute(&MaxZ, hipDeviceAttributeMaxBlockDimZ,
                                         hDevice->get()));
    detail::ur::assertion(MaxZ >= 0);

    return_sizes.sizes[0] = size_t(MaxX);
    return_sizes.sizes[1] = size_t(MaxY);
    return_sizes.sizes[2] = size_t(MaxZ);
    return ReturnValue(return_sizes);
  }

  case UR_DEVICE_INFO_MAX_WORK_GROUPS_3D: {
    struct {
      size_t sizes[MaxWorkItemDimensions];
    } return_sizes;

    int MaxX = 0, MaxY = 0, MaxZ = 0;
    UR_CHECK_ERROR(hipDeviceGetAttribute(&MaxX, hipDeviceAttributeMaxGridDimX,
                                         hDevice->get()));
    detail::ur::assertion(MaxX >= 0);

    UR_CHECK_ERROR(hipDeviceGetAttribute(&MaxY, hipDeviceAttributeMaxGridDimY,
                                         hDevice->get()));
    detail::ur::assertion(MaxY >= 0);

    UR_CHECK_ERROR(hipDeviceGetAttribute(&MaxZ, hipDeviceAttributeMaxGridDimZ,
                                         hDevice->get()));
    detail::ur::assertion(MaxZ >= 0);

    return_sizes.sizes[0] = size_t(MaxX);
    return_sizes.sizes[1] = size_t(MaxY);
    return_sizes.sizes[2] = size_t(MaxZ);
    return ReturnValue(return_sizes);
  }

  case UR_DEVICE_INFO_MAX_WORK_GROUP_SIZE: {
    int MaxWorkGroupSize = 0;
    UR_CHECK_ERROR(hipDeviceGetAttribute(&MaxWorkGroupSize,
                                         hipDeviceAttributeMaxThreadsPerBlock,
                                         hDevice->get()));

    detail::ur::assertion(MaxWorkGroupSize >= 0);

    return ReturnValue(size_t(MaxWorkGroupSize));
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
    int MaxThreads = 0;
    UR_CHECK_ERROR(hipDeviceGetAttribute(
        &MaxThreads, hipDeviceAttributeMaxThreadsPerBlock, hDevice->get()));
    int WarpSize = 0;
    UR_CHECK_ERROR(hipDeviceGetAttribute(&WarpSize, hipDeviceAttributeWarpSize,
                                         hDevice->get()));
    int MaxWarps = (MaxThreads + WarpSize - 1) / WarpSize;
    return ReturnValue(MaxWarps);
  }
  case UR_DEVICE_INFO_SUB_GROUP_INDEPENDENT_FORWARD_PROGRESS: {
    // Volta provides independent thread scheduling
    // TODO: Revisit for previous generation GPUs
    int Major = 0;
    UR_CHECK_ERROR(hipDeviceGetAttribute(
        &Major, hipDeviceAttributeComputeCapabilityMajor, hDevice->get()));
    bool IFP = (Major >= 7);
    return ReturnValue(IFP);
  }
  case UR_DEVICE_INFO_SUB_GROUP_SIZES_INTEL: {
    int WarpSize = 0;
    UR_CHECK_ERROR(hipDeviceGetAttribute(&WarpSize, hipDeviceAttributeWarpSize,
                                         hDevice->get()));
    size_t Sizes[1] = {static_cast<size_t>(WarpSize)};
    return ReturnValue(Sizes, 1);
  }
  case UR_DEVICE_INFO_MAX_CLOCK_FREQUENCY: {
    int ClockFreq = 0;
    UR_CHECK_ERROR(hipDeviceGetAttribute(
        &ClockFreq, hipDeviceAttributeClockRate, hDevice->get()));
    detail::ur::assertion(ClockFreq >= 0);
    return ReturnValue(static_cast<uint32_t>(ClockFreq) / 1000u);
  }
  case UR_DEVICE_INFO_ADDRESS_BITS: {
    auto Bits = uint32_t{std::numeric_limits<uintptr_t>::digits};
    return ReturnValue(Bits);
  }
  case UR_DEVICE_INFO_MAX_MEM_ALLOC_SIZE: {
    // Max size of memory object allocation in bytes.
    // The minimum value is max(min(1024 × 1024 ×
    // 1024, 1/4th of CL_DEVICE_GLOBAL_MEM_SIZE),
    // 32 × 1024 × 1024) for devices that are not of type
    // CL_DEVICE_TYPE_CUSTOM.

    size_t Global = 0;
    detail::ur::assertion(hipDeviceTotalMem(&Global, hDevice->get()) ==
                          hipSuccess);

    auto QuarterGlobal = static_cast<uint32_t>(Global / 4u);

    auto MaxAlloc = std::max(std::min(1024u * 1024u * 1024u, QuarterGlobal),
                             32u * 1024u * 1024u);

    return ReturnValue(uint64_t{MaxAlloc});
  }
  case UR_DEVICE_INFO_IMAGE_SUPPORTED: {
    return ReturnValue(true);
  }
  case UR_DEVICE_INFO_MAX_READ_IMAGE_ARGS: {
    // This call doesn't match to HIP as it doesn't have images, but instead
    // surfaces and textures. No clear call in the HIP API to determine this,
    // but some searching found as of SM 2.x 128 are supported.
    return ReturnValue(128u);
  }
  case UR_DEVICE_INFO_MAX_READ_WRITE_IMAGE_ARGS: {
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
    int TexHeight = 0;
    UR_CHECK_ERROR(hipDeviceGetAttribute(
        &TexHeight, hipDeviceAttributeMaxTexture2DHeight, hDevice->get()));
    detail::ur::assertion(TexHeight >= 0);
    int SurfHeight = 0;
    UR_CHECK_ERROR(hipDeviceGetAttribute(
        &SurfHeight, hipDeviceAttributeMaxTexture2DHeight, hDevice->get()));
    detail::ur::assertion(SurfHeight >= 0);

    int Min = std::min(TexHeight, SurfHeight);

    return ReturnValue(static_cast<size_t>(Min));
  }
  case UR_DEVICE_INFO_IMAGE2D_MAX_WIDTH: {
    // Take the smaller of maximum surface and maximum texture width.
    int TexWidth = 0;
    UR_CHECK_ERROR(hipDeviceGetAttribute(
        &TexWidth, hipDeviceAttributeMaxTexture2DWidth, hDevice->get()));
    detail::ur::assertion(TexWidth >= 0);
    int SurfWidth = 0;
    UR_CHECK_ERROR(hipDeviceGetAttribute(
        &SurfWidth, hipDeviceAttributeMaxTexture2DWidth, hDevice->get()));
    detail::ur::assertion(SurfWidth >= 0);

    int Min = std::min(TexWidth, SurfWidth);

    return ReturnValue(static_cast<size_t>(Min));
  }
  case UR_DEVICE_INFO_IMAGE3D_MAX_HEIGHT: {
    // Take the smaller of maximum surface and maximum texture height.
    int TexHeight = 0;
    UR_CHECK_ERROR(hipDeviceGetAttribute(
        &TexHeight, hipDeviceAttributeMaxTexture3DHeight, hDevice->get()));
    detail::ur::assertion(TexHeight >= 0);
    int SurfHeight = 0;
    UR_CHECK_ERROR(hipDeviceGetAttribute(
        &SurfHeight, hipDeviceAttributeMaxTexture3DHeight, hDevice->get()));
    detail::ur::assertion(SurfHeight >= 0);

    int Min = std::min(TexHeight, SurfHeight);

    return ReturnValue(static_cast<size_t>(Min));
  }
  case UR_DEVICE_INFO_IMAGE3D_MAX_WIDTH: {
    // Take the smaller of maximum surface and maximum texture width.
    int TexWidth = 0;
    UR_CHECK_ERROR(hipDeviceGetAttribute(
        &TexWidth, hipDeviceAttributeMaxTexture3DWidth, hDevice->get()));
    detail::ur::assertion(TexWidth >= 0);
    int SurfWidth = 0;
    UR_CHECK_ERROR(hipDeviceGetAttribute(
        &SurfWidth, hipDeviceAttributeMaxTexture3DWidth, hDevice->get()));
    detail::ur::assertion(SurfWidth >= 0);

    int Min = std::min(TexWidth, SurfWidth);

    return ReturnValue(static_cast<size_t>(Min));
  }
  case UR_DEVICE_INFO_IMAGE3D_MAX_DEPTH: {
    // Take the smaller of maximum surface and maximum texture depth.
    int TexDepth = 0;
    UR_CHECK_ERROR(hipDeviceGetAttribute(
        &TexDepth, hipDeviceAttributeMaxTexture3DDepth, hDevice->get()));
    detail::ur::assertion(TexDepth >= 0);
    int SurfDepth = 0;
    UR_CHECK_ERROR(hipDeviceGetAttribute(
        &SurfDepth, hipDeviceAttributeMaxTexture3DDepth, hDevice->get()));
    detail::ur::assertion(SurfDepth >= 0);

    int Min = std::min(TexDepth, SurfDepth);

    return ReturnValue(static_cast<size_t>(Min));
  }
  case UR_DEVICE_INFO_IMAGE_MAX_BUFFER_SIZE: {
    // Take the smaller of maximum surface and maximum texture width.
    int TexWidth = 0;
    UR_CHECK_ERROR(hipDeviceGetAttribute(
        &TexWidth, hipDeviceAttributeMaxTexture1DWidth, hDevice->get()));
    detail::ur::assertion(TexWidth >= 0);
    int SurfWidth = 0;
    UR_CHECK_ERROR(hipDeviceGetAttribute(
        &SurfWidth, hipDeviceAttributeMaxTexture1DWidth, hDevice->get()));
    detail::ur::assertion(SurfWidth >= 0);

    int Min = std::min(TexWidth, SurfWidth);

    return ReturnValue(static_cast<size_t>(Min));
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
    int MemBaseAddrAlign = 0;
    UR_CHECK_ERROR(hipDeviceGetAttribute(
        &MemBaseAddrAlign, hipDeviceAttributeTextureAlignment, hDevice->get()));
    // Multiply by 8 as clGetDeviceInfo returns this value in bits
    MemBaseAddrAlign *= 8;
    return ReturnValue(MemBaseAddrAlign);
  }
  case UR_DEVICE_INFO_HALF_FP_CONFIG: {
    return ReturnValue(0u);
  }
  case UR_DEVICE_INFO_SINGLE_FP_CONFIG: {
    ur_device_fp_capability_flags_t Config =
        UR_DEVICE_FP_CAPABILITY_FLAG_DENORM |
        UR_DEVICE_FP_CAPABILITY_FLAG_INF_NAN |
        UR_DEVICE_FP_CAPABILITY_FLAG_ROUND_TO_NEAREST |
        UR_DEVICE_FP_CAPABILITY_FLAG_ROUND_TO_ZERO |
        UR_DEVICE_FP_CAPABILITY_FLAG_ROUND_TO_INF |
        UR_DEVICE_FP_CAPABILITY_FLAG_FMA |
        UR_DEVICE_FP_CAPABILITY_FLAG_CORRECTLY_ROUNDED_DIVIDE_SQRT;
    return ReturnValue(Config);
  }
  case UR_DEVICE_INFO_DOUBLE_FP_CONFIG: {
    ur_device_fp_capability_flags_t Config =
        UR_DEVICE_FP_CAPABILITY_FLAG_DENORM |
        UR_DEVICE_FP_CAPABILITY_FLAG_INF_NAN |
        UR_DEVICE_FP_CAPABILITY_FLAG_ROUND_TO_NEAREST |
        UR_DEVICE_FP_CAPABILITY_FLAG_ROUND_TO_ZERO |
        UR_DEVICE_FP_CAPABILITY_FLAG_ROUND_TO_INF |
        UR_DEVICE_FP_CAPABILITY_FLAG_FMA;
    return ReturnValue(Config);
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
    int CacheSize = 0;
    UR_CHECK_ERROR(hipDeviceGetAttribute(
        &CacheSize, hipDeviceAttributeL2CacheSize, hDevice->get()));
    detail::ur::assertion(CacheSize >= 0);
    // The L2 cache is global to the GPU.
    return ReturnValue(static_cast<uint64_t>(CacheSize));
  }
  case UR_DEVICE_INFO_GLOBAL_MEM_SIZE: {
    size_t Bytes = 0;
    // Runtime API has easy access to this value, driver API info is scarse.
    UR_CHECK_ERROR(hipDeviceTotalMem(&Bytes, hDevice->get()));
    return ReturnValue(uint64_t{Bytes});
  }
  case UR_DEVICE_INFO_MAX_CONSTANT_BUFFER_SIZE: {
    int ConstantMemory = 0;

    // hipDeviceGetAttribute takes a int*, however the size of the constant
    // memory on AMD GPU may be larger than what can fit in the positive part
    // of a signed integer, so use an unsigned integer and cast the pointer to
    // int*.
    UR_CHECK_ERROR(hipDeviceGetAttribute(&ConstantMemory,
                                         hipDeviceAttributeTotalConstantMemory,
                                         hDevice->get()));
    detail::ur::assertion(ConstantMemory >= 0);

    return ReturnValue(static_cast<uint64_t>(ConstantMemory));
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
    int LocalMemSize = 0;
    UR_CHECK_ERROR(hipDeviceGetAttribute(
        &LocalMemSize, hipDeviceAttributeMaxSharedMemoryPerBlock,
        hDevice->get()));
    detail::ur::assertion(LocalMemSize >= 0);
    return ReturnValue(static_cast<uint64_t>(LocalMemSize));
  }
  case UR_DEVICE_INFO_ERROR_CORRECTION_SUPPORT: {
    int EccEnabled = 0;
    UR_CHECK_ERROR(hipDeviceGetAttribute(
        &EccEnabled, hipDeviceAttributeEccEnabled, hDevice->get()));

    detail::ur::assertion((EccEnabled == 0) | (EccEnabled == 1));
    auto Result = static_cast<bool>(EccEnabled);
    return ReturnValue(Result);
  }
  case UR_DEVICE_INFO_HOST_UNIFIED_MEMORY: {
    int IsIntegrated = 0;
    UR_CHECK_ERROR(hipDeviceGetAttribute(
        &IsIntegrated, hipDeviceAttributeIntegrated, hDevice->get()));

    detail::ur::assertion((IsIntegrated == 0) | (IsIntegrated == 1));
    auto Result = static_cast<bool>(IsIntegrated);
    return ReturnValue(Result);
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
    auto Capability = ur_device_exec_capability_flags_t{
        UR_DEVICE_EXEC_CAPABILITY_FLAG_KERNEL};
    return ReturnValue(Capability);
  }
  case UR_DEVICE_INFO_QUEUE_ON_DEVICE_PROPERTIES: {
    // The mandated minimum capability:
    ur_queue_flags_t Capability = UR_QUEUE_FLAG_PROFILING_ENABLE |
                                  UR_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE;
    return ReturnValue(Capability);
  }
  case UR_DEVICE_INFO_QUEUE_ON_HOST_PROPERTIES:
  case UR_DEVICE_INFO_QUEUE_PROPERTIES: {
    // The mandated minimum capability:
    ur_queue_flags_t Capability = UR_QUEUE_FLAG_PROFILING_ENABLE;
    return ReturnValue(Capability);
  }
  case UR_DEVICE_INFO_BUILT_IN_KERNELS: {
    // An empty string is returned if no built-in kernels are supported by the
    // device.
    return ReturnValue("");
  }
  case UR_DEVICE_INFO_PLATFORM: {
    return ReturnValue(hDevice->getPlatform());
  }
  case UR_DEVICE_INFO_NAME: {
    static constexpr size_t MAX_DEVICE_NAME_LENGTH = 256u;
    char Name[MAX_DEVICE_NAME_LENGTH];
    UR_CHECK_ERROR(
        hipDeviceGetName(Name, MAX_DEVICE_NAME_LENGTH, hDevice->get()));
    // On AMD GPUs hipDeviceGetName returns an empty string, so return the arch
    // name instead, this is also what AMD OpenCL devices return.
    if (strlen(Name) == 0) {
      hipDeviceProp_t Props;
      detail::ur::assertion(hipGetDeviceProperties(&Props, hDevice->get()) ==
                            hipSuccess);

      return ReturnValue(Props.gcnArchName, strlen(Props.gcnArchName) + 1);
    }
    return ReturnValue(Name, strlen(Name) + 1);
  }
  case UR_DEVICE_INFO_VENDOR: {
    return ReturnValue("AMD Corporation");
  }
  case UR_DEVICE_INFO_DRIVER_VERSION: {
    std::string Version;
    UR_CHECK_ERROR(getHipVersionString(Version));
    return ReturnValue(Version.c_str());
  }
  case UR_DEVICE_INFO_PROFILE: {
    return ReturnValue("HIP");
  }
  case UR_DEVICE_INFO_REFERENCE_COUNT: {
    return ReturnValue(hDevice->getReferenceCount());
  }
  case UR_DEVICE_INFO_VERSION: {
    std::stringstream S;

    hipDeviceProp_t Props;
    detail::ur::assertion(hipGetDeviceProperties(&Props, hDevice->get()) ==
                          hipSuccess);
#if defined(__HIP_PLATFORM_NVIDIA__)
    S << Props.major << "." << Props.minor;
#elif defined(__HIP_PLATFORM_AMD__)
    S << Props.gcnArchName;
#else
#error("Must define exactly one of __HIP_PLATFORM_AMD__ or __HIP_PLATFORM_NVIDIA__");
#endif
    return ReturnValue(S.str().c_str());
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

    hipDeviceProp_t Props;
    detail::ur::assertion(hipGetDeviceProperties(&Props, hDevice->get()) ==
                          hipSuccess);

    if (Props.arch.hasDoubles) {
      SupportedExtensions += "cl_khr_fp64 ";
    }

    SupportedExtensions += "cl_khr_fp16 ";

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
  case UR_DEVICE_INFO_SUPPORTED_PARTITIONS: {
    if (pPropSizeRet) {
      *pPropSizeRet = 0;
    }
    return UR_RESULT_SUCCESS;
  }

  case UR_DEVICE_INFO_PARTITION_AFFINITY_DOMAIN: {
    return ReturnValue(0u);
  }
  case UR_DEVICE_INFO_PARTITION_TYPE: {
    if (pPropSizeRet) {
      *pPropSizeRet = 0;
    }
    return UR_RESULT_SUCCESS;
  }

  // Intel USM extensions
  case UR_DEVICE_INFO_USM_HOST_SUPPORT: {
    // from cl_intel_unified_shared_memory: "The host memory access capabilities
    // apply to any host allocation."
    //
    // query if/how the device can access page-locked host memory, possibly
    // through PCIe, using the same pointer as the host
    ur_device_usm_access_capability_flags_t Value = {};
    // if (getAttribute(device, HIP_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING)) {
    // the device shares a unified address space with the host
    if (getAttribute(hDevice, hipDeviceAttributeComputeCapabilityMajor) >= 6) {
      // compute capability 6.x introduces operations that are atomic with
      // respect to other CPUs and GPUs in the system
      Value = UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_ACCESS |
              UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_ATOMIC_ACCESS |
              UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_CONCURRENT_ACCESS |
              UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_ATOMIC_CONCURRENT_ACCESS;
    } else {
      // on GPU architectures with compute capability lower than 6.x, atomic
      // operations from the GPU to CPU memory will not be atomic with respect
      // to CPU initiated atomic operations
      Value = UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_ACCESS |
              UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_CONCURRENT_ACCESS;
    }
    return ReturnValue(Value);
  }
  case UR_DEVICE_INFO_USM_DEVICE_SUPPORT: {
    // from cl_intel_unified_shared_memory:
    // "The device memory access capabilities apply to any device allocation
    // associated with this device."
    //
    // query how the device can access memory allocated on the device itself (?)
    ur_device_usm_access_capability_flags_t Value =
        UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_ACCESS |
        UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_ATOMIC_ACCESS |
        UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_CONCURRENT_ACCESS |
        UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_ATOMIC_CONCURRENT_ACCESS;
    return ReturnValue(Value);
  }
  case UR_DEVICE_INFO_USM_SINGLE_SHARED_SUPPORT: {
    // from cl_intel_unified_shared_memory:
    // "The single device shared memory access capabilities apply to any shared
    // allocation associated with this device."
    //
    // query if/how the device can access managed memory associated to it
    ur_device_usm_access_capability_flags_t Value = {};
    if (getAttribute(hDevice, hipDeviceAttributeManagedMemory)) {
      // the device can allocate managed memory on this system
      Value = UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_ACCESS |
              UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_ATOMIC_ACCESS;
    }
    if (getAttribute(hDevice, hipDeviceAttributeConcurrentManagedAccess)) {
      // the device can coherently access managed memory concurrently with the
      // CPU
      Value |= UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_CONCURRENT_ACCESS;
      if (getAttribute(hDevice, hipDeviceAttributeComputeCapabilityMajor) >=
          6) {
        // compute capability 6.x introduces operations that are atomic with
        // respect to other CPUs and GPUs in the system
        Value |= UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_ATOMIC_CONCURRENT_ACCESS;
      }
    }
    return ReturnValue(Value);
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
    ur_device_usm_access_capability_flags_t Value = {};
    if (getAttribute(hDevice, hipDeviceAttributeManagedMemory)) {
      // the device can allocate managed memory on this system
      Value |= UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_ACCESS;
    }
    if (getAttribute(hDevice, hipDeviceAttributeConcurrentManagedAccess)) {
      // all devices with the CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS
      // attribute can coherently access managed memory concurrently with the
      // CPU
      Value |= UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_CONCURRENT_ACCESS;
    }
    if (getAttribute(hDevice, hipDeviceAttributeComputeCapabilityMajor) >= 6) {
      // compute capability 6.x introduces operations that are atomic with
      // respect to other CPUs and GPUs in the system
      if (Value & UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_ACCESS)
        Value |= UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_ATOMIC_ACCESS;
      if (Value & UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_CONCURRENT_ACCESS)
        Value |= UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_ATOMIC_CONCURRENT_ACCESS;
    }
    return ReturnValue(Value);
  }
  case UR_DEVICE_INFO_USM_SYSTEM_SHARED_SUPPORT: {
    // from cl_intel_unified_shared_memory:
    // "The shared system memory access capabilities apply to any allocations
    // made by a system allocator, such as malloc or new."
    //
    // query if/how the device can access pageable host memory allocated by the
    // system allocator
    ur_device_usm_access_capability_flags_t Value = {};
    if (getAttribute(hDevice, hipDeviceAttributePageableMemoryAccess)) {
      // the link between the device and the host does not support native
      // atomic operations
      Value = UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_ACCESS |
              UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_CONCURRENT_ACCESS;
    }
    return ReturnValue(Value);
  }

  case UR_DEVICE_INFO_BACKEND_RUNTIME_VERSION: {
    int Major = 0, Minor = 0;
    UR_CHECK_ERROR(hipDeviceComputeCapability(&Major, &Minor, hDevice->get()));
    std::string Result = std::to_string(Major) + "." + std::to_string(Minor);
    return ReturnValue(Result.c_str());
  }

  case UR_DEVICE_INFO_ATOMIC_64: {
    hipDeviceProp_t Props;
    detail::ur::assertion(hipGetDeviceProperties(&Props, hDevice->get()) ==
                          hipSuccess);
    return ReturnValue(Props.arch.hasGlobalInt64Atomics &&
                       Props.arch.hasSharedInt64Atomics);
  }

  case UR_DEVICE_INFO_GLOBAL_MEM_FREE: {
    size_t FreeMemory = 0;
    size_t TotalMemory = 0;
    detail::ur::assertion(hipMemGetInfo(&FreeMemory, &TotalMemory) ==
                              hipSuccess,
                          "failed hipMemGetInfo() API.");
    return ReturnValue(FreeMemory);
  }

  case UR_DEVICE_INFO_MEMORY_CLOCK_RATE: {
    int Value = 0;
    UR_CHECK_ERROR(hipDeviceGetAttribute(
        &Value, hipDeviceAttributeMemoryClockRate, hDevice->get()));
    detail::ur::assertion(Value >= 0);
    // Convert kilohertz to megahertz when returning.
    return ReturnValue(Value / 1000);
  }

  case UR_DEVICE_INFO_MEMORY_BUS_WIDTH: {
    int Value = 0;
    UR_CHECK_ERROR(hipDeviceGetAttribute(
        &Value, hipDeviceAttributeMemoryBusWidth, hDevice->get()));
    detail::ur::assertion(Value >= 0);
    return ReturnValue(Value);
  }
  case UR_DEVICE_INFO_MAX_COMPUTE_QUEUE_INDICES: {
    return ReturnValue(int32_t{1});
  }

  case UR_DEVICE_INFO_ATOMIC_MEMORY_ORDER_CAPABILITIES: {
    ur_memory_order_capability_flags_t Capabilities =
        UR_MEMORY_ORDER_CAPABILITY_FLAG_RELAXED |
        UR_MEMORY_ORDER_CAPABILITY_FLAG_ACQUIRE |
        UR_MEMORY_ORDER_CAPABILITY_FLAG_RELEASE;
    return ReturnValue(Capabilities);
  }
  case UR_DEVICE_INFO_ATOMIC_MEMORY_SCOPE_CAPABILITIES:
  case UR_DEVICE_INFO_ATOMIC_FENCE_SCOPE_CAPABILITIES: {
    // SYCL2020 4.6.4.2 minimum mandated capabilities for
    // atomic_fence/memory_scope_capabilities.
    // Because scopes are hierarchical, wider scopes support all narrower
    // scopes. At a minimum, each device must support WORK_ITEM, SUB_GROUP and
    // WORK_GROUP. (https://github.com/KhronosGroup/SYCL-Docs/pull/382)
    uint64_t Capabilities = UR_MEMORY_SCOPE_CAPABILITY_FLAG_WORK_ITEM |
                            UR_MEMORY_SCOPE_CAPABILITY_FLAG_SUB_GROUP |
                            UR_MEMORY_SCOPE_CAPABILITY_FLAG_WORK_GROUP;
    return ReturnValue(Capabilities);
  }
  case UR_DEVICE_INFO_ATOMIC_FENCE_ORDER_CAPABILITIES: {
    // SYCL2020 4.6.4.2 minimum mandated capabilities for
    // atomic_fence_order_capabilities.
    ur_memory_order_capability_flags_t Capabilities =
        UR_MEMORY_ORDER_CAPABILITY_FLAG_RELAXED |
        UR_MEMORY_ORDER_CAPABILITY_FLAG_ACQUIRE |
        UR_MEMORY_ORDER_CAPABILITY_FLAG_RELEASE |
        UR_MEMORY_ORDER_CAPABILITY_FLAG_ACQ_REL;
    return ReturnValue(Capabilities);
  }
  case UR_DEVICE_INFO_DEVICE_ID: {
    int Value = 0;
    UR_CHECK_ERROR(hipDeviceGetAttribute(&Value, hipDeviceAttributePciDeviceId,
                                         hDevice->get()));
    detail::ur::assertion(Value >= 0);
    return ReturnValue(Value);
  }
  case UR_DEVICE_INFO_UUID: {
#if ((HIP_VERSION_MAJOR == 5 && HIP_VERSION_MINOR >= 2) ||                     \
     HIP_VERSION_MAJOR > 5)
    hipUUID UUID = {};
    // Supported since 5.2+
    detail::ur::assertion(hipDeviceGetUuid(&UUID, hDevice->get()) ==
                          hipSuccess);
    std::array<unsigned char, 16> Name;
    std::copy(UUID.bytes, UUID.bytes + 16, Name.begin());
    return ReturnValue(Name.data(), 16);
#endif
    return UR_RESULT_ERROR_INVALID_VALUE;
  }
  case UR_DEVICE_INFO_MAX_REGISTERS_PER_WORK_GROUP: {
    // Maximum number of 32-bit registers available to a thread block.
    // Note: This number is shared by all thread blocks simultaneously resident
    // on a multiprocessor.
    int MaxRegisters{-1};
    UR_CHECK_ERROR(hipDeviceGetAttribute(
        &MaxRegisters, hipDeviceAttributeMaxRegistersPerBlock, hDevice->get()));

    detail::ur::assertion(MaxRegisters >= 0);

    return ReturnValue(static_cast<uint32_t>(MaxRegisters));
  }
  case UR_DEVICE_INFO_MEM_CHANNEL_SUPPORT:
    return ReturnValue(false);
  case UR_DEVICE_INFO_IMAGE_SRGB:
    return ReturnValue(false);
  case UR_DEVICE_INFO_PCI_ADDRESS: {
    constexpr size_t AddressBufferSize = 13;
    char AddressBuffer[AddressBufferSize];
    UR_CHECK_ERROR(
        hipDeviceGetPCIBusId(AddressBuffer, AddressBufferSize, hDevice->get()));
    // A typical PCI address is 12 bytes + \0: "1234:67:90.2", but the HIP API
    // is not guaranteed to use this format. In practice, it uses this format,
    // at least in 5.3-5.5. To be on the safe side, we make sure the terminating
    // \0 is set.
    AddressBuffer[AddressBufferSize - 1] = '\0';
    detail::ur::assertion(strnlen(AddressBuffer, AddressBufferSize) > 0);
    return ReturnValue(AddressBuffer,
                       strnlen(AddressBuffer, AddressBufferSize - 1) + 1);
  }
  case UR_DEVICE_INFO_HOST_PIPE_READ_WRITE_SUPPORTED:
    return ReturnValue(false);
  case UR_DEVICE_INFO_VIRTUAL_MEMORY_SUPPORT:
    return ReturnValue(false);
  case UR_DEVICE_INFO_ESIMD_SUPPORT:
    return ReturnValue(false);
  case UR_DEVICE_INFO_COMPONENT_DEVICES:
  case UR_DEVICE_INFO_COMPOSITE_DEVICE:
    // These two are exclusive of L0.
    return ReturnValue(0);

  // TODO: Investigate if this information is available on HIP.
  case UR_DEVICE_INFO_GPU_EU_COUNT:
  case UR_DEVICE_INFO_GPU_EU_SIMD_WIDTH:
  case UR_DEVICE_INFO_GPU_EU_SLICES:
  case UR_DEVICE_INFO_GPU_SUBSLICES_PER_SLICE:
  case UR_DEVICE_INFO_GPU_EU_COUNT_PER_SUBSLICE:
  case UR_DEVICE_INFO_GPU_HW_THREADS_PER_EU:
  case UR_DEVICE_INFO_MAX_MEMORY_BANDWIDTH:
  case UR_DEVICE_INFO_BFLOAT16:
  case UR_DEVICE_INFO_IL_VERSION:
  case UR_DEVICE_INFO_ASYNC_BARRIER:
    return UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION;

  case UR_DEVICE_INFO_COMMAND_BUFFER_SUPPORT_EXP:
  case UR_DEVICE_INFO_COMMAND_BUFFER_UPDATE_SUPPORT_EXP:
    return ReturnValue(false);

  default:
    break;
  }
  return UR_RESULT_ERROR_INVALID_ENUMERATION;
}

/// \return UR_RESULT_SUCCESS if the function is executed successfully
/// HIP devices are always root devices so retain always returns success.
UR_APIEXPORT ur_result_t UR_APICALL urDeviceRetain(ur_device_handle_t) {
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urDevicePartition(ur_device_handle_t, const ur_device_partition_properties_t *,
                  uint32_t, ur_device_handle_t *, uint32_t *) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

/// \return UR_RESULT_SUCCESS always since HIP devices are always root
/// devices.
UR_APIEXPORT ur_result_t UR_APICALL urDeviceRelease(ur_device_handle_t) {
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urDeviceGet(ur_platform_handle_t hPlatform,
                                                ur_device_type_t DeviceType,
                                                uint32_t NumEntries,
                                                ur_device_handle_t *phDevices,
                                                uint32_t *pNumDevices) {
  ur_result_t Err = UR_RESULT_SUCCESS;
  const bool AskingForDefault = DeviceType == UR_DEVICE_TYPE_DEFAULT;
  const bool AskingForGPU = DeviceType == UR_DEVICE_TYPE_GPU;
  const bool AskingForAll = DeviceType == UR_DEVICE_TYPE_ALL;
  const bool ReturnDevices = AskingForDefault || AskingForGPU || AskingForAll;

  size_t NumDevices = ReturnDevices ? hPlatform->Devices.size() : 0;

  try {
    UR_ASSERT(pNumDevices || phDevices, UR_RESULT_ERROR_INVALID_VALUE);

    if (pNumDevices) {
      *pNumDevices = NumDevices;
    }

    if (ReturnDevices && phDevices) {
      for (size_t i = 0; i < std::min(size_t(NumEntries), NumDevices); ++i) {
        phDevices[i] = hPlatform->Devices[i].get();
      }
    }

    return Err;
  } catch (ur_result_t Err) {
    return Err;
  } catch (...) {
    return UR_RESULT_ERROR_OUT_OF_RESOURCES;
  }
}

/// Gets the native HIP handle of a UR device object
///
/// \param[in] hDevice The UR device to get the native HIP object of.
/// \param[out] phNativeHandle Set to the native handle of the UR device object.
///
/// \return UR_RESULT_SUCCESS
UR_APIEXPORT ur_result_t UR_APICALL urDeviceGetNativeHandle(
    ur_device_handle_t hDevice, ur_native_handle_t *phNativeHandle) {
  *phNativeHandle = reinterpret_cast<ur_native_handle_t>(hDevice->get());
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urDeviceCreateWithNativeHandle(
    ur_native_handle_t, ur_platform_handle_t,
    const ur_device_native_properties_t *, ur_device_handle_t *) {
  return UR_RESULT_ERROR_INVALID_OPERATION;
}

/// \return UR_RESULT_SUCCESS If available, the first binary that is PTX
///
UR_APIEXPORT ur_result_t UR_APICALL
urDeviceSelectBinary(ur_device_handle_t, const ur_device_binary_t *pBinaries,
                     uint32_t NumBinaries, uint32_t *pSelectedBinary) {
  // Ignore unused parameter
  UR_ASSERT(NumBinaries > 0, UR_RESULT_ERROR_INVALID_ARGUMENT);

  // Look for an image for the HIP target, and return the first one that is
  // found
#if defined(__HIP_PLATFORM_AMD__)
  const char *BinaryType = UR_DEVICE_BINARY_TARGET_AMDGCN;
#elif defined(__HIP_PLATFORM_NVIDIA__)
  const char *BinaryType = UR_DEVICE_BINARY_TARGET_NVPTX64;
#else
#error("Must define exactly one of __HIP_PLATFORM_AMD__ or __HIP_PLATFORM_NVIDIA__");
#endif
  for (uint32_t i = 0; i < NumBinaries; i++) {
    if (strcmp(pBinaries[i].pDeviceTargetSpec, BinaryType) == 0) {
      *pSelectedBinary = i;
      return UR_RESULT_SUCCESS;
    }
  }

  // No image can be loaded for the given device
  return UR_RESULT_ERROR_INVALID_BINARY;
}

ur_result_t UR_APICALL urDeviceGetGlobalTimestamps(ur_device_handle_t hDevice,
                                                   uint64_t *pDeviceTimestamp,
                                                   uint64_t *pHostTimestamp) {
  if (!pDeviceTimestamp && !pHostTimestamp)
    return UR_RESULT_SUCCESS;

  ur_event_handle_t_::native_type Event;
  ScopedContext Active(hDevice);

  if (pDeviceTimestamp) {
    UR_CHECK_ERROR(hipEventCreateWithFlags(&Event, hipEventDefault));
    UR_CHECK_ERROR(hipEventRecord(Event));
    UR_CHECK_ERROR(hipEventSynchronize(Event));
    float ElapsedTime = 0.0f;
    UR_CHECK_ERROR(hipEventElapsedTime(&ElapsedTime,
                                       ur_platform_handle_t_::EvBase, Event));
    *pDeviceTimestamp = (uint64_t)(ElapsedTime * (double)1e6);
  }

  if (pHostTimestamp) {
    using namespace std::chrono;
    *pHostTimestamp =
        duration_cast<nanoseconds>(steady_clock::now().time_since_epoch())
            .count();
  }
  return UR_RESULT_SUCCESS;
}
