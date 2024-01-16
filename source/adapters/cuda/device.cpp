//===--------- device.cpp - CUDA Adapter ----------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <array>
#include <cassert>
#include <sstream>

#include "adapter.hpp"
#include "context.hpp"
#include "device.hpp"
#include "platform.hpp"
#include "ur_util.hpp"

int getAttribute(ur_device_handle_t device, CUdevice_attribute attribute) {
  int value;

  UR_CHECK_ERROR(cuDeviceGetAttribute(&value, attribute, device->get()));
  return value;
}

uint64_t ur_device_handle_t_::getElapsedTime(CUevent ev) const {
  float Milliseconds = 0.0f;

  // cuEventSynchronize waits till the event is ready for call to
  // cuEventElapsedTime.
  UR_CHECK_ERROR(cuEventSynchronize(EvBase));
  UR_CHECK_ERROR(cuEventSynchronize(ev));
  UR_CHECK_ERROR(cuEventElapsedTime(&Milliseconds, EvBase, ev));

  return static_cast<uint64_t>(Milliseconds * 1.0e6);
}

UR_APIEXPORT ur_result_t UR_APICALL urDeviceGetInfo(ur_device_handle_t hDevice,
                                                    ur_device_info_t propName,
                                                    size_t propSize,
                                                    void *pPropValue,
                                                    size_t *pPropSizeRet) try {
  UrReturnHelper ReturnValue(propSize, pPropValue, pPropSizeRet);

  static constexpr uint32_t MaxWorkItemDimensions = 3u;

  ScopedContext Active(hDevice->getContext());

  switch ((uint32_t)propName) {
  case UR_DEVICE_INFO_TYPE: {
    return ReturnValue(UR_DEVICE_TYPE_GPU);
  }
  case UR_DEVICE_INFO_VENDOR_ID: {
    return ReturnValue(4318u);
  }
  case UR_DEVICE_INFO_MAX_COMPUTE_UNITS: {
    int ComputeUnits = 0;
    UR_CHECK_ERROR(cuDeviceGetAttribute(
        &ComputeUnits, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
        hDevice->get()));
    detail::ur::assertion(ComputeUnits >= 0);
    return ReturnValue(static_cast<uint32_t>(ComputeUnits));
  }
  case UR_DEVICE_INFO_MAX_WORK_ITEM_DIMENSIONS: {
    return ReturnValue(MaxWorkItemDimensions);
  }
  case UR_DEVICE_INFO_MAX_WORK_ITEM_SIZES: {
    struct {
      size_t Sizes[MaxWorkItemDimensions];
    } ReturnSizes;

    int MaxX = 0, MaxY = 0, MaxZ = 0;
    UR_CHECK_ERROR(cuDeviceGetAttribute(
        &MaxX, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, hDevice->get()));
    detail::ur::assertion(MaxX >= 0);

    UR_CHECK_ERROR(cuDeviceGetAttribute(
        &MaxY, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y, hDevice->get()));
    detail::ur::assertion(MaxY >= 0);

    UR_CHECK_ERROR(cuDeviceGetAttribute(
        &MaxZ, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z, hDevice->get()));
    detail::ur::assertion(MaxZ >= 0);

    ReturnSizes.Sizes[0] = size_t(MaxX);
    ReturnSizes.Sizes[1] = size_t(MaxY);
    ReturnSizes.Sizes[2] = size_t(MaxZ);
    return ReturnValue(ReturnSizes);
  }

  case UR_DEVICE_INFO_MAX_WORK_GROUPS_3D: {
    struct {
      size_t Sizes[MaxWorkItemDimensions];
    } ReturnSizes;
    int MaxX = 0, MaxY = 0, MaxZ = 0;
    UR_CHECK_ERROR(cuDeviceGetAttribute(
        &MaxX, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, hDevice->get()));
    detail::ur::assertion(MaxX >= 0);

    UR_CHECK_ERROR(cuDeviceGetAttribute(
        &MaxY, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y, hDevice->get()));
    detail::ur::assertion(MaxY >= 0);

    UR_CHECK_ERROR(cuDeviceGetAttribute(
        &MaxZ, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z, hDevice->get()));
    detail::ur::assertion(MaxZ >= 0);

    ReturnSizes.Sizes[0] = size_t(MaxX);
    ReturnSizes.Sizes[1] = size_t(MaxY);
    ReturnSizes.Sizes[2] = size_t(MaxZ);
    return ReturnValue(ReturnSizes);
  }

  case UR_DEVICE_INFO_MAX_WORK_GROUP_SIZE: {
    int MaxWorkGroupSize = 0;
    UR_CHECK_ERROR(cuDeviceGetAttribute(
        &MaxWorkGroupSize, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
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
    UR_CHECK_ERROR(cuDeviceGetAttribute(
        &MaxThreads, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
        hDevice->get()));
    int WarpSize = 0;
    UR_CHECK_ERROR(cuDeviceGetAttribute(
        &WarpSize, CU_DEVICE_ATTRIBUTE_WARP_SIZE, hDevice->get()));
    int MaxWarps = (MaxThreads + WarpSize - 1) / WarpSize;
    return ReturnValue(MaxWarps);
  }
  case UR_DEVICE_INFO_SUB_GROUP_INDEPENDENT_FORWARD_PROGRESS: {
    // Volta provides independent thread scheduling
    // TODO: Revisit for previous generation GPUs
    int Major = 0;
    UR_CHECK_ERROR(cuDeviceGetAttribute(
        &Major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, hDevice->get()));
    bool IFP = (Major >= 7);
    return ReturnValue(IFP);
  }

  case UR_DEVICE_INFO_ATOMIC_64: {
    int Major = 0;
    UR_CHECK_ERROR(cuDeviceGetAttribute(
        &Major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, hDevice->get()));

    bool Atomic64 = (Major >= 6) ? true : false;
    return ReturnValue(Atomic64);
  }
  case UR_DEVICE_INFO_ATOMIC_MEMORY_ORDER_CAPABILITIES: {
    ur_memory_order_capability_flags_t Capabilities =
        UR_MEMORY_ORDER_CAPABILITY_FLAG_RELAXED |
        UR_MEMORY_ORDER_CAPABILITY_FLAG_ACQUIRE |
        UR_MEMORY_ORDER_CAPABILITY_FLAG_RELEASE |
        UR_MEMORY_ORDER_CAPABILITY_FLAG_ACQ_REL;
    return ReturnValue(Capabilities);
  }
  case UR_DEVICE_INFO_ATOMIC_MEMORY_SCOPE_CAPABILITIES: {
    int Major = 0;
    UR_CHECK_ERROR(cuDeviceGetAttribute(
        &Major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, hDevice->get()));
    uint64_t Capabilities =
        (Major >= 7) ? UR_MEMORY_SCOPE_CAPABILITY_FLAG_WORK_ITEM |
                           UR_MEMORY_SCOPE_CAPABILITY_FLAG_SUB_GROUP |
                           UR_MEMORY_SCOPE_CAPABILITY_FLAG_WORK_GROUP |
                           UR_MEMORY_SCOPE_CAPABILITY_FLAG_DEVICE |
                           UR_MEMORY_SCOPE_CAPABILITY_FLAG_SYSTEM
                     : UR_MEMORY_SCOPE_CAPABILITY_FLAG_WORK_ITEM |
                           UR_MEMORY_SCOPE_CAPABILITY_FLAG_SUB_GROUP |
                           UR_MEMORY_SCOPE_CAPABILITY_FLAG_WORK_GROUP |
                           UR_MEMORY_SCOPE_CAPABILITY_FLAG_DEVICE;
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
  case UR_DEVICE_INFO_ATOMIC_FENCE_SCOPE_CAPABILITIES: {
    // SYCL2020 4.6.4.2 minimum mandated capabilities for
    // atomic_fence/memory_scope_capabilities.
    // Because scopes are hierarchical, wider scopes support all narrower
    // scopes. At a minimum, each device must support WORK_ITEM, SUB_GROUP and
    // WORK_GROUP. (https://github.com/KhronosGroup/SYCL-Docs/pull/382)
    ur_memory_scope_capability_flags_t Capabilities =
        UR_MEMORY_SCOPE_CAPABILITY_FLAG_WORK_ITEM |
        UR_MEMORY_SCOPE_CAPABILITY_FLAG_SUB_GROUP |
        UR_MEMORY_SCOPE_CAPABILITY_FLAG_WORK_GROUP;
    return ReturnValue(Capabilities);
  }
  case UR_DEVICE_INFO_BFLOAT16: {
    int Major = 0;
    UR_CHECK_ERROR(cuDeviceGetAttribute(
        &Major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, hDevice->get()));

    bool BFloat16 = (Major >= 8) ? true : false;
    return ReturnValue(BFloat16);
  }
  case UR_DEVICE_INFO_SUB_GROUP_SIZES_INTEL: {
    // NVIDIA devices only support one sub-group size (the warp size)
    int WarpSize = 0;
    UR_CHECK_ERROR(cuDeviceGetAttribute(
        &WarpSize, CU_DEVICE_ATTRIBUTE_WARP_SIZE, hDevice->get()));
    size_t Sizes[1] = {static_cast<size_t>(WarpSize)};
    return ReturnValue(Sizes, 1);
  }
  case UR_DEVICE_INFO_MAX_CLOCK_FREQUENCY: {
    int ClockFreq = 0;
    UR_CHECK_ERROR(cuDeviceGetAttribute(
        &ClockFreq, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, hDevice->get()));
    detail::ur::assertion(ClockFreq >= 0);
    return ReturnValue(static_cast<uint32_t>(ClockFreq) / 1000u);
  }
  case UR_DEVICE_INFO_ADDRESS_BITS: {
    auto Bits = uint32_t{std::numeric_limits<uintptr_t>::digits};
    return ReturnValue(Bits);
  }
  case UR_DEVICE_INFO_MAX_MEM_ALLOC_SIZE: {
    return ReturnValue(uint64_t{hDevice->getMaxAllocSize()});
  }
  case UR_DEVICE_INFO_IMAGE_SUPPORTED: {
    bool Enabled = false;

    if (std::getenv("SYCL_PI_CUDA_ENABLE_IMAGE_SUPPORT") != nullptr ||
        std::getenv("UR_CUDA_ENABLE_IMAGE_SUPPORT") != nullptr) {
      Enabled = true;
    } else {
      detail::ur::cuPrint(
          "Images are not fully supported by the CUDA BE, their support is "
          "disabled by default. Their partial support can be activated by "
          "setting SYCL_PI_CUDA_ENABLE_IMAGE_SUPPORT environment variable at "
          "runtime.");
    }

    return ReturnValue(Enabled);
  }
  case UR_DEVICE_INFO_MAX_READ_IMAGE_ARGS: {
    // This call doesn't match to CUDA as it doesn't have images, but instead
    // surfaces and textures. No clear call in the CUDA API to determine this,
    // but some searching found as of SM 2.x 128 are supported.
    return ReturnValue(128u);
  }
  case UR_DEVICE_INFO_MAX_WRITE_IMAGE_ARGS: {
    // This call doesn't match to CUDA as it doesn't have images, but instead
    // surfaces and textures. No clear call in the CUDA API to determine this,
    // but some searching found as of SM 2.x 128 are supported.
    return ReturnValue(128u);
  }
  case UR_DEVICE_INFO_IMAGE2D_MAX_HEIGHT: {
    // Take the smaller of maximum surface and maximum texture height.
    int TexHeight = 0;
    UR_CHECK_ERROR(cuDeviceGetAttribute(
        &TexHeight, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT,
        hDevice->get()));
    detail::ur::assertion(TexHeight >= 0);
    int SurfHeight = 0;
    UR_CHECK_ERROR(cuDeviceGetAttribute(
        &SurfHeight, CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT,
        hDevice->get()));
    detail::ur::assertion(SurfHeight >= 0);

    int Min = std::min(TexHeight, SurfHeight);

    return ReturnValue(static_cast<size_t>(Min));
  }
  case UR_DEVICE_INFO_IMAGE2D_MAX_WIDTH: {
    // Take the smaller of maximum surface and maximum texture width.
    int TexWidth = 0;
    UR_CHECK_ERROR(cuDeviceGetAttribute(
        &TexWidth, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH,
        hDevice->get()));
    detail::ur::assertion(TexWidth >= 0);
    int SurfWidth = 0;
    UR_CHECK_ERROR(cuDeviceGetAttribute(
        &SurfWidth, CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH,
        hDevice->get()));
    detail::ur::assertion(SurfWidth >= 0);

    int Min = std::min(TexWidth, SurfWidth);

    return ReturnValue(static_cast<size_t>(Min));
  }
  case UR_DEVICE_INFO_IMAGE3D_MAX_HEIGHT: {
    // Take the smaller of maximum surface and maximum texture height.
    int TexHeight = 0;
    UR_CHECK_ERROR(cuDeviceGetAttribute(
        &TexHeight, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT,
        hDevice->get()));
    detail::ur::assertion(TexHeight >= 0);
    int SurfHeight = 0;
    UR_CHECK_ERROR(cuDeviceGetAttribute(
        &SurfHeight, CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT,
        hDevice->get()));
    detail::ur::assertion(SurfHeight >= 0);

    int Min = std::min(TexHeight, SurfHeight);

    return ReturnValue(static_cast<size_t>(Min));
  }
  case UR_DEVICE_INFO_IMAGE3D_MAX_WIDTH: {
    // Take the smaller of maximum surface and maximum texture width.
    int TexWidth = 0;
    UR_CHECK_ERROR(cuDeviceGetAttribute(
        &TexWidth, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH,
        hDevice->get()));
    detail::ur::assertion(TexWidth >= 0);
    int SurfWidth = 0;
    UR_CHECK_ERROR(cuDeviceGetAttribute(
        &SurfWidth, CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH,
        hDevice->get()));
    detail::ur::assertion(SurfWidth >= 0);

    int Min = std::min(TexWidth, SurfWidth);

    return ReturnValue(static_cast<size_t>(Min));
  }
  case UR_DEVICE_INFO_IMAGE3D_MAX_DEPTH: {
    // Take the smaller of maximum surface and maximum texture depth.
    int TexDepth = 0;
    UR_CHECK_ERROR(cuDeviceGetAttribute(
        &TexDepth, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH,
        hDevice->get()));
    detail::ur::assertion(TexDepth >= 0);
    int SurfDepth = 0;
    UR_CHECK_ERROR(cuDeviceGetAttribute(
        &SurfDepth, CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH,
        hDevice->get()));
    detail::ur::assertion(SurfDepth >= 0);

    int Min = std::min(TexDepth, SurfDepth);

    return ReturnValue(static_cast<size_t>(Min));
  }
  case UR_DEVICE_INFO_IMAGE_MAX_BUFFER_SIZE: {
    // Take the smaller of maximum surface and maximum texture width.
    int TexWidth = 0;
    UR_CHECK_ERROR(cuDeviceGetAttribute(
        &TexWidth, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH,
        hDevice->get()));
    detail::ur::assertion(TexWidth >= 0);
    int SurfWidth = 0;
    UR_CHECK_ERROR(cuDeviceGetAttribute(
        &SurfWidth, CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH,
        hDevice->get()));
    detail::ur::assertion(SurfWidth >= 0);

    int Min = std::min(TexWidth, SurfWidth);

    return ReturnValue(static_cast<size_t>(Min));
  }
  case UR_DEVICE_INFO_IMAGE_MAX_ARRAY_SIZE: {
    return ReturnValue(0lu);
  }
  case UR_DEVICE_INFO_MAX_SAMPLERS: {
    // This call is kind of meaningless for cuda, as samplers don't exist.
    // Closest thing is textures, which is 128.
    return ReturnValue(128u);
  }
  case UR_DEVICE_INFO_MAX_PARAMETER_SIZE: {
    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/#function-parameters
    // __global__ function parameters are passed to the device via constant
    // memory and are limited to 4 KB.
    return ReturnValue(4000lu);
  }
  case UR_DEVICE_INFO_MEM_BASE_ADDR_ALIGN: {
    int MemBaseAddrAlign = 0;
    UR_CHECK_ERROR(cuDeviceGetAttribute(&MemBaseAddrAlign,
                                        CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT,
                                        hDevice->get()));
    // Multiply by 8 as clGetDeviceInfo returns this value in bits
    MemBaseAddrAlign *= 8;
    return ReturnValue(MemBaseAddrAlign);
  }
  case UR_DEVICE_INFO_HALF_FP_CONFIG: {
    // TODO: is this config consistent across all NVIDIA GPUs?
    return ReturnValue(0u);
  }
  case UR_DEVICE_INFO_SINGLE_FP_CONFIG: {
    // TODO: is this config consistent across all NVIDIA GPUs?
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
    // TODO: is this config consistent across all NVIDIA GPUs?
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
    // TODO: is this config consistent across all NVIDIA GPUs?
    return ReturnValue(UR_DEVICE_MEM_CACHE_TYPE_READ_WRITE_CACHE);
  }
  case UR_DEVICE_INFO_GLOBAL_MEM_CACHELINE_SIZE: {
    // The value is documented for all existing GPUs in the CUDA programming
    // guidelines, section "H.3.2. Global Memory".
    return ReturnValue(128u);
  }
  case UR_DEVICE_INFO_GLOBAL_MEM_CACHE_SIZE: {
    int CacheSize = 0;
    UR_CHECK_ERROR(cuDeviceGetAttribute(
        &CacheSize, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, hDevice->get()));
    detail::ur::assertion(CacheSize >= 0);
    // The L2 cache is global to the GPU.
    return ReturnValue(static_cast<uint64_t>(CacheSize));
  }
  case UR_DEVICE_INFO_GLOBAL_MEM_SIZE: {
    size_t Bytes = 0;
    // Runtime API has easy access to this value, driver API info is scarse.
    detail::ur::assertion(cuDeviceTotalMem(&Bytes, hDevice->get()) ==
                          CUDA_SUCCESS);
    return ReturnValue(uint64_t{Bytes});
  }
  case UR_DEVICE_INFO_MAX_CONSTANT_BUFFER_SIZE: {
    int ConstantMemory = 0;
    UR_CHECK_ERROR(cuDeviceGetAttribute(
        &ConstantMemory, CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY,
        hDevice->get()));
    detail::ur::assertion(ConstantMemory >= 0);

    return ReturnValue(static_cast<uint64_t>(ConstantMemory));
  }
  case UR_DEVICE_INFO_MAX_CONSTANT_ARGS: {
    // TODO: is there a way to retrieve this from CUDA driver API?
    // Hard coded to value returned by clinfo for OpenCL 1.2 CUDA | GeForce GTX
    // 1060 3GB
    return ReturnValue(9u);
  }
  case UR_DEVICE_INFO_LOCAL_MEM_TYPE: {
    return ReturnValue(UR_DEVICE_LOCAL_MEM_TYPE_LOCAL);
  }
  case UR_DEVICE_INFO_LOCAL_MEM_SIZE: {
    // OpenCL's "local memory" maps most closely to CUDA's "shared memory".
    // CUDA has its own definition of "local memory", which maps to OpenCL's
    // "private memory".
    if (hDevice->maxLocalMemSizeChosen()) {
      return ReturnValue(
          static_cast<uint64_t>(hDevice->getMaxChosenLocalMem()));
    } else {
      return ReturnValue(
          static_cast<uint64_t>(hDevice->getMaxCapacityLocalMem()));
    }
  }
  case UR_DEVICE_INFO_ERROR_CORRECTION_SUPPORT: {
    int ECCEnabled = 0;
    UR_CHECK_ERROR(cuDeviceGetAttribute(
        &ECCEnabled, CU_DEVICE_ATTRIBUTE_ECC_ENABLED, hDevice->get()));

    detail::ur::assertion((ECCEnabled == 0) | (ECCEnabled == 1));
    auto Result = static_cast<bool>(ECCEnabled);
    return ReturnValue(Result);
  }
  case UR_DEVICE_INFO_HOST_UNIFIED_MEMORY: {
    int IsIntegrated = 0;
    UR_CHECK_ERROR(cuDeviceGetAttribute(
        &IsIntegrated, CU_DEVICE_ATTRIBUTE_INTEGRATED, hDevice->get()));

    detail::ur::assertion((IsIntegrated == 0) | (IsIntegrated == 1));
    auto result = static_cast<bool>(IsIntegrated);
    return ReturnValue(result);
  }
  case UR_DEVICE_INFO_PROFILING_TIMER_RESOLUTION: {
    // Hard coded to value returned by clinfo for OpenCL 1.2 CUDA | GeForce GTX
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
  case UR_DEVICE_INFO_QUEUE_PROPERTIES:
    return ReturnValue(
        ur_queue_flag_t(UR_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE |
                        UR_QUEUE_FLAG_PROFILING_ENABLE));
  case UR_DEVICE_INFO_QUEUE_ON_DEVICE_PROPERTIES: {
    // The mandated minimum capability:
    ur_queue_flags_t Capability = UR_QUEUE_FLAG_PROFILING_ENABLE |
                                  UR_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE;
    return ReturnValue(Capability);
  }
  case UR_DEVICE_INFO_QUEUE_ON_HOST_PROPERTIES: {
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
    static constexpr size_t MaxDeviceNameLength = 256u;
    char Name[MaxDeviceNameLength];
    UR_CHECK_ERROR(cuDeviceGetName(Name, MaxDeviceNameLength, hDevice->get()));
    return ReturnValue(Name, strlen(Name) + 1);
  }
  case UR_DEVICE_INFO_VENDOR: {
    return ReturnValue("NVIDIA Corporation");
  }
  case UR_DEVICE_INFO_DRIVER_VERSION: {
    auto Version = getCudaVersionString();
    return ReturnValue(Version.c_str());
  }
  case UR_DEVICE_INFO_PROFILE: {
    return ReturnValue("CUDA");
  }
  case UR_DEVICE_INFO_REFERENCE_COUNT: {
    return ReturnValue(hDevice->getReferenceCount());
  }
  case UR_DEVICE_INFO_VERSION: {
    std::stringstream SS;
    int Major;
    UR_CHECK_ERROR(cuDeviceGetAttribute(
        &Major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, hDevice->get()));
    SS << Major;
    int Minor;
    UR_CHECK_ERROR(cuDeviceGetAttribute(
        &Minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, hDevice->get()));
    SS << "." << Minor;
    return ReturnValue(SS.str().c_str());
  }
  case UR_EXT_DEVICE_INFO_OPENCL_C_VERSION: {
    return ReturnValue("");
  }
  case UR_DEVICE_INFO_EXTENSIONS: {

    std::string SupportedExtensions = "cl_khr_fp64 cl_khr_subgroups ";
    SupportedExtensions += "pi_ext_intel_devicelib_assert ";
    // Return supported for the UR command-buffer experimental feature
    SupportedExtensions += "ur_exp_command_buffer ";
    SupportedExtensions += " ";

    int Major = 0;
    int Minor = 0;

    UR_CHECK_ERROR(cuDeviceGetAttribute(
        &Major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, hDevice->get()));
    UR_CHECK_ERROR(cuDeviceGetAttribute(
        &Minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, hDevice->get()));

    if ((Major >= 6) || ((Major == 5) && (Minor >= 3))) {
      SupportedExtensions += "cl_khr_fp16 ";
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
    uint32_t Value = {};
    if (getAttribute(hDevice, CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING)) {
      // the device shares a unified address space with the host
      if (getAttribute(hDevice, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR) >=
          6) {
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
    }
    return ReturnValue(Value);
  }
  case UR_DEVICE_INFO_USM_DEVICE_SUPPORT: {
    // from cl_intel_unified_shared_memory:
    // "The device memory access capabilities apply to any device allocation
    // associated with this device."
    //
    // query how the device can access memory allocated on the device itself (?)
    uint32_t Value =
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
    uint32_t Value = {};
    if (getAttribute(hDevice, CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY)) {
      // the device can allocate managed memory on this system
      Value = UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_ACCESS |
              UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_ATOMIC_ACCESS;
    }
    if (getAttribute(hDevice, CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS)) {
      // the device can coherently access managed memory concurrently with the
      // CPU
      Value |= UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_CONCURRENT_ACCESS;
      if (getAttribute(hDevice, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR) >=
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
    uint32_t Value = {};
    if (getAttribute(hDevice, CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY)) {
      // the device can allocate managed memory on this system
      Value |= UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_ACCESS;
    }
    if (getAttribute(hDevice, CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS)) {
      // all devices with the CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS
      // attribute can coherently access managed memory concurrently with the
      // CPU
      Value |= UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_CONCURRENT_ACCESS;
    }
    if (getAttribute(hDevice, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR) >=
        6) {
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
    uint32_t Value = {};
    if (getAttribute(hDevice, CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS)) {
      // the device suppports coherently accessing pageable memory without
      // calling cuMemHostRegister/cudaHostRegister on it
      if (getAttribute(hDevice,
                       CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED)) {
        // the link between the device and the host supports native atomic
        // operations
        Value = UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_ACCESS |
                UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_ATOMIC_ACCESS |
                UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_CONCURRENT_ACCESS |
                UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_ATOMIC_CONCURRENT_ACCESS;
      } else {
        // the link between the device and the host does not support native
        // atomic operations
        Value = UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_ACCESS |
                UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_CONCURRENT_ACCESS;
      }
    }
    return ReturnValue(Value);
  }
  case UR_DEVICE_INFO_ASYNC_BARRIER: {
    int Value = getAttribute(hDevice,
                             CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR) >= 8;
    return ReturnValue(static_cast<bool>(Value));
  }
  case UR_DEVICE_INFO_BACKEND_RUNTIME_VERSION: {
    int Major =
        getAttribute(hDevice, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR);
    int Minor =
        getAttribute(hDevice, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR);
    std::string Result = std::to_string(Major) + "." + std::to_string(Minor);
    return ReturnValue(Result.c_str());
  }

  case UR_DEVICE_INFO_GLOBAL_MEM_FREE: {
    size_t FreeMemory = 0;
    size_t TotalMemory = 0;
    detail::ur::assertion(cuMemGetInfo(&FreeMemory, &TotalMemory) ==
                              CUDA_SUCCESS,
                          "failed cuMemGetInfo() API.");
    return ReturnValue(FreeMemory);
  }
  case UR_DEVICE_INFO_MEMORY_CLOCK_RATE: {
    int Value = 0;
    UR_CHECK_ERROR(cuDeviceGetAttribute(
        &Value, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, hDevice->get()));
    detail::ur::assertion(Value >= 0);
    // Convert kilohertz to megahertz when returning.
    return ReturnValue(Value / 1000);
  }
  case UR_DEVICE_INFO_MEMORY_BUS_WIDTH: {
    int Value = 0;
    UR_CHECK_ERROR(cuDeviceGetAttribute(
        &Value, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, hDevice->get()));
    detail::ur::assertion(Value >= 0);
    return ReturnValue(Value);
  }
  case UR_DEVICE_INFO_MAX_COMPUTE_QUEUE_INDICES: {
    return ReturnValue(int32_t{1});
  }
  case UR_DEVICE_INFO_BINDLESS_IMAGES_SUPPORT_EXP: {
    // On CUDA bindless images are supported.
    return ReturnValue(true);
  }
  case UR_DEVICE_INFO_BINDLESS_IMAGES_SHARED_USM_SUPPORT_EXP: {
    // On CUDA bindless images can be backed by shared (managed) USM.
    return ReturnValue(true);
  }
  case UR_DEVICE_INFO_BINDLESS_IMAGES_1D_USM_SUPPORT_EXP: {
    // On CUDA 1D bindless image USM is not supported.
    // More specifically, linear filtering is not supported.
    return ReturnValue(false);
  }
  case UR_DEVICE_INFO_BINDLESS_IMAGES_2D_USM_SUPPORT_EXP: {
    // On CUDA 2D bindless image USM is supported.
    return ReturnValue(true);
  }
  case UR_DEVICE_INFO_IMAGE_PITCH_ALIGN_EXP: {
    int32_t tex_pitch_align = 0;
    UR_CHECK_ERROR(cuDeviceGetAttribute(
        &tex_pitch_align, CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT,
        hDevice->get()));
    return ReturnValue(tex_pitch_align);
  }
  case UR_DEVICE_INFO_MAX_IMAGE_LINEAR_WIDTH_EXP: {
    int32_t tex_max_linear_width = 0;
    UR_CHECK_ERROR(cuDeviceGetAttribute(
        &tex_max_linear_width,
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH, hDevice->get()));
    return ReturnValue(tex_max_linear_width);
  }
  case UR_DEVICE_INFO_MAX_IMAGE_LINEAR_HEIGHT_EXP: {
    int32_t tex_max_linear_height = 0;
    UR_CHECK_ERROR(cuDeviceGetAttribute(
        &tex_max_linear_height,
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT, hDevice->get()));
    return ReturnValue(tex_max_linear_height);
  }
  case UR_DEVICE_INFO_MAX_IMAGE_LINEAR_PITCH_EXP: {
    int32_t tex_max_linear_pitch = 0;
    UR_CHECK_ERROR(cuDeviceGetAttribute(
        &tex_max_linear_pitch,
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH, hDevice->get()));
    return ReturnValue(tex_max_linear_pitch);
  }
  case UR_DEVICE_INFO_MIPMAP_SUPPORT_EXP: {
    // CUDA supports mipmaps.
    return ReturnValue(true);
  }
  case UR_DEVICE_INFO_MIPMAP_ANISOTROPY_SUPPORT_EXP: {
    // CUDA supports anisotropic filtering.
    return ReturnValue(true);
  }
  case UR_DEVICE_INFO_MIPMAP_MAX_ANISOTROPY_EXP: {
    // CUDA has no query for this, but documentation states max value is 16.
    return ReturnValue(16.f);
  }
  case UR_DEVICE_INFO_MIPMAP_LEVEL_REFERENCE_SUPPORT_EXP: {
    // CUDA supports creation of images from individual mipmap levels.
    return ReturnValue(true);
  }

  case UR_DEVICE_INFO_INTEROP_MEMORY_IMPORT_SUPPORT_EXP: {
    // CUDA supports importing external memory.
    return ReturnValue(true);
  }
  case UR_DEVICE_INFO_INTEROP_MEMORY_EXPORT_SUPPORT_EXP: {
    // CUDA does not support exporting it's own device memory.
    return ReturnValue(false);
  }
  case UR_DEVICE_INFO_INTEROP_SEMAPHORE_IMPORT_SUPPORT_EXP: {
    // CUDA supports importing external semaphores.
    return ReturnValue(true);
  }
  case UR_DEVICE_INFO_INTEROP_SEMAPHORE_EXPORT_SUPPORT_EXP: {
    // CUDA does not support exporting semaphores or events.
    return ReturnValue(false);
  }
  case UR_DEVICE_INFO_DEVICE_ID: {
    int Value = 0;
    UR_CHECK_ERROR(cuDeviceGetAttribute(
        &Value, CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, hDevice->get()));
    detail::ur::assertion(Value >= 0);
    return ReturnValue(Value);
  }
  case UR_DEVICE_INFO_UUID: {
    CUuuid UUID;
#if (CUDA_VERSION >= 11040)
    detail::ur::assertion(cuDeviceGetUuid_v2(&UUID, hDevice->get()) ==
                          CUDA_SUCCESS);
#else
    detail::ur::assertion(cuDeviceGetUuid(&UUID, hDevice->get()) ==
                          CUDA_SUCCESS);
#endif
    std::array<unsigned char, 16> Name;
    std::copy(UUID.bytes, UUID.bytes + 16, Name.begin());
    return ReturnValue(Name.data(), 16);
  }
  case UR_DEVICE_INFO_MAX_MEMORY_BANDWIDTH: {
    int Major = 0;
    UR_CHECK_ERROR(cuDeviceGetAttribute(
        &Major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, hDevice->get()));

    int Minor = 0;
    UR_CHECK_ERROR(cuDeviceGetAttribute(
        &Minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, hDevice->get()));

    // Some specific devices seem to need special handling. See reference
    // https://github.com/jeffhammond/HPCInfo/blob/master/cuda/gpu-detect.cu
    bool IsXavierAGX = Major == 7 && Minor == 2;
    bool IsOrinAGX = Major == 8 && Minor == 7;

    int MemoryClockKHz = 0;
    if (IsXavierAGX) {
      MemoryClockKHz = 2133000;
    } else if (IsOrinAGX) {
      MemoryClockKHz = 3200000;
    } else {
      UR_CHECK_ERROR(cuDeviceGetAttribute(&MemoryClockKHz,
                                          CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE,
                                          hDevice->get()));
    }

    int MemoryBusWidth = 0;
    if (IsOrinAGX) {
      MemoryBusWidth = 256;
    } else {
      UR_CHECK_ERROR(cuDeviceGetAttribute(
          &MemoryBusWidth, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH,
          hDevice->get()));
    }

    uint32_t MemoryBandwidth = MemoryClockKHz * MemoryBusWidth * 250;

    return ReturnValue(MemoryBandwidth);
  }
  case UR_DEVICE_INFO_IL_VERSION: {
    std::string ILVersion = "nvptx-";

    int DriverVersion = 0;
    cuDriverGetVersion(&DriverVersion);
    int Major = DriverVersion / 1000;
    int Minor = DriverVersion % 1000 / 10;

    // We can work out which ptx ISA version we support based on the versioning
    // table published here
    // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#release-notes
    // Major versions that we support are consistent in how they line up, so we
    // can derive that easily. The minor versions for version 10 don't line up
    // the same so it needs a special case. This is not ideal but it does seem
    // to be the best bet to avoid a maintenance burden here.
    ILVersion += std::to_string(Major - 4) + ".";
    if (Major == 10) {
      ILVersion += std::to_string(Minor + 3);
    } else if (Major >= 11) {
      ILVersion += std::to_string(Minor);
    } else {
      return UR_RESULT_ERROR_INVALID_VALUE;
    }

    return ReturnValue(ILVersion.data(), ILVersion.size());
  }
  case UR_DEVICE_INFO_MAX_REGISTERS_PER_WORK_GROUP: {
    // Maximum number of 32-bit registers available to a thread block.
    // Note: This number is shared by all thread blocks simultaneously resident
    // on a multiprocessor.
    int MaxRegisters{-1};
    UR_CHECK_ERROR(cuDeviceGetAttribute(
        &MaxRegisters, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK,
        hDevice->get()));

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
        cuDeviceGetPCIBusId(AddressBuffer, AddressBufferSize, hDevice->get()));
    // CUDA API (8.x - 12.1) guarantees 12 bytes + \0 are written
    detail::ur::assertion(strnlen(AddressBuffer, AddressBufferSize) == 12);
    return ReturnValue(AddressBuffer,
                       strnlen(AddressBuffer, AddressBufferSize - 1) + 1);
  }
  case UR_DEVICE_INFO_KERNEL_SET_SPECIALIZATION_CONSTANTS:
    return ReturnValue(false);
    // TODO: Investigate if this information is available on CUDA.
  case UR_DEVICE_INFO_HOST_PIPE_READ_WRITE_SUPPORTED:
    return ReturnValue(false);
  case UR_DEVICE_INFO_VIRTUAL_MEMORY_SUPPORT:
    return ReturnValue(true);
  case UR_DEVICE_INFO_ESIMD_SUPPORT:
    return ReturnValue(false);
  case UR_DEVICE_INFO_MAX_READ_WRITE_IMAGE_ARGS:
  case UR_DEVICE_INFO_GPU_EU_COUNT:
  case UR_DEVICE_INFO_GPU_EU_SIMD_WIDTH:
  case UR_DEVICE_INFO_GPU_EU_SLICES:
  case UR_DEVICE_INFO_GPU_SUBSLICES_PER_SLICE:
  case UR_DEVICE_INFO_GPU_EU_COUNT_PER_SUBSLICE:
  case UR_DEVICE_INFO_GPU_HW_THREADS_PER_EU:
    return UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION;

  default:
    break;
  }
  return UR_RESULT_ERROR_INVALID_ENUMERATION;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

/// \return PI_SUCCESS if the function is executed successfully
/// CUDA devices are always root devices so retain always returns success.
UR_APIEXPORT ur_result_t UR_APICALL urDeviceRetain(ur_device_handle_t hDevice) {
  std::ignore = hDevice;
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urDevicePartition(ur_device_handle_t, const ur_device_partition_properties_t *,
                  uint32_t, ur_device_handle_t *, uint32_t *) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

/// \return UR_RESULT_SUCCESS always since CUDA devices are always root
/// devices.
UR_APIEXPORT ur_result_t UR_APICALL
urDeviceRelease(ur_device_handle_t hDevice) {
  std::ignore = hDevice;
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urDeviceGet(ur_platform_handle_t hPlatform,
                                                ur_device_type_t DeviceType,
                                                uint32_t NumEntries,
                                                ur_device_handle_t *phDevices,
                                                uint32_t *pNumDevices) {
  ur_result_t Result = UR_RESULT_SUCCESS;
  const bool AskingForAll = DeviceType == UR_DEVICE_TYPE_ALL;
  const bool AskingForDefault = DeviceType == UR_DEVICE_TYPE_DEFAULT;
  const bool AskingForGPU = DeviceType == UR_DEVICE_TYPE_GPU;
  const bool ReturnDevices = AskingForDefault || AskingForAll || AskingForGPU;

  size_t NumDevices = ReturnDevices ? hPlatform->Devices.size() : 0;

  try {
    if (pNumDevices) {
      *pNumDevices = NumDevices;
    }

    if (ReturnDevices && phDevices) {
      for (size_t i = 0; i < std::min(size_t(NumEntries), NumDevices); ++i) {
        phDevices[i] = hPlatform->Devices[i].get();
      }
    }

    return Result;
  } catch (ur_result_t Err) {
    return Err;
  } catch (...) {
    return UR_RESULT_ERROR_OUT_OF_RESOURCES;
  }
}

/// Gets the native CUDA handle of a UR device object
///
/// \param[in] device The UR device to get the native CUDA object of.
/// \param[out] nativeHandle Set to the native handle of the UR device object.
///
/// \return PI_SUCCESS

UR_APIEXPORT ur_result_t UR_APICALL urDeviceGetNativeHandle(
    ur_device_handle_t hDevice, ur_native_handle_t *phNativeHandle) {
  *phNativeHandle = reinterpret_cast<ur_native_handle_t>(
      static_cast<std::uintptr_t>(hDevice->get()));
  return UR_RESULT_SUCCESS;
}

/// Created a UR device object from a CUDA device handle.
/// NOTE: The created UR object does not take ownership of the native handle.
///
/// \param[in] nativeHandle The native handle to create UR device object from.
/// \param[in] platform is the UR platform of the device.
/// \param[out] device Set to the UR device object created from native handle.
///
/// \return TBD

UR_APIEXPORT ur_result_t UR_APICALL urDeviceCreateWithNativeHandle(
    ur_native_handle_t hNativeDevice, ur_platform_handle_t hPlatform,
    const ur_device_native_properties_t *pProperties,
    ur_device_handle_t *phDevice) {
  std::ignore = pProperties;

  // We can't cast between ur_native_handle_t and CUdevice, so memcpy the bits
  // instead
  CUdevice CuDevice = 0;
  memcpy(&CuDevice, &hNativeDevice, sizeof(CUdevice));

  auto IsDevice = [=](std::unique_ptr<ur_device_handle_t_> &Dev) {
    return Dev->get() == CuDevice;
  };

  // If a platform is provided just check if the device is in it
  if (hPlatform) {
    auto SearchRes = std::find_if(begin(hPlatform->Devices),
                                  end(hPlatform->Devices), IsDevice);
    if (SearchRes != end(hPlatform->Devices)) {
      *phDevice = SearchRes->get();
      return UR_RESULT_SUCCESS;
    }
  }

  // Get list of platforms
  uint32_t NumPlatforms = 0;
  ur_adapter_handle_t AdapterHandle = &adapter;
  ur_result_t Result =
      urPlatformGet(&AdapterHandle, 1, 0, nullptr, &NumPlatforms);
  if (Result != UR_RESULT_SUCCESS)
    return Result;

  std::vector<ur_platform_handle_t> Platforms(NumPlatforms);

  Result =
      urPlatformGet(&AdapterHandle, 1, NumPlatforms, Platforms.data(), nullptr);
  if (Result != UR_RESULT_SUCCESS)
    return Result;

  // Iterate through platforms to find device that matches nativeHandle
  for (const auto Platform : Platforms) {
    auto SearchRes = std::find_if(std::begin(Platform->Devices),
                                  std::end(Platform->Devices), IsDevice);
    if (SearchRes != end(Platform->Devices)) {
      *phDevice = static_cast<ur_device_handle_t>((*SearchRes).get());
      return UR_RESULT_SUCCESS;
    }
  }

  // If the provided nativeHandle cannot be matched to an
  // existing device return error
  return UR_RESULT_ERROR_INVALID_OPERATION;
}

ur_result_t UR_APICALL urDeviceGetGlobalTimestamps(ur_device_handle_t hDevice,
                                                   uint64_t *pDeviceTimestamp,
                                                   uint64_t *pHostTimestamp) {
  CUevent Event;
  ScopedContext Active(hDevice->getContext());

  if (pDeviceTimestamp) {
    UR_CHECK_ERROR(cuEventCreate(&Event, CU_EVENT_DEFAULT));
    UR_CHECK_ERROR(cuEventRecord(Event, 0));
  }
  if (pHostTimestamp) {

    using namespace std::chrono;
    *pHostTimestamp =
        duration_cast<nanoseconds>(steady_clock::now().time_since_epoch())
            .count();
  }

  if (pDeviceTimestamp) {
    UR_CHECK_ERROR(cuEventSynchronize(Event));
    *pDeviceTimestamp = hDevice->getElapsedTime(Event);
  }

  return UR_RESULT_SUCCESS;
}

/// \return If available, the first binary that is PTX
///
UR_APIEXPORT ur_result_t UR_APICALL urDeviceSelectBinary(
    ur_device_handle_t hDevice, const ur_device_binary_t *pBinaries,
    uint32_t NumBinaries, uint32_t *pSelectedBinary) {
  std::ignore = hDevice;

  // Look for an image for the NVPTX64 target, and return the first one that is
  // found
  for (uint32_t i = 0; i < NumBinaries; i++) {
    if (strcmp(pBinaries[i].pDeviceTargetSpec,
               UR_DEVICE_BINARY_TARGET_NVPTX64) == 0) {
      *pSelectedBinary = i;
      return UR_RESULT_SUCCESS;
    }
  }

  // No image can be loaded for the given device
  return UR_RESULT_ERROR_INVALID_BINARY;
}
