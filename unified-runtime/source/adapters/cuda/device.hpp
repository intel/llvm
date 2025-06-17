//===--------- device.hpp - CUDA Adapter ----------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include <ur/ur.hpp>

#include <umf/memory_pool.h>
#include <umf/memory_provider.h>

#include "common.hpp"

struct ur_device_handle_t_ : ur::cuda::handle_base {
private:
  using native_type = CUdevice;

  native_type CuDevice;
  CUcontext CuContext;
  CUevent EvBase; // CUDA event used as base counter
  std::atomic_uint32_t RefCount;
  ur_platform_handle_t Platform;
  uint32_t DeviceIndex;

  static constexpr uint32_t MaxWorkItemDimensions = 3u;
  size_t MaxWorkItemSizes[MaxWorkItemDimensions];
  size_t MaxWorkGroupSize{0};
  size_t MaxAllocSize{0};
  int MaxRegsPerBlock{0};
  int MaxCapacityLocalMem{0};
  int MaxChosenLocalMem{0};
  uint32_t NumComputeUnits{0};
  std::once_flag NVMLInitFlag;
  std::optional<nvmlDevice_t> NVMLDevice;

public:
  ur_device_handle_t_(native_type cuDevice, CUcontext cuContext, CUevent evBase,
                      ur_platform_handle_t platform, uint32_t DevIndex)
      : handle_base(), CuDevice(cuDevice), CuContext(cuContext), EvBase(evBase),
        RefCount{1}, Platform(platform), DeviceIndex{DevIndex} {
    UR_CHECK_ERROR(cuDeviceGetAttribute(
        &MaxRegsPerBlock, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK,
        cuDevice));
    UR_CHECK_ERROR(cuDeviceGetAttribute(
        &MaxCapacityLocalMem,
        CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, cuDevice));

    UR_CHECK_ERROR(urDeviceGetInfo(this, UR_DEVICE_INFO_MAX_WORK_ITEM_SIZES,
                                   sizeof(MaxWorkItemSizes), MaxWorkItemSizes,
                                   nullptr));

    UR_CHECK_ERROR(urDeviceGetInfo(this, UR_DEVICE_INFO_MAX_WORK_GROUP_SIZE,
                                   sizeof(MaxWorkGroupSize), &MaxWorkGroupSize,
                                   nullptr));

    UR_CHECK_ERROR(cuDeviceGetAttribute(
        reinterpret_cast<int *>(&NumComputeUnits),
        CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, cuDevice));

    // Set local mem max size if env var is present
    static const char *LocalMemSizePtrUR =
        std::getenv("UR_CUDA_MAX_LOCAL_MEM_SIZE");
    static const char *LocalMemSizePtrPI =
        std::getenv("SYCL_PI_CUDA_MAX_LOCAL_MEM_SIZE");
    static const char *LocalMemSizePtr =
        LocalMemSizePtrUR ? LocalMemSizePtrUR : LocalMemSizePtrPI;

    if (LocalMemSizePtr) {
      MaxChosenLocalMem = std::atoi(LocalMemSizePtr);
      if (MaxChosenLocalMem <= 0) {
        setErrorMessage(LocalMemSizePtrUR ? "Invalid value specified for "
                                            "UR_CUDA_MAX_LOCAL_MEM_SIZE"
                                          : "Invalid value specified for "
                                            "SYCL_PI_CUDA_MAX_LOCAL_MEM_SIZE",
                        UR_RESULT_ERROR_INVALID_VALUE);
        throw UR_RESULT_ERROR_ADAPTER_SPECIFIC;
      }

      // Cap chosen local mem size to device capacity, kernel enqueue will fail
      // if it actually needs more.
      MaxChosenLocalMem = std::min(MaxChosenLocalMem, MaxCapacityLocalMem);
    }

    // Max size of memory object allocation in bytes.
    // The minimum value is max (1/4th of info::device::global_mem_size,
    // 128*1024*1024) if this SYCL device is not device_type::custom.
    // CUDA doesn't really have this concept, and could allow almost 100% of
    // global memory in one allocation, but is dependent on device usage.
    UR_CHECK_ERROR(cuDeviceTotalMem(&MaxAllocSize, cuDevice));

    MemoryProviderDevice = nullptr;
    MemoryProviderShared = nullptr;
    MemoryPoolDevice = nullptr;
    MemoryPoolShared = nullptr;
  }

  ~ur_device_handle_t_() {
    if (MemoryPoolDevice) {
      umfPoolDestroy(MemoryPoolDevice);
    }
    if (MemoryPoolShared) {
      umfPoolDestroy(MemoryPoolShared);
    }
    if (MemoryProviderDevice) {
      umfMemoryProviderDestroy(MemoryProviderDevice);
    }
    if (MemoryProviderShared) {
      umfMemoryProviderDestroy(MemoryProviderShared);
    }
    if (NVMLDevice.has_value()) {
      UR_CHECK_ERROR(nvmlShutdown());
    }
    cuDevicePrimaryCtxRelease(CuDevice);
  }

  native_type get() const noexcept { return CuDevice; };

  nvmlDevice_t getNVML() {
    // Initialization happens lazily once per device object. Call to nvmlInit by
    // different objects will just increase the reference count. Each object's
    // destructor calls shutdown method, so once there will be no NVML users
    // left, resources will be released.
    std::call_once(NVMLInitFlag, [this]() {
      UR_CHECK_ERROR(nvmlInit());
      nvmlDevice_t Handle;
      UR_CHECK_ERROR(nvmlDeviceGetHandleByIndex(DeviceIndex, &Handle));
      NVMLDevice = Handle;
    });
    return NVMLDevice.value();
  };

  CUcontext getNativeContext() const noexcept { return CuContext; };

  uint32_t getReferenceCount() const noexcept { return RefCount; }

  ur_platform_handle_t getPlatform() const noexcept { return Platform; };

  // Returns the index of the device relative to the other devices in the same
  // platform
  uint32_t getIndex() const noexcept { return DeviceIndex; }

  uint64_t getElapsedTime(CUevent) const;

  size_t getMaxWorkItemSizes(int index) const noexcept {
    return MaxWorkItemSizes[index];
  }

  const size_t *getMaxWorkItemSizes() const noexcept {
    return MaxWorkItemSizes;
  }

  size_t getMaxWorkGroupSize() const noexcept { return MaxWorkGroupSize; };

  size_t getMaxRegsPerBlock() const noexcept { return MaxRegsPerBlock; };

  size_t getMaxAllocSize() const noexcept { return MaxAllocSize; };

  int getMaxCapacityLocalMem() const noexcept { return MaxCapacityLocalMem; };

  int getMaxChosenLocalMem() const noexcept { return MaxChosenLocalMem; };

  uint32_t getNumComputeUnits() const noexcept { return NumComputeUnits; };

  // bookkeeping for mipmappedArray leaks in Mapping external Memory
  std::map<CUarray, CUmipmappedArray> ChildCuarrayFromMipmapMap;

  // UMF CUDA memory provider and pool for the device memory
  // (UMF_MEMORY_TYPE_DEVICE)
  umf_memory_provider_handle_t MemoryProviderDevice;
  umf_memory_pool_handle_t MemoryPoolDevice;

  // UMF CUDA memory provider and pool for the shared memory
  // (UMF_MEMORY_TYPE_SHARED)
  umf_memory_provider_handle_t MemoryProviderShared;
  umf_memory_pool_handle_t MemoryPoolShared;
};

int getAttribute(ur_device_handle_t Device, CUdevice_attribute Attribute);
