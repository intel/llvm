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

#include "common.hpp"

struct ur_device_handle_t_ {
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
  bool MaxLocalMemSizeChosen{false};
  uint32_t NumComputeUnits{0};

public:
  ur_device_handle_t_(native_type cuDevice, CUcontext cuContext, CUevent evBase,
                      ur_platform_handle_t platform, uint32_t DevIndex)
      : CuDevice(cuDevice), CuContext(cuContext), EvBase(evBase), RefCount{1},
        Platform(platform), DeviceIndex{DevIndex} {

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
        LocalMemSizePtrUR ? LocalMemSizePtrUR
                          : (LocalMemSizePtrPI ? LocalMemSizePtrPI : nullptr);

    if (LocalMemSizePtr) {
      MaxChosenLocalMem = std::atoi(LocalMemSizePtr);
      MaxLocalMemSizeChosen = true;
    }

    // Max size of memory object allocation in bytes.
    // The minimum value is max (1/4th of info::device::global_mem_size,
    // 128*1024*1024) if this SYCL device is not device_type::custom.
    // CUDA doesn't really have this concept, and could allow almost 100% of
    // global memory in one allocation, but is dependent on device usage.
    UR_CHECK_ERROR(cuDeviceTotalMem(&MaxAllocSize, cuDevice));
  }

  ~ur_device_handle_t_() { cuDevicePrimaryCtxRelease(CuDevice); }

  native_type get() const noexcept { return CuDevice; };

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

  size_t getMaxWorkGroupSize() const noexcept { return MaxWorkGroupSize; };

  size_t getMaxRegsPerBlock() const noexcept { return MaxRegsPerBlock; };

  size_t getMaxAllocSize() const noexcept { return MaxAllocSize; };

  int getMaxCapacityLocalMem() const noexcept { return MaxCapacityLocalMem; };

  int getMaxChosenLocalMem() const noexcept { return MaxChosenLocalMem; };

  bool maxLocalMemSizeChosen() { return MaxLocalMemSizeChosen; };

  uint32_t getNumComputeUnits() const noexcept { return NumComputeUnits; };

  // bookkeeping for mipmappedArray leaks in Mapping external Memory
  std::map<CUarray, CUmipmappedArray> ChildCuarrayFromMipmapMap;
};

int getAttribute(ur_device_handle_t Device, CUdevice_attribute Attribute);
