//===----------- memory.hpp - LLVM Offload Adapter  -----------------------===//
//
// Copyright (C) 2025 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "ur_api.h"

#include "common.hpp"
#include "context.hpp"
#include "queue.hpp"
#include "ur2offload.hpp"

struct BufferMem {

  enum class BufferStrategy {
    // When pinned host memory isn't used, make a separate allocation on each
    // device, and migrate data from the last used device to the current device
    // when a buffer is used in an enqueue command.
    DiscreteDeviceAllocs,
    // When pinned host memory isn't used, make a single shared USM allocation
    // that is visible on all devices. When a buffer is used in an enqueue
    // command, prefetch the data to the active device.
    // TODO: Implement this.
    SingleSharedAlloc
  };

  // This is the most conventional implementation of buffers.
  static constexpr auto Strategy = BufferStrategy::DiscreteDeviceAllocs;

  enum class AllocMode {
    Default,
    UseHostPtr,
    CopyIn,
    AllocHostPtr,
  };

  struct BufferMap {
    size_t MapSize;
    size_t MapOffset;
    ur_map_flags_t MapFlags;
    // Allocated host memory used exclusively for this map.
    std::unique_ptr<unsigned char[]> MapMem;

    BufferMap(size_t MapSize, size_t MapOffset, ur_map_flags_t MapFlags)
        : MapSize(MapSize), MapOffset(MapOffset), MapFlags(MapFlags),
          MapMem(nullptr) {}

    BufferMap(size_t MapSize, size_t MapOffset, ur_map_flags_t MapFlags,
              std::unique_ptr<unsigned char[]> &&MapMem)
        : MapSize(MapSize), MapOffset(MapOffset), MapFlags(MapFlags),
          MapMem(std::move(MapMem)) {}
  };

  ur_mem_handle_t Parent;
  // Outer UR mem holding this BufferMem in variant
  ur_mem_handle_t OuterMemStruct;

  // Underlying device pointers
  std::vector<void *> Ptrs;

  // Pointer associated with this device on the host
  void *HostPtr;
  size_t Size;

  AllocMode MemAllocMode;
  std::unordered_map<void *, BufferMap> PtrToBufferMap;

  BufferMem(ur_mem_handle_t Parent, ur_mem_handle_t OuterMemStruct,
            ur_context_handle_t Context, BufferMem::AllocMode Mode,
            void *HostPtr, size_t Size)
      : Parent{Parent}, OuterMemStruct{OuterMemStruct},
        Ptrs(Context->Devices.size(), nullptr), HostPtr{HostPtr}, Size{Size},
        MemAllocMode{Mode} {};

  void *getPtr(ur_device_handle_t Device) const noexcept;

  size_t getSize() const noexcept { return Size; }

  BufferMap *getMapDetails(void *Map) {
    auto Details = PtrToBufferMap.find(Map);
    if (Details != PtrToBufferMap.end()) {
      return &Details->second;
    }
    return nullptr;
  }

  void *mapToPtr(size_t MapSize, size_t MapOffset,
                 ur_map_flags_t MapFlags) noexcept {

    void *MapPtr = nullptr;
    // If the buffer already has a host pointer we can just use it, otherwise
    // create a new host allocation
    if (HostPtr == nullptr) {
      auto MapMem = std::make_unique<unsigned char[]>(MapSize);
      MapPtr = MapMem.get();
      PtrToBufferMap.insert(
          {MapPtr, BufferMap(MapSize, MapOffset, MapFlags, std::move(MapMem))});
    } else {
      MapPtr = static_cast<char *>(HostPtr) + MapOffset;
      PtrToBufferMap.insert({MapPtr, BufferMap(MapSize, MapOffset, MapFlags)});
    }
    return MapPtr;
  }

  void unmap(void *MapPtr) noexcept {
    assert(MapPtr != nullptr);
    PtrToBufferMap.erase(MapPtr);
  }
};

struct ur_mem_handle_t_ : RefCounted {
  ur_context_handle_t Context;

  enum class Type { Buffer } MemType;
  ur_mem_flags_t MemFlags;

  ur_mutex MemoryAllocationMutex;

  // For now we only support BufferMem. Eventually we'll support images, so use
  // a variant to store the underlying object.
  std::variant<BufferMem> Mem;

  // For every device in the context, is it known to have the latest copy of the
  // data. Operations modifying the buffer should invalidate it on all devices
  // but the the one the operation occurred on.
  std::vector<bool> DeviceIsUpToDate;

  ur_queue_handle_t LastQueueWritingToMemObj{nullptr};

  ur_mem_handle_t_(ur_context_handle_t Context, ur_mem_handle_t Parent,
                   ur_mem_flags_t MemFlags, BufferMem::AllocMode Mode,
                   void *HostPtr, size_t Size)
      : Context{Context}, MemType{Type::Buffer}, MemFlags{MemFlags},
        Mem{BufferMem{Parent, this, Context, Mode, HostPtr, Size}},
        DeviceIsUpToDate(Context->Devices.size(), false) {
    urContextRetain(Context);
  };

  ~ur_mem_handle_t_() { urContextRelease(Context); }

  ur_context_handle_t getContext() const noexcept { return Context; }

  void *getDevicePointer(const ur_device_handle_t Device) {
    auto DeviceIdx = Context->getDeviceIndex(Device);
    auto &Buffer = std::get<BufferMem>(Mem);

    if (auto *Ptr = Buffer.Ptrs[DeviceIdx]) {
      return Ptr;
    } else {
      olMemAlloc(Device->OffloadDevice, OL_ALLOC_TYPE_DEVICE, Buffer.Size,
                 &Buffer.Ptrs[DeviceIdx]);
      return Buffer.Ptrs[DeviceIdx];
    }
  }

  ur_result_t prepareDeviceAllocation(ur_device_handle_t Device) {
    // Lock to prevent duplicate allocations in race conditions
    ur_lock LockGuard(MemoryAllocationMutex);

    auto DeviceIdx = Context->getDeviceIndex(Device);
    auto &Buffer = std::get<BufferMem>(Mem);
    auto DevPtr = Buffer.Ptrs[DeviceIdx];

    // Allocation has already been made
    if (DevPtr != nullptr) {
      return UR_RESULT_SUCCESS;
    }

    if (Buffer.MemAllocMode == BufferMem::AllocMode::AllocHostPtr) {
      // Host allocation has already been made by this point.
      // TODO: We (probably) need something like cuMemHostGetDevicePointer
      // for this to work everywhere. For now assume the managed host pointer is
      // always device-accessible.
      DevPtr = Buffer.HostPtr;

    } else if (Buffer.MemAllocMode == BufferMem::AllocMode::UseHostPtr) {
      // TODO: This code path is never used (same as the cuda adapter)
      DevPtr = Buffer.HostPtr;
    } else {
      auto Res = olMemAlloc(Device->OffloadDevice, OL_ALLOC_TYPE_DEVICE,
                            Buffer.Size, &DevPtr);
      if (Res) {
        return offloadResultToUR(Res);
      }
    }

    Buffer.Ptrs[DeviceIdx] = DevPtr;
    return UR_RESULT_SUCCESS;
  }

  void setLastQueueWritingToMemObj(ur_queue_handle_t WritingQueue) {
    urQueueRetain(WritingQueue);
    if (LastQueueWritingToMemObj != nullptr) {
      urQueueRelease(LastQueueWritingToMemObj);
    }
    LastQueueWritingToMemObj = WritingQueue;
    for (const auto &Device : Context->Devices) {
      DeviceIsUpToDate[Context->getDeviceIndex(Device)] =
          Device == WritingQueue->Device;
    }
  }

  ur_result_t
  enqueueMigrateMemoryToDeviceIfNeeded(const ur_device_handle_t Device,
                                       ol_queue_handle_t Queue);
};
