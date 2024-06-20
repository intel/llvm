//===--------- memory.hpp - CUDA Adapter ----------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include <cassert>
#include <cuda.h>
#include <memory>
#include <ur_api.h>
#include <variant>

#include "common.hpp"
#include "context.hpp"
#include "device.hpp"
#include "event.hpp"

ur_result_t allocateMemObjOnDeviceIfNeeded(ur_mem_handle_t,
                                           const ur_device_handle_t);
ur_result_t enqueueMigrateMemoryToDeviceIfNeeded(ur_mem_handle_t,
                                                 const ur_device_handle_t,
                                                 CUstream);

// Handler for plain, pointer-based CUDA allocations
struct BufferMem {

  struct BufferMap {
    /// Size of the active mapped region.
    size_t MapSize;
    /// Offset of the active mapped region.
    size_t MapOffset;
    /// Original flags for the mapped region
    ur_map_flags_t MapFlags;
    /// Allocated host memory used exclusively for this map.
    std::shared_ptr<unsigned char[]> MapMem;

    BufferMap(size_t MapSize, size_t MapOffset, ur_map_flags_t MapFlags)
        : MapSize(MapSize), MapOffset(MapOffset), MapFlags(MapFlags),
          MapMem(nullptr) {}

    BufferMap(size_t MapSize, size_t MapOffset, ur_map_flags_t MapFlags,
              std::unique_ptr<unsigned char[]> &&MapMem)
        : MapSize(MapSize), MapOffset(MapOffset), MapFlags(MapFlags),
          MapMem(std::move(MapMem)) {}

    size_t getMapSize() const noexcept { return MapSize; }

    size_t getMapOffset() const noexcept { return MapOffset; }

    ur_map_flags_t getMapFlags() const noexcept { return MapFlags; }
  };

  /** AllocMode
   * classic: Just a normal buffer allocated on the device via cuda malloc
   * use_host_ptr: Use an address on the host for the device
   * copy_in: The data for the device comes from the host but the host pointer
   * is not available later for re-use alloc_host_ptr: Uses pinned-memory
   * allocation
   */
  enum class AllocMode {
    Classic,
    UseHostPtr,
    CopyIn,
    AllocHostPtr,
  };

  using native_type = CUdeviceptr;

private:
  /// CUDA handler for the pointer
  std::vector<native_type> Ptrs;

public:
  /// If this allocation is a sub-buffer (i.e., a view on an existing
  /// allocation), this is the pointer to the parent handler structure
  ur_mem_handle_t Parent = nullptr;
  /// Outer UR mem holding this BufferMem in variant
  ur_mem_handle_t OuterMemStruct;
  /// Pointer associated with this device on the host
  void *HostPtr;
  /// Size of the allocation in bytes
  size_t Size;
  /// A map that contains all the active mappings for this buffer.
  std::unordered_map<void *, BufferMap> PtrToBufferMap;

  AllocMode MemAllocMode;

  BufferMem(ur_context_handle_t Context, ur_mem_handle_t OuterMemStruct,
            AllocMode Mode, void *HostPtr, size_t Size)
      : Ptrs(Context->getDevices().size(), native_type{0}),
        OuterMemStruct{OuterMemStruct}, HostPtr{HostPtr}, Size{Size},
        MemAllocMode{Mode} {};

  BufferMem(const BufferMem &Buffer) = default;

  native_type getPtrWithOffset(const ur_device_handle_t Device, size_t Offset);

  native_type getPtr(const ur_device_handle_t Device) {
    return getPtrWithOffset(Device, 0);
  }

  void *getVoid(const ur_device_handle_t Device) {
    return reinterpret_cast<void *>(getPtrWithOffset(Device, 0));
  }

  bool isSubBuffer() const noexcept { return Parent != nullptr; }

  size_t getSize() const noexcept { return Size; }

  BufferMap *getMapDetails(void *Map) {
    auto details = PtrToBufferMap.find(Map);
    if (details != PtrToBufferMap.end()) {
      return &details->second;
    }
    return nullptr;
  }

  /// Returns a pointer to data visible on the host that contains
  /// the data on the device associated with this allocation.
  /// The offset is used to index into the CUDA allocation.
  void *mapToPtr(size_t MapSize, size_t MapOffset,
                 ur_map_flags_t MapFlags) noexcept {

    void *MapPtr = nullptr;
    if (HostPtr == nullptr) {
      /// If HostPtr is invalid, we need to create a Mapping that owns its own
      /// memory on the host.
      auto MapMem = std::make_unique<unsigned char[]>(MapSize);
      MapPtr = MapMem.get();
      PtrToBufferMap.insert(
          {MapPtr, BufferMap(MapSize, MapOffset, MapFlags, std::move(MapMem))});
    } else {
      /// However, if HostPtr already has valid memory (e.g. pinned allocation),
      /// we can just use that memory for the mapping.
      MapPtr = static_cast<char *>(HostPtr) + MapOffset;
      PtrToBufferMap.insert({MapPtr, BufferMap(MapSize, MapOffset, MapFlags)});
    }
    return MapPtr;
  }

  /// Detach the allocation from the host memory.
  void unmap(void *MapPtr) noexcept {
    assert(MapPtr != nullptr);
    PtrToBufferMap.erase(MapPtr);
  }

  ur_result_t clear() {
    if (Parent != nullptr) {
      return UR_RESULT_SUCCESS;
    }

    switch (MemAllocMode) {
    case AllocMode::CopyIn:
    case AllocMode::Classic:
      for (auto &DevPtr : Ptrs) {
        if (DevPtr != native_type{0}) {
          UR_CHECK_ERROR(cuMemFree(DevPtr));
        }
      }
      break;
    case AllocMode::UseHostPtr:
      UR_CHECK_ERROR(cuMemHostUnregister(HostPtr));
      break;
    case AllocMode::AllocHostPtr:
      UR_CHECK_ERROR(cuMemFreeHost(HostPtr));
    }
    return UR_RESULT_SUCCESS;
  }

  friend struct ur_mem_handle_t_;
  friend ur_result_t allocateMemObjOnDeviceIfNeeded(ur_mem_handle_t,
                                                    const ur_device_handle_t);
};

// Handler data for surface object (i.e. Images)
struct SurfaceMem {
private:
  std::vector<CUarray> Arrays;
  std::vector<CUsurfObject> SurfObjs;

public:
  ur_mem_handle_t OuterMemStruct;

  ur_image_format_t ImageFormat;
  ur_image_desc_t ImageDesc;
  CUDA_ARRAY3D_DESCRIPTOR ArrayDesc;
  size_t PixelTypeSizeBytes;
  void *HostPtr;
  ur_result_t error = UR_RESULT_SUCCESS;

  SurfaceMem(ur_context_handle_t Context, ur_mem_handle_t OuterMemStruct,
             ur_image_format_t ImageFormat, ur_image_desc_t ImageDesc,
             void *HostPtr)
      : Arrays(Context->Devices.size(), CUarray{0}),
        SurfObjs(Context->Devices.size(), CUsurfObject{0}),
        OuterMemStruct{OuterMemStruct},
        ImageFormat{ImageFormat}, ImageDesc{ImageDesc}, HostPtr{HostPtr} {
    // We have to use hipArray3DCreate, which has some caveats. The height and
    // depth parameters must be set to 0 produce 1D or 2D arrays. image_desc
    // gives a minimum value of 1, so we need to convert the answer.
    ArrayDesc.NumChannels = 4; // Only support 4 channel image
    ArrayDesc.Flags = 0;       // No flags required
    ArrayDesc.Width = ImageDesc.width;
    if (ImageDesc.type == UR_MEM_TYPE_IMAGE1D) {
      ArrayDesc.Height = 0;
      ArrayDesc.Depth = 0;
    } else if (ImageDesc.type == UR_MEM_TYPE_IMAGE2D) {
      ArrayDesc.Height = ImageDesc.height;
      ArrayDesc.Depth = 0;
    } else if (ImageDesc.type == UR_MEM_TYPE_IMAGE3D) {
      ArrayDesc.Height = ImageDesc.height;
      ArrayDesc.Depth = ImageDesc.depth;
    }

    // We need to get PixelTypeSizeBytes for calculating the total image size
    // later
    switch (ImageFormat.channelType) {

    case UR_IMAGE_CHANNEL_TYPE_UNORM_INT8:
    case UR_IMAGE_CHANNEL_TYPE_UNSIGNED_INT8:
      ArrayDesc.Format = CU_AD_FORMAT_UNSIGNED_INT8;
      PixelTypeSizeBytes = 1;
      break;
    case UR_IMAGE_CHANNEL_TYPE_SNORM_INT8:
    case UR_IMAGE_CHANNEL_TYPE_SIGNED_INT8:
      ArrayDesc.Format = CU_AD_FORMAT_SIGNED_INT8;
      PixelTypeSizeBytes = 1;
      break;
    case UR_IMAGE_CHANNEL_TYPE_UNORM_INT16:
    case UR_IMAGE_CHANNEL_TYPE_UNSIGNED_INT16:
      ArrayDesc.Format = CU_AD_FORMAT_UNSIGNED_INT16;
      PixelTypeSizeBytes = 2;
      break;
    case UR_IMAGE_CHANNEL_TYPE_SNORM_INT16:
    case UR_IMAGE_CHANNEL_TYPE_SIGNED_INT16:
      ArrayDesc.Format = CU_AD_FORMAT_SIGNED_INT16;
      PixelTypeSizeBytes = 2;
      break;
    case UR_IMAGE_CHANNEL_TYPE_HALF_FLOAT:
      ArrayDesc.Format = CU_AD_FORMAT_HALF;
      PixelTypeSizeBytes = 2;
      break;
    case UR_IMAGE_CHANNEL_TYPE_UNSIGNED_INT32:
      ArrayDesc.Format = CU_AD_FORMAT_UNSIGNED_INT32;
      PixelTypeSizeBytes = 4;
      break;
    case UR_IMAGE_CHANNEL_TYPE_SIGNED_INT32:
      ArrayDesc.Format = CU_AD_FORMAT_SIGNED_INT32;
      PixelTypeSizeBytes = 4;
      break;
    case UR_IMAGE_CHANNEL_TYPE_FLOAT:
      ArrayDesc.Format = CU_AD_FORMAT_FLOAT;
      PixelTypeSizeBytes = 4;
      break;
    default:
      break;
    }
  }

  // Will allocate a new array on device if not already allocated
  CUarray getArray(const ur_device_handle_t Device);

  // Will allocate a new surface on device if not already allocated
  CUsurfObject getSurface(const ur_device_handle_t Device);

  ur_mem_type_t getType() { return ImageDesc.type; }

  ur_result_t clear() {
    for (auto Array : Arrays) {
      if (Array) {
        UR_CHECK_ERROR(cuArrayDestroy(Array));
      }
    }
    for (auto Surf : SurfObjs) {
      if (Surf != CUsurfObject{0}) {
        UR_CHECK_ERROR(cuSurfObjectDestroy(Surf));
      }
    }
    return UR_RESULT_SUCCESS;
  }
  friend ur_result_t allocateMemObjOnDeviceIfNeeded(ur_mem_handle_t,
                                                    const ur_device_handle_t);
};

/// UR Mem mapping to CUDA memory allocations, both data and texture/surface.
/// \brief Represents non-SVM allocations on the CUDA backend.
/// Keeps tracks of all mapped regions used for Map/Unmap calls.
/// Only one region can be active at the same time per allocation.
///
/// The ur_mem_handle_t is responsible for memory allocation and migration
/// across devices in the same ur_context_handle_t. If a kernel writes to a
/// ur_mem_handle_t then it will write to LastQueueWritingToMemObj. Then all
/// subsequent operations that want to read from the ur_mem_handle_t must wait
/// on the event referring to the last write.
///
/// Since urMemBufferCreate/urMemImageCreate do not take a queue or device
/// object, only a ur_context_handle_t, at mem obj creation we don't know which
/// device we must make a native image/allocation on. Therefore no allocations
/// are made at urMemBufferCreate/urMemImageCreate. Instead device
/// images/allocations are made lazily. These allocations are made implicitly
/// with a call to getPtr/getArray which will allocate a new allocation/image on
/// device if need be.
///
/// Memory migration between native allocations for devices in the same
/// ur_context_handle_t will occur at:
///
///   1. urEnqueueKernelLaunch
///   2. urEnqueueMem(Buffer|Image)Read(Rect)
///
/// Migrations will occur in both cases if the most recent version of data
/// is on a different device, marked by
/// LastQueueWritingToMemObj->getDevice()
///
struct ur_mem_handle_t_ {
  // Context where the memory object is accessible
  ur_context_handle_t Context;

  /// Reference counting of the handler
  std::atomic_uint32_t RefCount;

  // Original mem flags passed
  ur_mem_flags_t MemFlags;

  // If we make a ur_mem_handle_t_ from a native allocation, it can be useful to
  // associate it with the device that holds the native allocation.
  ur_device_handle_t DeviceWithNativeAllocation{nullptr};

  // Has the memory been migrated to a device since the last write?
  std::vector<bool> HaveMigratedToDeviceSinceLastWrite;

  // Queue with most up to date data of ur_mem_handle_t_
  ur_queue_handle_t LastQueueWritingToMemObj{nullptr};

  // Enumerates all possible types of accesses.
  enum access_mode_t { unknown, read_write, read_only, write_only };

  ur_mutex MemoryAllocationMutex; // A mutex for allocations

  /// A UR Memory object represents either plain memory allocations ("Buffers"
  /// in OpenCL) or typed allocations ("Images" in OpenCL).
  /// In CUDA their API handlers are different. Whereas "Buffers" are allocated
  /// as pointer-like structs, "Images" are stored in Textures or Surfaces.
  /// This variant allows implementation to use either from the same handler.
  std::variant<BufferMem, SurfaceMem> Mem;

  /// Constructs the UR mem handler for a non-typed allocation ("buffer")
  ur_mem_handle_t_(ur_context_handle_t Ctxt, ur_mem_flags_t MemFlags,
                   BufferMem::AllocMode Mode, void *HostPtr, size_t Size)
      : Context{Ctxt}, RefCount{1}, MemFlags{MemFlags},
        HaveMigratedToDeviceSinceLastWrite(Context->Devices.size(), false),
        Mem{std::in_place_type<BufferMem>, Ctxt, this, Mode, HostPtr, Size} {
    urContextRetain(Context);
  };

  // Subbuffer constructor
  ur_mem_handle_t_(ur_mem_handle_t Parent, size_t SubBufferOffset)
      : Context{Parent->Context}, RefCount{1}, MemFlags{Parent->MemFlags},
        HaveMigratedToDeviceSinceLastWrite(Parent->Context->Devices.size(),
                                           false),
        Mem{BufferMem{std::get<BufferMem>(Parent->Mem)}} {
    auto &SubBuffer = std::get<BufferMem>(Mem);
    SubBuffer.Parent = Parent;
    SubBuffer.OuterMemStruct = this;
    if (SubBuffer.HostPtr) {
      SubBuffer.HostPtr =
          static_cast<char *>(SubBuffer.HostPtr) + SubBufferOffset;
    }
    for (auto &DevPtr : SubBuffer.Ptrs) {
      if (DevPtr) {
        DevPtr += SubBufferOffset;
      }
    }
    urMemRetain(Parent);
  };

  /// Constructs the UR mem handler for an Image object
  ur_mem_handle_t_(ur_context_handle_t Ctxt, ur_mem_flags_t MemFlags,
                   ur_image_format_t ImageFormat, ur_image_desc_t ImageDesc,
                   void *HostPtr)
      : Context{Ctxt}, RefCount{1}, MemFlags{MemFlags},
        HaveMigratedToDeviceSinceLastWrite(Context->Devices.size(), false),
        Mem{std::in_place_type<SurfaceMem>,
            Ctxt,
            this,
            ImageFormat,
            ImageDesc,
            HostPtr} {
    urContextRetain(Context);
  }

  ~ur_mem_handle_t_() {
    clear();
    if (isBuffer() && isSubBuffer()) {
      urMemRelease(std::get<BufferMem>(Mem).Parent);
      return;
    }
    urContextRelease(Context);
  }

  bool isBuffer() const noexcept {
    return std::holds_alternative<BufferMem>(Mem);
  }

  bool isSubBuffer() const noexcept {
    return (isBuffer() && (std::get<BufferMem>(Mem).Parent != nullptr));
  }

  bool isImage() const noexcept {
    return std::holds_alternative<SurfaceMem>(Mem);
  }

  ur_result_t clear() {
    if (isBuffer()) {
      return std::get<BufferMem>(Mem).clear();
    }
    return std::get<SurfaceMem>(Mem).clear();
  }

  ur_context_handle_t getContext() const noexcept { return Context; }

  uint32_t incrementReferenceCount() noexcept { return ++RefCount; }

  uint32_t decrementReferenceCount() noexcept { return --RefCount; }

  uint32_t getReferenceCount() const noexcept { return RefCount; }

  void setLastQueueWritingToMemObj(ur_queue_handle_t WritingQueue) {
    urQueueRetain(WritingQueue);
    if (LastQueueWritingToMemObj != nullptr) {
      urQueueRelease(LastQueueWritingToMemObj);
    }
    LastQueueWritingToMemObj = WritingQueue;
    for (const auto &Device : Context->getDevices()) {
      // This event is never an interop event so will always have an associated
      // queue
      HaveMigratedToDeviceSinceLastWrite[Context->getDeviceIndex(Device)] =
          Device == WritingQueue->getDevice();
    }
  }
};

ur_result_t migrateMemoryToDeviceIfNeeded(ur_mem_handle_t,
                                          const ur_device_handle_t);
