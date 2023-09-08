//===--------- memory.hpp - HIP Adapter -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include "common.hpp"
#include "context.hpp"
#include "event.hpp"
#include <cassert>

/// UR Mem mapping to HIP memory allocations, both data and texture/surface.
/// \brief Represents non-SVM allocations on the HIP backend.
/// Keeps tracks of all mapped regions used for Map/Unmap calls.
/// Only one region can be active at the same time per allocation.
///
/// The ur_mem_handle_t is responsible for memory allocation and migration
/// across devices in the same ur_context_handle_t. If a kernel writes to a
/// ur_mem_handle_t then it will write to LastEventWritingToMemObj. Then all
/// subsequent operations that want to read from the ur_mem_handle_t must wait
/// on the event referring to the last write.
///
/// Since urMemBufferCreate/urMemImageCreate do not take a queue or device
/// object, only a ur_context_handle_t, at mem obj creation we don't know which
/// device we must make a native image/allocation on. Therefore no allocations
/// are made at urMemBufferCreate/urMemImageCreate. Instead device
/// images/allocations are made lazily. These allocations are made with
/// allocateMemObjOnDeviceIfNeeded which should be called either at:
///
///   1. urEnqueueMem(Buffer|Image)Write(Rect)
///       - in the case of a buffer/image that is initialized with host data.
///   2. urKernelSetArgMemObj
///       - in the case of an uninitialized buffer/image.
///       - in the case where a memObj has already been initialized using
///         urEnqueueMem(Buffer|Image)Write but needs to make an additional
///         allocation on a different device in the context.
///
/// Memory migration between native allocations for devices in the same
/// ur_context_handle_t will occur at:
///
///   1. urEnqueueKernelLaunch
///   2. urEnqueueMem(Buffer|Image)Read(Rect)
///
/// Migrations will occur in both cases if the most recent version of data
/// is on a different device, marked by LastEventWritingToMemObj->getDevice().
///
/// Example trace:
/// ~~~~~~~~~~~~~~
///
/// =====> urContextCreate([device0, device1], ...) // associated with [q0, q1]
///             -> OUT: hContext
///
/// =====> urMemBufferCreate(hContext,...);
///             -> No native allocations made
///             -> OUT: hBuffer
///
/// =====> urEnqueueMemBufferWrite(q0, hBuffer,...);
///             -> Allocation made on q0 ie device0
///             -> New allocation initialized with host data.
///
/// =====> urKernelSetArgMemObj(hKernel0, hBuffer, ...);
///             -> ur_kernel_handle_t associated with a ur_program_handle_t,
///                which is in turn unique to a device. So we can set the kernel
///                arg with the ptr of the device specific allocation.
///             -> hKernel0->getProgram()->getDevice() == device0
///             -> allocateMemObjOnDeviceIfNeeded(device0);
///                   -> Native allocation already made on device0, continue.
///
/// =====> urEnqueueKernelLaunch(q0, hKernel0, ...);
///             -> Suppose that hKernel0 writes to hBuffer.
///             -> Call hBuffer->setLastEventWritingToMemObj with return event
///                from this operation
///             -> Enqueue native kernel launch
///
/// =====> urKernelSetArgMemObj(hKernel1, hBuffer, ...);
///             -> hKernel1->getProgram()->getDevice() == device1
///             -> allocateMemObjOnDeviceIfNeeded(device1);
///                   -> No native allocation on device1
///                   -> Make native allocation on device1
///
/// =====> urEnqueueKernelLaunch(q1, hKernel1, ...);
///             -> Suppose hKernel1 wants to read from hBuffer and not write.
///             -> migrateMemoryToDeviceIfNeeded(device1);
///                   -> hBuffer->LastEventWritingToMemObj is not nullptr
///                   -> Check if memory has been migrated to device1 since the
///                      last write
///                        -> Hasn't been migrated
///                   -> Wait on LastEventWritingToMemObj.
///                   -> Migrate memory from device0's native allocation to
///                      device1's native allocation.
///             -> Enqueue native kernel launch
///
/// =====> urEnqueueKernelLaunch(q0, hKernel0, ...);
///             -> migrateMemoryToDeviceIfNeeded(device0);
///                   -> hBuffer->LastEventWritingToMemObj refers to an event
///                      from q0
///                        -> Migration not necessary
///             -> Enqueue native kernel launch
///
struct ur_mem_handle_t_ {

  using ur_context = ur_context_handle_t_ *;
  using ur_mem = ur_mem_handle_t_ *;

  // Context where the memory object is accessible
  ur_context Context;

  // If we make a ur_mem_handle_t_ from a native allocation, it can be useful to
  // associate it with the device that holds the native allocation.
  ur_device_handle_t DeviceWithNativeAllocation{nullptr};

  std::atomic_uint32_t RefCount{1};

  // Original mem flags passed
  ur_mem_flags_t MemFlags;

  // Has the memory been migrated to a device since the last write?
  std::vector<bool> HaveMigratedToDeviceSinceLastWrite;

  // We should wait on this event prior to migrating memory across allocations
  // in this ur_mem_handle_t_
  ur_event_handle_t LastEventWritingToMemObj{nullptr};

  // Enumerates all possible types of accesses.
  enum access_mode_t { unknown, read_write, read_only, write_only };

  // Methods to get type of the derived object (image or buffer)
  virtual bool isBuffer() const = 0;
  virtual bool isImage() const = 0;

  ur_context_handle_t getContext() const noexcept { return Context; }

  uint32_t incrementReferenceCount() noexcept { return ++RefCount; }

  uint32_t decrementReferenceCount() noexcept { return --RefCount; }

  uint32_t getReferenceCount() const noexcept { return RefCount; }

  virtual ur_result_t allocateMemObjOnDeviceIfNeeded(ur_device_handle_t) = 0;
  virtual ur_result_t migrateMemoryToDeviceIfNeeded(ur_device_handle_t) = 0;

  virtual ur_result_t clear() = 0;

  virtual ~ur_mem_handle_t_() = 0;

  ur_mutex MemoryAllocationMutex; // A mutex for allocations
  ur_mutex MemoryMigrationMutex;  // A mutex for memory transfers

  ur_mem_handle_t_(ur_context_handle_t Context, ur_mem_flags_t MemFlags)
      : Context{Context}, MemFlags{MemFlags},
        HaveMigratedToDeviceSinceLastWrite(Context->NumDevices, false) {
    urContextRetain(Context);
  };

  // In the case that a ur_mem_handle_t is created with a native allocation,
  // it can be useful to keep track of the device that has the original native
  // allocation so that we know not to free the memory on destruction
  ur_mem_handle_t_(ur_context_handle_t Context, ur_device_handle_t Device,
                   ur_mem_flags_t MemFlags)
      : Context{Context}, DeviceWithNativeAllocation{Device},
        MemFlags{MemFlags},
        HaveMigratedToDeviceSinceLastWrite(Context->NumDevices, false) {
    urContextRetain(Context);
    urDeviceRetain(Device);
  };

  void setLastEventWritingToMemObj(ur_event_handle_t NewEvent) {
    assert(NewEvent && "Invalid event!");
    // We only need this functionality for multi device context
    if (Context->NumDevices == 1) {
      return;
    }
    if (LastEventWritingToMemObj != nullptr) {
      urEventRelease(LastEventWritingToMemObj);
    }
    urEventRetain(NewEvent);
    LastEventWritingToMemObj = NewEvent;
    for (auto i = 0u; i < Context->NumDevices; ++i) {
      if (i == NewEvent->getDevice()->getIndex()) {
        HaveMigratedToDeviceSinceLastWrite[i] = true;
      } else {
        HaveMigratedToDeviceSinceLastWrite[i] = false;
      }
    }
  }
};

// Handle for plain, pointer-based HIP allocations.
//
// Since a ur_buffer_ is associated with a ur_context_handle_t_, which may
// contain multiple devices, each ur_buffer_ contains a vector of native
// allocations, one allocation for each device in the ur_context_handle_t_.
// Native allocations are made lazily, before a `ur_buffer_` is needed on a
// particular device.
//
// The ur_buffer_ is also responsible for migrating memory between native
// allocations. This migration happens lazily. The ur_buffer_ relies on knowing
// which event was the last to write to the mem obj `LastEventWritingToMemObj`.
// All subsequent reads must wait on this event.
//
struct ur_buffer_ final : ur_mem_handle_t_ {
  using native_type = hipDeviceptr_t;

  // If this is a subbuffer then this will point to the parent buffer
  ur_buffer_ *Parent{nullptr};

  // CUDA handler for the pointers. We hold a ptr for each device in our
  // context. Each device in the context is identified by its index
  std::vector<native_type> Ptrs;

  /// Pointer associated with this device on the host
  void *HostPtr;
  /// Size of the allocation in bytes
  size_t Size;
  /// Size of the active mapped region.
  size_t MapSize;
  /// Offset of the active mapped region.
  size_t MapOffset;
  /// Pointer to the active mapped region, if any
  void *MapPtr;
  /// Original flags for the mapped region
  ur_map_flags_t MapFlags;

  /** AllocMode
   * classic: Just a normal buffer allocated on the device via cuda malloc
   * use_host_ptr: Use an address on the host for the device
   * copy_in: The data for the device comes from the host but the host
   pointer is not available later for re-use
   * alloc_host_ptr: Uses pinned-memory allocation
  */
  enum class AllocMode {
    Classic,
    UseHostPtr,
    CopyIn,
    AllocHostPtr,
  } MemAllocMode;

  ur_buffer_(ur_context_handle_t Context, ur_buffer_ *Parent,
             ur_mem_flags_t MemFlags, AllocMode Mode, void *HostPtr,
             size_t Size)
      : ur_mem_handle_t_{Context, MemFlags}, Parent{Parent},
        Ptrs(Context->NumDevices, native_type{0}), HostPtr{HostPtr}, Size{Size},
        MapSize{0}, MapOffset{0}, MapPtr{nullptr}, MapFlags{UR_MAP_FLAG_WRITE},
        MemAllocMode{Mode} {
    if (isSubBuffer()) {
      urMemRetain(Parent);
    }
  }
  ur_buffer_(ur_context_handle_t Context, ur_device_handle_t Device,
             ur_buffer_ *Parent, ur_mem_flags_t MemFlags, AllocMode Mode,
             void *HostPtr, size_t Size)
      : ur_mem_handle_t_{Context, Device, MemFlags}, Parent{Parent},
        Ptrs(Context->NumDevices, native_type{0}), HostPtr{HostPtr}, Size{Size},
        MapSize{0}, MapOffset{0}, MapPtr{nullptr}, MapFlags{UR_MAP_FLAG_WRITE},
        MemAllocMode{Mode} {
    if (isSubBuffer()) {
      urMemRetain(Parent);
    }
  }

  ~ur_buffer_() override {
    if (isSubBuffer()) {
      urMemRelease(Parent);
    }
  }

  bool isBuffer() const noexcept override { return true; }
  bool isImage() const noexcept override { return false; }
  bool isSubBuffer() const noexcept { return Parent != nullptr; }

  native_type getWithOffset(size_t Offset, ur_device_handle_t Device) {
    return reinterpret_cast<native_type>(
        reinterpret_cast<uint8_t *>(Ptrs[Device->getIndex()]) + Offset);
  }

  std::vector<native_type> &getPtrs() noexcept { return Ptrs; }

  native_type &getPtr(ur_device_handle_t Device) noexcept {
    return Ptrs[Device->getIndex()];
  }

  size_t getSize() const noexcept { return Size; }

  void *getMapPtr() const noexcept { return MapPtr; }

  size_t getMapSize() const noexcept { return MapSize; }

  size_t getMapOffset() const noexcept { return MapOffset; }

  /// Returns a pointer to data visible on the host that contains
  /// the data on the device associated with this allocation.
  /// The offset is used to index into the CUDA allocation.
  void *mapToPtr(size_t Size, size_t Offset, ur_map_flags_t Flags) noexcept {
    assert(MapPtr == nullptr);
    MapSize = Size;
    MapOffset = Offset;
    MapFlags = Flags;
    if (HostPtr) {
      MapPtr = static_cast<char *>(HostPtr) + Offset;
    } else {
      // TODO: Allocate only what is needed based on the offset
      MapPtr = static_cast<void *>(malloc(this->getSize()));
    }
    return MapPtr;
  }

  /// Detach the allocation from the host memory.
  void unmap(void *) noexcept {
    assert(MapPtr != nullptr);

    if (MapPtr != HostPtr) {
      free(MapPtr);
    }
    MapSize = 0;
    MapPtr = nullptr;
    MapOffset = 0;
  }

  ur_map_flags_t getMapFlags() const noexcept {
    assert(MapPtr != nullptr);
    return MapFlags;
  }

  ur_result_t clear() override {
    if (isSubBuffer()) {
      return UR_RESULT_SUCCESS;
    }
    ur_result_t Result = UR_RESULT_SUCCESS;

    switch (MemAllocMode) {
    case ur_buffer_::AllocMode::CopyIn:
    case ur_buffer_::AllocMode::Classic:
      for (auto i = 0u; i < getContext()->NumDevices; ++i) {
        if (getPtrs()[i] != ur_buffer_::native_type{0}) {
          ScopedDevice Active(getContext()->getDevices()[i]);
          Result = UR_CHECK_ERROR(hipFree(Ptrs[i]));
        }
      }
      break;
    case ur_buffer_::AllocMode::UseHostPtr:
      Result = UR_CHECK_ERROR(hipHostUnregister(HostPtr));
      break;
    case ur_buffer_::AllocMode::AllocHostPtr:
      Result = UR_CHECK_ERROR(hipFreeHost(HostPtr));
    };
    return Result;
  };

  ur_result_t allocateMemObjOnDeviceIfNeeded(ur_device_handle_t) override;
  ur_result_t migrateMemoryToDeviceIfNeeded(ur_device_handle_t) override;

  struct {
    size_t Origin; // only valid if Parent != nullptr
  } SubBuffer;
};

// Handler data for image object (i.e. surface/textures)
struct ur_image_ final : ur_mem_handle_t_ {
  ur_image_format_t ImageFormat;
  ur_image_desc_t ImageDesc;
  void *HostPtr{nullptr};
  std::vector<hipArray *> Arrays;
  std::vector<hipSurfaceObject_t> SurfObjs;
  size_t PixelTypeSizeBytes;
  HIP_ARRAY3D_DESCRIPTOR ArrayDesc;

  // TODO(hdelan): this should probably not take a CUArray at all, should be
  // allocated and migrated to lazily using the allocateMemObjOnDeviceIfNeeded
  // and migrateMemoryToDeviceIfNeeded methods
  ur_image_(ur_context_handle_t Context, ur_mem_flags_t MemFlags,
            ur_image_format_t ImageFormat, ur_image_desc_t ImageDesc,
            void *HostPtr)
      : ur_mem_handle_t_{Context, MemFlags}, ImageFormat{ImageFormat},
        ImageDesc{ImageDesc}, HostPtr{HostPtr},
        Arrays(Context->NumDevices, nullptr),
        SurfObjs(Context->NumDevices, hipSurfaceObject_t{0}) {

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

    // We need to get this now in bytes for calculating the total image size
    // later

    switch (ImageFormat.channelType) {

    case UR_IMAGE_CHANNEL_TYPE_UNORM_INT8:
    case UR_IMAGE_CHANNEL_TYPE_UNSIGNED_INT8:
      ArrayDesc.Format = HIP_AD_FORMAT_UNSIGNED_INT8;
      PixelTypeSizeBytes = 1;
      break;
    case UR_IMAGE_CHANNEL_TYPE_SIGNED_INT8:
      ArrayDesc.Format = HIP_AD_FORMAT_SIGNED_INT8;
      PixelTypeSizeBytes = 1;
      break;
    case UR_IMAGE_CHANNEL_TYPE_UNORM_INT16:
    case UR_IMAGE_CHANNEL_TYPE_UNSIGNED_INT16:
      ArrayDesc.Format = HIP_AD_FORMAT_UNSIGNED_INT16;
      PixelTypeSizeBytes = 2;
      break;
    case UR_IMAGE_CHANNEL_TYPE_SIGNED_INT16:
      ArrayDesc.Format = HIP_AD_FORMAT_SIGNED_INT16;
      PixelTypeSizeBytes = 2;
      break;
    case UR_IMAGE_CHANNEL_TYPE_HALF_FLOAT:
      ArrayDesc.Format = HIP_AD_FORMAT_HALF;
      PixelTypeSizeBytes = 2;
      break;
    case UR_IMAGE_CHANNEL_TYPE_UNSIGNED_INT32:
      ArrayDesc.Format = HIP_AD_FORMAT_UNSIGNED_INT32;
      PixelTypeSizeBytes = 4;
      break;
    case UR_IMAGE_CHANNEL_TYPE_SIGNED_INT32:
      ArrayDesc.Format = HIP_AD_FORMAT_SIGNED_INT32;
      PixelTypeSizeBytes = 4;
      break;
    case UR_IMAGE_CHANNEL_TYPE_FLOAT:
      ArrayDesc.Format = HIP_AD_FORMAT_FLOAT;
      PixelTypeSizeBytes = 4;
      break;
    default:
      // urMemImageCreate given unsupported image_channel_data_type
      detail::ur::die("Bad image format given to ur_image_ constructor");
    }
  }

  ~ur_image_() override{};

  ur_result_t allocateMemObjOnDeviceIfNeeded(ur_device_handle_t) override;
  ur_result_t migrateMemoryToDeviceIfNeeded(ur_device_handle_t) override;

  bool isBuffer() const noexcept override { return false; }
  bool isImage() const noexcept override { return true; }

  ur_result_t clear() override {
    for (auto Array : Arrays) {
      if (Array) {
        UR_CHECK_ERROR(hipFreeArray(Array));
      }
    }
    for (auto Surf : SurfObjs) {
      if (Surf != hipSurfaceObject_t{0}) {
        UR_CHECK_ERROR(hipDestroySurfaceObject(Surf));
      }
    }
    return UR_RESULT_SUCCESS;
  }

  std::vector<hipArray *> &getArrays() noexcept { return Arrays; }
  std::vector<hipSurfaceObject_t> &getSurfaces() noexcept { return SurfObjs; }
  hipArray *&getArray(ur_device_handle_t Device) noexcept {
    return Arrays[Device->getIndex()];
  }
  hipSurfaceObject_t &getSurface(ur_device_handle_t Device) noexcept {
    return SurfObjs[Device->getIndex()];
  }
  ur_mem_type_t getImageType() const noexcept { return ImageDesc.type; }
};
