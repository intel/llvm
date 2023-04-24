//===--------- ur_level_zero_mem.hpp - Level Zero Adapter -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//
#pragma once

#include "ur_level_zero_common.hpp"
#include <cassert>
#include <list>
#include <map>
#include <stdarg.h>
#include <string>
#include <unordered_map>
#include <vector>

#include <sycl/detail/pi.h>
#include <ur/ur.hpp>
#include <ur_api.h>
#include <ze_api.h>
#include <zes_api.h>

#include "ur_level_zero.hpp"

struct ur_device_handle_t_;

bool IsDevicePointer(ur_context_handle_t Context, const void *Ptr);

// This is an experimental option to test performance of device to device copy
// operations on copy engines (versus compute engine)
const bool UseCopyEngineForD2DCopy = [] {
  const char *CopyEngineForD2DCopy =
      std::getenv("SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE_FOR_D2D_COPY");
  return (CopyEngineForD2DCopy && (std::stoi(CopyEngineForD2DCopy) != 0));
}();

// Shared by all memory read/write/copy PI interfaces.
// PI interfaces must have queue's and destination buffer's mutexes locked for
// exclusive use and source buffer's mutex locked for shared use on entry.
ur_result_t enqueueMemCopyHelper(ur_command_t CommandType,
                                 ur_queue_handle_t Queue, void *Dst,
                                 pi_bool BlockingWrite, size_t Size,
                                 const void *Src, uint32_t NumEventsInWaitList,
                                 const ur_event_handle_t *EventWaitList,
                                 ur_event_handle_t *OutEvent,
                                 bool PreferCopyEngine);

ur_result_t enqueueMemCopyRectHelper(
    ur_command_t CommandType, ur_queue_handle_t Queue, const void *SrcBuffer,
    void *DstBuffer, ur_rect_offset_t SrcOrigin, ur_rect_offset_t DstOrigin,
    ur_rect_region_t Region, size_t SrcRowPitch, size_t DstRowPitch,
    size_t SrcSlicePitch, size_t DstSlicePitch, pi_bool Blocking,
    uint32_t NumEventsInWaitList, const ur_event_handle_t *EventWaitList,
    ur_event_handle_t *OutEvent, bool PreferCopyEngine = false);

// Exception type to pass allocation errors
class UsmAllocationException {
  const ur_result_t Error;

public:
  UsmAllocationException(ur_result_t Err) : Error{Err} {}
  ur_result_t getError() const { return Error; }
};

struct ur_mem_handle_t_ : _ur_object {
  // Keeps the PI context of this memory handle.
  ur_context_handle_t UrContext;

  // Keeps device of this memory handle
  ur_device_handle_t UrDevice;

  // Enumerates all possible types of accesses.
  enum access_mode_t { unknown, read_write, read_only, write_only };

  // Interface of the _ur_mem object

  // Get the Level Zero handle of the current memory object
  virtual ur_result_t getZeHandle(char *&ZeHandle, access_mode_t,
                                  ur_device_handle_t Device = nullptr) = 0;

  // Get a pointer to the Level Zero handle of the current memory object
  virtual ur_result_t getZeHandlePtr(char **&ZeHandlePtr, access_mode_t,
                                     ur_device_handle_t Device = nullptr) = 0;

  // Method to get type of the derived object (image or buffer)
  virtual bool isImage() const = 0;

  virtual ~ur_mem_handle_t_() = default;

protected:
  ur_mem_handle_t_(ur_context_handle_t Context) : UrContext{Context} {}

  ur_mem_handle_t_(ur_context_handle_t Context, ur_device_handle_t Device)
      : UrContext{Context}, UrDevice(Device) {}
};

struct _ur_buffer final : ur_mem_handle_t_ {
  // Buffer constructor
  _ur_buffer(ur_context_handle_t Context, ur_device_handle_t UrDevice,
             size_t Size);

  _ur_buffer(ur_context_handle_t Context, size_t Size, char *HostPtr,
             bool ImportedHostPtr);

  // Sub-buffer constructor
  _ur_buffer(_ur_buffer *Parent, size_t Origin, size_t Size)
      : ur_mem_handle_t_(Parent->UrContext), Size(Size),
        SubBuffer{Parent, Origin} {}

  // Interop-buffer constructor
  _ur_buffer(ur_context_handle_t Context, size_t Size,
             ur_device_handle_t Device, char *ZeMemHandle, bool OwnZeMemHandle);

  // Returns a pointer to the USM allocation representing this PI buffer
  // on the specified Device. If Device is nullptr then the returned
  // USM allocation is on the device where this buffer was used the latest.
  // The returned allocation is always valid, i.e. its contents is
  // up-to-date and any data copies needed for that are performed under
  // the hood.
  //
  virtual ur_result_t getZeHandle(char *&ZeHandle, access_mode_t,
                                  ur_device_handle_t Device = nullptr) override;
  virtual ur_result_t
  getZeHandlePtr(char **&ZeHandlePtr, access_mode_t,
                 ur_device_handle_t Device = nullptr) override;

  bool isImage() const override { return false; }

  bool isSubBuffer() const { return SubBuffer.Parent != nullptr; }

  // Frees all allocations made for the buffer.
  ur_result_t free();

  // Information about a single allocation representing this buffer.
  struct allocation_t {
    // Level Zero memory handle is really just a naked pointer.
    // It is just convenient to have it char * to simplify offset arithmetics.
    char *ZeHandle{nullptr};
    // Indicates if this allocation's data is valid.
    bool Valid{false};
    // Specifies the action that needs to be taken for this
    // allocation at buffer destruction.
    enum {
      keep,       // do nothing, the allocation is not owned by us
      unimport,   // release of the imported allocation
      free,       // free from the pooling context (default)
      free_native // free with a native call
    } ReleaseAction{free};
  };

  // We maintain multiple allocations on possibly all devices in the context.
  // The "nullptr" device identifies a host allocation representing buffer.
  // Sub-buffers don't maintain own allocations but rely on parent buffer.
  std::unordered_map<ur_device_handle_t, allocation_t> Allocations;
  ur_device_handle_t LastDeviceWithValidAllocation{nullptr};

  // Flag to indicate that this memory is allocated in host memory.
  // Integrated device accesses this memory.
  bool OnHost{false};

  // Tells the host allocation to use for buffer map operations.
  char *MapHostPtr{nullptr};

  // Supplementary data to keep track of the mappings of this buffer
  // created with piEnqueueMemBufferMap.
  struct Mapping {
    // The offset in the buffer giving the start of the mapped region.
    size_t Offset;
    // The size of the mapped region.
    size_t Size;
  };

  // The key is the host pointer representing an active mapping.
  // The value is the information needed to maintain/undo the mapping.
  std::unordered_map<void *, Mapping> Mappings;

  // The size and alignment of the buffer
  size_t Size;
  size_t getAlignment() const;

  struct {
    _ur_buffer *Parent;
    size_t Origin; // only valid if Parent != nullptr
  } SubBuffer;
};

struct _ur_image final : ur_mem_handle_t_ {
  // Image constructor
  _ur_image(ur_context_handle_t UrContext, ze_image_handle_t ZeImage)
      : ur_mem_handle_t_(UrContext), ZeImage{ZeImage} {}

  _ur_image(ur_context_handle_t UrContext, ze_image_handle_t ZeImage,
            bool OwnNativeHandle)
      : ur_mem_handle_t_(UrContext), ZeImage{ZeImage},
        OwnZeMemHandle{OwnNativeHandle} {}

  virtual ur_result_t getZeHandle(char *&ZeHandle, access_mode_t,
                                  ur_device_handle_t = nullptr) override {
    ZeHandle = reinterpret_cast<char *>(ZeImage);
    return UR_RESULT_SUCCESS;
  }
  virtual ur_result_t getZeHandlePtr(char **&ZeHandlePtr, access_mode_t,
                                     ur_device_handle_t = nullptr) override {
    ZeHandlePtr = reinterpret_cast<char **>(&ZeImage);
    return UR_RESULT_SUCCESS;
  }

  bool isImage() const override { return true; }

#ifndef NDEBUG
  // Keep the descriptor of the image (for debugging purposes)
  ZeStruct<ze_image_desc_t> ZeImageDesc;
#endif // !NDEBUG

  // Level Zero image handle.
  ze_image_handle_t ZeImage;

  bool OwnZeMemHandle = true;
};

// Implements memory allocation via L0 RT for USM allocator interface.
class USMMemoryAllocBase : public SystemMemory {
protected:
  ur_context_handle_t Context;
  ur_device_handle_t Device;
  // Internal allocation routine which must be implemented for each allocation
  // type
  virtual ur_result_t allocateImpl(void **ResultPtr, size_t Size,
                                   uint32_t Alignment) = 0;

public:
  USMMemoryAllocBase(ur_context_handle_t Ctx, ur_device_handle_t Dev)
      : Context{Ctx}, Device{Dev} {}
  void *allocate(size_t Size) override final;
  void *allocate(size_t Size, size_t Alignment) override final;
  void deallocate(void *Ptr) override final;
};

// Allocation routines for shared memory type
class USMSharedMemoryAlloc : public USMMemoryAllocBase {
protected:
  ur_result_t allocateImpl(void **ResultPtr, size_t Size,
                           uint32_t Alignment) override;

public:
  USMSharedMemoryAlloc(ur_context_handle_t Ctx, ur_device_handle_t Dev)
      : USMMemoryAllocBase(Ctx, Dev) {}
};

// Allocation routines for shared memory type that is only modified from host.
class USMSharedReadOnlyMemoryAlloc : public USMMemoryAllocBase {
protected:
  ur_result_t allocateImpl(void **ResultPtr, size_t Size,
                           uint32_t Alignment) override;

public:
  USMSharedReadOnlyMemoryAlloc(ur_context_handle_t Ctx, ur_device_handle_t Dev)
      : USMMemoryAllocBase(Ctx, Dev) {}
};

// Allocation routines for device memory type
class USMDeviceMemoryAlloc : public USMMemoryAllocBase {
protected:
  ur_result_t allocateImpl(void **ResultPtr, size_t Size,
                           uint32_t Alignment) override;

public:
  USMDeviceMemoryAlloc(ur_context_handle_t Ctx, ur_device_handle_t Dev)
      : USMMemoryAllocBase(Ctx, Dev) {}
};

// Allocation routines for host memory type
class USMHostMemoryAlloc : public USMMemoryAllocBase {
protected:
  ur_result_t allocateImpl(void **ResultPtr, size_t Size,
                           uint32_t Alignment) override;

public:
  USMHostMemoryAlloc(ur_context_handle_t Ctx)
      : USMMemoryAllocBase(Ctx, nullptr) {}
};

ur_result_t USMDeviceAllocImpl(void **ResultPtr, ur_context_handle_t Context,
                               ur_device_handle_t Device,
                               ur_usm_flags_t *Properties, size_t Size,
                               uint32_t Alignment);

ur_result_t USMSharedAllocImpl(void **ResultPtr, ur_context_handle_t Context,
                               ur_device_handle_t Device, ur_usm_flags_t *,
                               size_t Size, uint32_t Alignment);

ur_result_t USMHostAllocImpl(void **ResultPtr, ur_context_handle_t Context,
                             ur_usm_flags_t *Properties, size_t Size,
                             uint32_t Alignment);

// If indirect access tracking is not enabled then this functions just performs
// zeMemFree. If indirect access tracking is enabled then reference counting is
// performed.
ur_result_t ZeMemFreeHelper(ur_context_handle_t Context, void *Ptr);

ur_result_t USMFreeHelper(ur_context_handle_t Context, void *Ptr,
                          bool OwnZeMemHandle = true);

bool ShouldUseUSMAllocator();

extern const bool UseUSMAllocator;
