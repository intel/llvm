//===--------- memory.hpp - Level Zero Adapter ----------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include "common.hpp"
#include <cassert>
#include <list>
#include <map>
#include <optional>
#include <stdarg.h>
#include <string>
#include <unordered_map>
#include <vector>

#include <ur/ur.hpp>
#include <ur_ddi.h>
#include <ze_api.h>
#include <zes_api.h>

#include "ur_level_zero.hpp"

struct ur_device_handle_t_;

bool IsDevicePointer(ur_context_handle_t Context, const void *Ptr);
bool IsSharedPointer(ur_context_handle_t Context, const void *Ptr);
bool PreferCopyEngineUsage(ur_device_handle_t Device,
                           ur_context_handle_t Context, const void *Src,
                           void *Dst);

// This is an experimental option to test performance of device to device copy
// operations on copy engines (versus compute engine)
const bool UseCopyEngineForD2DCopy = [] {
  const char *UrRet = std::getenv("UR_L0_USE_COPY_ENGINE_FOR_D2D_COPY");
  const char *PiRet =
      std::getenv("SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE_FOR_D2D_COPY");
  const char *CopyEngineForD2DCopy = UrRet ? UrRet : (PiRet ? PiRet : nullptr);
  return (CopyEngineForD2DCopy && (std::stoi(CopyEngineForD2DCopy) != 0));
}();

// Shared by all memory read/write/copy PI interfaces.
// PI interfaces must have queue's and destination buffer's mutexes locked for
// exclusive use and source buffer's mutex locked for shared use on entry.
ur_result_t enqueueMemCopyHelper(ur_command_t CommandType,
                                 ur_queue_handle_t Queue, void *Dst,
                                 ur_bool_t BlockingWrite, size_t Size,
                                 const void *Src, uint32_t NumEventsInWaitList,
                                 const ur_event_handle_t *EventWaitList,
                                 ur_event_handle_t *OutEvent,
                                 bool PreferCopyEngine);

ur_result_t enqueueMemCopyRectHelper(
    ur_command_t CommandType, ur_queue_handle_t Queue, const void *SrcBuffer,
    void *DstBuffer, ur_rect_offset_t SrcOrigin, ur_rect_offset_t DstOrigin,
    ur_rect_region_t Region, size_t SrcRowPitch, size_t DstRowPitch,
    size_t SrcSlicePitch, size_t DstSlicePitch, ur_bool_t Blocking,
    uint32_t NumEventsInWaitList, const ur_event_handle_t *EventWaitList,
    ur_event_handle_t *OutEvent, bool PreferCopyEngine = false);

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
                                  ur_device_handle_t Device,
                                  const ur_event_handle_t *phWaitEvents,
                                  uint32_t numWaitEvents) = 0;

  // Get a pointer to the Level Zero handle of the current memory object
  virtual ur_result_t getZeHandlePtr(char **&ZeHandlePtr, access_mode_t,
                                     ur_device_handle_t Device,
                                     const ur_event_handle_t *phWaitEvents,
                                     uint32_t numWaitEvents) = 0;

  // Method to get type of the derived object (image or buffer)
  virtual bool isImage() const = 0;

  virtual ~ur_mem_handle_t_() = default;

protected:
  ur_mem_handle_t_(ur_context_handle_t Context)
      : UrContext{Context}, UrDevice{nullptr} {}

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
        SubBuffer{{Parent, Origin}} {
    // Retain the Parent Buffer due to the Creation of the SubBuffer.
    Parent->RefCount.increment();
  }

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
                                  ur_device_handle_t Device,
                                  const ur_event_handle_t *phWaitEvents,
                                  uint32_t numWaitEvents) override;
  virtual ur_result_t getZeHandlePtr(char **&ZeHandlePtr, access_mode_t,
                                     ur_device_handle_t Device,
                                     const ur_event_handle_t *phWaitEvents,
                                     uint32_t numWaitEvents) override;

  bool isImage() const override { return false; }
  bool isSubBuffer() const { return SubBuffer != std::nullopt; }

  // Frees all allocations made for the buffer.
  ur_result_t free();

  // Tracks if this buffer is freed already or should be considered valid.
  bool isFreed{false};

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

  // Pointer to the original native buffer handle given this memory is a proxy
  // device buffer.
  void *DeviceMappedHostNativePtr{nullptr};

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

  struct SubBuffer_t {
    _ur_buffer *Parent;
    size_t Origin;
  };
  std::optional<SubBuffer_t> SubBuffer;
};

struct _ur_image final : ur_mem_handle_t_ {
  // Image constructor
  _ur_image(ur_context_handle_t UrContext, ze_image_handle_t ZeImage)
      : ur_mem_handle_t_(UrContext), ZeImage{ZeImage} {}

  _ur_image(ur_context_handle_t UrContext, ze_image_handle_t ZeImage,
            bool OwnZeMemHandle)
      : ur_mem_handle_t_(UrContext), ZeImage{ZeImage} {
    OwnNativeHandle = OwnZeMemHandle;
  }

  virtual ur_result_t getZeHandle(char *&ZeHandle, access_mode_t,
                                  ur_device_handle_t,
                                  const ur_event_handle_t *phWaitEvents,
                                  uint32_t numWaitEvents) override {
    std::ignore = phWaitEvents;
    std::ignore = numWaitEvents;
    ZeHandle = reinterpret_cast<char *>(ZeImage);
    return UR_RESULT_SUCCESS;
  }
  virtual ur_result_t getZeHandlePtr(char **&ZeHandlePtr, access_mode_t,
                                     ur_device_handle_t,
                                     const ur_event_handle_t *phWaitEvents,
                                     uint32_t numWaitEvents) override {
    std::ignore = phWaitEvents;
    std::ignore = numWaitEvents;
    ZeHandlePtr = reinterpret_cast<char **>(&ZeImage);
    return UR_RESULT_SUCCESS;
  }

  bool isImage() const override { return true; }

  // Keep the descriptor of the image
  ZeStruct<ze_image_desc_t> ZeImageDesc;

  // Level Zero image handle.
  ze_image_handle_t ZeImage;
};

template <typename T>
ur_result_t
createUrMemFromZeImage(ur_context_handle_t Context, ze_image_handle_t ZeImage,
                       bool OwnZeMemHandle,
                       const ZeStruct<ze_image_desc_t> &ZeImageDesc, T *UrMem) {
  try {
    auto UrImage = new _ur_image(Context, ZeImage, OwnZeMemHandle);
    UrImage->ZeImageDesc = ZeImageDesc;
    *UrMem = reinterpret_cast<T>(UrImage);
  } catch (const std::bad_alloc &) {
    return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }
  return UR_RESULT_SUCCESS;
}
