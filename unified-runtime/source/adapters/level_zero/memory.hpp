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

#include <cassert>
#include <list>
#include <map>
#include <optional>
#include <stdarg.h>
#include <string>
#include <unordered_map>
#include <vector>

#include "common.hpp"
#include "context.hpp"
#include "event.hpp"
#include "program.hpp"
#include "queue.hpp"
#include "sampler.hpp"

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

struct ur_mem_handle_t_ : ur_object {
  // Keeps the PI context of this memory handle.
  ur_context_handle_t UrContext;

  // Keeps device of this memory handle
  ur_device_handle_t UrDevice;

  // Whether this is an image or buffer
  enum mem_type_t { image, buffer };
  mem_type_t mem_type;

  // Enumerates all possible types of accesses.
  enum access_mode_t { unknown, read_write, read_only, write_only };

  // Interface of the _ur_mem object

  // Get the Level Zero handle of the current memory object
  ur_result_t getZeHandle(char *&ZeHandle, access_mode_t,
                          ur_device_handle_t Device,
                          const ur_event_handle_t *phWaitEvents,
                          uint32_t numWaitEvents);

  // Get a pointer to the Level Zero handle of the current memory object
  ur_result_t getZeHandlePtr(char **&ZeHandlePtr, access_mode_t,
                             ur_device_handle_t Device,
                             const ur_event_handle_t *phWaitEvents,
                             uint32_t numWaitEvents);

  // Method to get type of the derived object (image or buffer)
  bool isImage() const { return mem_type == mem_type_t::image; }

protected:
  ur_mem_handle_t_(mem_type_t type, ur_context_handle_t Context)
      : UrContext{Context}, UrDevice{nullptr}, mem_type(type) {}

  ur_mem_handle_t_(mem_type_t type, ur_context_handle_t Context,
                   ur_device_handle_t Device)
      : UrContext{Context}, UrDevice(Device), mem_type(type) {}

  // Since the destructor isn't virtual, callers must destruct it via ur_buffer
  // or ur_image
  ~ur_mem_handle_t_() {};
};

struct ur_buffer final : ur_mem_handle_t_ {
  // Buffer constructor
  ur_buffer(ur_context_handle_t Context, ur_device_handle_t UrDevice,
            size_t Size);

  ur_buffer(ur_context_handle_t Context, size_t Size, char *HostPtr,
            bool ImportedHostPtr);

  // Sub-buffer constructor
  ur_buffer(ur_buffer *Parent, size_t Origin, size_t Size)
      : ur_mem_handle_t_(mem_type_t::buffer, Parent->UrContext), Size(Size),
        SubBuffer{{Parent, Origin}} {
    // Retain the Parent Buffer due to the Creation of the SubBuffer.
    Parent->RefCount.increment();
  }

  // Interop-buffer constructor
  ur_buffer(ur_context_handle_t Context, size_t Size, ur_device_handle_t Device,
            char *ZeMemHandle, bool OwnZeMemHandle);

  ~ur_buffer();

  // Returns a pointer to the USM allocation representing this PI buffer
  // on the specified Device. If Device is nullptr then the returned
  // USM allocation is on the device where this buffer was used the latest.
  // The returned allocation is always valid, i.e. its contents is
  // up-to-date and any data copies needed for that are performed under
  // the hood.
  //
  ur_result_t getBufferZeHandle(char *&ZeHandle, access_mode_t,
                                ur_device_handle_t Device,
                                const ur_event_handle_t *phWaitEvents,
                                uint32_t numWaitEvents);
  ur_result_t getBufferZeHandlePtr(char **&ZeHandlePtr, access_mode_t,
                                   ur_device_handle_t Device,
                                   const ur_event_handle_t *phWaitEvents,
                                   uint32_t numWaitEvents);

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
    ur_buffer *Parent;
    size_t Origin;
  };
  std::optional<SubBuffer_t> SubBuffer;
};

struct ur_image final : ur_mem_handle_t_ {
  // Image constructor
  ur_image(ur_context_handle_t UrContext, ze_image_handle_t ZeImage)
      : ur_mem_handle_t_(mem_type_t::image, UrContext), ZeImage{ZeImage} {}

  ur_image(ur_context_handle_t UrContext, ze_image_handle_t ZeImage,
           bool OwnZeMemHandle)
      : ur_mem_handle_t_(mem_type_t::image, UrContext), ZeImage{ZeImage} {
    OwnNativeHandle = OwnZeMemHandle;
  }

  ur_result_t getImageZeHandle(char *&ZeHandle, access_mode_t,
                               ur_device_handle_t,
                               const ur_event_handle_t * /* phWaitEvents*/,
                               uint32_t /*numWaitEvents*/) {
    ZeHandle = reinterpret_cast<char *>(ZeImage);
    return UR_RESULT_SUCCESS;
  }
  ur_result_t getImageZeHandlePtr(char **&ZeHandlePtr, access_mode_t,
                                  ur_device_handle_t,
                                  const ur_event_handle_t * /*phWaitEvents*/,
                                  uint32_t /*numWaitEvents*/) {
    ZeHandlePtr = reinterpret_cast<char **>(&ZeImage);
    return UR_RESULT_SUCCESS;
  }

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
    auto UrImage = new ur_image(Context, ZeImage, OwnZeMemHandle);
    UrImage->ZeImageDesc = ZeImageDesc;
    *UrMem = reinterpret_cast<T>(UrImage);
  } catch (const std::bad_alloc &) {
    return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }
  return UR_RESULT_SUCCESS;
}
