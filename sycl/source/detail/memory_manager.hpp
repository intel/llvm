//==-------------- memory_manager.hpp - SYCL standard header file ----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <detail/sycl_mem_obj_i.hpp>
#include <sycl/access/access.hpp>
#include <sycl/detail/export.hpp>
#include <sycl/id.hpp>
#include <sycl/property_list.hpp>
#include <sycl/range.hpp>

#include <ur_api.h>

#include <memory>
#include <vector>

namespace sycl {
inline namespace _V1 {
namespace detail {

class queue_impl;
class event_impl;
class events_range;
class context_impl;

using EventImplPtr = std::shared_ptr<detail::event_impl>;

// The class contains methods that work with memory. All operations with
// device memory should go through MemoryManager.

class MemoryManager {
public:
  // The following method releases memory allocation of memory object.
  // Depending on the context it releases memory on host or on device.
  static void release(context_impl *TargetContext, SYCLMemObjI *MemObj,
                      void *MemAllocation, events_range DepEvents,
                      ur_event_handle_t &OutEvent);

  // The following method allocates memory allocation of memory object.
  // Depending on the context it allocates memory on host or on device.
  static void *allocate(context_impl *TargetContext, SYCLMemObjI *MemObj,
                        bool InitFromUserData, void *HostPtr,
                        events_range DepEvents, ur_event_handle_t &OutEvent);

  // The following method creates OpenCL sub buffer for specified
  // offset, range, and memory object.
  static void *allocateMemSubBuffer(context_impl *TargetContext,
                                    void *ParentMemObj, size_t ElemSize,
                                    size_t Offset, range<3> Range,
                                    events_range DepEvents,
                                    ur_event_handle_t &OutEvent);

  // Allocates buffer in specified context taking into account situations such
  // as host ptr or cl_mem provided by user. TargetContext should be device
  // one(not host).
  static void *allocateMemBuffer(context_impl *TargetContext,
                                 SYCLMemObjI *MemObj, void *UserPtr,
                                 bool HostPtrReadOnly, size_t Size,
                                 const EventImplPtr &InteropEvent,
                                 context_impl *InteropContext,
                                 const sycl::property_list &PropsList,
                                 ur_event_handle_t &OutEventToWait);

  // Allocates images in specified context taking into account situations such
  // as host ptr or cl_mem provided by user. TargetContext should be device
  // one(not host).
  static void *allocateMemImage(
      context_impl *TargetContext, SYCLMemObjI *MemObj, void *UserPtr,
      bool HostPtrReadOnly, size_t Size, const ur_image_desc_t &Desc,
      const ur_image_format_t &Format, const EventImplPtr &InteropEvent,
      context_impl *InteropContext, const sycl::property_list &PropsList,
      ur_event_handle_t &OutEventToWait);

  // Releases memory object(buffer or image). TargetContext should be device
  // one(not host).
  static void releaseMemObj(context_impl *TargetContext, SYCLMemObjI *MemObj,
                            void *MemAllocation, void *UserPtr);

  static void *allocateHostMemory(SYCLMemObjI *MemObj, void *UserPtr,
                                  bool HostPtrReadOnly, size_t Size,
                                  const sycl::property_list &PropsList);

  static void *allocateInteropMemObject(context_impl *TargetContext,
                                        void *UserPtr,
                                        const EventImplPtr &InteropEvent,
                                        context_impl *InteropContext,
                                        const sycl::property_list &PropsList,
                                        ur_event_handle_t &OutEventToWait);

  static void *allocateImageObject(context_impl *TargetContext, void *UserPtr,
                                   bool HostPtrReadOnly,
                                   const ur_image_desc_t &Desc,
                                   const ur_image_format_t &Format,
                                   const sycl::property_list &PropsList);

  static void *allocateBufferObject(context_impl *TargetContext, void *UserPtr,
                                    bool HostPtrReadOnly, const size_t Size,
                                    const sycl::property_list &PropsList);

  // Copies memory between: host and device, host and host,
  // device and device if memory objects bound to the one context.
  static void copy(SYCLMemObjI *SYCLMemObj, void *SrcMem, queue_impl *SrcQueue,
                   unsigned int DimSrc, sycl::range<3> SrcSize,
                   sycl::range<3> SrcAccessRange, sycl::id<3> SrcOffset,
                   unsigned int SrcElemSize, void *DstMem, queue_impl *TgtQueue,
                   unsigned int DimDst, sycl::range<3> DstSize,
                   sycl::range<3> DstAccessRange, sycl::id<3> DstOffset,
                   unsigned int DstElemSize,
                   std::vector<ur_event_handle_t> DepEvents,
                   ur_event_handle_t &OutEvent);

  static void fill(SYCLMemObjI *SYCLMemObj, void *Mem, queue_impl &Queue,
                   size_t PatternSize, const unsigned char *Pattern,
                   unsigned int Dim, sycl::range<3> Size,
                   sycl::range<3> AccessRange, sycl::id<3> AccessOffset,
                   unsigned int ElementSize,
                   std::vector<ur_event_handle_t> DepEvents,
                   ur_event_handle_t &OutEvent);

  static void *map(SYCLMemObjI *SYCLMemObj, void *Mem, queue_impl &Queue,
                   access::mode AccessMode, unsigned int Dim,
                   sycl::range<3> Size, sycl::range<3> AccessRange,
                   sycl::id<3> AccessOffset, unsigned int ElementSize,
                   std::vector<ur_event_handle_t> DepEvents,
                   ur_event_handle_t &OutEvent);

  static void unmap(SYCLMemObjI *SYCLMemObj, void *Mem, queue_impl &Queue,
                    void *MappedPtr, std::vector<ur_event_handle_t> DepEvents,
                    ur_event_handle_t &OutEvent);

  static void copy_usm(const void *SrcMem, queue_impl &Queue, size_t Len,
                       void *DstMem, std::vector<ur_event_handle_t> DepEvents,
                       ur_event_handle_t *OutEvent);

  static void context_copy_usm(const void *SrcMem, context_impl *Context,
                               size_t Len, void *DstMem);

  static void fill_usm(void *DstMem, queue_impl &Queue, size_t Len,
                       const std::vector<unsigned char> &Pattern,
                       std::vector<ur_event_handle_t> DepEvents,
                       ur_event_handle_t *OutEvent);

  static void prefetch_usm(void *Ptr, queue_impl &Queue, size_t Len,
                           std::vector<ur_event_handle_t> DepEvents,
                           ur_event_handle_t *OutEvent);

  static void advise_usm(const void *Ptr, queue_impl &Queue, size_t Len,
                         ur_usm_advice_flags_t Advice,
                         std::vector<ur_event_handle_t> DepEvents,
                         ur_event_handle_t *OutEvent);

  static void copy_2d_usm(const void *SrcMem, size_t SrcPitch,
                          queue_impl &Queue, void *DstMem, size_t DstPitch,
                          size_t Width, size_t Height,
                          std::vector<ur_event_handle_t> DepEvents,
                          ur_event_handle_t *OutEvent);

  static void fill_2d_usm(void *DstMem, queue_impl &Queue, size_t Pitch,
                          size_t Width, size_t Height,
                          const std::vector<unsigned char> &Pattern,
                          std::vector<ur_event_handle_t> DepEvents,
                          ur_event_handle_t *OutEvent);

  static void memset_2d_usm(void *DstMem, queue_impl &Queue, size_t Pitch,
                            size_t Width, size_t Height, char Value,
                            std::vector<ur_event_handle_t> DepEvents,
                            ur_event_handle_t *OutEvent);

  static void
  copy_to_device_global(const void *DeviceGlobalPtr, bool IsDeviceImageScoped,
                        queue_impl &Queue, size_t NumBytes, size_t Offset,
                        const void *SrcMem,
                        const std::vector<ur_event_handle_t> &DepEvents,
                        ur_event_handle_t *OutEvent);

  static void
  copy_from_device_global(const void *DeviceGlobalPtr, bool IsDeviceImageScoped,
                          queue_impl &Queue, size_t NumBytes, size_t Offset,
                          void *DstMem,
                          const std::vector<ur_event_handle_t> &DepEvents,
                          ur_event_handle_t *OutEvent);

  // Command buffer extension methods
  static void ext_oneapi_copyD2D_cmd_buffer(
      sycl::detail::context_impl *Context,
      ur_exp_command_buffer_handle_t CommandBuffer, SYCLMemObjI *SYCLMemObj,
      void *SrcMem, unsigned int DimSrc, sycl::range<3> SrcSize,
      sycl::range<3> SrcAccessRange, sycl::id<3> SrcOffset,
      unsigned int SrcElemSize, void *DstMem, unsigned int DimDst,
      sycl::range<3> DstSize, sycl::range<3> DstAccessRange,
      sycl::id<3> DstOffset, unsigned int DstElemSize,
      std::vector<ur_exp_command_buffer_sync_point_t> Deps,
      ur_exp_command_buffer_sync_point_t *OutSyncPoint);

  static void ext_oneapi_copyD2H_cmd_buffer(
      sycl::detail::context_impl *Context,
      ur_exp_command_buffer_handle_t CommandBuffer, SYCLMemObjI *SYCLMemObj,
      void *SrcMem, unsigned int DimSrc, sycl::range<3> SrcSize,
      sycl::range<3> SrcAccessRange, sycl::id<3> SrcOffset,
      unsigned int SrcElemSize, char *DstMem, unsigned int DimDst,
      sycl::range<3> DstSize, sycl::id<3> DstOffset, unsigned int DstElemSize,
      std::vector<ur_exp_command_buffer_sync_point_t> Deps,
      ur_exp_command_buffer_sync_point_t *OutSyncPoint);

  static void ext_oneapi_copyH2D_cmd_buffer(
      sycl::detail::context_impl *Context,
      ur_exp_command_buffer_handle_t CommandBuffer, SYCLMemObjI *SYCLMemObj,
      char *SrcMem, unsigned int DimSrc, sycl::range<3> SrcSize,
      sycl::id<3> SrcOffset, unsigned int SrcElemSize, void *DstMem,
      unsigned int DimDst, sycl::range<3> DstSize,
      sycl::range<3> DstAccessRange, sycl::id<3> DstOffset,
      unsigned int DstElemSize,
      std::vector<ur_exp_command_buffer_sync_point_t> Deps,
      ur_exp_command_buffer_sync_point_t *OutSyncPoint);

  static void ext_oneapi_copy_usm_cmd_buffer(
      context_impl *Context, const void *SrcMem,
      ur_exp_command_buffer_handle_t CommandBuffer, size_t Len, void *DstMem,
      std::vector<ur_exp_command_buffer_sync_point_t> Deps,
      ur_exp_command_buffer_sync_point_t *OutSyncPoint);

  static void ext_oneapi_fill_usm_cmd_buffer(
      sycl::detail::context_impl *Context,
      ur_exp_command_buffer_handle_t CommandBuffer, void *DstMem, size_t Len,
      const std::vector<unsigned char> &Pattern,
      std::vector<ur_exp_command_buffer_sync_point_t> Deps,
      ur_exp_command_buffer_sync_point_t *OutSyncPoint);

  static void ext_oneapi_fill_cmd_buffer(
      sycl::detail::context_impl *Context,
      ur_exp_command_buffer_handle_t CommandBuffer, SYCLMemObjI *SYCLMemObj,
      void *Mem, size_t PatternSize, const unsigned char *Pattern,
      unsigned int Dim, sycl::range<3> Size, sycl::range<3> AccessRange,
      sycl::id<3> AccessOffset, unsigned int ElementSize,
      std::vector<ur_exp_command_buffer_sync_point_t> Deps,
      ur_exp_command_buffer_sync_point_t *OutSyncPoint);

  static void ext_oneapi_prefetch_usm_cmd_buffer(
      sycl::detail::context_impl *Context,
      ur_exp_command_buffer_handle_t CommandBuffer, void *Mem, size_t Length,
      std::vector<ur_exp_command_buffer_sync_point_t> Deps,
      ur_exp_command_buffer_sync_point_t *OutSyncPoint);

  static void ext_oneapi_advise_usm_cmd_buffer(
      sycl::detail::context_impl *Context,
      ur_exp_command_buffer_handle_t CommandBuffer, const void *Mem,
      size_t Length, ur_usm_advice_flags_t Advice,
      std::vector<ur_exp_command_buffer_sync_point_t> Deps,
      ur_exp_command_buffer_sync_point_t *OutSyncPoint);

  static void copy_image_bindless(
      queue_impl &Queue, const void *Src, void *Dst,
      const ur_image_desc_t &SrcDesc, const ur_image_desc_t &DstDesc,
      const ur_image_format_t &SrcFormat, const ur_image_format_t &DstFormat,
      const ur_exp_image_copy_flags_t Flags, ur_rect_offset_t SrcOffset,
      ur_rect_offset_t DstOffset, ur_rect_region_t CopyExtent,
      const std::vector<ur_event_handle_t> &DepEvents,
      ur_event_handle_t *OutEvent);
};
} // namespace detail
} // namespace _V1
} // namespace sycl
