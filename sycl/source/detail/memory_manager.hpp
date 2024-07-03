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

#include <memory>
#include <vector>

namespace sycl {
inline namespace _V1 {
namespace detail {

class queue_impl;
class event_impl;
class context_impl;

using QueueImplPtr = std::shared_ptr<detail::queue_impl>;
using EventImplPtr = std::shared_ptr<detail::event_impl>;
using ContextImplPtr = std::shared_ptr<detail::context_impl>;

// The class contains methods that work with memory. All operations with
// device memory should go through MemoryManager.

class MemoryManager {
public:
  // The following method releases memory allocation of memory object.
  // Depending on the context it releases memory on host or on device.
  static void release(ContextImplPtr TargetContext, SYCLMemObjI *MemObj,
                      void *MemAllocation, std::vector<EventImplPtr> DepEvents,
                      sycl::detail::pi::PiEvent &OutEvent);

  // The following method allocates memory allocation of memory object.
  // Depending on the context it allocates memory on host or on device.
  static void *allocate(ContextImplPtr TargetContext, SYCLMemObjI *MemObj,
                        bool InitFromUserData, void *HostPtr,
                        std::vector<EventImplPtr> DepEvents,
                        sycl::detail::pi::PiEvent &OutEvent);

  // The following method creates OpenCL sub buffer for specified
  // offset, range, and memory object.
  static void *allocateMemSubBuffer(ContextImplPtr TargetContext,
                                    void *ParentMemObj, size_t ElemSize,
                                    size_t Offset, range<3> Range,
                                    std::vector<EventImplPtr> DepEvents,
                                    sycl::detail::pi::PiEvent &OutEvent);

  // Allocates buffer in specified context taking into account situations such
  // as host ptr or cl_mem provided by user. TargetContext should be device
  // one(not host).
  static void *allocateMemBuffer(ContextImplPtr TargetContext,
                                 SYCLMemObjI *MemObj, void *UserPtr,
                                 bool HostPtrReadOnly, size_t Size,
                                 const EventImplPtr &InteropEvent,
                                 const ContextImplPtr &InteropContext,
                                 const sycl::property_list &PropsList,
                                 sycl::detail::pi::PiEvent &OutEventToWait);

  // Allocates images in specified context taking into account situations such
  // as host ptr or cl_mem provided by user. TargetContext should be device
  // one(not host).
  static void *
  allocateMemImage(ContextImplPtr TargetContext, SYCLMemObjI *MemObj,
                   void *UserPtr, bool HostPtrReadOnly, size_t Size,
                   const sycl::detail::pi::PiMemImageDesc &Desc,
                   const sycl::detail::pi::PiMemImageFormat &Format,
                   const EventImplPtr &InteropEvent,
                   const ContextImplPtr &InteropContext,
                   const sycl::property_list &PropsList,
                   sycl::detail::pi::PiEvent &OutEventToWait);

  // Releases memory object(buffer or image). TargetContext should be device
  // one(not host).
  static void releaseMemObj(ContextImplPtr TargetContext, SYCLMemObjI *MemObj,
                            void *MemAllocation, void *UserPtr);

  static void *allocateHostMemory(SYCLMemObjI *MemObj, void *UserPtr,
                                  bool HostPtrReadOnly, size_t Size,
                                  const sycl::property_list &PropsList);
  static void *
  allocateInteropMemObject(ContextImplPtr TargetContext, void *UserPtr,
                           const EventImplPtr &InteropEvent,
                           const ContextImplPtr &InteropContext,
                           const sycl::property_list &PropsList,
                           sycl::detail::pi::PiEvent &OutEventToWait);

  static void *
  allocateImageObject(ContextImplPtr TargetContext, void *UserPtr,
                      bool HostPtrReadOnly,
                      const sycl::detail::pi::PiMemImageDesc &Desc,
                      const sycl::detail::pi::PiMemImageFormat &Format,
                      const sycl::property_list &PropsList);

  static void *allocateBufferObject(ContextImplPtr TargetContext, void *UserPtr,
                                    bool HostPtrReadOnly, const size_t Size,
                                    const sycl::property_list &PropsList);

  // Copies memory between: host and device, host and host,
  // device and device if memory objects bound to the one context.
  static void copy(SYCLMemObjI *SYCLMemObj, void *SrcMem, QueueImplPtr SrcQueue,
                   unsigned int DimSrc, sycl::range<3> SrcSize,
                   sycl::range<3> SrcAccessRange, sycl::id<3> SrcOffset,
                   unsigned int SrcElemSize, void *DstMem,
                   QueueImplPtr TgtQueue, unsigned int DimDst,
                   sycl::range<3> DstSize, sycl::range<3> DstAccessRange,
                   sycl::id<3> DstOffset, unsigned int DstElemSize,
                   std::vector<sycl::detail::pi::PiEvent> DepEvents,
                   sycl::detail::pi::PiEvent &OutEvent,
                   const detail::EventImplPtr &OutEventImpl);

  static void fill(SYCLMemObjI *SYCLMemObj, void *Mem, QueueImplPtr Queue,
                   size_t PatternSize, const char *Pattern, unsigned int Dim,
                   sycl::range<3> Size, sycl::range<3> AccessRange,
                   sycl::id<3> AccessOffset, unsigned int ElementSize,
                   std::vector<sycl::detail::pi::PiEvent> DepEvents,
                   sycl::detail::pi::PiEvent &OutEvent,
                   const detail::EventImplPtr &OutEventImpl);

  static void *map(SYCLMemObjI *SYCLMemObj, void *Mem, QueueImplPtr Queue,
                   access::mode AccessMode, unsigned int Dim,
                   sycl::range<3> Size, sycl::range<3> AccessRange,
                   sycl::id<3> AccessOffset, unsigned int ElementSize,
                   std::vector<sycl::detail::pi::PiEvent> DepEvents,
                   sycl::detail::pi::PiEvent &OutEvent);

  static void unmap(SYCLMemObjI *SYCLMemObj, void *Mem, QueueImplPtr Queue,
                    void *MappedPtr,
                    std::vector<sycl::detail::pi::PiEvent> DepEvents,
                    sycl::detail::pi::PiEvent &OutEvent);

  static void copy_usm(const void *SrcMem, QueueImplPtr Queue, size_t Len,
                       void *DstMem,
                       std::vector<sycl::detail::pi::PiEvent> DepEvents,
                       sycl::detail::pi::PiEvent *OutEvent,
                       const detail::EventImplPtr &OutEventImpl);

  static void fill_usm(void *DstMem, QueueImplPtr Queue, size_t Len,
                       int Pattern,
                       std::vector<sycl::detail::pi::PiEvent> DepEvents,
                       sycl::detail::pi::PiEvent *OutEvent,
                       const detail::EventImplPtr &OutEventImpl);

  static void prefetch_usm(void *Ptr, QueueImplPtr Queue, size_t Len,
                           std::vector<sycl::detail::pi::PiEvent> DepEvents,
                           sycl::detail::pi::PiEvent *OutEvent,
                           const detail::EventImplPtr &OutEventImpl);

  static void advise_usm(const void *Ptr, QueueImplPtr Queue, size_t Len,
                         pi_mem_advice Advice,
                         std::vector<sycl::detail::pi::PiEvent> DepEvents,
                         sycl::detail::pi::PiEvent *OutEvent,
                         const detail::EventImplPtr &OutEventImpl);

  static void copy_2d_usm(const void *SrcMem, size_t SrcPitch,
                          QueueImplPtr Queue, void *DstMem, size_t DstPitch,
                          size_t Width, size_t Height,
                          std::vector<sycl::detail::pi::PiEvent> DepEvents,
                          sycl::detail::pi::PiEvent *OutEvent,
                          const detail::EventImplPtr &OutEventImpl);

  static void fill_2d_usm(void *DstMem, QueueImplPtr Queue, size_t Pitch,
                          size_t Width, size_t Height,
                          const std::vector<char> &Pattern,
                          std::vector<sycl::detail::pi::PiEvent> DepEvents,
                          sycl::detail::pi::PiEvent *OutEvent,
                          const detail::EventImplPtr &OutEventImpl);

  static void memset_2d_usm(void *DstMem, QueueImplPtr Queue, size_t Pitch,
                            size_t Width, size_t Height, char Value,
                            std::vector<sycl::detail::pi::PiEvent> DepEvents,
                            sycl::detail::pi::PiEvent *OutEvent,
                            const detail::EventImplPtr &OutEventImpl);

  static void
  copy_to_device_global(const void *DeviceGlobalPtr, bool IsDeviceImageScoped,
                        QueueImplPtr Queue, size_t NumBytes, size_t Offset,
                        const void *SrcMem,
                        const std::vector<sycl::detail::pi::PiEvent> &DepEvents,
                        sycl::detail::pi::PiEvent *OutEvent,
                        const detail::EventImplPtr &OutEventImpl);

  static void copy_from_device_global(
      const void *DeviceGlobalPtr, bool IsDeviceImageScoped, QueueImplPtr Queue,
      size_t NumBytes, size_t Offset, void *DstMem,
      const std::vector<sycl::detail::pi::PiEvent> &DepEvents,
      sycl::detail::pi::PiEvent *OutEvent,
      const detail::EventImplPtr &OutEventImpl);

  // Command buffer extension methods
  static void ext_oneapi_copyD2D_cmd_buffer(
      sycl::detail::ContextImplPtr Context,
      sycl::detail::pi::PiExtCommandBuffer CommandBuffer,
      SYCLMemObjI *SYCLMemObj, void *SrcMem, unsigned int DimSrc,
      sycl::range<3> SrcSize, sycl::range<3> SrcAccessRange,
      sycl::id<3> SrcOffset, unsigned int SrcElemSize, void *DstMem,
      unsigned int DimDst, sycl::range<3> DstSize,
      sycl::range<3> DstAccessRange, sycl::id<3> DstOffset,
      unsigned int DstElemSize,
      std::vector<sycl::detail::pi::PiExtSyncPoint> Deps,
      sycl::detail::pi::PiExtSyncPoint *OutSyncPoint);

  static void ext_oneapi_copyD2H_cmd_buffer(
      sycl::detail::ContextImplPtr Context,
      sycl::detail::pi::PiExtCommandBuffer CommandBuffer,
      SYCLMemObjI *SYCLMemObj, void *SrcMem, unsigned int DimSrc,
      sycl::range<3> SrcSize, sycl::range<3> SrcAccessRange,
      sycl::id<3> SrcOffset, unsigned int SrcElemSize, char *DstMem,
      unsigned int DimDst, sycl::range<3> DstSize, sycl::id<3> DstOffset,
      unsigned int DstElemSize,
      std::vector<sycl::detail::pi::PiExtSyncPoint> Deps,
      sycl::detail::pi::PiExtSyncPoint *OutSyncPoint);

  static void ext_oneapi_copyH2D_cmd_buffer(
      sycl::detail::ContextImplPtr Context,
      sycl::detail::pi::PiExtCommandBuffer CommandBuffer,
      SYCLMemObjI *SYCLMemObj, char *SrcMem, unsigned int DimSrc,
      sycl::range<3> SrcSize, sycl::id<3> SrcOffset, unsigned int SrcElemSize,
      void *DstMem, unsigned int DimDst, sycl::range<3> DstSize,
      sycl::range<3> DstAccessRange, sycl::id<3> DstOffset,
      unsigned int DstElemSize,
      std::vector<sycl::detail::pi::PiExtSyncPoint> Deps,
      sycl::detail::pi::PiExtSyncPoint *OutSyncPoint);

  static void ext_oneapi_copy_usm_cmd_buffer(
      ContextImplPtr Context, const void *SrcMem,
      sycl::detail::pi::PiExtCommandBuffer CommandBuffer, size_t Len,
      void *DstMem, std::vector<sycl::detail::pi::PiExtSyncPoint> Deps,
      sycl::detail::pi::PiExtSyncPoint *OutSyncPoint);

  static void ext_oneapi_fill_usm_cmd_buffer(
      sycl::detail::ContextImplPtr Context,
      sycl::detail::pi::PiExtCommandBuffer CommandBuffer, void *DstMem,
      size_t Len, int Pattern,
      std::vector<sycl::detail::pi::PiExtSyncPoint> Deps,
      sycl::detail::pi::PiExtSyncPoint *OutSyncPoint);

  static void
  ext_oneapi_fill_cmd_buffer(sycl::detail::ContextImplPtr Context,
                             sycl::detail::pi::PiExtCommandBuffer CommandBuffer,
                             SYCLMemObjI *SYCLMemObj, void *Mem,
                             size_t PatternSize, const char *Pattern,
                             unsigned int Dim, sycl::range<3> Size,
                             sycl::range<3> AccessRange,
                             sycl::id<3> AccessOffset, unsigned int ElementSize,
                             std::vector<sycl::detail::pi::PiExtSyncPoint> Deps,
                             sycl::detail::pi::PiExtSyncPoint *OutSyncPoint);

  static void ext_oneapi_prefetch_usm_cmd_buffer(
      sycl::detail::ContextImplPtr Context,
      sycl::detail::pi::PiExtCommandBuffer CommandBuffer, void *Mem,
      size_t Length, std::vector<sycl::detail::pi::PiExtSyncPoint> Deps,
      sycl::detail::pi::PiExtSyncPoint *OutSyncPoint);

  static void ext_oneapi_advise_usm_cmd_buffer(
      sycl::detail::ContextImplPtr Context,
      sycl::detail::pi::PiExtCommandBuffer CommandBuffer, const void *Mem,
      size_t Length, pi_mem_advice Advice,
      std::vector<sycl::detail::pi::PiExtSyncPoint> Deps,
      sycl::detail::pi::PiExtSyncPoint *OutSyncPoint);

  static void
  copy_image_bindless(void *Src, QueueImplPtr Queue, void *Dst,
                      const sycl::detail::pi::PiMemImageDesc &Desc,
                      const sycl::detail::pi::PiMemImageFormat &Format,
                      const sycl::detail::pi::PiImageCopyFlags Flags,
                      sycl::detail::pi::PiImageOffset SrcOffset,
                      sycl::detail::pi::PiImageOffset DstOffset,
                      sycl::detail::pi::PiImageRegion CopyExtent,
                      sycl::detail::pi::PiImageRegion HostExtent,
                      const std::vector<sycl::detail::pi::PiEvent> &DepEvents,
                      sycl::detail::pi::PiEvent *OutEvent);
};
} // namespace detail
} // namespace _V1
} // namespace sycl
