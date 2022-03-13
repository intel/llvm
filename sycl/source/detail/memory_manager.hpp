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
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace detail {

class queue_impl;
class event_impl;
class context_impl;

using QueueImplPtr = std::shared_ptr<detail::queue_impl>;
using EventImplPtr = std::shared_ptr<detail::event_impl>;
using ContextImplPtr = std::shared_ptr<detail::context_impl>;

// The class contains methods that work with memory. All operations with
// device memory should go through MemoryManager.

class __SYCL_EXPORT MemoryManager {
public:
  // The following method releases memory allocation of memory object.
  // Depending on the context it releases memory on host or on device.
  static void release(ContextImplPtr TargetContext, SYCLMemObjI *MemObj,
                      void *MemAllocation, std::vector<EventImplPtr> DepEvents,
                      RT::PiEvent &OutEvent);

  // The following method allocates memory allocation of memory object.
  // Depending on the context it allocates memory on host or on device.
  static void *allocate(ContextImplPtr TargetContext, SYCLMemObjI *MemObj,
                        bool InitFromUserData, void *HostPtr,
                        std::vector<EventImplPtr> DepEvents,
                        RT::PiEvent &OutEvent);

  // The following method creates OpenCL sub buffer for specified
  // offset, range, and memory object.
  static void *allocateMemSubBuffer(ContextImplPtr TargetContext,
                                    void *ParentMemObj, size_t ElemSize,
                                    size_t Offset, range<3> Range,
                                    std::vector<EventImplPtr> DepEvents,
                                    RT::PiEvent &OutEvent);

  // Allocates buffer in specified context taking into account situations such
  // as host ptr or cl_mem provided by user. TargetContext should be device
  // one(not host).
  static void *allocateMemBuffer(ContextImplPtr TargetContext,
                                 SYCLMemObjI *MemObj, void *UserPtr,
                                 bool HostPtrReadOnly, size_t Size,
                                 const EventImplPtr &InteropEvent,
                                 const ContextImplPtr &InteropContext,
                                 const sycl::property_list &PropsList,
                                 RT::PiEvent &OutEventToWait);

  // Allocates images in specified context taking into account situations such
  // as host ptr or cl_mem provided by user. TargetContext should be device
  // one(not host).
  static void *allocateMemImage(
      ContextImplPtr TargetContext, SYCLMemObjI *MemObj, void *UserPtr,
      bool HostPtrReadOnly, size_t Size, const RT::PiMemImageDesc &Desc,
      const RT::PiMemImageFormat &Format, const EventImplPtr &InteropEvent,
      const ContextImplPtr &InteropContext,
      const sycl::property_list &PropsList, RT::PiEvent &OutEventToWait);

  // Releases memory object(buffer or image). TargetContext should be device
  // one(not host).
  static void releaseMemObj(ContextImplPtr TargetContext, SYCLMemObjI *MemObj,
                            void *MemAllocation, void *UserPtr);

  static void *allocateHostMemory(SYCLMemObjI *MemObj, void *UserPtr,
                                  bool HostPtrReadOnly, size_t Size,
                                  const sycl::property_list &PropsList);

  static void *allocateInteropMemObject(ContextImplPtr TargetContext,
                                        void *UserPtr,
                                        const EventImplPtr &InteropEvent,
                                        const ContextImplPtr &InteropContext,
                                        const sycl::property_list &PropsList,
                                        RT::PiEvent &OutEventToWait);

  static void *allocateImageObject(ContextImplPtr TargetContext, void *UserPtr,
                                   bool HostPtrReadOnly,
                                   const RT::PiMemImageDesc &Desc,
                                   const RT::PiMemImageFormat &Format,
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
                   std::vector<RT::PiEvent> DepEvents, RT::PiEvent &OutEvent);

  static void fill(SYCLMemObjI *SYCLMemObj, void *Mem, QueueImplPtr Queue,
                   size_t PatternSize, const char *Pattern, unsigned int Dim,
                   sycl::range<3> Size, sycl::range<3> AccessRange,
                   sycl::id<3> AccessOffset, unsigned int ElementSize,
                   std::vector<RT::PiEvent> DepEvents, RT::PiEvent &OutEvent);

  static void *map(SYCLMemObjI *SYCLMemObj, void *Mem, QueueImplPtr Queue,
                   access::mode AccessMode, unsigned int Dim,
                   sycl::range<3> Size, sycl::range<3> AccessRange,
                   sycl::id<3> AccessOffset, unsigned int ElementSize,
                   std::vector<RT::PiEvent> DepEvents, RT::PiEvent &OutEvent);

  static void unmap(SYCLMemObjI *SYCLMemObj, void *Mem, QueueImplPtr Queue,
                    void *MappedPtr, std::vector<RT::PiEvent> DepEvents,
                    RT::PiEvent &OutEvent);

  static void copy_usm(const void *SrcMem, QueueImplPtr Queue, size_t Len,
                       void *DstMem, std::vector<RT::PiEvent> DepEvents,
                       RT::PiEvent *OutEvent);

  static void fill_usm(void *DstMem, QueueImplPtr Queue, size_t Len,
                       int Pattern, std::vector<RT::PiEvent> DepEvents,
                       RT::PiEvent *OutEvent);

  static void prefetch_usm(void *Ptr, QueueImplPtr Queue, size_t Len,
                           std::vector<RT::PiEvent> DepEvents,
                           RT::PiEvent *OutEvent);

  static void advise_usm(const void *Ptr, QueueImplPtr Queue, size_t Len,
                         pi_mem_advice Advice,
                         std::vector<RT::PiEvent> DepEvents,
                         RT::PiEvent *OutEvent);

  static void copy_2d_usm(const void *SrcMem, size_t SrcPitch,
                          QueueImplPtr Queue, void *DstMem, size_t DstPitch,
                          size_t Width, size_t Height,
                          std::vector<RT::PiEvent> DepEvents,
                          RT::PiEvent *OutEvent);

  static void fill_2d_usm(void *DstMem, QueueImplPtr Queue, size_t Pitch,
                          size_t Width, size_t Height,
                          const std::vector<char> &Pattern,
                          std::vector<RT::PiEvent> DepEvents,
                          RT::PiEvent *OutEvent);

  static void memset_2d_usm(void *DstMem, QueueImplPtr Queue, size_t Pitch,
                            size_t Width, size_t Height, char Value,
                            std::vector<RT::PiEvent> DepEvents,
                            RT::PiEvent *OutEvent);

  static void copy_to_device_global(
      const void *DeviceGlobalPtr, bool IsDeviceImageScoped, QueueImplPtr Queue,
      size_t NumBytes, size_t Offset, const void *SrcMem, OSModuleHandle M,
      std::vector<RT::PiEvent> DepEvents, RT::PiEvent *OutEvent);

  static void copy_from_device_global(
      const void *DeviceGlobalPtr, bool IsDeviceImageScoped, QueueImplPtr Queue,
      size_t NumBytes, size_t Offset, void *DstMem, OSModuleHandle M,
      std::vector<RT::PiEvent> DepEvents, RT::PiEvent *OutEvent);
};
} // namespace detail
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
