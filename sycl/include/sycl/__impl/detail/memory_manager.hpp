//==-------------- memory_manager.hpp - SYCL standard header file ----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/__impl/access/access.hpp>
#include <sycl/__impl/detail/cl.h>
#include <sycl/__impl/detail/export.hpp>
#include <sycl/__impl/detail/sycl_mem_obj_i.hpp>
#include <sycl/__impl/property_list.hpp>
#include <sycl/__impl/range.hpp>

#include <memory>
#include <vector>

namespace __sycl_internal {
inline namespace __v1 {
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

  // Allocates memory buffer wrapped into an image. MemObj must be a buffer,
  // not an image.
  // TODO not used - remove.
  static void *wrapIntoImageBuffer(ContextImplPtr TargetContext, void *MemBuf,
                                   SYCLMemObjI *MemObj);

  // Releases the image buffer created by wrapIntoImageBuffer.
  // TODO not used - remove.
  static void releaseImageBuffer(ContextImplPtr TargetContext, void *ImageBuf);

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
                                 const __sycl_internal::__v1::property_list &PropsList,
                                 RT::PiEvent &OutEventToWait);

  // Allocates images in specified context taking into account situations such
  // as host ptr or cl_mem provided by user. TargetContext should be device
  // one(not host).
  static void *allocateMemImage(
      ContextImplPtr TargetContext, SYCLMemObjI *MemObj, void *UserPtr,
      bool HostPtrReadOnly, size_t Size, const RT::PiMemImageDesc &Desc,
      const RT::PiMemImageFormat &Format, const EventImplPtr &InteropEvent,
      const ContextImplPtr &InteropContext,
      const __sycl_internal::__v1::property_list &PropsList, RT::PiEvent &OutEventToWait);

  // Releases memory object(buffer or image). TargetContext should be device
  // one(not host).
  static void releaseMemObj(ContextImplPtr TargetContext, SYCLMemObjI *MemObj,
                            void *MemAllocation, void *UserPtr);

  static void *allocateHostMemory(SYCLMemObjI *MemObj, void *UserPtr,
                                  bool HostPtrReadOnly, size_t Size,
                                  const __sycl_internal::__v1::property_list &PropsList);

  static void *allocateInteropMemObject(ContextImplPtr TargetContext,
                                        void *UserPtr,
                                        const EventImplPtr &InteropEvent,
                                        const ContextImplPtr &InteropContext,
                                        const __sycl_internal::__v1::property_list &PropsList,
                                        RT::PiEvent &OutEventToWait);

  static void *allocateImageObject(ContextImplPtr TargetContext, void *UserPtr,
                                   bool HostPtrReadOnly,
                                   const RT::PiMemImageDesc &Desc,
                                   const RT::PiMemImageFormat &Format,
                                   const __sycl_internal::__v1::property_list &PropsList);

  static void *allocateBufferObject(ContextImplPtr TargetContext, void *UserPtr,
                                    bool HostPtrReadOnly, const size_t Size,
                                    const __sycl_internal::__v1::property_list &PropsList);

  // Copies memory between: host and device, host and host,
  // device and device if memory objects bound to the one context.
  static void copy(SYCLMemObjI *SYCLMemObj, void *SrcMem, QueueImplPtr SrcQueue,
                   unsigned int DimSrc, __sycl_internal::__v1::range<3> SrcSize,
                   __sycl_internal::__v1::range<3> SrcAccessRange, __sycl_internal::__v1::id<3> SrcOffset,
                   unsigned int SrcElemSize, void *DstMem,
                   QueueImplPtr TgtQueue, unsigned int DimDst,
                   __sycl_internal::__v1::range<3> DstSize, __sycl_internal::__v1::range<3> DstAccessRange,
                   __sycl_internal::__v1::id<3> DstOffset, unsigned int DstElemSize,
                   std::vector<RT::PiEvent> DepEvents, RT::PiEvent &OutEvent);

  static void fill(SYCLMemObjI *SYCLMemObj, void *Mem, QueueImplPtr Queue,
                   size_t PatternSize, const char *Pattern, unsigned int Dim,
                   __sycl_internal::__v1::range<3> Size, __sycl_internal::__v1::range<3> AccessRange,
                   __sycl_internal::__v1::id<3> AccessOffset, unsigned int ElementSize,
                   std::vector<RT::PiEvent> DepEvents, RT::PiEvent &OutEvent);

  static void *map(SYCLMemObjI *SYCLMemObj, void *Mem, QueueImplPtr Queue,
                   access::mode AccessMode, unsigned int Dim,
                   __sycl_internal::__v1::range<3> Size, __sycl_internal::__v1::range<3> AccessRange,
                   __sycl_internal::__v1::id<3> AccessOffset, unsigned int ElementSize,
                   std::vector<RT::PiEvent> DepEvents, RT::PiEvent &OutEvent);

  static void unmap(SYCLMemObjI *SYCLMemObj, void *Mem, QueueImplPtr Queue,
                    void *MappedPtr, std::vector<RT::PiEvent> DepEvents,
                    RT::PiEvent &OutEvent);

  static void copy_usm(const void *SrcMem, QueueImplPtr Queue, size_t Len,
                       void *DstMem, std::vector<RT::PiEvent> DepEvents,
                       RT::PiEvent &OutEvent);

  static void fill_usm(void *DstMem, QueueImplPtr Queue, size_t Len,
                       int Pattern, std::vector<RT::PiEvent> DepEvents,
                       RT::PiEvent &OutEvent);

  static void prefetch_usm(void *Ptr, QueueImplPtr Queue, size_t Len,
                           std::vector<RT::PiEvent> DepEvents,
                           RT::PiEvent &OutEvent);
};
} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
