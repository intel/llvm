//==-------------- memory_manager.cpp --------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/detail/context_impl.hpp>
#include <CL/sycl/detail/event_impl.hpp>
#include <CL/sycl/detail/memory_manager.hpp>
#include <CL/sycl/detail/queue_impl.hpp>
#include <CL/sycl/detail/usm_dispatch.hpp>

#include <algorithm>
#include <cassert>
#include <cstring>
#include <vector>

namespace cl {
namespace sycl {
namespace detail {

static void waitForEvents(const std::vector<RT::PiEvent> &Events) {
  if (!Events.empty())
    PI_CALL(RT::piEventsWait, Events.size(), &Events[0]);
}

void MemoryManager::release(ContextImplPtr TargetContext, SYCLMemObjI *MemObj,
                            void *MemAllocation,
                            std::vector<RT::PiEvent> DepEvents,
                            RT::PiEvent &OutEvent) {
  // There is no async API for memory releasing. Explicitly wait for all
  // dependency events and return empty event.
  waitForEvents(DepEvents);
  OutEvent = nullptr;
  MemObj->releaseMem(TargetContext, MemAllocation);
}

void MemoryManager::releaseMemObj(ContextImplPtr TargetContext,
                                  SYCLMemObjI *MemObj, void *MemAllocation,
                                  void *UserPtr) {
  if (UserPtr == MemAllocation) {
    // Do nothing as it's user provided memory.
    return;
  }

  if (TargetContext->is_host()) {
    MemObj->releaseHostMem(MemAllocation);
    return;
  }

  PI_CALL(RT::piMemRelease, pi::cast<RT::PiMem>(MemAllocation));
}

void *MemoryManager::allocate(ContextImplPtr TargetContext, SYCLMemObjI *MemObj,
                              bool InitFromUserData,
                              std::vector<RT::PiEvent> DepEvents,
                              RT::PiEvent &OutEvent) {
  // There is no async API for memory allocation. Explicitly wait for all
  // dependency events and return empty event.
  waitForEvents(DepEvents);
  OutEvent = nullptr;

  return MemObj->allocateMem(TargetContext, InitFromUserData, OutEvent);
}

void *MemoryManager::allocateHostMemory(SYCLMemObjI *MemObj, void *UserPtr,
                                        bool HostPtrReadOnly, size_t Size) {
  // Can return user pointer directly if it points to writable memory.
  if (UserPtr && HostPtrReadOnly == false)
    return UserPtr;

  void *NewMem = MemObj->allocateHostMem();

  // Need to initialize new memory if user provides pointer to read only
  // memory.
  if (UserPtr && HostPtrReadOnly == true)
    std::memcpy((char *)NewMem, (char *)UserPtr, Size);
  return NewMem;
}

void *MemoryManager::allocateInteropMemObject(
    ContextImplPtr TargetContext, void *UserPtr,
    const EventImplPtr &InteropEvent, const ContextImplPtr &InteropContext,
    RT::PiEvent &OutEventToWait) {
  // If memory object is created with interop c'tor.
  // Return cl_mem as is if contexts match.
  if (TargetContext == InteropContext) {
    OutEventToWait = InteropEvent->getHandleRef();
    return UserPtr;
  }
  // Allocate new cl_mem and initialize from user provided one.
  assert(!"Not implemented");
  return nullptr;
}

void *MemoryManager::allocateImageObject(ContextImplPtr TargetContext,
                                         void *UserPtr, bool HostPtrReadOnly,
                                         const RT::PiMemImageDesc &Desc,
                                         const RT::PiMemImageFormat &Format) {
  // Create read_write mem object by default to handle arbitrary uses.
  RT::PiMemFlags CreationFlags = PI_MEM_FLAGS_ACCESS_RW;
  if (UserPtr)
    CreationFlags |= HostPtrReadOnly ? PI_MEM_FLAGS_HOST_PTR_COPY
                                     : PI_MEM_FLAGS_HOST_PTR_USE;

  RT::PiMem NewMem;
  PI_CALL(RT::piMemImageCreate, TargetContext->getHandleRef(), CreationFlags,
          &Format, &Desc, UserPtr, &NewMem);
  return NewMem;
}

void *MemoryManager::allocateBufferObject(ContextImplPtr TargetContext,
                                          void *UserPtr, bool HostPtrReadOnly,
                                          const size_t Size) {
  // Create read_write mem object by default to handle arbitrary uses.
  RT::PiMemFlags CreationFlags = PI_MEM_FLAGS_ACCESS_RW;
  if (UserPtr)
    CreationFlags |= HostPtrReadOnly ? PI_MEM_FLAGS_HOST_PTR_COPY
                                     : PI_MEM_FLAGS_HOST_PTR_USE;

  RT::PiMem NewMem;
  PI_CALL(RT::piMemBufferCreate, TargetContext->getHandleRef(), CreationFlags,
          Size, UserPtr, &NewMem);
  return NewMem;
}

void *MemoryManager::allocateMemBuffer(ContextImplPtr TargetContext,
                                       SYCLMemObjI *MemObj, void *UserPtr,
                                       bool HostPtrReadOnly, size_t Size,
                                       const EventImplPtr &InteropEvent,
                                       const ContextImplPtr &InteropContext,
                                       RT::PiEvent &OutEventToWait) {
  if (TargetContext->is_host())
    return allocateHostMemory(MemObj, UserPtr, HostPtrReadOnly, Size);
  if (UserPtr && InteropContext)
    return allocateInteropMemObject(TargetContext, UserPtr, InteropEvent,
                                    InteropContext, OutEventToWait);
  return allocateBufferObject(TargetContext, UserPtr, HostPtrReadOnly, Size);
}

void *MemoryManager::allocateMemImage(
    ContextImplPtr TargetContext, SYCLMemObjI *MemObj, void *UserPtr,
    bool HostPtrReadOnly, size_t Size, const RT::PiMemImageDesc &Desc,
    const RT::PiMemImageFormat &Format, const EventImplPtr &InteropEvent,
    const ContextImplPtr &InteropContext, RT::PiEvent &OutEventToWait) {
  if (TargetContext->is_host())
    return allocateHostMemory(MemObj, UserPtr, HostPtrReadOnly, Size);
  if (UserPtr && InteropContext)
    return allocateInteropMemObject(TargetContext, UserPtr, InteropEvent,
                                    InteropContext, OutEventToWait);
  return allocateImageObject(TargetContext, UserPtr, HostPtrReadOnly, Desc,
                             Format);
}

void *MemoryManager::allocateMemSubBuffer(ContextImplPtr TargetContext,
                                          void *ParentMemObj, size_t ElemSize,
                                          size_t Offset, range<3> Range,
                                          std::vector<RT::PiEvent> DepEvents,
                                          RT::PiEvent &OutEvent) {
  waitForEvents(DepEvents);
  OutEvent = nullptr;

  if (TargetContext->is_host())
    return static_cast<void *>(static_cast<char *>(ParentMemObj) + Offset);

  size_t SizeInBytes = ElemSize;
  for (size_t I = 0; I < 3; ++I)
    SizeInBytes *= Range[I];

  RT::PiResult Error = PI_SUCCESS;
  // TODO replace with pi_buffer_region
  cl_buffer_region Region{Offset, SizeInBytes};
  RT::PiMem NewMem;
  Error = PI_CALL_RESULT(
      RT::piMemBufferPartition, pi::cast<RT::PiMem>(ParentMemObj),
      PI_MEM_FLAGS_ACCESS_RW, PI_BUFFER_CREATE_TYPE_REGION, &Region, &NewMem);
  if (Error == PI_MISALIGNED_SUB_BUFFER_OFFSET)
    throw invalid_object_error(
        "Specified offset of the sub-buffer being constructed is not a "
        "multiple of the memory base address alignment");
  return NewMem;
}

void copyH2D(SYCLMemObjI *SYCLMemObj, char *SrcMem, QueueImplPtr SrcQueue,
             unsigned int DimSrc, sycl::range<3> SrcSize,
             sycl::range<3> SrcAccessRange, sycl::id<3> SrcOffset,
             unsigned int SrcElemSize, RT::PiMem DstMem, QueueImplPtr TgtQueue,
             unsigned int DimDst, sycl::range<3> DstSize,
             sycl::range<3> DstAccessRange, sycl::id<3> DstOffset,
             unsigned int DstElemSize, std::vector<RT::PiEvent> DepEvents,
             bool UseExclusiveQueue, RT::PiEvent &OutEvent) {
  assert(SYCLMemObj && "The SYCLMemObj is nullptr");

  RT::PiQueue Queue = UseExclusiveQueue ? TgtQueue->getExclusiveQueueHandleRef()
                                        : TgtQueue->getHandleRef();
  // Adjust first dimension of copy range and offset as OpenCL expects size in
  // bytes.
  DstSize[0] *= DstElemSize;
  if (SYCLMemObj->getType() == detail::SYCLMemObjI::MemObjType::BUFFER) {
    DstOffset[0] *= DstElemSize;
    SrcOffset[0] *= SrcElemSize;
    SrcAccessRange[0] *= SrcElemSize;
    DstAccessRange[0] *= DstElemSize;
    SrcSize[0] *= SrcElemSize;

    if (1 == DimDst && 1 == DimSrc) {
      PI_CALL(RT::piEnqueueMemBufferWrite, Queue, DstMem,
              /*blocking_write=*/CL_FALSE, DstOffset[0], DstAccessRange[0],
              SrcMem + SrcOffset[0], DepEvents.size(), &DepEvents[0],
              &OutEvent);
    } else {
      size_t BufferRowPitch = (1 == DimDst) ? 0 : DstSize[0];
      size_t BufferSlicePitch = (3 == DimDst) ? DstSize[0] * DstSize[1] : 0;
      size_t HostRowPitch = (1 == DimSrc) ? 0 : SrcSize[0];
      size_t HostSlicePitch = (3 == DimSrc) ? SrcSize[0] * SrcSize[1] : 0;
      PI_CALL(RT::piEnqueueMemBufferWriteRect, Queue, DstMem,
              /*blocking_write=*/CL_FALSE, &DstOffset[0], &SrcOffset[0],
              &DstAccessRange[0], BufferRowPitch, BufferSlicePitch,
              HostRowPitch, HostSlicePitch, SrcMem, DepEvents.size(),
              &DepEvents[0], &OutEvent);
    }
  } else {
    size_t InputRowPitch = (1 == DimDst) ? 0 : DstSize[0];
    size_t InputSlicePitch = (3 == DimDst) ? DstSize[0] * DstSize[1] : 0;
    PI_CALL(RT::piEnqueueMemImageWrite, Queue, DstMem,
            /*blocking_write=*/CL_FALSE, &DstOffset[0], &DstAccessRange[0],
            InputRowPitch, InputSlicePitch, SrcMem, DepEvents.size(),
            &DepEvents[0], &OutEvent);
  }
}

void copyD2H(SYCLMemObjI *SYCLMemObj, RT::PiMem SrcMem, QueueImplPtr SrcQueue,
             unsigned int DimSrc, sycl::range<3> SrcSize,
             sycl::range<3> SrcAccessRange, sycl::id<3> SrcOffset,
             unsigned int SrcElemSize, char *DstMem, QueueImplPtr TgtQueue,
             unsigned int DimDst, sycl::range<3> DstSize,
             sycl::range<3> DstAccessRange, sycl::id<3> DstOffset,
             unsigned int DstElemSize, std::vector<RT::PiEvent> DepEvents,
             bool UseExclusiveQueue, RT::PiEvent &OutEvent) {
  assert(SYCLMemObj && "The SYCLMemObj is nullptr");

  RT::PiQueue Queue = UseExclusiveQueue ? SrcQueue->getExclusiveQueueHandleRef()
                                        : SrcQueue->getHandleRef();
  // Adjust sizes of 1 dimensions as OpenCL expects size in bytes.
  SrcSize[0] *= SrcElemSize;
  if (SYCLMemObj->getType() == detail::SYCLMemObjI::MemObjType::BUFFER) {
    DstOffset[0] *= DstElemSize;
    SrcOffset[0] *= SrcElemSize;
    SrcAccessRange[0] *= SrcElemSize;
    DstAccessRange[0] *= DstElemSize;
    DstSize[0] *= DstElemSize;

    if (1 == DimDst && 1 == DimSrc) {
      PI_CALL(RT::piEnqueueMemBufferRead, Queue, SrcMem,
              /*blocking_read=*/CL_FALSE, SrcOffset[0], SrcAccessRange[0],
              DstMem + DstOffset[0], DepEvents.size(), &DepEvents[0],
              &OutEvent);
    } else {
      size_t BufferRowPitch = (1 == DimSrc) ? 0 : SrcSize[0];
      size_t BufferSlicePitch = (3 == DimSrc) ? SrcSize[0] * SrcSize[1] : 0;

      size_t HostRowPitch = (1 == DimDst) ? 0 : DstSize[0];
      size_t HostSlicePitch = (3 == DimDst) ? DstSize[0] * DstSize[1] : 0;
      PI_CALL(RT::piEnqueueMemBufferReadRect, Queue, SrcMem,
              /*blocking_read=*/CL_FALSE, &SrcOffset[0], &DstOffset[0],
              &SrcAccessRange[0], BufferRowPitch, BufferSlicePitch,
              HostRowPitch, HostSlicePitch, DstMem, DepEvents.size(),
              &DepEvents[0], &OutEvent);
    }
  } else {
    size_t RowPitch = (1 == DimSrc) ? 0 : SrcSize[0];
    size_t SlicePitch = (3 == DimSrc) ? SrcSize[0] * SrcSize[1] : 0;
    PI_CALL(RT::piEnqueueMemImageRead, Queue, SrcMem, CL_FALSE, &SrcOffset[0],
            &SrcAccessRange[0], RowPitch, SlicePitch, DstMem, DepEvents.size(),
            &DepEvents[0], &OutEvent);
  }
}

void copyD2D(SYCLMemObjI *SYCLMemObj, RT::PiMem SrcMem, QueueImplPtr SrcQueue,
             unsigned int DimSrc, sycl::range<3> SrcSize,
             sycl::range<3> SrcAccessRange, sycl::id<3> SrcOffset,
             unsigned int SrcElemSize, RT::PiMem DstMem, QueueImplPtr TgtQueue,
             unsigned int DimDst, sycl::range<3> DstSize,
             sycl::range<3> DstAccessRange, sycl::id<3> DstOffset,
             unsigned int DstElemSize, std::vector<RT::PiEvent> DepEvents,
             bool UseExclusiveQueue, RT::PiEvent &OutEvent) {
  assert(SYCLMemObj && "The SYCLMemObj is nullptr");

  RT::PiQueue Queue = UseExclusiveQueue ? SrcQueue->getExclusiveQueueHandleRef()
                                        : SrcQueue->getHandleRef();
  if (SYCLMemObj->getType() == detail::SYCLMemObjI::MemObjType::BUFFER) {
    // Adjust sizes of 1 dimensions as OpenCL expects size in bytes.
    DstOffset[0] *= DstElemSize;
    SrcOffset[0] *= SrcElemSize;
    SrcAccessRange[0] *= SrcElemSize;
    SrcSize[0] *= SrcElemSize;
    DstSize[0] *= DstElemSize;

    if (1 == DimDst && 1 == DimSrc) {
      PI_CALL(RT::piEnqueueMemBufferCopy, Queue, SrcMem, DstMem, SrcOffset[0],
              DstOffset[0], SrcAccessRange[0], DepEvents.size(), &DepEvents[0],
              &OutEvent);
    } else {
      size_t SrcRowPitch = (1 == DimSrc) ? 0 : SrcSize[0];
      size_t SrcSlicePitch = (3 == DimSrc) ? SrcSize[0] * SrcSize[1] : 0;

      size_t DstRowPitch = (1 == DimDst) ? 0 : DstSize[0];
      size_t DstSlicePitch = (3 == DimDst) ? DstSize[0] * DstSize[1] : 0;

      PI_CALL(RT::piEnqueueMemBufferCopyRect, Queue, SrcMem, DstMem,
              &SrcOffset[0], &DstOffset[0], &SrcAccessRange[0], SrcRowPitch,
              SrcSlicePitch, DstRowPitch, DstSlicePitch, DepEvents.size(),
              &DepEvents[0], &OutEvent);
    }
  } else {
    PI_CALL(RT::piEnqueueMemImageCopy, Queue, SrcMem, DstMem, &SrcOffset[0],
            &DstOffset[0], &SrcAccessRange[0], DepEvents.size(), &DepEvents[0],
            &OutEvent);
  }
}

static void copyH2H(SYCLMemObjI *SYCLMemObj, char *SrcMem,
                    QueueImplPtr SrcQueue, unsigned int DimSrc,
                    sycl::range<3> SrcSize, sycl::range<3> SrcAccessRange,
                    sycl::id<3> SrcOffset, unsigned int SrcElemSize,
                    char *DstMem, QueueImplPtr TgtQueue, unsigned int DimDst,
                    sycl::range<3> DstSize, sycl::range<3> DstAccessRange,
                    sycl::id<3> DstOffset, unsigned int DstElemSize,
                    std::vector<RT::PiEvent> DepEvents, bool UseExclusiveQueue,
                    RT::PiEvent &OutEvent) {
  if ((DimSrc != 1 || DimDst != 1) &&
      (SrcOffset != id<3>{0, 0, 0} || DstOffset != id<3>{0, 0, 0} ||
       SrcSize != SrcAccessRange || DstSize != DstAccessRange)) {
    assert(!"Not supported configuration of memcpy requested");
    throw runtime_error("Not supported configuration of memcpy requested");
  }

  DstOffset[0] *= DstElemSize;
  SrcOffset[0] *= SrcElemSize;

  size_t BytesToCopy =
      SrcAccessRange[0] * SrcElemSize * SrcAccessRange[1] * SrcAccessRange[2];

  std::memcpy(DstMem + DstOffset[0], SrcMem + SrcOffset[0], BytesToCopy);
}

// Copies memory between: host and device, host and host,
// device and device if memory objects bound to the one context.
void MemoryManager::copy(SYCLMemObjI *SYCLMemObj, void *SrcMem,
                         QueueImplPtr SrcQueue, unsigned int DimSrc,
                         sycl::range<3> SrcSize, sycl::range<3> SrcAccessRange,
                         sycl::id<3> SrcOffset, unsigned int SrcElemSize,
                         void *DstMem, QueueImplPtr TgtQueue,
                         unsigned int DimDst, sycl::range<3> DstSize,
                         sycl::range<3> DstAccessRange, sycl::id<3> DstOffset,
                         unsigned int DstElemSize,
                         std::vector<RT::PiEvent> DepEvents,
                         bool UseExclusiveQueue, RT::PiEvent &OutEvent) {

  if (SrcQueue->is_host()) {
    if (TgtQueue->is_host())
      copyH2H(SYCLMemObj, (char *)SrcMem, std::move(SrcQueue), DimSrc, SrcSize,
              SrcAccessRange, SrcOffset, SrcElemSize, (char *)DstMem,
              std::move(TgtQueue), DimDst, DstSize, DstAccessRange, DstOffset,
              DstElemSize, std::move(DepEvents), UseExclusiveQueue, OutEvent);

    else
      copyH2D(SYCLMemObj, (char *)SrcMem, std::move(SrcQueue), DimSrc, SrcSize,
              SrcAccessRange, SrcOffset, SrcElemSize,
              pi::cast<RT::PiMem>(DstMem), std::move(TgtQueue), DimDst, DstSize,
              DstAccessRange, DstOffset, DstElemSize, std::move(DepEvents),
              UseExclusiveQueue, OutEvent);
  } else {
    if (TgtQueue->is_host())
      copyD2H(SYCLMemObj, pi::cast<RT::PiMem>(SrcMem), std::move(SrcQueue),
              DimSrc, SrcSize, SrcAccessRange, SrcOffset, SrcElemSize,
              (char *)DstMem, std::move(TgtQueue), DimDst, DstSize,
              DstAccessRange, DstOffset, DstElemSize, std::move(DepEvents),
              UseExclusiveQueue, OutEvent);
    else
      copyD2D(SYCLMemObj, pi::cast<RT::PiMem>(SrcMem), std::move(SrcQueue),
              DimSrc, SrcSize, SrcAccessRange, SrcOffset, SrcElemSize,
              pi::cast<RT::PiMem>(DstMem), std::move(TgtQueue), DimDst, DstSize,
              DstAccessRange, DstOffset, DstElemSize, std::move(DepEvents),
              UseExclusiveQueue, OutEvent);
  }
}

void MemoryManager::fill(SYCLMemObjI *SYCLMemObj, void *Mem, QueueImplPtr Queue,
                         size_t PatternSize, const char *Pattern,
                         unsigned int Dim, sycl::range<3> Size,
                         sycl::range<3> Range, sycl::id<3> Offset,
                         unsigned int ElementSize,
                         std::vector<RT::PiEvent> DepEvents,
                         RT::PiEvent &OutEvent) {
  assert(SYCLMemObj && "The SYCLMemObj is nullptr");

  if (SYCLMemObj->getType() == detail::SYCLMemObjI::MemObjType::BUFFER) {
    if (Dim == 1) {
      PI_CALL(RT::piEnqueueMemBufferFill, Queue->getHandleRef(),
              pi::cast<RT::PiMem>(Mem), Pattern, PatternSize,
              Offset[0] * ElementSize, Range[0] * ElementSize, DepEvents.size(),
              &DepEvents[0], &OutEvent);
      return;
    }
    assert(!"Not supported configuration of fill requested");
    throw runtime_error("Not supported configuration of fill requested");
  } else {
    PI_CALL(RT::piEnqueueMemImageFill, Queue->getHandleRef(),
            pi::cast<RT::PiMem>(Mem), Pattern, &Offset[0], &Range[0],
            DepEvents.size(), &DepEvents[0], &OutEvent);
  }
}

void *MemoryManager::map(SYCLMemObjI *SYCLMemObj, void *Mem, QueueImplPtr Queue,
                         access::mode AccessMode, unsigned int Dim,
                         sycl::range<3> Size, sycl::range<3> AccessRange,
                         sycl::id<3> AccessOffset, unsigned int ElementSize,
                         std::vector<RT::PiEvent> DepEvents,
                         RT::PiEvent &OutEvent) {
  if (Queue->is_host() || Dim != 1) {
    assert(!"Not supported configuration of map requested");
    throw runtime_error("Not supported configuration of map requested");
  }

  cl_map_flags Flags = 0;

  switch (AccessMode) {
  case access::mode::read:
    Flags |= CL_MAP_READ;
    break;
  case access::mode::write:
    Flags |= CL_MAP_WRITE;
    break;
  case access::mode::read_write:
  case access::mode::atomic:
    Flags = CL_MAP_WRITE | CL_MAP_READ;
    break;
  case access::mode::discard_write:
  case access::mode::discard_read_write:
    Flags |= CL_MAP_WRITE_INVALIDATE_REGION;
    break;
  }

  AccessOffset[0] *= ElementSize;
  AccessRange[0] *= ElementSize;

  void *MappedPtr;
  PI_CALL(RT::piEnqueueMemBufferMap, Queue->getHandleRef(),
          pi::cast<RT::PiMem>(Mem), CL_FALSE, Flags, AccessOffset[0],
          AccessRange[0], DepEvents.size(),
          DepEvents.empty() ? nullptr : &DepEvents[0], &OutEvent, &MappedPtr);
  return MappedPtr;
}

void MemoryManager::unmap(SYCLMemObjI *SYCLMemObj, void *Mem,
                          QueueImplPtr Queue, void *MappedPtr,
                          std::vector<RT::PiEvent> DepEvents,
                          bool UseExclusiveQueue, RT::PiEvent &OutEvent) {

  PI_CALL(RT::piEnqueueMemUnmap,
          UseExclusiveQueue ? Queue->getExclusiveQueueHandleRef()
                            : Queue->getHandleRef(),
          pi::cast<RT::PiMem>(Mem), MappedPtr, DepEvents.size(),
          DepEvents.empty() ? nullptr : &DepEvents[0], &OutEvent);
}

void MemoryManager::copy_usm(const void *SrcMem, QueueImplPtr SrcQueue,
                             size_t Len, void *DstMem,
                             std::vector<RT::PiEvent> DepEvents,
                             bool UseExclusiveQueue, RT::PiEvent &OutEvent) {
  sycl::context Context = SrcQueue->get_context();

  if (Context.is_host()) {
    std::memcpy(DstMem, SrcMem, Len);
  } else {
    RT::PiQueue Queue = UseExclusiveQueue
                            ? SrcQueue->getExclusiveQueueHandleRef()
                            : SrcQueue->getHandleRef();

    std::shared_ptr<usm::USMDispatcher> USMDispatch =
        getSyclObjImpl(Context)->getUSMDispatch();
    RT::piCheckResult(USMDispatch->enqueueMemcpy(Queue,
                                                 /* blocking */ false, DstMem,
                                                 SrcMem, Len, DepEvents.size(),
                                                 &DepEvents[0], &OutEvent));
  }
}

void MemoryManager::fill_usm(void *Mem, QueueImplPtr Queue, size_t Length,
                             int Pattern, std::vector<RT::PiEvent> DepEvents,
                             RT::PiEvent &OutEvent) {
  sycl::context Context = Queue->get_context();

  if (Context.is_host()) {
    std::memset(Mem, Pattern, Length);
  } else {
    std::shared_ptr<usm::USMDispatcher> USMDispatch =
        getSyclObjImpl(Context)->getUSMDispatch();
    RT::piCheckResult(
        USMDispatch->enqueueMemset(Queue->getHandleRef(), Mem, Pattern, Length,
                                   DepEvents.size(), &DepEvents[0], &OutEvent));
  }
}

void MemoryManager::prefetch_usm(void *Mem, QueueImplPtr Queue, size_t Length,
                                 std::vector<RT::PiEvent> DepEvents,
                                 RT::PiEvent &OutEvent) {
  sycl::context Context = Queue->get_context();

  if (Context.is_host()) {
    // TODO: Potentially implement prefetch on the host.
  } else {
    std::shared_ptr<usm::USMDispatcher> USMDispatch =
        getSyclObjImpl(Context)->getUSMDispatch();
    RT::piCheckResult(USMDispatch->enqueuePrefetch(Queue->getHandleRef(), Mem,
                                                   Length, DepEvents.size(),
                                                   &DepEvents[0], &OutEvent));
  }
}

} // namespace detail
} // namespace sycl
} // namespace cl
