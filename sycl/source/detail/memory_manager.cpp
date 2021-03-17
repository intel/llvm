//==-------------- memory_manager.cpp --------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/detail/memory_manager.hpp>
#include <detail/context_impl.hpp>
#include <detail/event_impl.hpp>
#include <detail/queue_impl.hpp>

#include <algorithm>
#include <cassert>
#include <cstring>
#include <vector>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

static void waitForEvents(const std::vector<EventImplPtr> &Events) {
  // Assuming all events will be on the same device or
  // devices associated with the same Backend.
  if (!Events.empty()) {
    const detail::plugin &Plugin = Events[0]->getPlugin();
    std::vector<RT::PiEvent> PiEvents(Events.size());
    std::transform(Events.begin(), Events.end(), PiEvents.begin(),
                   [](const EventImplPtr &EventImpl) {
                     return EventImpl->getHandleRef();
                   });
    Plugin.call<PiApiKind::piEventsWait>(PiEvents.size(), &PiEvents[0]);
  }
}

void MemoryManager::release(ContextImplPtr TargetContext, SYCLMemObjI *MemObj,
                            void *MemAllocation,
                            std::vector<EventImplPtr> DepEvents,
                            RT::PiEvent &OutEvent) {
  // There is no async API for memory releasing. Explicitly wait for all
  // dependency events and return empty event.
  waitForEvents(DepEvents);
  OutEvent = nullptr;
  MemObj->releaseMem(TargetContext, MemAllocation);
}

void MemoryManager::releaseImageBuffer(ContextImplPtr TargetContext,
                                       void *ImageBuf) {
  (void)TargetContext;
  (void)ImageBuf;
  // TODO remove when ABI breaking changes are allowed.
  throw runtime_error("Deprecated release operation", PI_INVALID_OPERATION);
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

  const detail::plugin &Plugin = TargetContext->getPlugin();
  Plugin.call<PiApiKind::piMemRelease>(pi::cast<RT::PiMem>(MemAllocation));
}

void *MemoryManager::allocate(ContextImplPtr TargetContext, SYCLMemObjI *MemObj,
                              bool InitFromUserData, void *HostPtr,
                              std::vector<EventImplPtr> DepEvents,
                              RT::PiEvent &OutEvent) {
  // There is no async API for memory allocation. Explicitly wait for all
  // dependency events and return empty event.
  waitForEvents(DepEvents);
  OutEvent = nullptr;

  return MemObj->allocateMem(TargetContext, InitFromUserData, HostPtr,
                             OutEvent);
}

void *MemoryManager::wrapIntoImageBuffer(ContextImplPtr TargetContext,
                                         void *MemBuf, SYCLMemObjI *MemObj) {
  (void)TargetContext;
  (void)MemBuf;
  (void)MemObj;
  // TODO remove when ABI breaking changes are allowed.
  throw runtime_error("Deprecated allocation operation", PI_INVALID_OPERATION);
}

void *MemoryManager::allocateHostMemory(SYCLMemObjI *MemObj, void *UserPtr,
                                        bool HostPtrReadOnly, size_t Size,
                                        const sycl::property_list &) {
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
    const sycl::property_list &, RT::PiEvent &OutEventToWait) {
  (void)TargetContext;
  (void)InteropContext;
  // If memory object is created with interop c'tor return cl_mem as is.
  assert(TargetContext == InteropContext && "Expected matching contexts");
  OutEventToWait = InteropEvent->getHandleRef();
  // Retain the event since it will be released during alloca command
  // destruction
  if (nullptr != OutEventToWait) {
    const detail::plugin &Plugin = InteropEvent->getPlugin();
    Plugin.call<PiApiKind::piEventRetain>(OutEventToWait);
  }
  return UserPtr;
}

static RT::PiMemFlags getMemObjCreationFlags(void *UserPtr,
                                             bool HostPtrReadOnly) {
  // Create read_write mem object to handle arbitrary uses.
  RT::PiMemFlags Result = PI_MEM_FLAGS_ACCESS_RW;
  if (UserPtr)
    Result |= HostPtrReadOnly ? PI_MEM_FLAGS_HOST_PTR_COPY
                              : PI_MEM_FLAGS_HOST_PTR_USE;
  return Result;
}

void *MemoryManager::allocateImageObject(ContextImplPtr TargetContext,
                                         void *UserPtr, bool HostPtrReadOnly,
                                         const RT::PiMemImageDesc &Desc,
                                         const RT::PiMemImageFormat &Format,
                                         const sycl::property_list &) {
  RT::PiMemFlags CreationFlags =
      getMemObjCreationFlags(UserPtr, HostPtrReadOnly);

  RT::PiMem NewMem;
  const detail::plugin &Plugin = TargetContext->getPlugin();
  Plugin.call<PiApiKind::piMemImageCreate>(TargetContext->getHandleRef(),
                                           CreationFlags, &Format, &Desc,
                                           UserPtr, &NewMem);
  return NewMem;
}

void *
MemoryManager::allocateBufferObject(ContextImplPtr TargetContext, void *UserPtr,
                                    bool HostPtrReadOnly, const size_t Size,
                                    const sycl::property_list &PropsList) {
  RT::PiMemFlags CreationFlags =
      getMemObjCreationFlags(UserPtr, HostPtrReadOnly);
  if (PropsList.has_property<
          sycl::ext::oneapi::property::buffer::use_pinned_host_memory>())
    CreationFlags |= PI_MEM_FLAGS_HOST_PTR_ALLOC;

  RT::PiMem NewMem = nullptr;
  const detail::plugin &Plugin = TargetContext->getPlugin();
  Plugin.call<PiApiKind::piMemBufferCreate>(TargetContext->getHandleRef(),
                                            CreationFlags, Size, UserPtr,
                                            &NewMem, nullptr);
  return NewMem;
}

void *MemoryManager::allocateMemBuffer(ContextImplPtr TargetContext,
                                       SYCLMemObjI *MemObj, void *UserPtr,
                                       bool HostPtrReadOnly, size_t Size,
                                       const EventImplPtr &InteropEvent,
                                       const ContextImplPtr &InteropContext,
                                       const sycl::property_list &PropsList,
                                       RT::PiEvent &OutEventToWait) {
  if (TargetContext->is_host())
    return allocateHostMemory(MemObj, UserPtr, HostPtrReadOnly, Size,
                              PropsList);
  if (UserPtr && InteropContext)
    return allocateInteropMemObject(TargetContext, UserPtr, InteropEvent,
                                    InteropContext, PropsList, OutEventToWait);
  return allocateBufferObject(TargetContext, UserPtr, HostPtrReadOnly, Size,
                              PropsList);
}

void *MemoryManager::allocateMemImage(
    ContextImplPtr TargetContext, SYCLMemObjI *MemObj, void *UserPtr,
    bool HostPtrReadOnly, size_t Size, const RT::PiMemImageDesc &Desc,
    const RT::PiMemImageFormat &Format, const EventImplPtr &InteropEvent,
    const ContextImplPtr &InteropContext, const sycl::property_list &PropsList,
    RT::PiEvent &OutEventToWait) {
  if (TargetContext->is_host())
    return allocateHostMemory(MemObj, UserPtr, HostPtrReadOnly, Size,
                              PropsList);
  if (UserPtr && InteropContext)
    return allocateInteropMemObject(TargetContext, UserPtr, InteropEvent,
                                    InteropContext, PropsList, OutEventToWait);
  return allocateImageObject(TargetContext, UserPtr, HostPtrReadOnly, Desc,
                             Format, PropsList);
}

void *MemoryManager::allocateMemSubBuffer(ContextImplPtr TargetContext,
                                          void *ParentMemObj, size_t ElemSize,
                                          size_t Offset, range<3> Range,
                                          std::vector<EventImplPtr> DepEvents,
                                          RT::PiEvent &OutEvent) {
  waitForEvents(DepEvents);
  OutEvent = nullptr;

  if (TargetContext->is_host())
    return static_cast<void *>(static_cast<char *>(ParentMemObj) + Offset);

  size_t SizeInBytes = ElemSize;
  for (size_t I = 0; I < 3; ++I)
    SizeInBytes *= Range[I];

  RT::PiResult Error = PI_SUCCESS;
  pi_buffer_region_struct Region{Offset, SizeInBytes};
  RT::PiMem NewMem;
  const detail::plugin &Plugin = TargetContext->getPlugin();
  Error = Plugin.call_nocheck<PiApiKind::piMemBufferPartition>(
      pi::cast<RT::PiMem>(ParentMemObj), PI_MEM_FLAGS_ACCESS_RW,
      PI_BUFFER_CREATE_TYPE_REGION, &Region, &NewMem);
  if (Error == PI_MISALIGNED_SUB_BUFFER_OFFSET)
    throw invalid_object_error(
        "Specified offset of the sub-buffer being constructed is not a "
        "multiple of the memory base address alignment",
        PI_INVALID_VALUE);
  return NewMem;
}

struct TermPositions {
  int XTerm;
  int YTerm;
  int ZTerm;
};
void prepTermPositions(TermPositions &pos, int Dimensions,
                       detail::SYCLMemObjI::MemObjType Type) {
  // For buffers, the offsets/ranges coming from accessor are always
  // id<3>/range<3> But their organization varies by dimension:
  //  1 ==>  {width, 1, 1}
  //  2 ==>  {height, width, 1}
  //  3 ==>  {depth, height, width}
  // Some callers schedule 0 as DimDst/DimSrc.

  if (Type == detail::SYCLMemObjI::MemObjType::BUFFER) {
    if (Dimensions == 3) {
      pos.XTerm = 2, pos.YTerm = 1, pos.ZTerm = 0;
    } else if (Dimensions == 2) {
      pos.XTerm = 1, pos.YTerm = 0, pos.ZTerm = 2;
    } else { // Dimension is 1 or 0
      pos.XTerm = 0, pos.YTerm = 1, pos.ZTerm = 2;
    }
  } else { // While range<>/id<> use by images is different than buffers, it's
           // consistent with their accessors.
    pos.XTerm = 0;
    pos.YTerm = 1;
    pos.ZTerm = 2;
  }
}

void copyH2D(SYCLMemObjI *SYCLMemObj, char *SrcMem, QueueImplPtr,
             unsigned int DimSrc, sycl::range<3> SrcSize,
             sycl::range<3> SrcAccessRange, sycl::id<3> SrcOffset,
             unsigned int SrcElemSize, RT::PiMem DstMem, QueueImplPtr TgtQueue,
             unsigned int DimDst, sycl::range<3> DstSize,
             sycl::range<3> DstAccessRange, sycl::id<3> DstOffset,
             unsigned int DstElemSize, std::vector<RT::PiEvent> DepEvents,
             RT::PiEvent &OutEvent) {
  (void)SrcAccessRange;
  assert(SYCLMemObj && "The SYCLMemObj is nullptr");

  const RT::PiQueue Queue = TgtQueue->getHandleRef();
  const detail::plugin &Plugin = TgtQueue->getPlugin();

  detail::SYCLMemObjI::MemObjType MemType = SYCLMemObj->getType();
  TermPositions SrcPos, DstPos;
  prepTermPositions(SrcPos, DimSrc, MemType);
  prepTermPositions(DstPos, DimDst, MemType);

  size_t DstXOffBytes = DstOffset[DstPos.XTerm] * DstElemSize;
  size_t SrcXOffBytes = SrcOffset[SrcPos.XTerm] * SrcElemSize;
  size_t DstAccessRangeWidthBytes = DstAccessRange[DstPos.XTerm] * DstElemSize;
  size_t DstSzWidthBytes = DstSize[DstPos.XTerm] * DstElemSize;
  size_t SrcSzWidthBytes = SrcSize[SrcPos.XTerm] * SrcElemSize;

  if (MemType == detail::SYCLMemObjI::MemObjType::BUFFER) {
    if (1 == DimDst && 1 == DimSrc) {
      Plugin.call<PiApiKind::piEnqueueMemBufferWrite>(
          Queue, DstMem,
          /*blocking_write=*/CL_FALSE, DstXOffBytes, DstAccessRangeWidthBytes,
          SrcMem + SrcXOffBytes, DepEvents.size(), DepEvents.data(), &OutEvent);
    } else {
      size_t BufferRowPitch = (1 == DimDst) ? 0 : DstSzWidthBytes;
      size_t BufferSlicePitch =
          (3 == DimDst) ? DstSzWidthBytes * DstSize[DstPos.YTerm] : 0;
      size_t HostRowPitch = (1 == DimSrc) ? 0 : SrcSzWidthBytes;
      size_t HostSlicePitch =
          (3 == DimSrc) ? SrcSzWidthBytes * SrcSize[SrcPos.YTerm] : 0;

      pi_buff_rect_offset_struct BufferOffset{
          DstXOffBytes, DstOffset[DstPos.YTerm], DstOffset[DstPos.ZTerm]};
      pi_buff_rect_offset_struct HostOffset{
          SrcXOffBytes, SrcOffset[SrcPos.YTerm], SrcOffset[SrcPos.ZTerm]};
      pi_buff_rect_region_struct RectRegion{DstAccessRangeWidthBytes,
                                            DstAccessRange[DstPos.YTerm],
                                            DstAccessRange[DstPos.ZTerm]};

      Plugin.call<PiApiKind::piEnqueueMemBufferWriteRect>(
          Queue, DstMem,
          /*blocking_write=*/CL_FALSE, &BufferOffset, &HostOffset, &RectRegion,
          BufferRowPitch, BufferSlicePitch, HostRowPitch, HostSlicePitch,
          SrcMem, DepEvents.size(), DepEvents.data(), &OutEvent);
    }
  } else {
    size_t InputRowPitch = (1 == DimDst) ? 0 : DstSzWidthBytes;
    size_t InputSlicePitch =
        (3 == DimDst) ? DstSzWidthBytes * DstSize[DstPos.YTerm] : 0;

    pi_image_offset_struct Origin{DstOffset[DstPos.XTerm],
                                  DstOffset[DstPos.YTerm],
                                  DstOffset[DstPos.ZTerm]};
    pi_image_region_struct Region{DstAccessRange[DstPos.XTerm],
                                  DstAccessRange[DstPos.YTerm],
                                  DstAccessRange[DstPos.ZTerm]};

    Plugin.call<PiApiKind::piEnqueueMemImageWrite>(
        Queue, DstMem,
        /*blocking_write=*/CL_FALSE, &Origin, &Region, InputRowPitch,
        InputSlicePitch, SrcMem, DepEvents.size(), DepEvents.data(), &OutEvent);
  }
}

void copyD2H(SYCLMemObjI *SYCLMemObj, RT::PiMem SrcMem, QueueImplPtr SrcQueue,
             unsigned int DimSrc, sycl::range<3> SrcSize,
             sycl::range<3> SrcAccessRange, sycl::id<3> SrcOffset,
             unsigned int SrcElemSize, char *DstMem, QueueImplPtr,
             unsigned int DimDst, sycl::range<3> DstSize,
             sycl::range<3> DstAccessRange, sycl::id<3> DstOffset,
             unsigned int DstElemSize, std::vector<RT::PiEvent> DepEvents,
             RT::PiEvent &OutEvent) {
  (void)DstAccessRange;
  assert(SYCLMemObj && "The SYCLMemObj is nullptr");

  const RT::PiQueue Queue = SrcQueue->getHandleRef();
  const detail::plugin &Plugin = SrcQueue->getPlugin();

  detail::SYCLMemObjI::MemObjType MemType = SYCLMemObj->getType();
  TermPositions SrcPos, DstPos;
  prepTermPositions(SrcPos, DimSrc, MemType);
  prepTermPositions(DstPos, DimDst, MemType);

  //  For a given buffer, the various mem copy routines (copyD2H, copyH2D,
  //  copyD2D) will usually have the same values for AccessRange, Size,
  //  Dimension, Offset, etc. EXCEPT when the dtor for ~SYCLMemObjT is called.
  //  Essentially, it schedules a copyBack of chars thus in copyD2H the
  //  Dimension will then be 1 and DstAccessRange[0] and DstSize[0] will be
  //  sized to bytes with a DstElemSize of 1.
  size_t DstXOffBytes = DstOffset[DstPos.XTerm] * DstElemSize;
  size_t SrcXOffBytes = SrcOffset[SrcPos.XTerm] * SrcElemSize;
  size_t SrcAccessRangeWidthBytes = SrcAccessRange[SrcPos.XTerm] * SrcElemSize;
  size_t DstSzWidthBytes = DstSize[DstPos.XTerm] * DstElemSize;
  size_t SrcSzWidthBytes = SrcSize[SrcPos.XTerm] * SrcElemSize;

  if (MemType == detail::SYCLMemObjI::MemObjType::BUFFER) {
    if (1 == DimDst && 1 == DimSrc) {
      Plugin.call<PiApiKind::piEnqueueMemBufferRead>(
          Queue, SrcMem,
          /*blocking_read=*/CL_FALSE, SrcXOffBytes, SrcAccessRangeWidthBytes,
          DstMem + DstXOffBytes, DepEvents.size(), DepEvents.data(), &OutEvent);
    } else {
      size_t BufferRowPitch = (1 == DimSrc) ? 0 : SrcSzWidthBytes;
      size_t BufferSlicePitch =
          (3 == DimSrc) ? SrcSzWidthBytes * SrcSize[SrcPos.YTerm] : 0;
      size_t HostRowPitch = (1 == DimDst) ? 0 : DstSzWidthBytes;
      size_t HostSlicePitch =
          (3 == DimDst) ? DstSzWidthBytes * DstSize[DstPos.YTerm] : 0;

      pi_buff_rect_offset_struct BufferOffset{
          SrcXOffBytes, SrcOffset[SrcPos.YTerm], SrcOffset[SrcPos.ZTerm]};
      pi_buff_rect_offset_struct HostOffset{
          DstXOffBytes, DstOffset[DstPos.YTerm], DstOffset[DstPos.ZTerm]};
      pi_buff_rect_region_struct RectRegion{SrcAccessRangeWidthBytes,
                                            SrcAccessRange[SrcPos.YTerm],
                                            SrcAccessRange[SrcPos.ZTerm]};

      Plugin.call<PiApiKind::piEnqueueMemBufferReadRect>(
          Queue, SrcMem,
          /*blocking_read=*/CL_FALSE, &BufferOffset, &HostOffset, &RectRegion,
          BufferRowPitch, BufferSlicePitch, HostRowPitch, HostSlicePitch,
          DstMem, DepEvents.size(), DepEvents.data(), &OutEvent);
    }
  } else {
    size_t RowPitch = (1 == DimSrc) ? 0 : SrcSzWidthBytes;
    size_t SlicePitch =
        (3 == DimSrc) ? SrcSzWidthBytes * SrcSize[SrcPos.YTerm] : 0;

    pi_image_offset_struct Offset{SrcOffset[SrcPos.XTerm],
                                  SrcOffset[SrcPos.YTerm],
                                  SrcOffset[SrcPos.ZTerm]};
    pi_image_region_struct Region{SrcAccessRange[SrcPos.XTerm],
                                  SrcAccessRange[SrcPos.YTerm],
                                  SrcAccessRange[SrcPos.ZTerm]};

    Plugin.call<PiApiKind::piEnqueueMemImageRead>(
        Queue, SrcMem, CL_FALSE, &Offset, &Region, RowPitch, SlicePitch, DstMem,
        DepEvents.size(), DepEvents.data(), &OutEvent);
  }
}

void copyD2D(SYCLMemObjI *SYCLMemObj, RT::PiMem SrcMem, QueueImplPtr SrcQueue,
             unsigned int DimSrc, sycl::range<3> SrcSize,
             sycl::range<3> SrcAccessRange, sycl::id<3> SrcOffset,
             unsigned int SrcElemSize, RT::PiMem DstMem, QueueImplPtr,
             unsigned int DimDst, sycl::range<3> DstSize, sycl::range<3>,
             sycl::id<3> DstOffset, unsigned int DstElemSize,
             std::vector<RT::PiEvent> DepEvents, RT::PiEvent &OutEvent) {
  assert(SYCLMemObj && "The SYCLMemObj is nullptr");

  const RT::PiQueue Queue = SrcQueue->getHandleRef();
  const detail::plugin &Plugin = SrcQueue->getPlugin();

  detail::SYCLMemObjI::MemObjType MemType = SYCLMemObj->getType();
  TermPositions SrcPos, DstPos;
  prepTermPositions(SrcPos, DimSrc, MemType);
  prepTermPositions(DstPos, DimDst, MemType);

  size_t DstXOffBytes = DstOffset[DstPos.XTerm] * DstElemSize;
  size_t SrcXOffBytes = SrcOffset[SrcPos.XTerm] * SrcElemSize;
  size_t SrcAccessRangeWidthBytes = SrcAccessRange[SrcPos.XTerm] * SrcElemSize;
  size_t DstSzWidthBytes = DstSize[DstPos.XTerm] * DstElemSize;
  size_t SrcSzWidthBytes = SrcSize[SrcPos.XTerm] * SrcElemSize;

  if (MemType == detail::SYCLMemObjI::MemObjType::BUFFER) {
    if (1 == DimDst && 1 == DimSrc) {
      Plugin.call<PiApiKind::piEnqueueMemBufferCopy>(
          Queue, SrcMem, DstMem, SrcXOffBytes, DstXOffBytes,
          SrcAccessRangeWidthBytes, DepEvents.size(), DepEvents.data(),
          &OutEvent);
    } else {
      // passing 0 for pitches not allowed. Because clEnqueueCopyBufferRect will
      // calculate both src and dest pitch using region[0], which is not correct
      // if src and dest are not the same size.
      size_t SrcRowPitch = SrcSzWidthBytes;
      size_t SrcSlicePitch = (DimSrc <= 1)
                                 ? SrcSzWidthBytes
                                 : SrcSzWidthBytes * SrcSize[SrcPos.YTerm];
      size_t DstRowPitch = DstSzWidthBytes;
      size_t DstSlicePitch = (DimDst <= 1)
                                 ? DstSzWidthBytes
                                 : DstSzWidthBytes * DstSize[DstPos.YTerm];

      pi_buff_rect_offset_struct SrcOrigin{
          SrcXOffBytes, SrcOffset[SrcPos.YTerm], SrcOffset[SrcPos.ZTerm]};
      pi_buff_rect_offset_struct DstOrigin{
          DstXOffBytes, DstOffset[DstPos.YTerm], DstOffset[DstPos.ZTerm]};
      pi_buff_rect_region_struct Region{SrcAccessRangeWidthBytes,
                                        SrcAccessRange[SrcPos.YTerm],
                                        SrcAccessRange[SrcPos.ZTerm]};

      Plugin.call<PiApiKind::piEnqueueMemBufferCopyRect>(
          Queue, SrcMem, DstMem, &SrcOrigin, &DstOrigin, &Region, SrcRowPitch,
          SrcSlicePitch, DstRowPitch, DstSlicePitch, DepEvents.size(),
          DepEvents.data(), &OutEvent);
    }
  } else {
    pi_image_offset_struct SrcOrigin{SrcOffset[SrcPos.XTerm],
                                     SrcOffset[SrcPos.YTerm],
                                     SrcOffset[SrcPos.ZTerm]};
    pi_image_offset_struct DstOrigin{DstOffset[DstPos.XTerm],
                                     DstOffset[DstPos.YTerm],
                                     DstOffset[DstPos.ZTerm]};
    pi_image_region_struct Region{SrcAccessRange[SrcPos.XTerm],
                                  SrcAccessRange[SrcPos.YTerm],
                                  SrcAccessRange[SrcPos.ZTerm]};

    Plugin.call<PiApiKind::piEnqueueMemImageCopy>(
        Queue, SrcMem, DstMem, &SrcOrigin, &DstOrigin, &Region,
        DepEvents.size(), DepEvents.data(), &OutEvent);
  }
}

static void copyH2H(SYCLMemObjI *, char *SrcMem, QueueImplPtr,
                    unsigned int DimSrc, sycl::range<3> SrcSize,
                    sycl::range<3> SrcAccessRange, sycl::id<3> SrcOffset,
                    unsigned int SrcElemSize, char *DstMem, QueueImplPtr,
                    unsigned int DimDst, sycl::range<3> DstSize,
                    sycl::range<3> DstAccessRange, sycl::id<3> DstOffset,
                    unsigned int DstElemSize, std::vector<RT::PiEvent>,
                    RT::PiEvent &) {
  if ((DimSrc != 1 || DimDst != 1) &&
      (SrcOffset != id<3>{0, 0, 0} || DstOffset != id<3>{0, 0, 0} ||
       SrcSize != SrcAccessRange || DstSize != DstAccessRange)) {
    throw runtime_error("Not supported configuration of memcpy requested",
                        PI_INVALID_OPERATION);
  }

  SrcMem += SrcOffset[0] * SrcElemSize;
  DstMem += DstOffset[0] * DstElemSize;

  if (SrcMem == DstMem)
    return;

  size_t BytesToCopy =
      SrcAccessRange[0] * SrcElemSize * SrcAccessRange[1] * SrcAccessRange[2];
  std::memcpy(DstMem, SrcMem, BytesToCopy);
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
                         RT::PiEvent &OutEvent) {

  if (SrcQueue->is_host()) {
    if (TgtQueue->is_host())
      copyH2H(SYCLMemObj, (char *)SrcMem, std::move(SrcQueue), DimSrc, SrcSize,
              SrcAccessRange, SrcOffset, SrcElemSize, (char *)DstMem,
              std::move(TgtQueue), DimDst, DstSize, DstAccessRange, DstOffset,
              DstElemSize, std::move(DepEvents), OutEvent);

    else
      copyH2D(SYCLMemObj, (char *)SrcMem, std::move(SrcQueue), DimSrc, SrcSize,
              SrcAccessRange, SrcOffset, SrcElemSize,
              pi::cast<RT::PiMem>(DstMem), std::move(TgtQueue), DimDst, DstSize,
              DstAccessRange, DstOffset, DstElemSize, std::move(DepEvents),
              OutEvent);
  } else {
    if (TgtQueue->is_host())
      copyD2H(SYCLMemObj, pi::cast<RT::PiMem>(SrcMem), std::move(SrcQueue),
              DimSrc, SrcSize, SrcAccessRange, SrcOffset, SrcElemSize,
              (char *)DstMem, std::move(TgtQueue), DimDst, DstSize,
              DstAccessRange, DstOffset, DstElemSize, std::move(DepEvents),
              OutEvent);
    else
      copyD2D(SYCLMemObj, pi::cast<RT::PiMem>(SrcMem), std::move(SrcQueue),
              DimSrc, SrcSize, SrcAccessRange, SrcOffset, SrcElemSize,
              pi::cast<RT::PiMem>(DstMem), std::move(TgtQueue), DimDst, DstSize,
              DstAccessRange, DstOffset, DstElemSize, std::move(DepEvents),
              OutEvent);
  }
}

void MemoryManager::fill(SYCLMemObjI *SYCLMemObj, void *Mem, QueueImplPtr Queue,
                         size_t PatternSize, const char *Pattern,
                         unsigned int Dim, sycl::range<3>, sycl::range<3> Range,
                         sycl::id<3> Offset, unsigned int ElementSize,
                         std::vector<RT::PiEvent> DepEvents,
                         RT::PiEvent &OutEvent) {
  assert(SYCLMemObj && "The SYCLMemObj is nullptr");

  const detail::plugin &Plugin = Queue->getPlugin();
  if (SYCLMemObj->getType() == detail::SYCLMemObjI::MemObjType::BUFFER) {
    if (Dim == 1) {
      Plugin.call<PiApiKind::piEnqueueMemBufferFill>(
          Queue->getHandleRef(), pi::cast<RT::PiMem>(Mem), Pattern, PatternSize,
          Offset[0] * ElementSize, Range[0] * ElementSize, DepEvents.size(),
          DepEvents.data(), &OutEvent);
      return;
    }
    throw runtime_error("Not supported configuration of fill requested",
                        PI_INVALID_OPERATION);
  } else {
    Plugin.call<PiApiKind::piEnqueueMemImageFill>(
        Queue->getHandleRef(), pi::cast<RT::PiMem>(Mem), Pattern, &Offset[0],
        &Range[0], DepEvents.size(), DepEvents.data(), &OutEvent);
  }
}

void *MemoryManager::map(SYCLMemObjI *, void *Mem, QueueImplPtr Queue,
                         access::mode AccessMode, unsigned int, sycl::range<3>,
                         sycl::range<3> AccessRange, sycl::id<3> AccessOffset,
                         unsigned int ElementSize,
                         std::vector<RT::PiEvent> DepEvents,
                         RT::PiEvent &OutEvent) {
  if (Queue->is_host()) {
    throw runtime_error("Not supported configuration of map requested",
                        PI_INVALID_OPERATION);
  }

  pi_map_flags Flags = 0;

  switch (AccessMode) {
  case access::mode::read:
    Flags |= PI_MAP_READ;
    break;
  case access::mode::write:
    Flags |= PI_MAP_WRITE;
    break;
  case access::mode::read_write:
  case access::mode::atomic:
    Flags = PI_MAP_WRITE | PI_MAP_READ;
    break;
  case access::mode::discard_write:
  case access::mode::discard_read_write:
    Flags |= PI_MAP_WRITE_INVALIDATE_REGION;
    break;
  }

  AccessOffset[0] *= ElementSize;
  AccessRange[0] *= ElementSize;

  // TODO: Handle offset
  assert(AccessOffset[0] == 0 && "Handle offset");

  void *MappedPtr = nullptr;
  const size_t BytesToMap = AccessRange[0] * AccessRange[1] * AccessRange[2];
  const detail::plugin &Plugin = Queue->getPlugin();
  Plugin.call<PiApiKind::piEnqueueMemBufferMap>(
      Queue->getHandleRef(), pi::cast<RT::PiMem>(Mem), CL_FALSE, Flags,
      AccessOffset[0], BytesToMap, DepEvents.size(), DepEvents.data(),
      &OutEvent, &MappedPtr);
  return MappedPtr;
}

void MemoryManager::unmap(SYCLMemObjI *, void *Mem, QueueImplPtr Queue,
                          void *MappedPtr, std::vector<RT::PiEvent> DepEvents,
                          RT::PiEvent &OutEvent) {

  // Host queue is not supported here.
  // All DepEvents are to the same Context.
  // Using the plugin of the Queue.

  const detail::plugin &Plugin = Queue->getPlugin();
  Plugin.call<PiApiKind::piEnqueueMemUnmap>(
      Queue->getHandleRef(), pi::cast<RT::PiMem>(Mem), MappedPtr,
      DepEvents.size(), DepEvents.data(), &OutEvent);
}

void MemoryManager::copy_usm(const void *SrcMem, QueueImplPtr SrcQueue,
                             size_t Len, void *DstMem,
                             std::vector<RT::PiEvent> DepEvents,
                             RT::PiEvent &OutEvent) {
  if (!Len)
    return;
  if (!SrcMem || !DstMem)
    throw runtime_error("NULL pointer argument in memory copy operation.",
                        PI_INVALID_VALUE);

  sycl::context Context = SrcQueue->get_context();

  if (Context.is_host()) {
    std::memcpy(DstMem, SrcMem, Len);
  } else {
    const detail::plugin &Plugin = SrcQueue->getPlugin();
    Plugin.call<PiApiKind::piextUSMEnqueueMemcpy>(SrcQueue->getHandleRef(),
                                                  /* blocking */ false, DstMem,
                                                  SrcMem, Len, DepEvents.size(),
                                                  DepEvents.data(), &OutEvent);
  }
}

void MemoryManager::fill_usm(void *Mem, QueueImplPtr Queue, size_t Length,
                             int Pattern, std::vector<RT::PiEvent> DepEvents,
                             RT::PiEvent &OutEvent) {
  if (!Length)
    return;
  if (!Mem)
    throw runtime_error("NULL pointer argument in memory fill operation.",
                        PI_INVALID_VALUE);

  sycl::context Context = Queue->get_context();

  if (Context.is_host()) {
    std::memset(Mem, Pattern, Length);
  } else {
    const detail::plugin &Plugin = Queue->getPlugin();
    Plugin.call<PiApiKind::piextUSMEnqueueMemset>(
        Queue->getHandleRef(), Mem, Pattern, Length, DepEvents.size(),
        DepEvents.data(), &OutEvent);
  }
}

void MemoryManager::prefetch_usm(void *Mem, QueueImplPtr Queue, size_t Length,
                                 std::vector<RT::PiEvent> DepEvents,
                                 RT::PiEvent &OutEvent) {
  sycl::context Context = Queue->get_context();

  if (Context.is_host()) {
    // TODO: Potentially implement prefetch on the host.
  } else {
    const detail::plugin &Plugin = Queue->getPlugin();
    Plugin.call<PiApiKind::piextUSMEnqueuePrefetch>(
        Queue->getHandleRef(), Mem, Length, PI_USM_MIGRATION_TBD0,
        DepEvents.size(), DepEvents.data(), &OutEvent);
  }
}

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
