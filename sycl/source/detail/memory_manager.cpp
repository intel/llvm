//==-------------- memory_manager.cpp --------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/context_impl.hpp>
#include <detail/device_image_impl.hpp>
#include <detail/event_impl.hpp>
#include <detail/mem_alloc_helper.hpp>
#include <detail/memory_manager.hpp>
#include <detail/pi_utils.hpp>
#include <detail/queue_impl.hpp>
#include <detail/xpti_registry.hpp>

#include <algorithm>
#include <cassert>
#include <cstring>
#include <vector>

#ifdef XPTI_ENABLE_INSTRUMENTATION
#include <xpti/xpti_data_types.h>
#include <xpti/xpti_trace_framework.hpp>
#endif

namespace sycl {
inline namespace _V1 {
namespace detail {

#ifdef XPTI_ENABLE_INSTRUMENTATION
uint8_t GMemAllocStreamID;
xpti::trace_event_data_t *GMemAllocEvent;
#endif

uint64_t emitMemAllocBeginTrace(uintptr_t ObjHandle, size_t AllocSize,
                                size_t GuardZone) {
  (void)ObjHandle;
  (void)AllocSize;
  (void)GuardZone;
  uint64_t CorrelationID = 0;
#ifdef XPTI_ENABLE_INSTRUMENTATION
  constexpr uint16_t NotificationTraceType =
      static_cast<uint16_t>(xpti::trace_point_type_t::mem_alloc_begin);
  if (xptiCheckTraceEnabled(GMemAllocStreamID, NotificationTraceType)) {
    xpti::mem_alloc_data_t MemAlloc{ObjHandle, 0 /* alloc ptr */, AllocSize,
                                    GuardZone};

    CorrelationID = xptiGetUniqueId();
    xptiNotifySubscribers(GMemAllocStreamID, NotificationTraceType,
                          GMemAllocEvent, nullptr, CorrelationID, &MemAlloc);
  }
#endif
  return CorrelationID;
}

void emitMemAllocEndTrace(uintptr_t ObjHandle, uintptr_t AllocPtr,
                          size_t AllocSize, size_t GuardZone,
                          uint64_t CorrelationID) {
  (void)ObjHandle;
  (void)AllocPtr;
  (void)AllocSize;
  (void)GuardZone;
  (void)CorrelationID;
#ifdef XPTI_ENABLE_INSTRUMENTATION
  constexpr uint16_t NotificationTraceType =
      static_cast<uint16_t>(xpti::trace_point_type_t::mem_alloc_end);
  if (xptiCheckTraceEnabled(GMemAllocStreamID, NotificationTraceType)) {
    xpti::mem_alloc_data_t MemAlloc{ObjHandle, AllocPtr, AllocSize, GuardZone};

    xptiNotifySubscribers(GMemAllocStreamID, NotificationTraceType,
                          GMemAllocEvent, nullptr, CorrelationID, &MemAlloc);
  }
#endif
}

uint64_t emitMemReleaseBeginTrace(uintptr_t ObjHandle, uintptr_t AllocPtr) {
  (void)ObjHandle;
  (void)AllocPtr;
  uint64_t CorrelationID = 0;
#ifdef XPTI_ENABLE_INSTRUMENTATION
  constexpr uint16_t NotificationTraceType =
      static_cast<uint16_t>(xpti::trace_point_type_t::mem_release_begin);
  if (xptiCheckTraceEnabled(GMemAllocStreamID, NotificationTraceType)) {
    xpti::mem_alloc_data_t MemAlloc{ObjHandle, AllocPtr, 0 /* alloc size */,
                                    0 /* guard zone */};

    CorrelationID = xptiGetUniqueId();
    xptiNotifySubscribers(GMemAllocStreamID, NotificationTraceType,
                          GMemAllocEvent, nullptr, CorrelationID, &MemAlloc);
  }
#endif
  return CorrelationID;
}

void emitMemReleaseEndTrace(uintptr_t ObjHandle, uintptr_t AllocPtr,
                            uint64_t CorrelationID) {
  (void)ObjHandle;
  (void)AllocPtr;
  (void)CorrelationID;
#ifdef XPTI_ENABLE_INSTRUMENTATION
  constexpr uint16_t NotificationTraceType =
      static_cast<uint16_t>(xpti::trace_point_type_t::mem_release_end);
  if (xptiCheckTraceEnabled(GMemAllocStreamID, NotificationTraceType)) {
    xpti::mem_alloc_data_t MemAlloc{ObjHandle, AllocPtr, 0 /* alloc size */,
                                    0 /* guard zone */};

    xptiNotifySubscribers(GMemAllocStreamID, NotificationTraceType,
                          GMemAllocEvent, nullptr, CorrelationID, &MemAlloc);
  }
#endif
}

static void waitForEvents(const std::vector<EventImplPtr> &Events) {
  // Assuming all events will be on the same device or
  // devices associated with the same Backend.
  if (!Events.empty()) {
    const PluginPtr &Plugin = Events[0]->getPlugin();
    std::vector<sycl::detail::pi::PiEvent> PiEvents(Events.size());
    std::transform(Events.begin(), Events.end(), PiEvents.begin(),
                   [](const EventImplPtr &EventImpl) {
                     return EventImpl->getHandleRef();
                   });
    Plugin->call<PiApiKind::piEventsWait>(PiEvents.size(), &PiEvents[0]);
  }
}

void memBufferCreateHelper(const PluginPtr &Plugin, pi_context Ctx,
                           pi_mem_flags Flags, size_t Size, void *HostPtr,
                           pi_mem *RetMem, const pi_mem_properties *Props) {
#ifdef XPTI_ENABLE_INSTRUMENTATION
  uint64_t CorrID = 0;
#endif
  // We only want to instrument piMemBufferCreate
  {
#ifdef XPTI_ENABLE_INSTRUMENTATION
    CorrID =
        emitMemAllocBeginTrace(0 /* mem object */, Size, 0 /* guard zone */);
    xpti::utils::finally _{[&] {
      // C-style cast is required for MSVC
      uintptr_t MemObjID = (uintptr_t)(*RetMem);
      pi_native_handle Ptr = 0;
      // Always use call_nocheck here, because call may throw an exception,
      // and this lambda will be called from destructor, which in combination
      // rewards us with UB.
      Plugin->call_nocheck<PiApiKind::piextMemGetNativeHandle>(*RetMem, &Ptr);
      emitMemAllocEndTrace(MemObjID, (uintptr_t)(Ptr), Size, 0 /* guard zone */,
                           CorrID);
    }};
#endif
    if (Size)
      Plugin->call<PiApiKind::piMemBufferCreate>(Ctx, Flags, Size, HostPtr,
                                                 RetMem, Props);
  }
}

void memReleaseHelper(const PluginPtr &Plugin, pi_mem Mem) {
  // FIXME piMemRelease does not guarante memory release. It is only true if
  // reference counter is 1. However, SYCL runtime currently only calls
  // piMemRetain only for OpenCL interop
#ifdef XPTI_ENABLE_INSTRUMENTATION
  uint64_t CorrID = 0;
  // C-style cast is required for MSVC
  uintptr_t MemObjID = (uintptr_t)(Mem);
  uintptr_t Ptr = 0;
  // Do not make unnecessary PI calls without instrumentation enabled
  if (xptiTraceEnabled()) {
    pi_native_handle PtrHandle = 0;
    Plugin->call<PiApiKind::piextMemGetNativeHandle>(Mem, &PtrHandle);
    Ptr = (uintptr_t)(PtrHandle);
  }
#endif
  // We only want to instrument piMemRelease
  {
#ifdef XPTI_ENABLE_INSTRUMENTATION
    CorrID = emitMemReleaseBeginTrace(MemObjID, Ptr);
    xpti::utils::finally _{
        [&] { emitMemReleaseEndTrace(MemObjID, Ptr, CorrID); }};
#endif
    Plugin->call<PiApiKind::piMemRelease>(Mem);
  }
}

void memBufferMapHelper(const PluginPtr &Plugin, pi_queue Queue, pi_mem Buffer,
                        pi_bool Blocking, pi_map_flags Flags, size_t Offset,
                        size_t Size, pi_uint32 NumEvents,
                        const pi_event *WaitList, pi_event *Event,
                        void **RetMap) {
#ifdef XPTI_ENABLE_INSTRUMENTATION
  uint64_t CorrID = 0;
  uintptr_t MemObjID = (uintptr_t)(Buffer);
#endif
  // We only want to instrument piEnqueueMemBufferMap

#ifdef XPTI_ENABLE_INSTRUMENTATION
  CorrID = emitMemAllocBeginTrace(MemObjID, Size, 0 /* guard zone */);
  xpti::utils::finally _{[&] {
    emitMemAllocEndTrace(MemObjID, (uintptr_t)(*RetMap), Size,
                         0 /* guard zone */, CorrID);
  }};
#endif
  Plugin->call<PiApiKind::piEnqueueMemBufferMap>(Queue, Buffer, Blocking, Flags,
                                                 Offset, Size, NumEvents,
                                                 WaitList, Event, RetMap);
}

void memUnmapHelper(const PluginPtr &Plugin, pi_queue Queue, pi_mem Mem,
                    void *MappedPtr, pi_uint32 NumEvents,
                    const pi_event *WaitList, pi_event *Event) {
#ifdef XPTI_ENABLE_INSTRUMENTATION
  uint64_t CorrID = 0;
  uintptr_t MemObjID = (uintptr_t)(Mem);
  uintptr_t Ptr = (uintptr_t)(MappedPtr);
#endif
  // We only want to instrument piEnqueueMemUnmap
  {
#ifdef XPTI_ENABLE_INSTRUMENTATION
    CorrID = emitMemReleaseBeginTrace(MemObjID, Ptr);
    xpti::utils::finally _{[&] {
      // There's no way for SYCL to know, when the pointer is freed, so we have
      // to explicitly wait for the end of data transfers here in order to
      // report correct events.
      // Always use call_nocheck here, because call may throw an exception,
      // and this lambda will be called from destructor, which in combination
      // rewards us with UB.
      Plugin->call_nocheck<PiApiKind::piEventsWait>(1, Event);
      emitMemReleaseEndTrace(MemObjID, Ptr, CorrID);
    }};
#endif
    Plugin->call<PiApiKind::piEnqueueMemUnmap>(Queue, Mem, MappedPtr, NumEvents,
                                               WaitList, Event);
  }
}

void MemoryManager::release(ContextImplPtr TargetContext, SYCLMemObjI *MemObj,
                            void *MemAllocation,
                            std::vector<EventImplPtr> DepEvents,
                            sycl::detail::pi::PiEvent &OutEvent) {
  // There is no async API for memory releasing. Explicitly wait for all
  // dependency events and return empty event.
  waitForEvents(DepEvents);
  OutEvent = nullptr;
  XPTIRegistry::bufferReleaseNotification(MemObj, MemAllocation);
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

  const PluginPtr &Plugin = TargetContext->getPlugin();
  memReleaseHelper(Plugin, pi::cast<sycl::detail::pi::PiMem>(MemAllocation));
}

void *MemoryManager::allocate(ContextImplPtr TargetContext, SYCLMemObjI *MemObj,
                              bool InitFromUserData, void *HostPtr,
                              std::vector<EventImplPtr> DepEvents,
                              sycl::detail::pi::PiEvent &OutEvent) {
  // There is no async API for memory allocation. Explicitly wait for all
  // dependency events and return empty event.
  waitForEvents(DepEvents);
  OutEvent = nullptr;

  return MemObj->allocateMem(TargetContext, InitFromUserData, HostPtr,
                             OutEvent);
}

void *MemoryManager::allocateHostMemory(SYCLMemObjI *MemObj, void *UserPtr,
                                        bool HostPtrReadOnly, size_t Size,
                                        const sycl::property_list &) {
  std::ignore = HostPtrReadOnly;
  std::ignore = Size;

  // Can return user pointer directly if it is not a nullptr.
  if (UserPtr)
    return UserPtr;

  return MemObj->allocateHostMem();
  ;
}

void *MemoryManager::allocateInteropMemObject(
    ContextImplPtr TargetContext, void *UserPtr,
    const EventImplPtr &InteropEvent, const ContextImplPtr &InteropContext,
    const sycl::property_list &, sycl::detail::pi::PiEvent &OutEventToWait) {
  (void)TargetContext;
  (void)InteropContext;
  // If memory object is created with interop c'tor return cl_mem as is.
  assert(TargetContext == InteropContext && "Expected matching contexts");
  OutEventToWait = InteropEvent->getHandleRef();
  // Retain the event since it will be released during alloca command
  // destruction
  if (nullptr != OutEventToWait) {
    const PluginPtr &Plugin = InteropEvent->getPlugin();
    Plugin->call<PiApiKind::piEventRetain>(OutEventToWait);
  }
  return UserPtr;
}

static sycl::detail::pi::PiMemFlags
getMemObjCreationFlags(void *UserPtr, bool HostPtrReadOnly) {
  // Create read_write mem object to handle arbitrary uses.
  sycl::detail::pi::PiMemFlags Result =
      HostPtrReadOnly ? PI_MEM_ACCESS_READ_ONLY : PI_MEM_FLAGS_ACCESS_RW;
  if (UserPtr)
    Result |= PI_MEM_FLAGS_HOST_PTR_USE;
  return Result;
}

void *MemoryManager::allocateImageObject(
    ContextImplPtr TargetContext, void *UserPtr, bool HostPtrReadOnly,
    const sycl::detail::pi::PiMemImageDesc &Desc,
    const sycl::detail::pi::PiMemImageFormat &Format,
    const sycl::property_list &) {
  sycl::detail::pi::PiMemFlags CreationFlags =
      getMemObjCreationFlags(UserPtr, HostPtrReadOnly);

  sycl::detail::pi::PiMem NewMem;
  const PluginPtr &Plugin = TargetContext->getPlugin();
  Plugin->call<PiApiKind::piMemImageCreate>(TargetContext->getHandleRef(),
                                            CreationFlags, &Format, &Desc,
                                            UserPtr, &NewMem);
  return NewMem;
}

void *
MemoryManager::allocateBufferObject(ContextImplPtr TargetContext, void *UserPtr,
                                    bool HostPtrReadOnly, const size_t Size,
                                    const sycl::property_list &PropsList) {
  sycl::detail::pi::PiMemFlags CreationFlags =
      getMemObjCreationFlags(UserPtr, HostPtrReadOnly);
  if (PropsList.has_property<
          sycl::ext::oneapi::property::buffer::use_pinned_host_memory>())
    CreationFlags |= PI_MEM_FLAGS_HOST_PTR_ALLOC;

  sycl::detail::pi::PiMem NewMem = nullptr;
  const PluginPtr &Plugin = TargetContext->getPlugin();

  std::vector<pi_mem_properties> AllocProps;

  if (PropsList.has_property<property::buffer::detail::buffer_location>() &&
      TargetContext->isBufferLocationSupported()) {
    auto Location =
        PropsList.get_property<property::buffer::detail::buffer_location>()
            .get_buffer_location();
    AllocProps.reserve(AllocProps.size() + 2);
    AllocProps.push_back(PI_MEM_PROPERTIES_ALLOC_BUFFER_LOCATION);
    AllocProps.push_back(Location);
  }

  if (PropsList.has_property<property::buffer::mem_channel>()) {
    auto Channel =
        PropsList.get_property<property::buffer::mem_channel>().get_channel();
    AllocProps.reserve(AllocProps.size() + 2);
    AllocProps.push_back(PI_MEM_PROPERTIES_CHANNEL);
    AllocProps.push_back(Channel);
  }

  pi_mem_properties *AllocPropsPtr = nullptr;
  if (!AllocProps.empty()) {
    // If there are allocation properties, push an end to the list and update
    // the properties pointer.
    AllocProps.push_back(0);
    AllocPropsPtr = AllocProps.data();
  }

  memBufferCreateHelper(Plugin, TargetContext->getHandleRef(), CreationFlags,
                        Size, UserPtr, &NewMem, AllocPropsPtr);
  return NewMem;
}

void *MemoryManager::allocateMemBuffer(
    ContextImplPtr TargetContext, SYCLMemObjI *MemObj, void *UserPtr,
    bool HostPtrReadOnly, size_t Size, const EventImplPtr &InteropEvent,
    const ContextImplPtr &InteropContext, const sycl::property_list &PropsList,
    sycl::detail::pi::PiEvent &OutEventToWait) {
  void *MemPtr;
  if (TargetContext->is_host())
    MemPtr =
        allocateHostMemory(MemObj, UserPtr, HostPtrReadOnly, Size, PropsList);
  else if (UserPtr && InteropContext)
    MemPtr =
        allocateInteropMemObject(TargetContext, UserPtr, InteropEvent,
                                 InteropContext, PropsList, OutEventToWait);
  else
    MemPtr = allocateBufferObject(TargetContext, UserPtr, HostPtrReadOnly, Size,
                                  PropsList);
  XPTIRegistry::bufferAssociateNotification(MemObj, MemPtr);
  return MemPtr;
}

void *MemoryManager::allocateMemImage(
    ContextImplPtr TargetContext, SYCLMemObjI *MemObj, void *UserPtr,
    bool HostPtrReadOnly, size_t Size,
    const sycl::detail::pi::PiMemImageDesc &Desc,
    const sycl::detail::pi::PiMemImageFormat &Format,
    const EventImplPtr &InteropEvent, const ContextImplPtr &InteropContext,
    const sycl::property_list &PropsList,
    sycl::detail::pi::PiEvent &OutEventToWait) {
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
                                          sycl::detail::pi::PiEvent &OutEvent) {
  waitForEvents(DepEvents);
  OutEvent = nullptr;

  if (TargetContext->is_host())
    return static_cast<void *>(static_cast<char *>(ParentMemObj) + Offset);

  size_t SizeInBytes = ElemSize;
  for (size_t I = 0; I < 3; ++I)
    SizeInBytes *= Range[I];

  sycl::detail::pi::PiResult Error = PI_SUCCESS;
  pi_buffer_region_struct Region{Offset, SizeInBytes};
  sycl::detail::pi::PiMem NewMem;
  const PluginPtr &Plugin = TargetContext->getPlugin();
  Error = Plugin->call_nocheck<PiApiKind::piMemBufferPartition>(
      pi::cast<sycl::detail::pi::PiMem>(ParentMemObj), PI_MEM_FLAGS_ACCESS_RW,
      PI_BUFFER_CREATE_TYPE_REGION, &Region, &NewMem);
  if (Error == PI_ERROR_MISALIGNED_SUB_BUFFER_OFFSET)
    throw invalid_object_error(
        "Specified offset of the sub-buffer being constructed is not a "
        "multiple of the memory base address alignment",
        PI_ERROR_INVALID_VALUE);

  if (Error != PI_SUCCESS) {
    Plugin->reportPiError(Error, "allocateMemSubBuffer()");
  }

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

  if (Type == detail::SYCLMemObjI::MemObjType::Buffer) {
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
             unsigned int SrcElemSize, sycl::detail::pi::PiMem DstMem,
             QueueImplPtr TgtQueue, unsigned int DimDst, sycl::range<3> DstSize,
             sycl::range<3> DstAccessRange, sycl::id<3> DstOffset,
             unsigned int DstElemSize,
             std::vector<sycl::detail::pi::PiEvent> DepEvents,
             sycl::detail::pi::PiEvent &OutEvent,
             const detail::EventImplPtr &OutEventImpl) {
  (void)SrcAccessRange;
  assert(SYCLMemObj && "The SYCLMemObj is nullptr");

  const sycl::detail::pi::PiQueue Queue = TgtQueue->getHandleRef();
  const PluginPtr &Plugin = TgtQueue->getPlugin();

  detail::SYCLMemObjI::MemObjType MemType = SYCLMemObj->getType();
  TermPositions SrcPos, DstPos;
  prepTermPositions(SrcPos, DimSrc, MemType);
  prepTermPositions(DstPos, DimDst, MemType);

  size_t DstXOffBytes = DstOffset[DstPos.XTerm] * DstElemSize;
  size_t SrcXOffBytes = SrcOffset[SrcPos.XTerm] * SrcElemSize;
  size_t DstAccessRangeWidthBytes = DstAccessRange[DstPos.XTerm] * DstElemSize;
  size_t DstSzWidthBytes = DstSize[DstPos.XTerm] * DstElemSize;
  size_t SrcSzWidthBytes = SrcSize[SrcPos.XTerm] * SrcElemSize;

  if (MemType == detail::SYCLMemObjI::MemObjType::Buffer) {
    if (1 == DimDst && 1 == DimSrc) {
      if (OutEventImpl != nullptr)
        OutEventImpl->setHostEnqueueTime();
      Plugin->call<PiApiKind::piEnqueueMemBufferWrite>(
          Queue, DstMem,
          /*blocking_write=*/PI_FALSE, DstXOffBytes, DstAccessRangeWidthBytes,
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
      if (OutEventImpl != nullptr)
        OutEventImpl->setHostEnqueueTime();
      Plugin->call<PiApiKind::piEnqueueMemBufferWriteRect>(
          Queue, DstMem,
          /*blocking_write=*/PI_FALSE, &BufferOffset, &HostOffset, &RectRegion,
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
    if (OutEventImpl != nullptr)
      OutEventImpl->setHostEnqueueTime();
    Plugin->call<PiApiKind::piEnqueueMemImageWrite>(
        Queue, DstMem,
        /*blocking_write=*/PI_FALSE, &Origin, &Region, InputRowPitch,
        InputSlicePitch, SrcMem, DepEvents.size(), DepEvents.data(), &OutEvent);
  }
}

void copyD2H(SYCLMemObjI *SYCLMemObj, sycl::detail::pi::PiMem SrcMem,
             QueueImplPtr SrcQueue, unsigned int DimSrc, sycl::range<3> SrcSize,
             sycl::range<3> SrcAccessRange, sycl::id<3> SrcOffset,
             unsigned int SrcElemSize, char *DstMem, QueueImplPtr,
             unsigned int DimDst, sycl::range<3> DstSize,
             sycl::range<3> DstAccessRange, sycl::id<3> DstOffset,
             unsigned int DstElemSize,
             std::vector<sycl::detail::pi::PiEvent> DepEvents,
             sycl::detail::pi::PiEvent &OutEvent,
             const detail::EventImplPtr &OutEventImpl) {
  (void)DstAccessRange;
  assert(SYCLMemObj && "The SYCLMemObj is nullptr");

  const sycl::detail::pi::PiQueue Queue = SrcQueue->getHandleRef();
  const PluginPtr &Plugin = SrcQueue->getPlugin();

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

  if (MemType == detail::SYCLMemObjI::MemObjType::Buffer) {
    if (1 == DimDst && 1 == DimSrc) {
      if (OutEventImpl != nullptr)
        OutEventImpl->setHostEnqueueTime();
      Plugin->call<PiApiKind::piEnqueueMemBufferRead>(
          Queue, SrcMem,
          /*blocking_read=*/PI_FALSE, SrcXOffBytes, SrcAccessRangeWidthBytes,
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
      if (OutEventImpl != nullptr)
        OutEventImpl->setHostEnqueueTime();
      Plugin->call<PiApiKind::piEnqueueMemBufferReadRect>(
          Queue, SrcMem,
          /*blocking_read=*/PI_FALSE, &BufferOffset, &HostOffset, &RectRegion,
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
    if (OutEventImpl != nullptr)
      OutEventImpl->setHostEnqueueTime();
    Plugin->call<PiApiKind::piEnqueueMemImageRead>(
        Queue, SrcMem, PI_FALSE, &Offset, &Region, RowPitch, SlicePitch, DstMem,
        DepEvents.size(), DepEvents.data(), &OutEvent);
  }
}

void copyD2D(SYCLMemObjI *SYCLMemObj, sycl::detail::pi::PiMem SrcMem,
             QueueImplPtr SrcQueue, unsigned int DimSrc, sycl::range<3> SrcSize,
             sycl::range<3> SrcAccessRange, sycl::id<3> SrcOffset,
             unsigned int SrcElemSize, sycl::detail::pi::PiMem DstMem,
             QueueImplPtr, unsigned int DimDst, sycl::range<3> DstSize,
             sycl::range<3>, sycl::id<3> DstOffset, unsigned int DstElemSize,
             std::vector<sycl::detail::pi::PiEvent> DepEvents,
             sycl::detail::pi::PiEvent &OutEvent,
             const detail::EventImplPtr &OutEventImpl) {
  assert(SYCLMemObj && "The SYCLMemObj is nullptr");

  const sycl::detail::pi::PiQueue Queue = SrcQueue->getHandleRef();
  const PluginPtr &Plugin = SrcQueue->getPlugin();

  detail::SYCLMemObjI::MemObjType MemType = SYCLMemObj->getType();
  TermPositions SrcPos, DstPos;
  prepTermPositions(SrcPos, DimSrc, MemType);
  prepTermPositions(DstPos, DimDst, MemType);

  size_t DstXOffBytes = DstOffset[DstPos.XTerm] * DstElemSize;
  size_t SrcXOffBytes = SrcOffset[SrcPos.XTerm] * SrcElemSize;
  size_t SrcAccessRangeWidthBytes = SrcAccessRange[SrcPos.XTerm] * SrcElemSize;
  size_t DstSzWidthBytes = DstSize[DstPos.XTerm] * DstElemSize;
  size_t SrcSzWidthBytes = SrcSize[SrcPos.XTerm] * SrcElemSize;

  if (MemType == detail::SYCLMemObjI::MemObjType::Buffer) {
    if (1 == DimDst && 1 == DimSrc) {
      if (OutEventImpl != nullptr)
        OutEventImpl->setHostEnqueueTime();
      Plugin->call<PiApiKind::piEnqueueMemBufferCopy>(
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
      if (OutEventImpl != nullptr)
        OutEventImpl->setHostEnqueueTime();
      Plugin->call<PiApiKind::piEnqueueMemBufferCopyRect>(
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
    if (OutEventImpl != nullptr)
      OutEventImpl->setHostEnqueueTime();
    Plugin->call<PiApiKind::piEnqueueMemImageCopy>(
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
                    unsigned int DstElemSize,
                    std::vector<sycl::detail::pi::PiEvent>,
                    sycl::detail::pi::PiEvent &, const detail::EventImplPtr &) {
  if ((DimSrc != 1 || DimDst != 1) &&
      (SrcOffset != id<3>{0, 0, 0} || DstOffset != id<3>{0, 0, 0} ||
       SrcSize != SrcAccessRange || DstSize != DstAccessRange)) {
    throw runtime_error("Not supported configuration of memcpy requested",
                        PI_ERROR_INVALID_OPERATION);
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
                         std::vector<sycl::detail::pi::PiEvent> DepEvents,
                         sycl::detail::pi::PiEvent &OutEvent,
                         const detail::EventImplPtr &OutEventImpl) {

  if (SrcQueue->is_host()) {
    if (TgtQueue->is_host())
      copyH2H(SYCLMemObj, (char *)SrcMem, std::move(SrcQueue), DimSrc, SrcSize,
              SrcAccessRange, SrcOffset, SrcElemSize, (char *)DstMem,
              std::move(TgtQueue), DimDst, DstSize, DstAccessRange, DstOffset,
              DstElemSize, std::move(DepEvents), OutEvent, OutEventImpl);
    else
      copyH2D(SYCLMemObj, (char *)SrcMem, std::move(SrcQueue), DimSrc, SrcSize,
              SrcAccessRange, SrcOffset, SrcElemSize,
              pi::cast<sycl::detail::pi::PiMem>(DstMem), std::move(TgtQueue),
              DimDst, DstSize, DstAccessRange, DstOffset, DstElemSize,
              std::move(DepEvents), OutEvent, OutEventImpl);
  } else {
    if (TgtQueue->is_host())
      copyD2H(SYCLMemObj, pi::cast<sycl::detail::pi::PiMem>(SrcMem),
              std::move(SrcQueue), DimSrc, SrcSize, SrcAccessRange, SrcOffset,
              SrcElemSize, (char *)DstMem, std::move(TgtQueue), DimDst, DstSize,
              DstAccessRange, DstOffset, DstElemSize, std::move(DepEvents),
              OutEvent, OutEventImpl);
    else
      copyD2D(SYCLMemObj, pi::cast<sycl::detail::pi::PiMem>(SrcMem),
              std::move(SrcQueue), DimSrc, SrcSize, SrcAccessRange, SrcOffset,
              SrcElemSize, pi::cast<sycl::detail::pi::PiMem>(DstMem),
              std::move(TgtQueue), DimDst, DstSize, DstAccessRange, DstOffset,
              DstElemSize, std::move(DepEvents), OutEvent, OutEventImpl);
  }
}

// TODO: This function will remain until ABI-breaking change
void MemoryManager::copy(SYCLMemObjI *SYCLMemObj, void *SrcMem,
                         QueueImplPtr SrcQueue, unsigned int DimSrc,
                         sycl::range<3> SrcSize, sycl::range<3> SrcAccessRange,
                         sycl::id<3> SrcOffset, unsigned int SrcElemSize,
                         void *DstMem, QueueImplPtr TgtQueue,
                         unsigned int DimDst, sycl::range<3> DstSize,
                         sycl::range<3> DstAccessRange, sycl::id<3> DstOffset,
                         unsigned int DstElemSize,
                         std::vector<sycl::detail::pi::PiEvent> DepEvents,
                         sycl::detail::pi::PiEvent &OutEvent) {
  MemoryManager::copy(SYCLMemObj, SrcMem, SrcQueue, DimSrc, SrcSize,
                      SrcAccessRange, SrcOffset, SrcElemSize, DstMem, TgtQueue,
                      DimDst, DstSize, DstAccessRange, DstOffset, DstElemSize,
                      DepEvents, OutEvent, nullptr);
}

void MemoryManager::fill(SYCLMemObjI *SYCLMemObj, void *Mem, QueueImplPtr Queue,
                         size_t PatternSize, const char *Pattern,
                         unsigned int Dim, sycl::range<3> MemRange,
                         sycl::range<3> AccRange, sycl::id<3> Offset,
                         unsigned int ElementSize,
                         std::vector<sycl::detail::pi::PiEvent> DepEvents,
                         sycl::detail::pi::PiEvent &OutEvent,
                         const detail::EventImplPtr &OutEventImpl) {
  assert(SYCLMemObj && "The SYCLMemObj is nullptr");

  const PluginPtr &Plugin = Queue->getPlugin();

  if (SYCLMemObj->getType() == detail::SYCLMemObjI::MemObjType::Buffer) {
    if (OutEventImpl != nullptr)
      OutEventImpl->setHostEnqueueTime();

    // 2D and 3D buffers accessors can't have custom range or the data will
    // likely be discontiguous.
    bool RangesUsable = (Dim <= 1) || (MemRange == AccRange);
    // For 2D and 3D buffers, the offset must be 0, or the data will be
    // discontiguous.
    bool OffsetUsable = (Dim <= 1) || (Offset == sycl::id<3>{0, 0, 0});
    size_t RangeMultiplier = AccRange[0] * AccRange[1] * AccRange[2];

    if (RangesUsable && OffsetUsable) {
      Plugin->call<PiApiKind::piEnqueueMemBufferFill>(
          Queue->getHandleRef(), pi::cast<sycl::detail::pi::PiMem>(Mem),
          Pattern, PatternSize, Offset[0] * ElementSize,
          RangeMultiplier * ElementSize, DepEvents.size(), DepEvents.data(),
          &OutEvent);
      return;
    }
    // The sycl::handler uses a parallel_for kernel in the case of unusable
    // Range or Offset, not CG:Fill. So we should not be here.
    throw runtime_error("Not supported configuration of fill requested",
                        PI_ERROR_INVALID_OPERATION);
  } else {
    if (OutEventImpl != nullptr)
      OutEventImpl->setHostEnqueueTime();
    // images don't support offset accessors and thus avoid issues of
    // discontinguous data
    Plugin->call<PiApiKind::piEnqueueMemImageFill>(
        Queue->getHandleRef(), pi::cast<sycl::detail::pi::PiMem>(Mem), Pattern,
        &Offset[0], &AccRange[0], DepEvents.size(), DepEvents.data(),
        &OutEvent);
  }
}

// TODO: This function will remain until ABI-breaking change
void MemoryManager::fill(SYCLMemObjI *SYCLMemObj, void *Mem, QueueImplPtr Queue,
                         size_t PatternSize, const char *Pattern,
                         unsigned int Dim, sycl::range<3> Size,
                         sycl::range<3> Range, sycl::id<3> Offset,
                         unsigned int ElementSize,
                         std::vector<sycl::detail::pi::PiEvent> DepEvents,
                         sycl::detail::pi::PiEvent &OutEvent) {
  MemoryManager::fill(SYCLMemObj, Mem, Queue, PatternSize, Pattern, Dim, Size,
                      Range, Offset, ElementSize, DepEvents, OutEvent, nullptr);
}

void *MemoryManager::map(SYCLMemObjI *, void *Mem, QueueImplPtr Queue,
                         access::mode AccessMode, unsigned int, sycl::range<3>,
                         sycl::range<3> AccessRange, sycl::id<3> AccessOffset,
                         unsigned int ElementSize,
                         std::vector<sycl::detail::pi::PiEvent> DepEvents,
                         sycl::detail::pi::PiEvent &OutEvent) {
  if (Queue->is_host()) {
    throw runtime_error("Not supported configuration of map requested",
                        PI_ERROR_INVALID_OPERATION);
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
  const PluginPtr &Plugin = Queue->getPlugin();
  memBufferMapHelper(Plugin, Queue->getHandleRef(),
                     pi::cast<sycl::detail::pi::PiMem>(Mem), PI_FALSE, Flags,
                     AccessOffset[0], BytesToMap, DepEvents.size(),
                     DepEvents.data(), &OutEvent, &MappedPtr);
  return MappedPtr;
}

void MemoryManager::unmap(SYCLMemObjI *, void *Mem, QueueImplPtr Queue,
                          void *MappedPtr,
                          std::vector<sycl::detail::pi::PiEvent> DepEvents,
                          sycl::detail::pi::PiEvent &OutEvent) {

  // Host queue is not supported here.
  // All DepEvents are to the same Context.
  // Using the plugin of the Queue.

  const PluginPtr &Plugin = Queue->getPlugin();
  memUnmapHelper(Plugin, Queue->getHandleRef(),
                 pi::cast<sycl::detail::pi::PiMem>(Mem), MappedPtr,
                 DepEvents.size(), DepEvents.data(), &OutEvent);
}

void MemoryManager::copy_usm(const void *SrcMem, QueueImplPtr SrcQueue,
                             size_t Len, void *DstMem,
                             std::vector<sycl::detail::pi::PiEvent> DepEvents,
                             sycl::detail::pi::PiEvent *OutEvent,
                             const detail::EventImplPtr &OutEventImpl) {
  assert(!SrcQueue->getContextImplPtr()->is_host() &&
         "Host queue not supported in fill_usm.");

  if (!Len) { // no-op, but ensure DepEvents will still be waited on
    if (!DepEvents.empty()) {
      if (OutEventImpl != nullptr)
        OutEventImpl->setHostEnqueueTime();
      SrcQueue->getPlugin()->call<PiApiKind::piEnqueueEventsWait>(
          SrcQueue->getHandleRef(), DepEvents.size(), DepEvents.data(),
          OutEvent);
    }
    return;
  }

  if (!SrcMem || !DstMem)
    throw runtime_error("NULL pointer argument in memory copy operation.",
                        PI_ERROR_INVALID_VALUE);

  const PluginPtr &Plugin = SrcQueue->getPlugin();
  if (OutEventImpl != nullptr)
    OutEventImpl->setHostEnqueueTime();
  Plugin->call<PiApiKind::piextUSMEnqueueMemcpy>(
      SrcQueue->getHandleRef(),
      /* blocking */ PI_FALSE, DstMem, SrcMem, Len, DepEvents.size(),
      DepEvents.data(), OutEvent);
}

// TODO: This function will remain until ABI-breaking change
void MemoryManager::copy_usm(const void *SrcMem, QueueImplPtr SrcQueue,
                             size_t Len, void *DstMem,
                             std::vector<sycl::detail::pi::PiEvent> DepEvents,
                             sycl::detail::pi::PiEvent *OutEvent) {
  MemoryManager::copy_usm(SrcMem, SrcQueue, Len, DstMem, DepEvents, OutEvent,
                          nullptr);
}

void MemoryManager::fill_usm(void *Mem, QueueImplPtr Queue, size_t Length,
                             int Pattern,
                             std::vector<sycl::detail::pi::PiEvent> DepEvents,
                             sycl::detail::pi::PiEvent *OutEvent,
                             const detail::EventImplPtr &OutEventImpl) {
  assert(!Queue->getContextImplPtr()->is_host() &&
         "Host queue not supported in fill_usm.");

  if (!Length) { // no-op, but ensure DepEvents will still be waited on
    if (!DepEvents.empty()) {
      if (OutEventImpl != nullptr)
        OutEventImpl->setHostEnqueueTime();
      Queue->getPlugin()->call<PiApiKind::piEnqueueEventsWait>(
          Queue->getHandleRef(), DepEvents.size(), DepEvents.data(), OutEvent);
    }
    return;
  }

  if (!Mem)
    throw runtime_error("NULL pointer argument in memory fill operation.",
                        PI_ERROR_INVALID_VALUE);
  if (OutEventImpl != nullptr)
    OutEventImpl->setHostEnqueueTime();
  const PluginPtr &Plugin = Queue->getPlugin();
  Plugin->call<PiApiKind::piextUSMEnqueueMemset>(
      Queue->getHandleRef(), Mem, Pattern, Length, DepEvents.size(),
      DepEvents.data(), OutEvent);
}

// TODO: This function will remain until ABI-breaking change
void MemoryManager::fill_usm(void *Mem, QueueImplPtr Queue, size_t Length,
                             int Pattern,
                             std::vector<sycl::detail::pi::PiEvent> DepEvents,
                             sycl::detail::pi::PiEvent *OutEvent) {
  MemoryManager::fill_usm(Mem, Queue, Length, Pattern, DepEvents, OutEvent,
                          nullptr); // OutEventImpl);
}

void MemoryManager::prefetch_usm(
    void *Mem, QueueImplPtr Queue, size_t Length,
    std::vector<sycl::detail::pi::PiEvent> DepEvents,
    sycl::detail::pi::PiEvent *OutEvent,
    const detail::EventImplPtr &OutEventImpl) {
  assert(!Queue->getContextImplPtr()->is_host() &&
         "Host queue not supported in prefetch_usm.");

  const PluginPtr &Plugin = Queue->getPlugin();
  if (OutEventImpl != nullptr)
    OutEventImpl->setHostEnqueueTime();
  Plugin->call<PiApiKind::piextUSMEnqueuePrefetch>(
      Queue->getHandleRef(), Mem, Length, _pi_usm_migration_flags(0),
      DepEvents.size(), DepEvents.data(), OutEvent);
}

// TODO: This function will remain until ABI-breaking change
void MemoryManager::prefetch_usm(
    void *Mem, QueueImplPtr Queue, size_t Length,
    std::vector<sycl::detail::pi::PiEvent> DepEvents,
    sycl::detail::pi::PiEvent *OutEvent) {
  MemoryManager::prefetch_usm(Mem, Queue, Length, DepEvents, OutEvent, nullptr);
}

void MemoryManager::advise_usm(
    const void *Mem, QueueImplPtr Queue, size_t Length, pi_mem_advice Advice,
    std::vector<sycl::detail::pi::PiEvent> /*DepEvents*/,
    sycl::detail::pi::PiEvent *OutEvent,
    const detail::EventImplPtr &OutEventImpl) {
  assert(!Queue->getContextImplPtr()->is_host() &&
         "Host queue not supported in advise_usm.");

  const PluginPtr &Plugin = Queue->getPlugin();
  if (OutEventImpl != nullptr)
    OutEventImpl->setHostEnqueueTime();
  Plugin->call<PiApiKind::piextUSMEnqueueMemAdvise>(Queue->getHandleRef(), Mem,
                                                    Length, Advice, OutEvent);
}

// TODO: This function will remain until ABI-breaking change
void MemoryManager::advise_usm(const void *Mem, QueueImplPtr Queue,
                               size_t Length, pi_mem_advice Advice,
                               std::vector<sycl::detail::pi::PiEvent> DepEvents,
                               sycl::detail::pi::PiEvent *OutEvent) {
  MemoryManager::advise_usm(Mem, Queue, Length, Advice, DepEvents, OutEvent,
                            nullptr);
}

void MemoryManager::copy_2d_usm(
    const void *SrcMem, size_t SrcPitch, QueueImplPtr Queue, void *DstMem,
    size_t DstPitch, size_t Width, size_t Height,
    std::vector<sycl::detail::pi::PiEvent> DepEvents,
    sycl::detail::pi::PiEvent *OutEvent,
    const detail::EventImplPtr &OutEventImpl) {
  assert(!Queue->getContextImplPtr()->is_host() &&
         "Host queue not supported in copy_2d_usm.");

  if (Width == 0 || Height == 0) {
    // no-op, but ensure DepEvents will still be waited on
    if (!DepEvents.empty()) {
      if (OutEventImpl != nullptr)
        OutEventImpl->setHostEnqueueTime();
      Queue->getPlugin()->call<PiApiKind::piEnqueueEventsWait>(
          Queue->getHandleRef(), DepEvents.size(), DepEvents.data(), OutEvent);
    }
    return;
  }

  if (!DstMem || !SrcMem)
    throw sycl::exception(sycl::make_error_code(errc::invalid),
                          "NULL pointer argument in 2D memory copy operation.");

  const PluginPtr &Plugin = Queue->getPlugin();

  pi_bool SupportsUSMMemcpy2D = false;
  Plugin->call<detail::PiApiKind::piContextGetInfo>(
      Queue->getContextImplPtr()->getHandleRef(),
      PI_EXT_ONEAPI_CONTEXT_INFO_USM_MEMCPY2D_SUPPORT, sizeof(pi_bool),
      &SupportsUSMMemcpy2D, nullptr);

  if (SupportsUSMMemcpy2D) {
    if (OutEventImpl != nullptr)
      OutEventImpl->setHostEnqueueTime();
    // Direct memcpy2D is supported so we use this function.
    Plugin->call<PiApiKind::piextUSMEnqueueMemcpy2D>(
        Queue->getHandleRef(), /*blocking=*/PI_FALSE, DstMem, DstPitch, SrcMem,
        SrcPitch, Width, Height, DepEvents.size(), DepEvents.data(), OutEvent);
    return;
  }

  // Otherwise we allow the special case where the copy is to or from host.
#ifndef NDEBUG
  context Ctx = createSyclObjFromImpl<context>(Queue->getContextImplPtr());
  usm::alloc SrcAllocType = get_pointer_type(SrcMem, Ctx);
  usm::alloc DstAllocType = get_pointer_type(DstMem, Ctx);
  bool SrcIsHost =
      SrcAllocType == usm::alloc::unknown || SrcAllocType == usm::alloc::host;
  bool DstIsHost =
      DstAllocType == usm::alloc::unknown || DstAllocType == usm::alloc::host;
  assert((SrcIsHost || DstIsHost) && "In fallback path for copy_2d_usm either "
                                     "source or destination must be on host.");
#endif // NDEBUG

  // The fallback in this case is to insert a copy per row.
  std::vector<OwnedPiEvent> CopyEventsManaged;
  CopyEventsManaged.reserve(Height);
  // We'll need continuous range of events for a wait later as well.
  std::vector<sycl::detail::pi::PiEvent> CopyEvents(Height);
  if (OutEventImpl != nullptr)
    OutEventImpl->setHostEnqueueTime();
  for (size_t I = 0; I < Height; ++I) {
    char *DstItBegin = static_cast<char *>(DstMem) + I * DstPitch;
    const char *SrcItBegin = static_cast<const char *>(SrcMem) + I * SrcPitch;
    Plugin->call<PiApiKind::piextUSMEnqueueMemcpy>(
        Queue->getHandleRef(), /* blocking */ PI_FALSE, DstItBegin, SrcItBegin,
        Width, DepEvents.size(), DepEvents.data(), CopyEvents.data() + I);
    CopyEventsManaged.emplace_back(CopyEvents[I], Plugin,
                                   /*TakeOwnership=*/true);
  }
  if (OutEventImpl != nullptr)
    OutEventImpl->setHostEnqueueTime();
  // Then insert a wait to coalesce the copy events.
  Queue->getPlugin()->call<PiApiKind::piEnqueueEventsWait>(
      Queue->getHandleRef(), CopyEvents.size(), CopyEvents.data(), OutEvent);
}

// TODO: This function will remain until ABI-breaking change
void MemoryManager::copy_2d_usm(
    const void *SrcMem, size_t SrcPitch, QueueImplPtr Queue, void *DstMem,
    size_t DstPitch, size_t Width, size_t Height,
    std::vector<sycl::detail::pi::PiEvent> DepEvents,
    sycl::detail::pi::PiEvent *OutEvent) {
  MemoryManager::copy_2d_usm(SrcMem, SrcPitch, Queue, DstMem, DstPitch, Width,
                             Height, DepEvents, OutEvent, nullptr);
}

void MemoryManager::fill_2d_usm(
    void *DstMem, QueueImplPtr Queue, size_t Pitch, size_t Width, size_t Height,
    const std::vector<char> &Pattern,
    std::vector<sycl::detail::pi::PiEvent> DepEvents,
    sycl::detail::pi::PiEvent *OutEvent,
    const detail::EventImplPtr &OutEventImpl) {
  assert(!Queue->getContextImplPtr()->is_host() &&
         "Host queue not supported in fill_2d_usm.");

  if (Width == 0 || Height == 0) {
    // no-op, but ensure DepEvents will still be waited on
    if (!DepEvents.empty()) {
      if (OutEventImpl != nullptr)
        OutEventImpl->setHostEnqueueTime();
      Queue->getPlugin()->call<PiApiKind::piEnqueueEventsWait>(
          Queue->getHandleRef(), DepEvents.size(), DepEvents.data(), OutEvent);
    }
    return;
  }

  if (!DstMem)
    throw sycl::exception(sycl::make_error_code(errc::invalid),
                          "NULL pointer argument in 2D memory fill operation.");
  if (OutEventImpl != nullptr)
    OutEventImpl->setHostEnqueueTime();
  const PluginPtr &Plugin = Queue->getPlugin();
  Plugin->call<PiApiKind::piextUSMEnqueueFill2D>(
      Queue->getHandleRef(), DstMem, Pitch, Pattern.size(), Pattern.data(),
      Width, Height, DepEvents.size(), DepEvents.data(), OutEvent);
}

// TODO: This function will remain until ABI-breaking change
void MemoryManager::fill_2d_usm(
    void *DstMem, QueueImplPtr Queue, size_t Pitch, size_t Width, size_t Height,
    const std::vector<char> &Pattern,
    std::vector<sycl::detail::pi::PiEvent> DepEvents,
    sycl::detail::pi::PiEvent *OutEvent) {
  MemoryManager::fill_2d_usm(DstMem, Queue, Pitch, Width, Height, Pattern,
                             DepEvents, OutEvent, nullptr);
}

void MemoryManager::memset_2d_usm(
    void *DstMem, QueueImplPtr Queue, size_t Pitch, size_t Width, size_t Height,
    char Value, std::vector<sycl::detail::pi::PiEvent> DepEvents,
    sycl::detail::pi::PiEvent *OutEvent,
    const detail::EventImplPtr &OutEventImpl) {
  assert(!Queue->getContextImplPtr()->is_host() &&
         "Host queue not supported in fill_2d_usm.");

  if (Width == 0 || Height == 0) {
    // no-op, but ensure DepEvents will still be waited on
    if (!DepEvents.empty()) {
      if (OutEventImpl != nullptr)
        OutEventImpl->setHostEnqueueTime();
      Queue->getPlugin()->call<PiApiKind::piEnqueueEventsWait>(
          Queue->getHandleRef(), DepEvents.size(), DepEvents.data(), OutEvent);
    }
    return;
  }

  if (!DstMem)
    throw sycl::exception(
        sycl::make_error_code(errc::invalid),
        "NULL pointer argument in 2D memory memset operation.");
  if (OutEventImpl != nullptr)
    OutEventImpl->setHostEnqueueTime();
  const PluginPtr &Plugin = Queue->getPlugin();
  Plugin->call<PiApiKind::piextUSMEnqueueMemset2D>(
      Queue->getHandleRef(), DstMem, Pitch, static_cast<int>(Value), Width,
      Height, DepEvents.size(), DepEvents.data(), OutEvent);
}

// TODO: This function will remain until ABI-breaking change
void MemoryManager::memset_2d_usm(
    void *DstMem, QueueImplPtr Queue, size_t Pitch, size_t Width, size_t Height,
    char Value, std::vector<sycl::detail::pi::PiEvent> DepEvents,
    sycl::detail::pi::PiEvent *OutEvent) {
  MemoryManager::memset_2d_usm(DstMem, Queue, Pitch, Width, Height, Value,
                               DepEvents, OutEvent, nullptr);
}

static void
memcpyToDeviceGlobalUSM(QueueImplPtr Queue,
                        DeviceGlobalMapEntry *DeviceGlobalEntry,
                        size_t NumBytes, size_t Offset, const void *Src,
                        const std::vector<sycl::detail::pi::PiEvent> &DepEvents,
                        sycl::detail::pi::PiEvent *OutEvent,
                        const detail::EventImplPtr &OutEventImpl) {
  // Get or allocate USM memory for the device_global.
  DeviceGlobalUSMMem &DeviceGlobalUSM =
      DeviceGlobalEntry->getOrAllocateDeviceGlobalUSM(Queue);
  void *Dest = DeviceGlobalUSM.getPtr();

  // OwnedPiEvent will keep the initialization event alive for the duration
  // of this function call.
  OwnedPiEvent ZIEvent = DeviceGlobalUSM.getInitEvent(Queue->getPlugin());

  // We may need addtional events, so create a non-const dependency events list
  // to use if we need to modify it.
  std::vector<sycl::detail::pi::PiEvent> AuxDepEventsStorage;
  const std::vector<sycl::detail::pi::PiEvent> &ActualDepEvents =
      ZIEvent ? AuxDepEventsStorage : DepEvents;

  // If there is a zero-initializer event the memory operation should wait for
  // it.
  if (ZIEvent) {
    AuxDepEventsStorage = DepEvents;
    AuxDepEventsStorage.push_back(ZIEvent.GetEvent());
  }

  MemoryManager::copy_usm(Src, Queue, NumBytes,
                          reinterpret_cast<char *>(Dest) + Offset,
                          ActualDepEvents, OutEvent, OutEventImpl);
}

static void memcpyFromDeviceGlobalUSM(
    QueueImplPtr Queue, DeviceGlobalMapEntry *DeviceGlobalEntry,
    size_t NumBytes, size_t Offset, void *Dest,
    const std::vector<sycl::detail::pi::PiEvent> &DepEvents,
    sycl::detail::pi::PiEvent *OutEvent,
    const detail::EventImplPtr &OutEventImpl) {
  // Get or allocate USM memory for the device_global. Since we are reading from
  // it, we need it initialized if it has not been yet.
  DeviceGlobalUSMMem &DeviceGlobalUSM =
      DeviceGlobalEntry->getOrAllocateDeviceGlobalUSM(Queue);
  void *Src = DeviceGlobalUSM.getPtr();

  // OwnedPiEvent will keep the initialization event alive for the duration
  // of this function call.
  OwnedPiEvent ZIEvent = DeviceGlobalUSM.getInitEvent(Queue->getPlugin());

  // We may need addtional events, so create a non-const dependency events list
  // to use if we need to modify it.
  std::vector<sycl::detail::pi::PiEvent> AuxDepEventsStorage;
  const std::vector<sycl::detail::pi::PiEvent> &ActualDepEvents =
      ZIEvent ? AuxDepEventsStorage : DepEvents;

  // If there is a zero-initializer event the memory operation should wait for
  // it.
  if (ZIEvent) {
    AuxDepEventsStorage = DepEvents;
    AuxDepEventsStorage.push_back(ZIEvent.GetEvent());
  }

  MemoryManager::copy_usm(reinterpret_cast<const char *>(Src) + Offset, Queue,
                          NumBytes, Dest, ActualDepEvents, OutEvent,
                          OutEventImpl);
}

static sycl::detail::pi::PiProgram
getOrBuildProgramForDeviceGlobal(QueueImplPtr Queue,
                                 DeviceGlobalMapEntry *DeviceGlobalEntry) {
  assert(DeviceGlobalEntry->MIsDeviceImageScopeDecorated &&
         "device_global is not device image scope decorated.");

  // If the device global is used in multiple device images we cannot proceed.
  if (DeviceGlobalEntry->MImageIdentifiers.size() > 1)
    throw sycl::exception(make_error_code(errc::invalid),
                          "More than one image exists with the device_global.");

  // If there are no kernels using the device_global we cannot proceed.
  if (DeviceGlobalEntry->MImageIdentifiers.size() == 0)
    throw sycl::exception(make_error_code(errc::invalid),
                          "No image exists with the device_global.");

  // Look for cached programs with the device_global.
  device Device = Queue->get_device();
  ContextImplPtr ContextImpl = Queue->getContextImplPtr();
  std::optional<sycl::detail::pi::PiProgram> CachedProgram =
      ContextImpl->getProgramForDeviceGlobal(Device, DeviceGlobalEntry);
  if (CachedProgram)
    return *CachedProgram;

  // If there was no cached program, build one.
  auto Context = createSyclObjFromImpl<context>(ContextImpl);
  ProgramManager &PM = ProgramManager::getInstance();
  RTDeviceBinaryImage &Img =
      PM.getDeviceImage(DeviceGlobalEntry->MImages, Context, Device);
  device_image_plain DeviceImage =
      PM.getDeviceImageFromBinaryImage(&Img, Context, Device);
  device_image_plain BuiltImage = PM.build(DeviceImage, {Device}, {});
  return getSyclObjImpl(BuiltImage)->get_program_ref();
}

static void memcpyToDeviceGlobalDirect(
    QueueImplPtr Queue, DeviceGlobalMapEntry *DeviceGlobalEntry,
    size_t NumBytes, size_t Offset, const void *Src,
    const std::vector<sycl::detail::pi::PiEvent> &DepEvents,
    sycl::detail::pi::PiEvent *OutEvent) {
  sycl::detail::pi::PiProgram Program =
      getOrBuildProgramForDeviceGlobal(Queue, DeviceGlobalEntry);
  const PluginPtr &Plugin = Queue->getPlugin();
  Plugin->call<PiApiKind::piextEnqueueDeviceGlobalVariableWrite>(
      Queue->getHandleRef(), Program, DeviceGlobalEntry->MUniqueId.c_str(),
      false, NumBytes, Offset, Src, DepEvents.size(), DepEvents.data(),
      OutEvent);
}

static void memcpyFromDeviceGlobalDirect(
    QueueImplPtr Queue, DeviceGlobalMapEntry *DeviceGlobalEntry,
    size_t NumBytes, size_t Offset, void *Dest,
    const std::vector<sycl::detail::pi::PiEvent> &DepEvents,
    sycl::detail::pi::PiEvent *OutEvent) {
  sycl::detail::pi::PiProgram Program =
      getOrBuildProgramForDeviceGlobal(Queue, DeviceGlobalEntry);
  const PluginPtr &Plugin = Queue->getPlugin();
  Plugin->call<PiApiKind::piextEnqueueDeviceGlobalVariableRead>(
      Queue->getHandleRef(), Program, DeviceGlobalEntry->MUniqueId.c_str(),
      false, NumBytes, Offset, Dest, DepEvents.size(), DepEvents.data(),
      OutEvent);
}

void MemoryManager::copy_to_device_global(
    const void *DeviceGlobalPtr, bool IsDeviceImageScoped, QueueImplPtr Queue,
    size_t NumBytes, size_t Offset, const void *SrcMem,
    const std::vector<sycl::detail::pi::PiEvent> &DepEvents,
    sycl::detail::pi::PiEvent *OutEvent,
    const detail::EventImplPtr &OutEventImpl) {
  DeviceGlobalMapEntry *DGEntry =
      detail::ProgramManager::getInstance().getDeviceGlobalEntry(
          DeviceGlobalPtr);
  assert(DGEntry &&
         DGEntry->MIsDeviceImageScopeDecorated == IsDeviceImageScoped &&
         "Invalid copy operation for device_global.");
  assert(DGEntry->MDeviceGlobalTSize >= Offset + NumBytes &&
         "Copy to device_global is out of bounds.");

  if (IsDeviceImageScoped)
    memcpyToDeviceGlobalDirect(Queue, DGEntry, NumBytes, Offset, SrcMem,
                               DepEvents, OutEvent);
  else
    memcpyToDeviceGlobalUSM(Queue, DGEntry, NumBytes, Offset, SrcMem, DepEvents,
                            OutEvent, OutEventImpl);
}

// TODO: This function will remain until ABI-breaking change
void MemoryManager::copy_to_device_global(
    const void *DeviceGlobalPtr, bool IsDeviceImageScoped, QueueImplPtr Queue,
    size_t NumBytes, size_t Offset, const void *SrcMem,
    const std::vector<sycl::detail::pi::PiEvent> &DepEvents,
    sycl::detail::pi::PiEvent *OutEvent) {
  copy_to_device_global(DeviceGlobalPtr, IsDeviceImageScoped, Queue, NumBytes,
                        Offset, SrcMem, DepEvents, OutEvent, nullptr);
}

void MemoryManager::copy_from_device_global(
    const void *DeviceGlobalPtr, bool IsDeviceImageScoped, QueueImplPtr Queue,
    size_t NumBytes, size_t Offset, void *DstMem,
    const std::vector<sycl::detail::pi::PiEvent> &DepEvents,
    sycl::detail::pi::PiEvent *OutEvent,
    const detail::EventImplPtr &OutEventImpl) {
  DeviceGlobalMapEntry *DGEntry =
      detail::ProgramManager::getInstance().getDeviceGlobalEntry(
          DeviceGlobalPtr);
  assert(DGEntry &&
         DGEntry->MIsDeviceImageScopeDecorated == IsDeviceImageScoped &&
         "Invalid copy operation for device_global.");
  assert(DGEntry->MDeviceGlobalTSize >= Offset + NumBytes &&
         "Copy from device_global is out of bounds.");

  if (IsDeviceImageScoped)
    memcpyFromDeviceGlobalDirect(Queue, DGEntry, NumBytes, Offset, DstMem,
                                 DepEvents, OutEvent);
  else
    memcpyFromDeviceGlobalUSM(Queue, DGEntry, NumBytes, Offset, DstMem,
                              DepEvents, OutEvent, OutEventImpl);
}

// TODO: This function will remain until ABI-breaking change
void MemoryManager::copy_from_device_global(
    const void *DeviceGlobalPtr, bool IsDeviceImageScoped, QueueImplPtr Queue,
    size_t NumBytes, size_t Offset, void *DstMem,
    const std::vector<sycl::detail::pi::PiEvent> &DepEvents,
    sycl::detail::pi::PiEvent *OutEvent) {
  copy_from_device_global(DeviceGlobalPtr, IsDeviceImageScoped, Queue, NumBytes,
                          Offset, DstMem, DepEvents, OutEvent, nullptr);
}

// Command buffer methods
void MemoryManager::ext_oneapi_copyD2D_cmd_buffer(
    sycl::detail::ContextImplPtr Context,
    sycl::detail::pi::PiExtCommandBuffer CommandBuffer, SYCLMemObjI *SYCLMemObj,
    void *SrcMem, unsigned int DimSrc, sycl::range<3> SrcSize,
    sycl::range<3> SrcAccessRange, sycl::id<3> SrcOffset,
    unsigned int SrcElemSize, void *DstMem, unsigned int DimDst,
    sycl::range<3> DstSize, sycl::range<3> DstAccessRange,
    sycl::id<3> DstOffset, unsigned int DstElemSize,
    std::vector<sycl::detail::pi::PiExtSyncPoint> Deps,
    sycl::detail::pi::PiExtSyncPoint *OutSyncPoint) {
  assert(SYCLMemObj && "The SYCLMemObj is nullptr");
  (void)DstAccessRange;

  const PluginPtr &Plugin = Context->getPlugin();

  detail::SYCLMemObjI::MemObjType MemType = SYCLMemObj->getType();
  TermPositions SrcPos, DstPos;
  prepTermPositions(SrcPos, DimSrc, MemType);
  prepTermPositions(DstPos, DimDst, MemType);

  size_t DstXOffBytes = DstOffset[DstPos.XTerm] * DstElemSize;
  size_t SrcXOffBytes = SrcOffset[SrcPos.XTerm] * SrcElemSize;
  size_t SrcAccessRangeWidthBytes = SrcAccessRange[SrcPos.XTerm] * SrcElemSize;
  size_t DstSzWidthBytes = DstSize[DstPos.XTerm] * DstElemSize;
  size_t SrcSzWidthBytes = SrcSize[SrcPos.XTerm] * SrcElemSize;

  if (MemType != detail::SYCLMemObjI::MemObjType::Buffer) {
    throw sycl::exception(sycl::make_error_code(sycl::errc::invalid),
                          "Images are not supported in Graphs");
  }

  if (1 == DimDst && 1 == DimSrc) {
    Plugin->call<PiApiKind::piextCommandBufferMemBufferCopy>(
        CommandBuffer, sycl::detail::pi::cast<sycl::detail::pi::PiMem>(SrcMem),
        sycl::detail::pi::cast<sycl::detail::pi::PiMem>(DstMem), SrcXOffBytes,
        DstXOffBytes, SrcAccessRangeWidthBytes, Deps.size(), Deps.data(),
        OutSyncPoint);
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

    pi_buff_rect_offset_struct SrcOrigin{SrcXOffBytes, SrcOffset[SrcPos.YTerm],
                                         SrcOffset[SrcPos.ZTerm]};
    pi_buff_rect_offset_struct DstOrigin{DstXOffBytes, DstOffset[DstPos.YTerm],
                                         DstOffset[DstPos.ZTerm]};
    pi_buff_rect_region_struct Region{SrcAccessRangeWidthBytes,
                                      SrcAccessRange[SrcPos.YTerm],
                                      SrcAccessRange[SrcPos.ZTerm]};

    Plugin->call<PiApiKind::piextCommandBufferMemBufferCopyRect>(
        CommandBuffer, sycl::detail::pi::cast<sycl::detail::pi::PiMem>(SrcMem),
        sycl::detail::pi::cast<sycl::detail::pi::PiMem>(DstMem), &SrcOrigin,
        &DstOrigin, &Region, SrcRowPitch, SrcSlicePitch, DstRowPitch,
        DstSlicePitch, Deps.size(), Deps.data(), OutSyncPoint);
  }
}

void MemoryManager::ext_oneapi_copyD2H_cmd_buffer(
    sycl::detail::ContextImplPtr Context,
    sycl::detail::pi::PiExtCommandBuffer CommandBuffer, SYCLMemObjI *SYCLMemObj,
    void *SrcMem, unsigned int DimSrc, sycl::range<3> SrcSize,
    sycl::range<3> SrcAccessRange, sycl::id<3> SrcOffset,
    unsigned int SrcElemSize, char *DstMem, unsigned int DimDst,
    sycl::range<3> DstSize, sycl::id<3> DstOffset, unsigned int DstElemSize,
    std::vector<sycl::detail::pi::PiExtSyncPoint> Deps,
    sycl::detail::pi::PiExtSyncPoint *OutSyncPoint) {
  assert(SYCLMemObj && "The SYCLMemObj is nullptr");

  const PluginPtr &Plugin = Context->getPlugin();

  detail::SYCLMemObjI::MemObjType MemType = SYCLMemObj->getType();
  TermPositions SrcPos, DstPos;
  prepTermPositions(SrcPos, DimSrc, MemType);
  prepTermPositions(DstPos, DimDst, MemType);

  size_t DstXOffBytes = DstOffset[DstPos.XTerm] * DstElemSize;
  size_t SrcXOffBytes = SrcOffset[SrcPos.XTerm] * SrcElemSize;
  size_t SrcAccessRangeWidthBytes = SrcAccessRange[SrcPos.XTerm] * SrcElemSize;
  size_t DstSzWidthBytes = DstSize[DstPos.XTerm] * DstElemSize;
  size_t SrcSzWidthBytes = SrcSize[SrcPos.XTerm] * SrcElemSize;

  if (MemType != detail::SYCLMemObjI::MemObjType::Buffer) {
    throw sycl::exception(sycl::make_error_code(sycl::errc::invalid),
                          "Images are not supported in Graphs");
  }

  if (1 == DimDst && 1 == DimSrc) {
    pi_result Result =
        Plugin->call_nocheck<PiApiKind::piextCommandBufferMemBufferRead>(
            CommandBuffer,
            sycl::detail::pi::cast<sycl::detail::pi::PiMem>(SrcMem),
            SrcXOffBytes, SrcAccessRangeWidthBytes, DstMem + DstXOffBytes,
            Deps.size(), Deps.data(), OutSyncPoint);

    if (Result == PI_ERROR_INVALID_OPERATION) {
      throw sycl::exception(
          sycl::make_error_code(sycl::errc::feature_not_supported),
          "Device-to-host buffer copy command not supported by graph backend");
    } else {
      Plugin->checkPiResult(Result);
    }
  } else {
    size_t BufferRowPitch = (1 == DimSrc) ? 0 : SrcSzWidthBytes;
    size_t BufferSlicePitch =
        (3 == DimSrc) ? SrcSzWidthBytes * SrcSize[SrcPos.YTerm] : 0;
    size_t HostRowPitch = (1 == DimDst) ? 0 : DstSzWidthBytes;
    size_t HostSlicePitch =
        (3 == DimDst) ? DstSzWidthBytes * DstSize[DstPos.YTerm] : 0;

    pi_buff_rect_offset_struct BufferOffset{
        SrcXOffBytes, SrcOffset[SrcPos.YTerm], SrcOffset[SrcPos.ZTerm]};
    pi_buff_rect_offset_struct HostOffset{DstXOffBytes, DstOffset[DstPos.YTerm],
                                          DstOffset[DstPos.ZTerm]};
    pi_buff_rect_region_struct RectRegion{SrcAccessRangeWidthBytes,
                                          SrcAccessRange[SrcPos.YTerm],
                                          SrcAccessRange[SrcPos.ZTerm]};

    pi_result Result =
        Plugin->call_nocheck<PiApiKind::piextCommandBufferMemBufferReadRect>(
            CommandBuffer,
            sycl::detail::pi::cast<sycl::detail::pi::PiMem>(SrcMem),
            &BufferOffset, &HostOffset, &RectRegion, BufferRowPitch,
            BufferSlicePitch, HostRowPitch, HostSlicePitch, DstMem, Deps.size(),
            Deps.data(), OutSyncPoint);
    if (Result == PI_ERROR_INVALID_OPERATION) {
      throw sycl::exception(
          sycl::make_error_code(sycl::errc::feature_not_supported),
          "Device-to-host buffer copy command not supported by graph backend");
    } else {
      Plugin->checkPiResult(Result);
    }
  }
}

void MemoryManager::ext_oneapi_copyH2D_cmd_buffer(
    sycl::detail::ContextImplPtr Context,
    sycl::detail::pi::PiExtCommandBuffer CommandBuffer, SYCLMemObjI *SYCLMemObj,
    char *SrcMem, unsigned int DimSrc, sycl::range<3> SrcSize,
    sycl::id<3> SrcOffset, unsigned int SrcElemSize, void *DstMem,
    unsigned int DimDst, sycl::range<3> DstSize, sycl::range<3> DstAccessRange,
    sycl::id<3> DstOffset, unsigned int DstElemSize,
    std::vector<sycl::detail::pi::PiExtSyncPoint> Deps,
    sycl::detail::pi::PiExtSyncPoint *OutSyncPoint) {
  assert(SYCLMemObj && "The SYCLMemObj is nullptr");

  const PluginPtr &Plugin = Context->getPlugin();

  detail::SYCLMemObjI::MemObjType MemType = SYCLMemObj->getType();
  TermPositions SrcPos, DstPos;
  prepTermPositions(SrcPos, DimSrc, MemType);
  prepTermPositions(DstPos, DimDst, MemType);

  size_t DstXOffBytes = DstOffset[DstPos.XTerm] * DstElemSize;
  size_t SrcXOffBytes = SrcOffset[SrcPos.XTerm] * SrcElemSize;
  size_t DstAccessRangeWidthBytes = DstAccessRange[DstPos.XTerm] * DstElemSize;
  size_t DstSzWidthBytes = DstSize[DstPos.XTerm] * DstElemSize;
  size_t SrcSzWidthBytes = SrcSize[SrcPos.XTerm] * SrcElemSize;

  if (MemType != detail::SYCLMemObjI::MemObjType::Buffer) {
    throw sycl::exception(sycl::make_error_code(sycl::errc::invalid),
                          "Images are not supported in Graphs");
  }

  if (1 == DimDst && 1 == DimSrc) {
    pi_result Result =
        Plugin->call_nocheck<PiApiKind::piextCommandBufferMemBufferWrite>(
            CommandBuffer,
            sycl::detail::pi::cast<sycl::detail::pi::PiMem>(DstMem),
            DstXOffBytes, DstAccessRangeWidthBytes, SrcMem + SrcXOffBytes,
            Deps.size(), Deps.data(), OutSyncPoint);

    if (Result == PI_ERROR_INVALID_OPERATION) {
      throw sycl::exception(
          sycl::make_error_code(sycl::errc::feature_not_supported),
          "Host-to-device buffer copy command not supported by graph backend");
    } else {
      Plugin->checkPiResult(Result);
    }
  } else {
    size_t BufferRowPitch = (1 == DimDst) ? 0 : DstSzWidthBytes;
    size_t BufferSlicePitch =
        (3 == DimDst) ? DstSzWidthBytes * DstSize[DstPos.YTerm] : 0;
    size_t HostRowPitch = (1 == DimSrc) ? 0 : SrcSzWidthBytes;
    size_t HostSlicePitch =
        (3 == DimSrc) ? SrcSzWidthBytes * SrcSize[SrcPos.YTerm] : 0;

    pi_buff_rect_offset_struct BufferOffset{
        DstXOffBytes, DstOffset[DstPos.YTerm], DstOffset[DstPos.ZTerm]};
    pi_buff_rect_offset_struct HostOffset{SrcXOffBytes, SrcOffset[SrcPos.YTerm],
                                          SrcOffset[SrcPos.ZTerm]};
    pi_buff_rect_region_struct RectRegion{DstAccessRangeWidthBytes,
                                          DstAccessRange[DstPos.YTerm],
                                          DstAccessRange[DstPos.ZTerm]};

    pi_result Result =
        Plugin->call_nocheck<PiApiKind::piextCommandBufferMemBufferWriteRect>(
            CommandBuffer,
            sycl::detail::pi::cast<sycl::detail::pi::PiMem>(DstMem),
            &BufferOffset, &HostOffset, &RectRegion, BufferRowPitch,
            BufferSlicePitch, HostRowPitch, HostSlicePitch, SrcMem, Deps.size(),
            Deps.data(), OutSyncPoint);

    if (Result == PI_ERROR_INVALID_OPERATION) {
      throw sycl::exception(
          sycl::make_error_code(sycl::errc::feature_not_supported),
          "Host-to-device buffer copy command not supported by graph backend");
    } else {
      Plugin->checkPiResult(Result);
    }
  }
}

void MemoryManager::ext_oneapi_copy_usm_cmd_buffer(
    ContextImplPtr Context, const void *SrcMem,
    sycl::detail::pi::PiExtCommandBuffer CommandBuffer, size_t Len,
    void *DstMem, std::vector<sycl::detail::pi::PiExtSyncPoint> Deps,
    sycl::detail::pi::PiExtSyncPoint *OutSyncPoint) {
  if (!SrcMem || !DstMem)
    throw runtime_error("NULL pointer argument in memory copy operation.",
                        PI_ERROR_INVALID_VALUE);

  const PluginPtr &Plugin = Context->getPlugin();
  pi_result Result =
      Plugin->call_nocheck<PiApiKind::piextCommandBufferMemcpyUSM>(
          CommandBuffer, DstMem, SrcMem, Len, Deps.size(), Deps.data(),
          OutSyncPoint);
  if (Result == PI_ERROR_INVALID_OPERATION) {
    throw sycl::exception(
        sycl::make_error_code(sycl::errc::feature_not_supported),
        "USM copy command not supported by graph backend");
  } else {
    Plugin->checkPiResult(Result);
  }
}

void MemoryManager::ext_oneapi_fill_usm_cmd_buffer(
    sycl::detail::ContextImplPtr Context,
    sycl::detail::pi::PiExtCommandBuffer CommandBuffer, void *DstMem,
    size_t Len, int Pattern, std::vector<sycl::detail::pi::PiExtSyncPoint> Deps,
    sycl::detail::pi::PiExtSyncPoint *OutSyncPoint) {

  if (!DstMem)
    throw runtime_error("NULL pointer argument in memory fill operation.",
                        PI_ERROR_INVALID_VALUE);

  const PluginPtr &Plugin = Context->getPlugin();
  // Pattern is interpreted as an unsigned char so pattern size is always 1.
  size_t PatternSize = 1;
  Plugin->call<PiApiKind::piextCommandBufferFillUSM>(
      CommandBuffer, DstMem, &Pattern, PatternSize, Len, Deps.size(),
      Deps.data(), OutSyncPoint);
}

void MemoryManager::ext_oneapi_fill_cmd_buffer(
    sycl::detail::ContextImplPtr Context,
    sycl::detail::pi::PiExtCommandBuffer CommandBuffer, SYCLMemObjI *SYCLMemObj,
    void *Mem, size_t PatternSize, const char *Pattern, unsigned int Dim,
    sycl::range<3> Size, sycl::range<3> AccessRange, sycl::id<3> AccessOffset,
    unsigned int ElementSize,
    std::vector<sycl::detail::pi::PiExtSyncPoint> Deps,
    sycl::detail::pi::PiExtSyncPoint *OutSyncPoint) {
  (void)Size;
  assert(SYCLMemObj && "The SYCLMemObj is nullptr");

  const PluginPtr &Plugin = Context->getPlugin();
  if (SYCLMemObj->getType() != detail::SYCLMemObjI::MemObjType::Buffer) {
    throw sycl::exception(sycl::make_error_code(sycl::errc::invalid),
                          "Images are not supported in Graphs");
  }
  if (Dim <= 1) {
    Plugin->call<PiApiKind::piextCommandBufferMemBufferFill>(
        CommandBuffer, pi::cast<sycl::detail::pi::PiMem>(Mem), Pattern,
        PatternSize, AccessOffset[0] * ElementSize,
        AccessRange[0] * ElementSize, Deps.size(), Deps.data(), OutSyncPoint);
    return;
  }
  throw runtime_error("Not supported configuration of fill requested",
                      PI_ERROR_INVALID_OPERATION);
}

void MemoryManager::ext_oneapi_prefetch_usm_cmd_buffer(
    sycl::detail::ContextImplPtr Context,
    sycl::detail::pi::PiExtCommandBuffer CommandBuffer, void *Mem,
    size_t Length, std::vector<sycl::detail::pi::PiExtSyncPoint> Deps,
    sycl::detail::pi::PiExtSyncPoint *OutSyncPoint) {
  assert(!Context->is_host() && "Host queue not supported in prefetch_usm.");

  const PluginPtr &Plugin = Context->getPlugin();
  Plugin->call<PiApiKind::piextCommandBufferPrefetchUSM>(
      CommandBuffer, Mem, Length, _pi_usm_migration_flags(0), Deps.size(),
      Deps.data(), OutSyncPoint);
}

void MemoryManager::ext_oneapi_advise_usm_cmd_buffer(
    sycl::detail::ContextImplPtr Context,
    sycl::detail::pi::PiExtCommandBuffer CommandBuffer, const void *Mem,
    size_t Length, pi_mem_advice Advice,
    std::vector<sycl::detail::pi::PiExtSyncPoint> Deps,
    sycl::detail::pi::PiExtSyncPoint *OutSyncPoint) {
  assert(!Context->is_host() && "Host queue not supported in advise_usm.");

  const PluginPtr &Plugin = Context->getPlugin();
  Plugin->call<PiApiKind::piextCommandBufferAdviseUSM>(
      CommandBuffer, Mem, Length, Advice, Deps.size(), Deps.data(),
      OutSyncPoint);
}

void MemoryManager::copy_image_bindless(
    void *Src, QueueImplPtr Queue, void *Dst,
    const sycl::detail::pi::PiMemImageDesc &Desc,
    const sycl::detail::pi::PiMemImageFormat &Format,
    const sycl::detail::pi::PiImageCopyFlags Flags,
    sycl::detail::pi::PiImageOffset SrcOffset,
    sycl::detail::pi::PiImageOffset DstOffset,
    sycl::detail::pi::PiImageRegion HostExtent,
    sycl::detail::pi::PiImageRegion CopyExtent,
    const std::vector<sycl::detail::pi::PiEvent> &DepEvents,
    sycl::detail::pi::PiEvent *OutEvent) {

  assert(!Queue->getContextImplPtr()->is_host() &&
         "Host queue not supported in copy_image_bindless.");
  assert((Flags == (sycl::detail::pi::PiImageCopyFlags)
                       ext::oneapi::experimental::image_copy_flags::HtoD ||
          Flags == (sycl::detail::pi::PiImageCopyFlags)
                       ext::oneapi::experimental::image_copy_flags::DtoH) &&
         "Invalid flags passed to copy_image_bindless.");
  if (!Dst || !Src)
    throw sycl::exception(
        sycl::make_error_code(errc::invalid),
        "NULL pointer argument in bindless image copy operation.");

  const detail::PluginPtr &Plugin = Queue->getPlugin();
  Plugin->call<PiApiKind::piextMemImageCopy>(
      Queue->getHandleRef(), Dst, Src, &Format, &Desc, Flags, &SrcOffset,
      &DstOffset, &CopyExtent, &HostExtent, DepEvents.size(), DepEvents.data(),
      OutEvent);
}

} // namespace detail
} // namespace _V1
} // namespace sycl
