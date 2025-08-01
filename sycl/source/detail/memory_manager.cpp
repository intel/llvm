//==-------------- memory_manager.cpp --------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ur_api.h"
#include <detail/context_impl.hpp>
#include <detail/device_image_impl.hpp>
#include <detail/event_impl.hpp>
#include <detail/mem_alloc_helper.hpp>
#include <detail/memory_manager.hpp>
#include <detail/queue_impl.hpp>
#include <detail/ur_utils.hpp>
#include <detail/xpti_registry.hpp>

#include <sycl/detail/ur.hpp>
#include <sycl/ext/oneapi/bindless_images_memory.hpp>
#include <sycl/usm/usm_enums.hpp>
#include <sycl/usm/usm_pointer_info.hpp>

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

static void waitForEvents(events_range Events) {
  // Assuming all events will be on the same device or
  // devices associated with the same Backend.
  if (!Events.empty()) {
    adapter_impl &Adapter = Events.front().getAdapter();
    std::vector<ur_event_handle_t> UrEvents(Events.size());
    std::transform(Events.begin(), Events.end(), UrEvents.begin(),
                   [](event_impl &Event) { return Event.getHandle(); });
    // TODO: Why this condition??? Added during PI Removal in
    // https://github.com/intel/llvm/pull/14145 with no explanation.
    // Should we just filter out all `nullptr`, not only the one in the first
    // element?
    assert(!UrEvents.empty() && UrEvents[0]);
    if (!UrEvents.empty() && UrEvents[0]) {
      Adapter.call<UrApiKind::urEventWait>(UrEvents.size(), &UrEvents[0]);
    }
  }
}

void memBufferCreateHelper(adapter_impl &Adapter, ur_context_handle_t Ctx,
                           ur_mem_flags_t Flags, size_t Size,
                           ur_mem_handle_t *RetMem,
                           const ur_buffer_properties_t *Props) {
#ifdef XPTI_ENABLE_INSTRUMENTATION
  uint64_t CorrID = 0;
#endif
  // We only want to instrument urMemBufferCreate
  {
#ifdef XPTI_ENABLE_INSTRUMENTATION
    CorrID =
        emitMemAllocBeginTrace(0 /* mem object */, Size, 0 /* guard zone */);
    xpti::utils::finally _{[&] {
      // C-style cast is required for MSVC
      uintptr_t MemObjID = (uintptr_t)(*RetMem);
      ur_native_handle_t Ptr = 0;
      // Always use call_nocheck here, because call may throw an exception,
      // and this lambda will be called from destructor, which in combination
      // rewards us with UB.
      // When doing buffer interop we don't know what device the memory should
      // be resident on, so pass nullptr for Device param. Buffer interop may
      // not be supported by all backends.
      Adapter.call_nocheck<UrApiKind::urMemGetNativeHandle>(
          *RetMem, /*Dev*/ nullptr, &Ptr);
      emitMemAllocEndTrace(MemObjID, (uintptr_t)(Ptr), Size, 0 /* guard zone */,
                           CorrID);
    }};
#endif
    if (Size)
      Adapter.call<UrApiKind::urMemBufferCreate>(Ctx, Flags, Size, Props,
                                                 RetMem);
  }
}

void memReleaseHelper(adapter_impl &Adapter, ur_mem_handle_t Mem) {
  // FIXME urMemRelease does not guarante memory release. It is only true if
  // reference counter is 1. However, SYCL runtime currently only calls
  // urMemRetain only for OpenCL interop
#ifdef XPTI_ENABLE_INSTRUMENTATION
  uint64_t CorrID = 0;
  // C-style cast is required for MSVC
  uintptr_t MemObjID = (uintptr_t)(Mem);
  uintptr_t Ptr = 0;
  // Do not make unnecessary UR calls without instrumentation enabled
  if (xptiTraceEnabled()) {
    ur_native_handle_t PtrHandle = 0;
    // When doing buffer interop we don't know what device the memory should be
    // resident on, so pass nullptr for Device param. Buffer interop may not be
    // supported by all backends.
    Adapter.call_nocheck<UrApiKind::urMemGetNativeHandle>(Mem, /*Dev*/ nullptr,
                                                          &PtrHandle);
    Ptr = (uintptr_t)(PtrHandle);
  }
#endif
  // We only want to instrument urMemRelease
  {
#ifdef XPTI_ENABLE_INSTRUMENTATION
    CorrID = emitMemReleaseBeginTrace(MemObjID, Ptr);
    xpti::utils::finally _{
        [&] { emitMemReleaseEndTrace(MemObjID, Ptr, CorrID); }};
#endif
    Adapter.call<UrApiKind::urMemRelease>(Mem);
  }
}

void memBufferMapHelper(adapter_impl &Adapter, ur_queue_handle_t Queue,
                        ur_mem_handle_t Buffer, bool Blocking,
                        ur_map_flags_t Flags, size_t Offset, size_t Size,
                        uint32_t NumEvents, const ur_event_handle_t *WaitList,
                        ur_event_handle_t *Event, void **RetMap) {
#ifdef XPTI_ENABLE_INSTRUMENTATION
  uint64_t CorrID = 0;
  uintptr_t MemObjID = (uintptr_t)(Buffer);
#endif
  // We only want to instrument urEnqueueMemBufferMap

#ifdef XPTI_ENABLE_INSTRUMENTATION
  CorrID = emitMemAllocBeginTrace(MemObjID, Size, 0 /* guard zone */);
  xpti::utils::finally _{[&] {
    emitMemAllocEndTrace(MemObjID, (uintptr_t)(*RetMap), Size,
                         0 /* guard zone */, CorrID);
  }};
#endif
  Adapter.call<UrApiKind::urEnqueueMemBufferMap>(Queue, Buffer, Blocking, Flags,
                                                 Offset, Size, NumEvents,
                                                 WaitList, Event, RetMap);
}

void memUnmapHelper(adapter_impl &Adapter, ur_queue_handle_t Queue,
                    ur_mem_handle_t Mem, void *MappedPtr, uint32_t NumEvents,
                    const ur_event_handle_t *WaitList,
                    ur_event_handle_t *Event) {
#ifdef XPTI_ENABLE_INSTRUMENTATION
  uint64_t CorrID = 0;
  uintptr_t MemObjID = (uintptr_t)(Mem);
  uintptr_t Ptr = (uintptr_t)(MappedPtr);
#endif
  // We only want to instrument urEnqueueMemUnmap
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
      Adapter.call_nocheck<UrApiKind::urEventWait>(1, Event);
      emitMemReleaseEndTrace(MemObjID, Ptr, CorrID);
    }};
#endif
    Adapter.call<UrApiKind::urEnqueueMemUnmap>(Queue, Mem, MappedPtr, NumEvents,
                                               WaitList, Event);
  }
}

void MemoryManager::release(context_impl *TargetContext, SYCLMemObjI *MemObj,
                            void *MemAllocation, events_range DepEvents,
                            ur_event_handle_t &OutEvent) {
  // There is no async API for memory releasing. Explicitly wait for all
  // dependency events and return empty event.
  waitForEvents(DepEvents);
  OutEvent = nullptr;
  XPTIRegistry::bufferReleaseNotification(MemObj, MemAllocation);
  MemObj->releaseMem(TargetContext, MemAllocation);
}

void MemoryManager::releaseMemObj(context_impl *TargetContext,
                                  SYCLMemObjI *MemObj, void *MemAllocation,
                                  void *UserPtr) {
  if (UserPtr == MemAllocation) {
    // Do nothing as it's user provided memory.
    return;
  }

  if (!TargetContext) {
    MemObj->releaseHostMem(MemAllocation);
    return;
  }

  adapter_impl &Adapter = TargetContext->getAdapter();
  memReleaseHelper(Adapter, ur::cast<ur_mem_handle_t>(MemAllocation));
}

void *MemoryManager::allocate(context_impl *TargetContext, SYCLMemObjI *MemObj,
                              bool InitFromUserData, void *HostPtr,
                              events_range DepEvents,
                              ur_event_handle_t &OutEvent) {
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
}

void *MemoryManager::allocateInteropMemObject(
    context_impl *TargetContext, void *UserPtr,
    const EventImplPtr &InteropEvent, context_impl *InteropContext,
    const sycl::property_list &, ur_event_handle_t &OutEventToWait) {
  (void)TargetContext;
  (void)InteropContext;
  // If memory object is created with interop c'tor return cl_mem as is.
  assert(TargetContext == InteropContext && "Expected matching contexts");

  OutEventToWait = InteropEvent->getHandle();
  // Retain the event since it will be released during alloca command
  // destruction
  if (nullptr != OutEventToWait) {
    adapter_impl &Adapter = InteropEvent->getAdapter();
    Adapter.call<UrApiKind::urEventRetain>(OutEventToWait);
  }
  return UserPtr;
}

static ur_mem_flags_t getMemObjCreationFlags(void *UserPtr,
                                             bool HostPtrReadOnly) {
  // Create read_write mem object to handle arbitrary uses.
  ur_mem_flags_t Result =
      HostPtrReadOnly ? UR_MEM_FLAG_READ_ONLY : UR_MEM_FLAG_READ_WRITE;
  if (UserPtr)
    Result |= UR_MEM_FLAG_USE_HOST_POINTER;
  return Result;
}

void *MemoryManager::allocateImageObject(context_impl *TargetContext,
                                         void *UserPtr, bool HostPtrReadOnly,
                                         const ur_image_desc_t &Desc,
                                         const ur_image_format_t &Format,
                                         const sycl::property_list &) {
  ur_mem_flags_t CreationFlags =
      getMemObjCreationFlags(UserPtr, HostPtrReadOnly);

  ur_mem_handle_t NewMem = nullptr;
  adapter_impl &Adapter = TargetContext->getAdapter();
  Adapter.call<UrApiKind::urMemImageCreate>(TargetContext->getHandleRef(),
                                            CreationFlags, &Format, &Desc,
                                            UserPtr, &NewMem);
  return NewMem;
}

void *
MemoryManager::allocateBufferObject(context_impl *TargetContext, void *UserPtr,
                                    bool HostPtrReadOnly, const size_t Size,
                                    const sycl::property_list &PropsList) {
  ur_mem_flags_t CreationFlags =
      getMemObjCreationFlags(UserPtr, HostPtrReadOnly);
  if (PropsList.has_property<
          sycl::ext::oneapi::property::buffer::use_pinned_host_memory>())
    CreationFlags |= UR_MEM_FLAG_ALLOC_HOST_POINTER;

  ur_mem_handle_t NewMem = nullptr;
  adapter_impl &Adapter = TargetContext->getAdapter();

  ur_buffer_properties_t AllocProps = {UR_STRUCTURE_TYPE_BUFFER_PROPERTIES,
                                       nullptr, UserPtr};

  void **Next = &AllocProps.pNext;
  ur_buffer_alloc_location_properties_t LocationProperties = {
      UR_STRUCTURE_TYPE_BUFFER_ALLOC_LOCATION_PROPERTIES, nullptr, 0};
  if (PropsList.has_property<property::buffer::detail::buffer_location>() &&
      TargetContext->isBufferLocationSupported()) {
    LocationProperties.location =
        PropsList.get_property<property::buffer::detail::buffer_location>()
            .get_buffer_location();
    *Next = &LocationProperties;
    Next = &LocationProperties.pNext;
  }

  ur_buffer_channel_properties_t ChannelProperties = {
      UR_STRUCTURE_TYPE_BUFFER_CHANNEL_PROPERTIES, nullptr, 0};
  if (PropsList.has_property<property::buffer::mem_channel>()) {
    ChannelProperties.channel =
        PropsList.get_property<property::buffer::mem_channel>().get_channel();
    *Next = &ChannelProperties;
  }

  memBufferCreateHelper(Adapter, TargetContext->getHandleRef(), CreationFlags,
                        Size, &NewMem, &AllocProps);
  return NewMem;
}

void *MemoryManager::allocateMemBuffer(context_impl *TargetContext,
                                       SYCLMemObjI *MemObj, void *UserPtr,
                                       bool HostPtrReadOnly, size_t Size,
                                       const EventImplPtr &InteropEvent,
                                       context_impl *InteropContext,
                                       const sycl::property_list &PropsList,
                                       ur_event_handle_t &OutEventToWait) {
  void *MemPtr;
  if (!TargetContext)
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
    context_impl *TargetContext, SYCLMemObjI *MemObj, void *UserPtr,
    bool HostPtrReadOnly, size_t Size, const ur_image_desc_t &Desc,
    const ur_image_format_t &Format, const EventImplPtr &InteropEvent,
    context_impl *InteropContext, const sycl::property_list &PropsList,
    ur_event_handle_t &OutEventToWait) {
  if (!TargetContext)
    return allocateHostMemory(MemObj, UserPtr, HostPtrReadOnly, Size,
                              PropsList);
  if (UserPtr && InteropContext)
    return allocateInteropMemObject(TargetContext, UserPtr, InteropEvent,
                                    InteropContext, PropsList, OutEventToWait);
  return allocateImageObject(TargetContext, UserPtr, HostPtrReadOnly, Desc,
                             Format, PropsList);
}

void *MemoryManager::allocateMemSubBuffer(context_impl *TargetContext,
                                          void *ParentMemObj, size_t ElemSize,
                                          size_t Offset, range<3> Range,
                                          events_range DepEvents,
                                          ur_event_handle_t &OutEvent) {
  waitForEvents(DepEvents);
  OutEvent = nullptr;

  if (!TargetContext)
    return static_cast<void *>(static_cast<char *>(ParentMemObj) + Offset);

  size_t SizeInBytes = ElemSize;
  for (size_t I = 0; I < 3; ++I)
    SizeInBytes *= Range[I];

  ur_result_t Error = UR_RESULT_SUCCESS;
  ur_buffer_region_t Region = {UR_STRUCTURE_TYPE_BUFFER_REGION, nullptr, Offset,
                               SizeInBytes};
  ur_mem_handle_t NewMem;
  adapter_impl &Adapter = TargetContext->getAdapter();
  Error = Adapter.call_nocheck<UrApiKind::urMemBufferPartition>(
      ur::cast<ur_mem_handle_t>(ParentMemObj), UR_MEM_FLAG_READ_WRITE,
      UR_BUFFER_CREATE_TYPE_REGION, &Region, &NewMem);
  if (Error == UR_RESULT_ERROR_MISALIGNED_SUB_BUFFER_OFFSET)
    throw detail::set_ur_error(
        exception(make_error_code(errc::invalid),
                  "Specified offset of the sub-buffer being constructed is not "
                  "a multiple of the memory base address alignment"),
        Error);

  Adapter.checkUrResult(Error);

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

void copyH2D(queue_impl &TgtQueue, SYCLMemObjI *SYCLMemObj, char *SrcMem,
             unsigned int DimSrc, sycl::range<3> SrcSize,
             sycl::range<3> SrcAccessRange, sycl::id<3> SrcOffset,
             unsigned int SrcElemSize, ur_mem_handle_t DstMem,
             unsigned int DimDst, sycl::range<3> DstSize,
             sycl::range<3> DstAccessRange, sycl::id<3> DstOffset,
             unsigned int DstElemSize, std::vector<ur_event_handle_t> DepEvents,
             ur_event_handle_t &OutEvent) {
  (void)SrcAccessRange;
  assert(SYCLMemObj && "The SYCLMemObj is nullptr");

  const ur_queue_handle_t Queue = TgtQueue.getHandleRef();
  adapter_impl &Adapter = TgtQueue.getAdapter();

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
      Adapter.call<UrApiKind::urEnqueueMemBufferWrite>(
          Queue, DstMem,
          /*blocking_write=*/false, DstXOffBytes, DstAccessRangeWidthBytes,
          SrcMem + SrcXOffBytes, DepEvents.size(), DepEvents.data(), &OutEvent);
    } else {
      size_t BufferRowPitch = (1 == DimDst) ? 0 : DstSzWidthBytes;
      size_t BufferSlicePitch =
          (3 == DimDst) ? DstSzWidthBytes * DstSize[DstPos.YTerm] : 0;
      size_t HostRowPitch = (1 == DimSrc) ? 0 : SrcSzWidthBytes;
      size_t HostSlicePitch =
          (3 == DimSrc) ? SrcSzWidthBytes * SrcSize[SrcPos.YTerm] : 0;

      ur_rect_offset_t BufferOffset{DstXOffBytes, DstOffset[DstPos.YTerm],
                                    DstOffset[DstPos.ZTerm]};
      ur_rect_offset_t HostOffset{SrcXOffBytes, SrcOffset[SrcPos.YTerm],
                                  SrcOffset[SrcPos.ZTerm]};
      ur_rect_region_t RectRegion{DstAccessRangeWidthBytes,
                                  DstAccessRange[DstPos.YTerm],
                                  DstAccessRange[DstPos.ZTerm]};
      Adapter.call<UrApiKind::urEnqueueMemBufferWriteRect>(
          Queue, DstMem,
          /*blocking_write=*/false, BufferOffset, HostOffset, RectRegion,
          BufferRowPitch, BufferSlicePitch, HostRowPitch, HostSlicePitch,
          SrcMem, DepEvents.size(), DepEvents.data(), &OutEvent);
    }
  } else {
    size_t InputRowPitch = (1 == DimDst) ? 0 : DstSzWidthBytes;
    size_t InputSlicePitch =
        (3 == DimDst) ? DstSzWidthBytes * DstSize[DstPos.YTerm] : 0;

    ur_rect_offset_t Origin{DstOffset[DstPos.XTerm], DstOffset[DstPos.YTerm],
                            DstOffset[DstPos.ZTerm]};
    ur_rect_region_t Region{DstAccessRange[DstPos.XTerm],
                            DstAccessRange[DstPos.YTerm],
                            DstAccessRange[DstPos.ZTerm]};
    Adapter.call<UrApiKind::urEnqueueMemImageWrite>(
        Queue, DstMem,
        /*blocking_write=*/false, Origin, Region, InputRowPitch,
        InputSlicePitch, SrcMem, DepEvents.size(), DepEvents.data(), &OutEvent);
  }
}

void copyD2H(queue_impl &SrcQueue, SYCLMemObjI *SYCLMemObj,
             ur_mem_handle_t SrcMem, unsigned int DimSrc,
             sycl::range<3> SrcSize, sycl::range<3> SrcAccessRange,
             sycl::id<3> SrcOffset, unsigned int SrcElemSize, char *DstMem,
             unsigned int DimDst, sycl::range<3> DstSize,
             sycl::range<3> DstAccessRange, sycl::id<3> DstOffset,
             unsigned int DstElemSize, std::vector<ur_event_handle_t> DepEvents,
             ur_event_handle_t &OutEvent) {
  (void)DstAccessRange;
  assert(SYCLMemObj && "The SYCLMemObj is nullptr");

  const ur_queue_handle_t Queue = SrcQueue.getHandleRef();
  adapter_impl &Adapter = SrcQueue.getAdapter();

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
      Adapter.call<UrApiKind::urEnqueueMemBufferRead>(
          Queue, SrcMem,
          /*blocking_read=*/false, SrcXOffBytes, SrcAccessRangeWidthBytes,
          DstMem + DstXOffBytes, DepEvents.size(), DepEvents.data(), &OutEvent);
    } else {
      size_t BufferRowPitch = (1 == DimSrc) ? 0 : SrcSzWidthBytes;
      size_t BufferSlicePitch =
          (3 == DimSrc) ? SrcSzWidthBytes * SrcSize[SrcPos.YTerm] : 0;
      size_t HostRowPitch = (1 == DimDst) ? 0 : DstSzWidthBytes;
      size_t HostSlicePitch =
          (3 == DimDst) ? DstSzWidthBytes * DstSize[DstPos.YTerm] : 0;

      ur_rect_offset_t BufferOffset{SrcXOffBytes, SrcOffset[SrcPos.YTerm],
                                    SrcOffset[SrcPos.ZTerm]};
      ur_rect_offset_t HostOffset{DstXOffBytes, DstOffset[DstPos.YTerm],
                                  DstOffset[DstPos.ZTerm]};
      ur_rect_region_t RectRegion{SrcAccessRangeWidthBytes,
                                  SrcAccessRange[SrcPos.YTerm],
                                  SrcAccessRange[SrcPos.ZTerm]};
      Adapter.call<UrApiKind::urEnqueueMemBufferReadRect>(
          Queue, SrcMem,
          /*blocking_read=*/false, BufferOffset, HostOffset, RectRegion,
          BufferRowPitch, BufferSlicePitch, HostRowPitch, HostSlicePitch,
          DstMem, DepEvents.size(), DepEvents.data(), &OutEvent);
    }
  } else {
    size_t RowPitch = (1 == DimSrc) ? 0 : SrcSzWidthBytes;
    size_t SlicePitch =
        (3 == DimSrc) ? SrcSzWidthBytes * SrcSize[SrcPos.YTerm] : 0;

    ur_rect_offset_t Offset{SrcOffset[SrcPos.XTerm], SrcOffset[SrcPos.YTerm],
                            SrcOffset[SrcPos.ZTerm]};
    ur_rect_region_t Region{SrcAccessRange[SrcPos.XTerm],
                            SrcAccessRange[SrcPos.YTerm],
                            SrcAccessRange[SrcPos.ZTerm]};
    Adapter.call<UrApiKind::urEnqueueMemImageRead>(
        Queue, SrcMem, false, Offset, Region, RowPitch, SlicePitch, DstMem,
        DepEvents.size(), DepEvents.data(), &OutEvent);
  }
}

// Only when memory objects are bound to the same context, so one queue_impl is
// all we need.
void copyD2D(queue_impl &SrcQueue, SYCLMemObjI *SYCLMemObj,
             ur_mem_handle_t SrcMem, unsigned int DimSrc,
             sycl::range<3> SrcSize, sycl::range<3> SrcAccessRange,
             sycl::id<3> SrcOffset, unsigned int SrcElemSize,
             ur_mem_handle_t DstMem, unsigned int DimDst,
             sycl::range<3> DstSize, sycl::range<3>, sycl::id<3> DstOffset,
             unsigned int DstElemSize, std::vector<ur_event_handle_t> DepEvents,
             ur_event_handle_t &OutEvent) {
  assert(SYCLMemObj && "The SYCLMemObj is nullptr");

  const ur_queue_handle_t Queue = SrcQueue.getHandleRef();
  adapter_impl &Adapter = SrcQueue.getAdapter();

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
      Adapter.call<UrApiKind::urEnqueueMemBufferCopy>(
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

      ur_rect_offset_t SrcOrigin{SrcXOffBytes, SrcOffset[SrcPos.YTerm],
                                 SrcOffset[SrcPos.ZTerm]};
      ur_rect_offset_t DstOrigin{DstXOffBytes, DstOffset[DstPos.YTerm],
                                 DstOffset[DstPos.ZTerm]};
      ur_rect_region_t Region{SrcAccessRangeWidthBytes,
                              SrcAccessRange[SrcPos.YTerm],
                              SrcAccessRange[SrcPos.ZTerm]};
      Adapter.call<UrApiKind::urEnqueueMemBufferCopyRect>(
          Queue, SrcMem, DstMem, SrcOrigin, DstOrigin, Region, SrcRowPitch,
          SrcSlicePitch, DstRowPitch, DstSlicePitch, DepEvents.size(),
          DepEvents.data(), &OutEvent);
    }
  } else {
    ur_rect_offset_t SrcOrigin{SrcOffset[SrcPos.XTerm], SrcOffset[SrcPos.YTerm],
                               SrcOffset[SrcPos.ZTerm]};
    ur_rect_offset_t DstOrigin{DstOffset[DstPos.XTerm], DstOffset[DstPos.YTerm],
                               DstOffset[DstPos.ZTerm]};
    ur_rect_region_t Region{SrcAccessRange[SrcPos.XTerm],
                            SrcAccessRange[SrcPos.YTerm],
                            SrcAccessRange[SrcPos.ZTerm]};
    Adapter.call<UrApiKind::urEnqueueMemImageCopy>(
        Queue, SrcMem, DstMem, SrcOrigin, DstOrigin, Region, DepEvents.size(),
        DepEvents.data(), &OutEvent);
  }
}

static void copyH2H(SYCLMemObjI *, char *SrcMem, unsigned int DimSrc,
                    sycl::range<3> SrcSize, sycl::range<3> SrcAccessRange,
                    sycl::id<3> SrcOffset, unsigned int SrcElemSize,
                    char *DstMem, unsigned int DimDst, sycl::range<3> DstSize,
                    sycl::range<3> DstAccessRange, sycl::id<3> DstOffset,
                    unsigned int DstElemSize, std::vector<ur_event_handle_t>,
                    ur_event_handle_t &) {
  if ((DimSrc != 1 || DimDst != 1) &&
      (SrcOffset != id<3>{0, 0, 0} || DstOffset != id<3>{0, 0, 0} ||
       SrcSize != SrcAccessRange || DstSize != DstAccessRange)) {
    throw exception(make_error_code(errc::feature_not_supported),
                    "Not supported configuration of memcpy requested");
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
                         queue_impl *SrcQueue, unsigned int DimSrc,
                         sycl::range<3> SrcSize, sycl::range<3> SrcAccessRange,
                         sycl::id<3> SrcOffset, unsigned int SrcElemSize,
                         void *DstMem, queue_impl *TgtQueue,
                         unsigned int DimDst, sycl::range<3> DstSize,
                         sycl::range<3> DstAccessRange, sycl::id<3> DstOffset,
                         unsigned int DstElemSize,
                         std::vector<ur_event_handle_t> DepEvents,
                         ur_event_handle_t &OutEvent) {

  if (!SrcQueue) {
    if (!TgtQueue)
      copyH2H(SYCLMemObj, (char *)SrcMem, DimSrc, SrcSize, SrcAccessRange,
              SrcOffset, SrcElemSize, (char *)DstMem, DimDst, DstSize,
              DstAccessRange, DstOffset, DstElemSize, std::move(DepEvents),
              OutEvent);
    else
      copyH2D(*TgtQueue, SYCLMemObj, (char *)SrcMem, DimSrc, SrcSize,
              SrcAccessRange, SrcOffset, SrcElemSize,
              ur::cast<ur_mem_handle_t>(DstMem), DimDst, DstSize,
              DstAccessRange, DstOffset, DstElemSize, std::move(DepEvents),
              OutEvent);
  } else {
    if (!TgtQueue)
      copyD2H(*SrcQueue, SYCLMemObj, ur::cast<ur_mem_handle_t>(SrcMem), DimSrc,
              SrcSize, SrcAccessRange, SrcOffset, SrcElemSize, (char *)DstMem,
              DimDst, DstSize, DstAccessRange, DstOffset, DstElemSize,
              std::move(DepEvents), OutEvent);
    else
      copyD2D(*SrcQueue, SYCLMemObj, ur::cast<ur_mem_handle_t>(SrcMem), DimSrc,
              SrcSize, SrcAccessRange, SrcOffset, SrcElemSize,
              ur::cast<ur_mem_handle_t>(DstMem), DimDst, DstSize,
              DstAccessRange, DstOffset, DstElemSize, std::move(DepEvents),
              OutEvent);
  }
}

void MemoryManager::fill(SYCLMemObjI *SYCLMemObj, void *Mem, queue_impl &Queue,
                         size_t PatternSize, const unsigned char *Pattern,
                         unsigned int Dim, sycl::range<3> MemRange,
                         sycl::range<3> AccRange, sycl::id<3> Offset,
                         unsigned int ElementSize,
                         std::vector<ur_event_handle_t> DepEvents,
                         ur_event_handle_t &OutEvent) {
  assert(SYCLMemObj && "The SYCLMemObj is nullptr");

  adapter_impl &Adapter = Queue.getAdapter();

  if (SYCLMemObj->getType() == detail::SYCLMemObjI::MemObjType::Buffer) {

    // 2D and 3D buffers accessors can't have custom range or the data will
    // likely be discontiguous.
    bool RangesUsable = (Dim <= 1) || (MemRange == AccRange);
    // For 2D and 3D buffers, the offset must be 0, or the data will be
    // discontiguous.
    bool OffsetUsable = (Dim <= 1) || (Offset == sycl::id<3>{0, 0, 0});
    size_t RangeMultiplier = AccRange[0] * AccRange[1] * AccRange[2];

    if (RangesUsable && OffsetUsable) {
      Adapter.call<UrApiKind::urEnqueueMemBufferFill>(
          Queue.getHandleRef(), ur::cast<ur_mem_handle_t>(Mem), Pattern,
          PatternSize, Offset[0] * ElementSize, RangeMultiplier * ElementSize,
          DepEvents.size(), DepEvents.data(), &OutEvent);
      return;
    }
    // The sycl::handler uses a parallel_for kernel in the case of unusable
    // Range or Offset, not CG:Fill. So we should not be here.
    throw exception(make_error_code(errc::runtime),
                    "Not supported configuration of fill requested");
  } else {
    // We don't have any backend implementations that support enqueueing a fill
    // on non-buffer mem objects like this. The old UR function was a stub with
    // an abort.
    throw exception(make_error_code(errc::runtime),
                    "Fill operation not supported for the given mem object");
  }
}

void *MemoryManager::map(SYCLMemObjI *, void *Mem, queue_impl &Queue,
                         access::mode AccessMode, unsigned int, sycl::range<3>,
                         sycl::range<3> AccessRange, sycl::id<3> AccessOffset,
                         unsigned int ElementSize,
                         std::vector<ur_event_handle_t> DepEvents,
                         ur_event_handle_t &OutEvent) {
  ur_map_flags_t Flags = 0;

  switch (AccessMode) {
  case access::mode::read:
    Flags |= UR_MAP_FLAG_READ;
    break;
  case access::mode::write:
    Flags |= UR_MAP_FLAG_WRITE;
    break;
  case access::mode::read_write:
  case access::mode::atomic:
    Flags = UR_MAP_FLAG_WRITE | UR_MAP_FLAG_READ;
    break;
  case access::mode::discard_write:
  case access::mode::discard_read_write:
    Flags |= UR_MAP_FLAG_WRITE_INVALIDATE_REGION;
    break;
  }

  AccessOffset[0] *= ElementSize;
  AccessRange[0] *= ElementSize;

  // TODO: Handle offset
  assert(AccessOffset[0] == 0 && "Handle offset");

  void *MappedPtr = nullptr;
  const size_t BytesToMap = AccessRange[0] * AccessRange[1] * AccessRange[2];
  adapter_impl &Adapter = Queue.getAdapter();
  memBufferMapHelper(Adapter, Queue.getHandleRef(),
                     ur::cast<ur_mem_handle_t>(Mem), false, Flags,
                     AccessOffset[0], BytesToMap, DepEvents.size(),
                     DepEvents.data(), &OutEvent, &MappedPtr);
  return MappedPtr;
}

void MemoryManager::unmap(SYCLMemObjI *, void *Mem, queue_impl &Queue,
                          void *MappedPtr,
                          std::vector<ur_event_handle_t> DepEvents,
                          ur_event_handle_t &OutEvent) {
  // All DepEvents are to the same Context.
  // Using the adapter of the Queue.

  adapter_impl &Adapter = Queue.getAdapter();
  memUnmapHelper(Adapter, Queue.getHandleRef(), ur::cast<ur_mem_handle_t>(Mem),
                 MappedPtr, DepEvents.size(), DepEvents.data(), &OutEvent);
}

void MemoryManager::copy_usm(const void *SrcMem, queue_impl &SrcQueue,
                             size_t Len, void *DstMem,
                             std::vector<ur_event_handle_t> DepEvents,
                             ur_event_handle_t *OutEvent) {
  adapter_impl &Adapter = SrcQueue.getAdapter();
  if (!Len) { // no-op, but ensure DepEvents will still be waited on
    if (!DepEvents.empty()) {
      Adapter.call<UrApiKind::urEnqueueEventsWait>(SrcQueue.getHandleRef(),
                                                   DepEvents.size(),
                                                   DepEvents.data(), OutEvent);
    }
    return;
  }

  if (!SrcMem || !DstMem)
    throw exception(make_error_code(errc::invalid),
                    "NULL pointer argument in memory copy operation.");

  Adapter.call<UrApiKind::urEnqueueUSMMemcpy>(SrcQueue.getHandleRef(),
                                              /* blocking */ false, DstMem,
                                              SrcMem, Len, DepEvents.size(),
                                              DepEvents.data(), OutEvent);
}

void MemoryManager::context_copy_usm(const void *SrcMem, context_impl *Context,
                                     size_t Len, void *DstMem) {
  if (!SrcMem || !DstMem)
    throw exception(make_error_code(errc::invalid),
                    "NULL pointer argument in memory copy operation.");
  adapter_impl &Adapter = Context->getAdapter();
  Adapter.call<UrApiKind::urUSMContextMemcpyExp>(Context->getHandleRef(),
                                                 DstMem, SrcMem, Len);
}

void MemoryManager::fill_usm(void *Mem, queue_impl &Queue, size_t Length,
                             const std::vector<unsigned char> &Pattern,
                             std::vector<ur_event_handle_t> DepEvents,
                             ur_event_handle_t *OutEvent) {
  if (!Length) { // no-op, but ensure DepEvents will still be waited on
    if (!DepEvents.empty()) {
      Queue.getAdapter().call<UrApiKind::urEnqueueEventsWait>(
          Queue.getHandleRef(), DepEvents.size(), DepEvents.data(), OutEvent);
    }
    return;
  }

  if (!Mem)
    throw exception(make_error_code(errc::invalid),
                    "NULL pointer argument in memory fill operation.");
  adapter_impl &Adapter = Queue.getAdapter();
  Adapter.call<UrApiKind::urEnqueueUSMFill>(
      Queue.getHandleRef(), Mem, Pattern.size(), Pattern.data(), Length,
      DepEvents.size(), DepEvents.data(), OutEvent);
}

void MemoryManager::prefetch_usm(void *Mem, queue_impl &Queue, size_t Length,
                                 std::vector<ur_event_handle_t> DepEvents,
                                 ur_event_handle_t *OutEvent) {
  adapter_impl &Adapter = Queue.getAdapter();
  Adapter.call<UrApiKind::urEnqueueUSMPrefetch>(Queue.getHandleRef(), Mem,
                                                Length, 0u, DepEvents.size(),
                                                DepEvents.data(), OutEvent);
}

void MemoryManager::advise_usm(const void *Mem, queue_impl &Queue,
                               size_t Length, ur_usm_advice_flags_t Advice,
                               std::vector<ur_event_handle_t> /*DepEvents*/,
                               ur_event_handle_t *OutEvent) {
  adapter_impl &Adapter = Queue.getAdapter();
  Adapter.call<UrApiKind::urEnqueueUSMAdvise>(Queue.getHandleRef(), Mem, Length,
                                              Advice, OutEvent);
}

void MemoryManager::copy_2d_usm(const void *SrcMem, size_t SrcPitch,
                                queue_impl &Queue, void *DstMem,
                                size_t DstPitch, size_t Width, size_t Height,
                                std::vector<ur_event_handle_t> DepEvents,
                                ur_event_handle_t *OutEvent) {
  if (Width == 0 || Height == 0) {
    // no-op, but ensure DepEvents will still be waited on
    if (!DepEvents.empty()) {
      Queue.getAdapter().call<UrApiKind::urEnqueueEventsWait>(
          Queue.getHandleRef(), DepEvents.size(), DepEvents.data(), OutEvent);
    }
    return;
  }

  if (!DstMem || !SrcMem)
    throw sycl::exception(sycl::make_error_code(errc::invalid),
                          "NULL pointer argument in 2D memory copy operation.");

  adapter_impl &Adapter = Queue.getAdapter();

  bool SupportsUSMMemcpy2D = false;
  Adapter.call<UrApiKind::urContextGetInfo>(
      Queue.getContextImpl().getHandleRef(),
      UR_CONTEXT_INFO_USM_MEMCPY2D_SUPPORT, sizeof(bool), &SupportsUSMMemcpy2D,
      nullptr);

  if (SupportsUSMMemcpy2D) {
    // Direct memcpy2D is supported so we use this function.
    Adapter.call<UrApiKind::urEnqueueUSMMemcpy2D>(
        Queue.getHandleRef(),
        /*blocking=*/false, DstMem, DstPitch, SrcMem, SrcPitch, Width, Height,
        DepEvents.size(), DepEvents.data(), OutEvent);
    return;
  }

  // Otherwise we allow the special case where the copy is to or from host.
#ifndef NDEBUG
  context Ctx = createSyclObjFromImpl<context>(Queue.getContextImpl());
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
  std::vector<OwnedUrEvent> CopyEventsManaged;
  CopyEventsManaged.reserve(Height);
  // We'll need continuous range of events for a wait later as well.
  std::vector<ur_event_handle_t> CopyEvents(Height);

  for (size_t I = 0; I < Height; ++I) {
    char *DstItBegin = static_cast<char *>(DstMem) + I * DstPitch;
    const char *SrcItBegin = static_cast<const char *>(SrcMem) + I * SrcPitch;
    Adapter.call<UrApiKind::urEnqueueUSMMemcpy>(
        Queue.getHandleRef(),
        /* blocking */ false, DstItBegin, SrcItBegin, Width, DepEvents.size(),
        DepEvents.data(), CopyEvents.data() + I);
    CopyEventsManaged.emplace_back(CopyEvents[I], Adapter,
                                   /*TakeOwnership=*/true);
  }
  // Then insert a wait to coalesce the copy events.
  Queue.getAdapter().call<UrApiKind::urEnqueueEventsWait>(
      Queue.getHandleRef(), CopyEvents.size(), CopyEvents.data(), OutEvent);
}

void MemoryManager::fill_2d_usm(void *DstMem, queue_impl &Queue, size_t Pitch,
                                size_t Width, size_t Height,
                                const std::vector<unsigned char> &Pattern,
                                std::vector<ur_event_handle_t> DepEvents,
                                ur_event_handle_t *OutEvent) {
  if (Width == 0 || Height == 0) {
    // no-op, but ensure DepEvents will still be waited on
    if (!DepEvents.empty()) {
      Queue.getAdapter().call<UrApiKind::urEnqueueEventsWait>(
          Queue.getHandleRef(), DepEvents.size(), DepEvents.data(), OutEvent);
    }
    return;
  }

  if (!DstMem)
    throw sycl::exception(sycl::make_error_code(errc::invalid),
                          "NULL pointer argument in 2D memory fill operation.");
  adapter_impl &Adapter = Queue.getAdapter();
  Adapter.call<UrApiKind::urEnqueueUSMFill2D>(
      Queue.getHandleRef(), DstMem, Pitch, Pattern.size(), Pattern.data(),
      Width, Height, DepEvents.size(), DepEvents.data(), OutEvent);
}

void MemoryManager::memset_2d_usm(void *DstMem, queue_impl &Queue, size_t Pitch,
                                  size_t Width, size_t Height, char Value,
                                  std::vector<ur_event_handle_t> DepEvents,
                                  ur_event_handle_t *OutEvent) {
  if (Width == 0 || Height == 0) {
    // no-op, but ensure DepEvents will still be waited on
    if (!DepEvents.empty()) {
      Queue.getAdapter().call<UrApiKind::urEnqueueEventsWait>(
          Queue.getHandleRef(), DepEvents.size(), DepEvents.data(), OutEvent);
    }
    return;
  }

  if (!DstMem)
    throw sycl::exception(
        sycl::make_error_code(errc::invalid),
        "NULL pointer argument in 2D memory memset operation.");
  MemoryManager::fill_2d_usm(DstMem, Queue, Pitch, Width, Height,
                             {static_cast<unsigned char>(Value)},
                             std::move(DepEvents), OutEvent);
}

static void
memcpyToDeviceGlobalUSM(queue_impl &Queue,
                        DeviceGlobalMapEntry *DeviceGlobalEntry,
                        size_t NumBytes, size_t Offset, const void *Src,
                        const std::vector<ur_event_handle_t> &DepEvents,
                        ur_event_handle_t *OutEvent) {
  // Get or allocate USM memory for the device_global.
  DeviceGlobalUSMMem &DeviceGlobalUSM =
      DeviceGlobalEntry->getOrAllocateDeviceGlobalUSM(Queue);
  void *Dest = DeviceGlobalUSM.getPtr();

  // OwnedPiEvent will keep the initialization event alive for the duration
  // of this function call.
  OwnedUrEvent ZIEvent = DeviceGlobalUSM.getInitEvent(Queue.getAdapter());

  // We may need addtional events, so create a non-const dependency events list
  // to use if we need to modify it.
  std::vector<ur_event_handle_t> AuxDepEventsStorage;
  const std::vector<ur_event_handle_t> &ActualDepEvents =
      ZIEvent ? AuxDepEventsStorage : DepEvents;

  // If there is a zero-initializer event the memory operation should wait for
  // it.
  if (ZIEvent) {
    AuxDepEventsStorage = DepEvents;
    AuxDepEventsStorage.push_back(ZIEvent.GetEvent());
  }

  MemoryManager::copy_usm(Src, Queue, NumBytes,
                          reinterpret_cast<char *>(Dest) + Offset,
                          ActualDepEvents, OutEvent);
}

static void memcpyFromDeviceGlobalUSM(
    queue_impl &Queue, DeviceGlobalMapEntry *DeviceGlobalEntry, size_t NumBytes,
    size_t Offset, void *Dest, const std::vector<ur_event_handle_t> &DepEvents,
    ur_event_handle_t *OutEvent) {
  // Get or allocate USM memory for the device_global. Since we are reading from
  // it, we need it initialized if it has not been yet.
  DeviceGlobalUSMMem &DeviceGlobalUSM =
      DeviceGlobalEntry->getOrAllocateDeviceGlobalUSM(Queue);
  void *Src = DeviceGlobalUSM.getPtr();

  // OwnedPiEvent will keep the initialization event alive for the duration
  // of this function call.
  OwnedUrEvent ZIEvent = DeviceGlobalUSM.getInitEvent(Queue.getAdapter());

  // We may need addtional events, so create a non-const dependency events list
  // to use if we need to modify it.
  std::vector<ur_event_handle_t> AuxDepEventsStorage;
  const std::vector<ur_event_handle_t> &ActualDepEvents =
      ZIEvent ? AuxDepEventsStorage : DepEvents;

  // If there is a zero-initializer event the memory operation should wait for
  // it.
  if (ZIEvent) {
    AuxDepEventsStorage = DepEvents;
    AuxDepEventsStorage.push_back(ZIEvent.GetEvent());
  }

  MemoryManager::copy_usm(reinterpret_cast<const char *>(Src) + Offset, Queue,
                          NumBytes, Dest, ActualDepEvents, OutEvent);
}

static ur_program_handle_t
getOrBuildProgramForDeviceGlobal(queue_impl &Queue,
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
  device Device = Queue.get_device();
  context_impl &ContextImpl = Queue.getContextImpl();
  std::optional<ur_program_handle_t> CachedProgram =
      ContextImpl.getProgramForDeviceGlobal(Device, DeviceGlobalEntry);
  if (CachedProgram)
    return *CachedProgram;

  // If there was no cached program, build one.
  auto Context = createSyclObjFromImpl<context>(ContextImpl);
  ProgramManager &PM = ProgramManager::getInstance();
  const RTDeviceBinaryImage &Img = PM.getDeviceImage(
      DeviceGlobalEntry->MImages, ContextImpl, *getSyclObjImpl(Device));

  device_image_plain DeviceImage =
      PM.getDeviceImageFromBinaryImage(&Img, Context, Device);
  device_image_plain BuiltImage =
      PM.build(std::move(DeviceImage), {std::move(Device)}, {});
  return getSyclObjImpl(BuiltImage)->get_ur_program();
}

static void
memcpyToDeviceGlobalDirect(queue_impl &Queue,
                           DeviceGlobalMapEntry *DeviceGlobalEntry,
                           size_t NumBytes, size_t Offset, const void *Src,
                           const std::vector<ur_event_handle_t> &DepEvents,
                           ur_event_handle_t *OutEvent) {
  ur_program_handle_t Program =
      getOrBuildProgramForDeviceGlobal(Queue, DeviceGlobalEntry);
  adapter_impl &Adapter = Queue.getAdapter();
  Adapter.call<UrApiKind::urEnqueueDeviceGlobalVariableWrite>(
      Queue.getHandleRef(), Program, DeviceGlobalEntry->MUniqueId.c_str(),
      false, NumBytes, Offset, Src, DepEvents.size(), DepEvents.data(),
      OutEvent);
}

static void memcpyFromDeviceGlobalDirect(
    queue_impl &Queue, DeviceGlobalMapEntry *DeviceGlobalEntry, size_t NumBytes,
    size_t Offset, void *Dest, const std::vector<ur_event_handle_t> &DepEvents,
    ur_event_handle_t *OutEvent) {
  ur_program_handle_t Program =
      getOrBuildProgramForDeviceGlobal(Queue, DeviceGlobalEntry);
  adapter_impl &Adapter = Queue.getAdapter();
  Adapter.call<UrApiKind::urEnqueueDeviceGlobalVariableRead>(
      Queue.getHandleRef(), Program, DeviceGlobalEntry->MUniqueId.c_str(),
      false, NumBytes, Offset, Dest, DepEvents.size(), DepEvents.data(),
      OutEvent);
}

void MemoryManager::copy_to_device_global(
    const void *DeviceGlobalPtr, bool IsDeviceImageScoped, queue_impl &Queue,
    size_t NumBytes, size_t Offset, const void *SrcMem,
    const std::vector<ur_event_handle_t> &DepEvents,
    ur_event_handle_t *OutEvent) {
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
                            OutEvent);
}

void MemoryManager::copy_from_device_global(
    const void *DeviceGlobalPtr, bool IsDeviceImageScoped, queue_impl &Queue,
    size_t NumBytes, size_t Offset, void *DstMem,
    const std::vector<ur_event_handle_t> &DepEvents,
    ur_event_handle_t *OutEvent) {
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
                              DepEvents, OutEvent);
}

// Command buffer methods
void MemoryManager::ext_oneapi_copyD2D_cmd_buffer(
    sycl::detail::context_impl *Context,
    ur_exp_command_buffer_handle_t CommandBuffer, SYCLMemObjI *SYCLMemObj,
    void *SrcMem, unsigned int DimSrc, sycl::range<3> SrcSize,
    sycl::range<3> SrcAccessRange, sycl::id<3> SrcOffset,
    unsigned int SrcElemSize, void *DstMem, unsigned int DimDst,
    sycl::range<3> DstSize, sycl::range<3> DstAccessRange,
    sycl::id<3> DstOffset, unsigned int DstElemSize,
    std::vector<ur_exp_command_buffer_sync_point_t> Deps,
    ur_exp_command_buffer_sync_point_t *OutSyncPoint) {
  assert(SYCLMemObj && "The SYCLMemObj is nullptr");
  (void)DstAccessRange;

  adapter_impl &Adapter = Context->getAdapter();

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
    Adapter.call<UrApiKind::urCommandBufferAppendMemBufferCopyExp>(
        CommandBuffer, sycl::detail::ur::cast<ur_mem_handle_t>(SrcMem),
        sycl::detail::ur::cast<ur_mem_handle_t>(DstMem), SrcXOffBytes,
        DstXOffBytes, SrcAccessRangeWidthBytes, Deps.size(), Deps.data(), 0u,
        nullptr, OutSyncPoint, nullptr, nullptr);
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

    ur_rect_offset_t SrcOrigin{SrcXOffBytes, SrcOffset[SrcPos.YTerm],
                               SrcOffset[SrcPos.ZTerm]};
    ur_rect_offset_t DstOrigin{DstXOffBytes, DstOffset[DstPos.YTerm],
                               DstOffset[DstPos.ZTerm]};
    ur_rect_region_t Region{SrcAccessRangeWidthBytes,
                            SrcAccessRange[SrcPos.YTerm],
                            SrcAccessRange[SrcPos.ZTerm]};

    Adapter.call<UrApiKind::urCommandBufferAppendMemBufferCopyRectExp>(
        CommandBuffer, sycl::detail::ur::cast<ur_mem_handle_t>(SrcMem),
        sycl::detail::ur::cast<ur_mem_handle_t>(DstMem), SrcOrigin, DstOrigin,
        Region, SrcRowPitch, SrcSlicePitch, DstRowPitch, DstSlicePitch,
        Deps.size(), Deps.data(), 0u, nullptr, OutSyncPoint, nullptr, nullptr);
  }
}

void MemoryManager::ext_oneapi_copyD2H_cmd_buffer(
    sycl::detail::context_impl *Context,
    ur_exp_command_buffer_handle_t CommandBuffer, SYCLMemObjI *SYCLMemObj,
    void *SrcMem, unsigned int DimSrc, sycl::range<3> SrcSize,
    sycl::range<3> SrcAccessRange, sycl::id<3> SrcOffset,
    unsigned int SrcElemSize, char *DstMem, unsigned int DimDst,
    sycl::range<3> DstSize, sycl::id<3> DstOffset, unsigned int DstElemSize,
    std::vector<ur_exp_command_buffer_sync_point_t> Deps,
    ur_exp_command_buffer_sync_point_t *OutSyncPoint) {
  assert(SYCLMemObj && "The SYCLMemObj is nullptr");

  adapter_impl &Adapter = Context->getAdapter();

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
    ur_result_t Result =
        Adapter.call_nocheck<UrApiKind::urCommandBufferAppendMemBufferReadExp>(
            CommandBuffer, sycl::detail::ur::cast<ur_mem_handle_t>(SrcMem),
            SrcXOffBytes, SrcAccessRangeWidthBytes, DstMem + DstXOffBytes,
            Deps.size(), Deps.data(), 0u, nullptr, OutSyncPoint, nullptr,
            nullptr);

    if (Result == UR_RESULT_ERROR_UNSUPPORTED_FEATURE) {
      throw sycl::exception(
          sycl::make_error_code(sycl::errc::feature_not_supported),
          "Device-to-host buffer copy command not supported by graph backend");
    } else {
      Adapter.checkUrResult(Result);
    }
  } else {
    size_t BufferRowPitch = (1 == DimSrc) ? 0 : SrcSzWidthBytes;
    size_t BufferSlicePitch =
        (3 == DimSrc) ? SrcSzWidthBytes * SrcSize[SrcPos.YTerm] : 0;
    size_t HostRowPitch = (1 == DimDst) ? 0 : DstSzWidthBytes;
    size_t HostSlicePitch =
        (3 == DimDst) ? DstSzWidthBytes * DstSize[DstPos.YTerm] : 0;

    ur_rect_offset_t BufferOffset{SrcXOffBytes, SrcOffset[SrcPos.YTerm],
                                  SrcOffset[SrcPos.ZTerm]};
    ur_rect_offset_t HostOffset{DstXOffBytes, DstOffset[DstPos.YTerm],
                                DstOffset[DstPos.ZTerm]};
    ur_rect_region_t RectRegion{SrcAccessRangeWidthBytes,
                                SrcAccessRange[SrcPos.YTerm],
                                SrcAccessRange[SrcPos.ZTerm]};

    ur_result_t Result =
        Adapter
            .call_nocheck<UrApiKind::urCommandBufferAppendMemBufferReadRectExp>(
                CommandBuffer, sycl::detail::ur::cast<ur_mem_handle_t>(SrcMem),
                BufferOffset, HostOffset, RectRegion, BufferRowPitch,
                BufferSlicePitch, HostRowPitch, HostSlicePitch, DstMem,
                Deps.size(), Deps.data(), 0u, nullptr, OutSyncPoint, nullptr,
                nullptr);
    if (Result == UR_RESULT_ERROR_UNSUPPORTED_FEATURE) {
      throw sycl::exception(
          sycl::make_error_code(sycl::errc::feature_not_supported),
          "Device-to-host buffer copy command not supported by graph backend");
    } else {
      Adapter.checkUrResult(Result);
    }
  }
}

void MemoryManager::ext_oneapi_copyH2D_cmd_buffer(
    sycl::detail::context_impl *Context,
    ur_exp_command_buffer_handle_t CommandBuffer, SYCLMemObjI *SYCLMemObj,
    char *SrcMem, unsigned int DimSrc, sycl::range<3> SrcSize,
    sycl::id<3> SrcOffset, unsigned int SrcElemSize, void *DstMem,
    unsigned int DimDst, sycl::range<3> DstSize, sycl::range<3> DstAccessRange,
    sycl::id<3> DstOffset, unsigned int DstElemSize,
    std::vector<ur_exp_command_buffer_sync_point_t> Deps,
    ur_exp_command_buffer_sync_point_t *OutSyncPoint) {
  assert(SYCLMemObj && "The SYCLMemObj is nullptr");

  adapter_impl &Adapter = Context->getAdapter();

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
    ur_result_t Result =
        Adapter.call_nocheck<UrApiKind::urCommandBufferAppendMemBufferWriteExp>(
            CommandBuffer, sycl::detail::ur::cast<ur_mem_handle_t>(DstMem),
            DstXOffBytes, DstAccessRangeWidthBytes, SrcMem + SrcXOffBytes,
            Deps.size(), Deps.data(), 0u, nullptr, OutSyncPoint, nullptr,
            nullptr);

    if (Result == UR_RESULT_ERROR_UNSUPPORTED_FEATURE) {
      throw sycl::exception(
          sycl::make_error_code(sycl::errc::feature_not_supported),
          "Host-to-device buffer copy command not supported by graph backend");
    } else {
      Adapter.checkUrResult(Result);
    }
  } else {
    size_t BufferRowPitch = (1 == DimDst) ? 0 : DstSzWidthBytes;
    size_t BufferSlicePitch =
        (3 == DimDst) ? DstSzWidthBytes * DstSize[DstPos.YTerm] : 0;
    size_t HostRowPitch = (1 == DimSrc) ? 0 : SrcSzWidthBytes;
    size_t HostSlicePitch =
        (3 == DimSrc) ? SrcSzWidthBytes * SrcSize[SrcPos.YTerm] : 0;

    ur_rect_offset_t BufferOffset{DstXOffBytes, DstOffset[DstPos.YTerm],
                                  DstOffset[DstPos.ZTerm]};
    ur_rect_offset_t HostOffset{SrcXOffBytes, SrcOffset[SrcPos.YTerm],
                                SrcOffset[SrcPos.ZTerm]};
    ur_rect_region_t RectRegion{DstAccessRangeWidthBytes,
                                DstAccessRange[DstPos.YTerm],
                                DstAccessRange[DstPos.ZTerm]};

    ur_result_t Result = Adapter.call_nocheck<
        UrApiKind::urCommandBufferAppendMemBufferWriteRectExp>(
        CommandBuffer, sycl::detail::ur::cast<ur_mem_handle_t>(DstMem),
        BufferOffset, HostOffset, RectRegion, BufferRowPitch, BufferSlicePitch,
        HostRowPitch, HostSlicePitch, SrcMem, Deps.size(), Deps.data(), 0u,
        nullptr, OutSyncPoint, nullptr, nullptr);

    if (Result == UR_RESULT_ERROR_UNSUPPORTED_FEATURE) {
      throw sycl::exception(
          sycl::make_error_code(sycl::errc::feature_not_supported),
          "Host-to-device buffer copy command not supported by graph backend");
    } else {
      Adapter.checkUrResult(Result);
    }
  }
}

void MemoryManager::ext_oneapi_copy_usm_cmd_buffer(
    context_impl *Context, const void *SrcMem,
    ur_exp_command_buffer_handle_t CommandBuffer, size_t Len, void *DstMem,
    std::vector<ur_exp_command_buffer_sync_point_t> Deps,
    ur_exp_command_buffer_sync_point_t *OutSyncPoint) {
  if (!SrcMem || !DstMem)
    throw exception(make_error_code(errc::invalid),
                    "NULL pointer argument in memory copy operation.");

  adapter_impl &Adapter = Context->getAdapter();
  ur_result_t Result =
      Adapter.call_nocheck<UrApiKind::urCommandBufferAppendUSMMemcpyExp>(
          CommandBuffer, DstMem, SrcMem, Len, Deps.size(), Deps.data(), 0u,
          nullptr, OutSyncPoint, nullptr, nullptr);
  if (Result == UR_RESULT_ERROR_UNSUPPORTED_FEATURE) {
    throw sycl::exception(
        sycl::make_error_code(sycl::errc::feature_not_supported),
        "USM copy command not supported by graph backend");
  } else {
    Adapter.checkUrResult(Result);
  }
}

void MemoryManager::ext_oneapi_fill_usm_cmd_buffer(
    sycl::detail::context_impl *Context,
    ur_exp_command_buffer_handle_t CommandBuffer, void *DstMem, size_t Len,
    const std::vector<unsigned char> &Pattern,
    std::vector<ur_exp_command_buffer_sync_point_t> Deps,
    ur_exp_command_buffer_sync_point_t *OutSyncPoint) {

  if (!DstMem)
    throw exception(make_error_code(errc::invalid),
                    "NULL pointer argument in memory fill operation.");

  adapter_impl &Adapter = Context->getAdapter();
  ur_result_t Result =
      Adapter.call_nocheck<UrApiKind::urCommandBufferAppendUSMFillExp>(
          CommandBuffer, DstMem, Pattern.data(), Pattern.size(), Len,
          Deps.size(), Deps.data(), 0u, nullptr, OutSyncPoint, nullptr,
          nullptr);
  if (Result == UR_RESULT_ERROR_UNSUPPORTED_FEATURE) {
    throw sycl::exception(
        sycl::make_error_code(sycl::errc::feature_not_supported),
        "USM fill command not supported by graph backend");
  } else {
    Adapter.checkUrResult(Result);
  }
}

void MemoryManager::ext_oneapi_fill_cmd_buffer(
    sycl::detail::context_impl *Context,
    ur_exp_command_buffer_handle_t CommandBuffer, SYCLMemObjI *SYCLMemObj,
    void *Mem, size_t PatternSize, const unsigned char *Pattern,
    unsigned int Dim, sycl::range<3> Size, sycl::range<3> AccessRange,
    sycl::id<3> AccessOffset, unsigned int ElementSize,
    std::vector<ur_exp_command_buffer_sync_point_t> Deps,
    ur_exp_command_buffer_sync_point_t *OutSyncPoint) {
  assert(SYCLMemObj && "The SYCLMemObj is nullptr");

  adapter_impl &Adapter = Context->getAdapter();
  if (SYCLMemObj->getType() != detail::SYCLMemObjI::MemObjType::Buffer) {
    throw sycl::exception(sycl::make_error_code(sycl::errc::invalid),
                          "Images are not supported in Graphs");
  }

  // 2D and 3D buffers accessors can't have custom range or the data will
  // likely be discontiguous.
  bool RangesUsable = (Dim <= 1) || (Size == AccessRange);
  // For 2D and 3D buffers, the offset must be 0, or the data will be
  // discontiguous.
  bool OffsetUsable = (Dim <= 1) || (AccessOffset == sycl::id<3>{0, 0, 0});
  size_t RangeMultiplier = AccessRange[0] * AccessRange[1] * AccessRange[2];

  if (RangesUsable && OffsetUsable) {
    Adapter.call<UrApiKind::urCommandBufferAppendMemBufferFillExp>(
        CommandBuffer, ur::cast<ur_mem_handle_t>(Mem), Pattern, PatternSize,
        AccessOffset[0] * ElementSize, RangeMultiplier * ElementSize,
        Deps.size(), Deps.data(), 0u, nullptr, OutSyncPoint, nullptr, nullptr);
    return;
  }
  // The sycl::handler uses a parallel_for kernel in the case of unusable
  // Range or Offset, not CG:Fill. So we should not be here.
  throw exception(make_error_code(errc::runtime),
                  "Not supported configuration of fill requested");
}

void MemoryManager::ext_oneapi_prefetch_usm_cmd_buffer(
    sycl::detail::context_impl *Context,
    ur_exp_command_buffer_handle_t CommandBuffer, void *Mem, size_t Length,
    std::vector<ur_exp_command_buffer_sync_point_t> Deps,
    ur_exp_command_buffer_sync_point_t *OutSyncPoint) {
  adapter_impl &Adapter = Context->getAdapter();
  Adapter.call<UrApiKind::urCommandBufferAppendUSMPrefetchExp>(
      CommandBuffer, Mem, Length, ur_usm_migration_flags_t(0), Deps.size(),
      Deps.data(), 0u, nullptr, OutSyncPoint, nullptr, nullptr);
}

void MemoryManager::ext_oneapi_advise_usm_cmd_buffer(
    sycl::detail::context_impl *Context,
    ur_exp_command_buffer_handle_t CommandBuffer, const void *Mem,
    size_t Length, ur_usm_advice_flags_t Advice,
    std::vector<ur_exp_command_buffer_sync_point_t> Deps,
    ur_exp_command_buffer_sync_point_t *OutSyncPoint) {
  adapter_impl &Adapter = Context->getAdapter();
  Adapter.call<UrApiKind::urCommandBufferAppendUSMAdviseExp>(
      CommandBuffer, Mem, Length, Advice, Deps.size(), Deps.data(), 0u, nullptr,
      OutSyncPoint, nullptr, nullptr);
}

void MemoryManager::copy_image_bindless(
    queue_impl &Queue, const void *Src, void *Dst,
    const ur_image_desc_t &SrcDesc, const ur_image_desc_t &DstDesc,
    const ur_image_format_t &SrcFormat, const ur_image_format_t &DstFormat,
    const ur_exp_image_copy_flags_t Flags, ur_rect_offset_t SrcOffset,
    ur_rect_offset_t DstOffset, ur_rect_region_t CopyExtent,
    const std::vector<ur_event_handle_t> &DepEvents,
    ur_event_handle_t *OutEvent) {
  assert((Flags == UR_EXP_IMAGE_COPY_FLAG_HOST_TO_DEVICE ||
          Flags == UR_EXP_IMAGE_COPY_FLAG_DEVICE_TO_HOST ||
          Flags == UR_EXP_IMAGE_COPY_FLAG_DEVICE_TO_DEVICE ||
          Flags == UR_EXP_IMAGE_COPY_FLAG_HOST_TO_HOST) &&
         "Invalid flags passed to copy_image_bindless.");
  if (!Dst || !Src)
    throw sycl::exception(
        sycl::make_error_code(errc::invalid),
        "NULL pointer argument in bindless image copy operation.");

  detail::adapter_impl &Adapter = Queue.getAdapter();

  ur_exp_image_copy_region_t CopyRegion{};
  CopyRegion.stype = UR_STRUCTURE_TYPE_EXP_IMAGE_COPY_REGION;
  CopyRegion.copyExtent = CopyExtent;
  CopyRegion.srcOffset = SrcOffset;
  CopyRegion.dstOffset = DstOffset;

  Adapter.call<UrApiKind::urBindlessImagesImageCopyExp>(
      Queue.getHandleRef(), Src, Dst, &SrcDesc, &DstDesc, &SrcFormat,
      &DstFormat, &CopyRegion, Flags, DepEvents.size(), DepEvents.data(),
      OutEvent);
}

} // namespace detail
} // namespace _V1
} // namespace sycl
