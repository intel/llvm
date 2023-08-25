//===--------- memory.cpp - Level Zero Adapter -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//

#include <algorithm>
#include <climits>
#include <string.h>

#include "context.hpp"
#include "event.hpp"
#include "ur_level_zero.hpp"

// Default to using compute engine for fill operation, but allow to
// override this with an environment variable.
static bool PreferCopyEngine = [] {
  const char *UrRet = std::getenv("UR_L0_USE_COPY_ENGINE_FOR_FILL");
  const char *PiRet =
      std::getenv("SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE_FOR_FILL");
  return (UrRet ? std::stoi(UrRet) : (PiRet ? std::stoi(PiRet) : 0));
}();

// Helper function to check if a pointer is a device pointer.
bool IsDevicePointer(ur_context_handle_t Context, const void *Ptr) {
  ze_device_handle_t ZeDeviceHandle;
  ZeStruct<ze_memory_allocation_properties_t> ZeMemoryAllocationProperties;

  // Query memory type of the pointer
  ZE2UR_CALL(zeMemGetAllocProperties,
             (Context->ZeContext, Ptr, &ZeMemoryAllocationProperties,
              &ZeDeviceHandle));

  return (ZeMemoryAllocationProperties.type == ZE_MEMORY_TYPE_DEVICE);
}

// Shared by all memory read/write/copy PI interfaces.
// PI interfaces must have queue's and destination buffer's mutexes locked for
// exclusive use and source buffer's mutex locked for shared use on entry.
ur_result_t enqueueMemCopyHelper(ur_command_t CommandType,
                                 ur_queue_handle_t Queue, void *Dst,
                                 ur_bool_t BlockingWrite, size_t Size,
                                 const void *Src, uint32_t NumEventsInWaitList,
                                 const ur_event_handle_t *EventWaitList,
                                 ur_event_handle_t *OutEvent,
                                 bool PreferCopyEngine) {
  bool UseCopyEngine = Queue->useCopyEngine(PreferCopyEngine);

  _ur_ze_event_list_t TmpWaitList;
  UR_CALL(TmpWaitList.createAndRetainUrZeEventList(
      NumEventsInWaitList, EventWaitList, Queue, UseCopyEngine));

  // We want to batch these commands to avoid extra submissions (costly)
  bool OkToBatch = true;

  // Get a new command list to be used on this call
  ur_command_list_ptr_t CommandList{};
  UR_CALL(Queue->Context->getAvailableCommandList(Queue, CommandList,
                                                  UseCopyEngine, OkToBatch));

  ze_event_handle_t ZeEvent = nullptr;
  ur_event_handle_t InternalEvent;
  bool IsInternal = OutEvent == nullptr;
  ur_event_handle_t *Event = OutEvent ? OutEvent : &InternalEvent;
  UR_CALL(createEventAndAssociateQueue(Queue, Event, CommandType, CommandList,
                                       IsInternal));
  ZeEvent = (*Event)->ZeEvent;
  (*Event)->WaitList = TmpWaitList;

  const auto &ZeCommandList = CommandList->first;
  const auto &WaitList = (*Event)->WaitList;

  urPrint("calling zeCommandListAppendMemoryCopy() with\n"
          "  ZeEvent %#" PRIxPTR "\n",
          ur_cast<std::uintptr_t>(ZeEvent));
  printZeEventList(WaitList);

  ZE2UR_CALL(zeCommandListAppendMemoryCopy,
             (ZeCommandList, Dst, Src, Size, ZeEvent, WaitList.Length,
              WaitList.ZeEventList));

  UR_CALL(Queue->executeCommandList(CommandList, BlockingWrite, OkToBatch));

  return UR_RESULT_SUCCESS;
}

// Shared by all memory read/write/copy rect PI interfaces.
// PI interfaces must have queue's and destination buffer's mutexes locked for
// exclusive use and source buffer's mutex locked for shared use on entry.
ur_result_t enqueueMemCopyRectHelper(
    ur_command_t CommandType, ur_queue_handle_t Queue, const void *SrcBuffer,
    void *DstBuffer, ur_rect_offset_t SrcOrigin, ur_rect_offset_t DstOrigin,
    ur_rect_region_t Region, size_t SrcRowPitch, size_t DstRowPitch,
    size_t SrcSlicePitch, size_t DstSlicePitch, ur_bool_t Blocking,
    uint32_t NumEventsInWaitList, const ur_event_handle_t *EventWaitList,
    ur_event_handle_t *OutEvent, bool PreferCopyEngine) {
  bool UseCopyEngine = Queue->useCopyEngine(PreferCopyEngine);

  _ur_ze_event_list_t TmpWaitList;
  UR_CALL(TmpWaitList.createAndRetainUrZeEventList(
      NumEventsInWaitList, EventWaitList, Queue, UseCopyEngine));

  // We want to batch these commands to avoid extra submissions (costly)
  bool OkToBatch = true;

  // Get a new command list to be used on this call
  ur_command_list_ptr_t CommandList{};
  UR_CALL(Queue->Context->getAvailableCommandList(Queue, CommandList,
                                                  UseCopyEngine, OkToBatch));

  ze_event_handle_t ZeEvent = nullptr;
  ur_event_handle_t InternalEvent;
  bool IsInternal = OutEvent == nullptr;
  ur_event_handle_t *Event = OutEvent ? OutEvent : &InternalEvent;
  UR_CALL(createEventAndAssociateQueue(Queue, Event, CommandType, CommandList,
                                       IsInternal));

  ZeEvent = (*Event)->ZeEvent;
  (*Event)->WaitList = TmpWaitList;

  const auto &ZeCommandList = CommandList->first;
  const auto &WaitList = (*Event)->WaitList;

  urPrint("calling zeCommandListAppendMemoryCopy() with\n"
          "  ZeEvent %#" PRIxPTR "\n",
          ur_cast<std::uintptr_t>(ZeEvent));
  printZeEventList(WaitList);

  uint32_t SrcOriginX = ur_cast<uint32_t>(SrcOrigin.x);
  uint32_t SrcOriginY = ur_cast<uint32_t>(SrcOrigin.y);
  uint32_t SrcOriginZ = ur_cast<uint32_t>(SrcOrigin.z);

  uint32_t SrcPitch = SrcRowPitch;
  if (SrcPitch == 0)
    SrcPitch = ur_cast<uint32_t>(Region.width);

  if (SrcSlicePitch == 0)
    SrcSlicePitch = ur_cast<uint32_t>(Region.height) * SrcPitch;

  uint32_t DstOriginX = ur_cast<uint32_t>(DstOrigin.x);
  uint32_t DstOriginY = ur_cast<uint32_t>(DstOrigin.y);
  uint32_t DstOriginZ = ur_cast<uint32_t>(DstOrigin.z);

  uint32_t DstPitch = DstRowPitch;
  if (DstPitch == 0)
    DstPitch = ur_cast<uint32_t>(Region.width);

  if (DstSlicePitch == 0)
    DstSlicePitch = ur_cast<uint32_t>(Region.height) * DstPitch;

  uint32_t Width = ur_cast<uint32_t>(Region.width);
  uint32_t Height = ur_cast<uint32_t>(Region.height);
  uint32_t Depth = ur_cast<uint32_t>(Region.depth);

  const ze_copy_region_t ZeSrcRegion = {SrcOriginX, SrcOriginY, SrcOriginZ,
                                        Width,      Height,     Depth};
  const ze_copy_region_t ZeDstRegion = {DstOriginX, DstOriginY, DstOriginZ,
                                        Width,      Height,     Depth};

  ZE2UR_CALL(zeCommandListAppendMemoryCopyRegion,
             (ZeCommandList, DstBuffer, &ZeDstRegion, DstPitch, DstSlicePitch,
              SrcBuffer, &ZeSrcRegion, SrcPitch, SrcSlicePitch, ZeEvent,
              WaitList.Length, WaitList.ZeEventList));

  urPrint("calling zeCommandListAppendMemoryCopyRegion()\n");

  UR_CALL(Queue->executeCommandList(CommandList, Blocking, OkToBatch));

  return UR_RESULT_SUCCESS;
}

// PI interfaces must have queue's and buffer's mutexes locked on entry.
static ur_result_t enqueueMemFillHelper(ur_command_t CommandType,
                                        ur_queue_handle_t Queue, void *Ptr,
                                        const void *Pattern, size_t PatternSize,
                                        size_t Size,
                                        uint32_t NumEventsInWaitList,
                                        const ur_event_handle_t *EventWaitList,
                                        ur_event_handle_t *OutEvent) {
  // Pattern size must be a power of two.
  UR_ASSERT((PatternSize > 0) && ((PatternSize & (PatternSize - 1)) == 0),
            UR_RESULT_ERROR_INVALID_VALUE);
  auto &Device = Queue->Device;

  // Make sure that pattern size matches the capability of the copy queues.
  // Check both main and link groups as we don't known which one will be used.
  //
  if (PreferCopyEngine && Device->hasCopyEngine()) {
    if (Device->hasMainCopyEngine() &&
        Device->QueueGroup[ur_device_handle_t_::queue_group_info_t::MainCopy]
                .ZeProperties.maxMemoryFillPatternSize < PatternSize) {
      PreferCopyEngine = false;
    }
    if (Device->hasLinkCopyEngine() &&
        Device->QueueGroup[ur_device_handle_t_::queue_group_info_t::LinkCopy]
                .ZeProperties.maxMemoryFillPatternSize < PatternSize) {
      PreferCopyEngine = false;
    }
  }

  bool UseCopyEngine = Queue->useCopyEngine(PreferCopyEngine);
  if (!UseCopyEngine) {
    // Pattern size must fit the compute queue capabilities.
    UR_ASSERT(
        PatternSize <=
            Device->QueueGroup[ur_device_handle_t_::queue_group_info_t::Compute]
                .ZeProperties.maxMemoryFillPatternSize,
        UR_RESULT_ERROR_INVALID_VALUE);
  }

  _ur_ze_event_list_t TmpWaitList;
  UR_CALL(TmpWaitList.createAndRetainUrZeEventList(
      NumEventsInWaitList, EventWaitList, Queue, UseCopyEngine));

  ur_command_list_ptr_t CommandList{};
  // We want to batch these commands to avoid extra submissions (costly)
  bool OkToBatch = true;
  UR_CALL(Queue->Context->getAvailableCommandList(Queue, CommandList,
                                                  UseCopyEngine, OkToBatch));

  ze_event_handle_t ZeEvent = nullptr;
  ur_event_handle_t InternalEvent;
  bool IsInternal = OutEvent == nullptr;
  ur_event_handle_t *Event = OutEvent ? OutEvent : &InternalEvent;
  UR_CALL(createEventAndAssociateQueue(Queue, Event, CommandType, CommandList,
                                       IsInternal));

  ZeEvent = (*Event)->ZeEvent;
  (*Event)->WaitList = TmpWaitList;

  const auto &ZeCommandList = CommandList->first;
  const auto &WaitList = (*Event)->WaitList;

  ZE2UR_CALL(zeCommandListAppendMemoryFill,
             (ZeCommandList, Ptr, Pattern, PatternSize, Size, ZeEvent,
              WaitList.Length, WaitList.ZeEventList));

  urPrint("calling zeCommandListAppendMemoryFill() with\n"
          "  ZeEvent %#" PRIxPTR "\n",
          ur_cast<uint64_t>(ZeEvent));
  printZeEventList(WaitList);

  // Execute command list asynchronously, as the event will be used
  // to track down its completion.
  UR_CALL(Queue->executeCommandList(CommandList, false, OkToBatch));

  return UR_RESULT_SUCCESS;
}

// If indirect access tracking is enabled then performs reference counting,
// otherwise just calls zeMemAllocHost.
static ur_result_t ZeHostMemAllocHelper(void **ResultPtr,
                                        ur_context_handle_t UrContext,
                                        size_t Size) {
  ur_platform_handle_t Plt = UrContext->getPlatform();
  std::unique_lock<ur_shared_mutex> ContextsLock(Plt->ContextsMutex,
                                                 std::defer_lock);
  if (IndirectAccessTrackingEnabled) {
    // Lock the mutex which is guarding contexts container in the platform.
    // This prevents new kernels from being submitted in any context while
    // we are in the process of allocating a memory, this is needed to
    // properly capture allocations by kernels with indirect access.
    ContextsLock.lock();
    // We are going to defer memory release if there are kernels with
    // indirect access, that is why explicitly retain context to be sure
    // that it is released after all memory allocations in this context are
    // released.
    UR_CALL(urContextRetain(UrContext));
  }

  ZeStruct<ze_host_mem_alloc_desc_t> ZeDesc;
  ZeDesc.flags = 0;
  ZE2UR_CALL(zeMemAllocHost,
             (UrContext->ZeContext, &ZeDesc, Size, 1, ResultPtr));

  if (IndirectAccessTrackingEnabled) {
    // Keep track of all memory allocations in the context
    UrContext->MemAllocs.emplace(
        std::piecewise_construct, std::forward_as_tuple(*ResultPtr),
        std::forward_as_tuple(
            reinterpret_cast<ur_context_handle_t>(UrContext)));
  }
  return UR_RESULT_SUCCESS;
}

static ur_result_t getImageRegionHelper(_ur_image *Mem,
                                        ur_rect_offset_t *Origin,
                                        ur_rect_region_t *Region,
                                        ze_image_region_t &ZeRegion) {
  UR_ASSERT(Mem, UR_RESULT_ERROR_INVALID_MEM_OBJECT);
  UR_ASSERT(Origin, UR_RESULT_ERROR_INVALID_VALUE);

#ifndef NDEBUG
  auto UrImage = static_cast<_ur_image *>(Mem);
  ze_image_desc_t &ZeImageDesc = UrImage->ZeImageDesc;

  UR_ASSERT(Mem->isImage(), UR_RESULT_ERROR_INVALID_MEM_OBJECT);
  UR_ASSERT((ZeImageDesc.type == ZE_IMAGE_TYPE_1D && Origin->y == 0 &&
             Origin->z == 0) ||
                (ZeImageDesc.type == ZE_IMAGE_TYPE_1DARRAY && Origin->z == 0) ||
                (ZeImageDesc.type == ZE_IMAGE_TYPE_2D && Origin->z == 0) ||
                (ZeImageDesc.type == ZE_IMAGE_TYPE_3D),
            UR_RESULT_ERROR_INVALID_VALUE);

  UR_ASSERT(Region->width && Region->height && Region->depth,
            UR_RESULT_ERROR_INVALID_VALUE);
  UR_ASSERT(
      (ZeImageDesc.type == ZE_IMAGE_TYPE_1D && Region->height == 1 &&
       Region->depth == 1) ||
          (ZeImageDesc.type == ZE_IMAGE_TYPE_1DARRAY && Region->depth == 1) ||
          (ZeImageDesc.type == ZE_IMAGE_TYPE_2D && Region->depth == 1) ||
          (ZeImageDesc.type == ZE_IMAGE_TYPE_3D),
      UR_RESULT_ERROR_INVALID_VALUE);
#endif // !NDEBUG

  uint32_t OriginX = ur_cast<uint32_t>(Origin->x);
  uint32_t OriginY = ur_cast<uint32_t>(Origin->y);
  uint32_t OriginZ = ur_cast<uint32_t>(Origin->z);

  uint32_t Width = ur_cast<uint32_t>(Region->width);
  uint32_t Height = ur_cast<uint32_t>(Region->height);
  uint32_t Depth = ur_cast<uint32_t>(Region->depth);

  ZeRegion = {OriginX, OriginY, OriginZ, Width, Height, Depth};

  return UR_RESULT_SUCCESS;
}

// Helper function to implement image read/write/copy.
// PI interfaces must have queue's and destination image's mutexes locked for
// exclusive use and source image's mutex locked for shared use on entry.
static ur_result_t enqueueMemImageCommandHelper(
    ur_command_t CommandType, ur_queue_handle_t Queue,
    const void *Src, // image or ptr
    void *Dst,       // image or ptr
    ur_bool_t IsBlocking, ur_rect_offset_t *SrcOrigin,
    ur_rect_offset_t *DstOrigin, ur_rect_region_t *Region, size_t RowPitch,
    size_t SlicePitch, uint32_t NumEventsInWaitList,
    const ur_event_handle_t *EventWaitList, ur_event_handle_t *OutEvent,
    bool PreferCopyEngine = false) {
  bool UseCopyEngine = Queue->useCopyEngine(PreferCopyEngine);

  _ur_ze_event_list_t TmpWaitList;
  UR_CALL(TmpWaitList.createAndRetainUrZeEventList(
      NumEventsInWaitList, EventWaitList, Queue, UseCopyEngine));

  // We want to batch these commands to avoid extra submissions (costly)
  bool OkToBatch = true;

  // Get a new command list to be used on this call
  ur_command_list_ptr_t CommandList{};
  UR_CALL(Queue->Context->getAvailableCommandList(Queue, CommandList,
                                                  UseCopyEngine, OkToBatch));

  ze_event_handle_t ZeEvent = nullptr;
  ur_event_handle_t InternalEvent;
  bool IsInternal = OutEvent == nullptr;
  ur_event_handle_t *Event = OutEvent ? OutEvent : &InternalEvent;
  UR_CALL(createEventAndAssociateQueue(Queue, Event, CommandType, CommandList,
                                       IsInternal));
  ZeEvent = (*Event)->ZeEvent;
  (*Event)->WaitList = TmpWaitList;

  const auto &ZeCommandList = CommandList->first;
  const auto &WaitList = (*Event)->WaitList;

  if (CommandType == UR_COMMAND_MEM_IMAGE_READ) {
    _ur_image *SrcMem = ur_cast<_ur_image *>(const_cast<void *>(Src));

    ze_image_region_t ZeSrcRegion;
    UR_CALL(getImageRegionHelper(SrcMem, SrcOrigin, Region, ZeSrcRegion));

    // TODO: Level Zero does not support row_pitch/slice_pitch for images yet.
    // Check that SYCL RT did not want pitch larger than default.
    std::ignore = RowPitch;
    std::ignore = SlicePitch;
    UR_ASSERT(SrcMem->isImage(), UR_RESULT_ERROR_INVALID_MEM_OBJECT);

#ifndef NDEBUG
    auto SrcImage = SrcMem;
    const ze_image_desc_t &ZeImageDesc = SrcImage->ZeImageDesc;
    UR_ASSERT(
        RowPitch == 0 ||
            // special case RGBA image pitch equal to region's width
            (ZeImageDesc.format.layout == ZE_IMAGE_FORMAT_LAYOUT_32_32_32_32 &&
             RowPitch == 4 * 4 * ZeSrcRegion.width) ||
            (ZeImageDesc.format.layout == ZE_IMAGE_FORMAT_LAYOUT_16_16_16_16 &&
             RowPitch == 4 * 2 * ZeSrcRegion.width) ||
            (ZeImageDesc.format.layout == ZE_IMAGE_FORMAT_LAYOUT_8_8_8_8 &&
             RowPitch == 4 * ZeSrcRegion.width),
        UR_RESULT_ERROR_INVALID_IMAGE_SIZE);
#endif
    UR_ASSERT(SlicePitch == 0 || SlicePitch == RowPitch * ZeSrcRegion.height,
              UR_RESULT_ERROR_INVALID_IMAGE_SIZE);

    char *ZeHandleSrc = nullptr;
    UR_CALL(SrcMem->getZeHandle(ZeHandleSrc, ur_mem_handle_t_::read_only,
                                Queue->Device));
    ZE2UR_CALL(zeCommandListAppendImageCopyToMemory,
               (ZeCommandList, Dst, ur_cast<ze_image_handle_t>(ZeHandleSrc),
                &ZeSrcRegion, ZeEvent, WaitList.Length, WaitList.ZeEventList));
  } else if (CommandType == UR_COMMAND_MEM_IMAGE_WRITE) {
    _ur_image *DstMem = ur_cast<_ur_image *>(Dst);
    ze_image_region_t ZeDstRegion;
    UR_CALL(getImageRegionHelper(DstMem, DstOrigin, Region, ZeDstRegion));

    // TODO: Level Zero does not support row_pitch/slice_pitch for images yet.
    // Check that SYCL RT did not want pitch larger than default.
    UR_ASSERT(DstMem->isImage(), UR_RESULT_ERROR_INVALID_MEM_OBJECT);

#ifndef NDEBUG
    auto DstImage = static_cast<_ur_image *>(DstMem);
    const ze_image_desc_t &ZeImageDesc = DstImage->ZeImageDesc;
    UR_ASSERT(
        RowPitch == 0 ||
            // special case RGBA image pitch equal to region's width
            (ZeImageDesc.format.layout == ZE_IMAGE_FORMAT_LAYOUT_32_32_32_32 &&
             RowPitch == 4 * 4 * ZeDstRegion.width) ||
            (ZeImageDesc.format.layout == ZE_IMAGE_FORMAT_LAYOUT_16_16_16_16 &&
             RowPitch == 4 * 2 * ZeDstRegion.width) ||
            (ZeImageDesc.format.layout == ZE_IMAGE_FORMAT_LAYOUT_8_8_8_8 &&
             RowPitch == 4 * ZeDstRegion.width),
        UR_RESULT_ERROR_INVALID_IMAGE_SIZE);
#endif
    UR_ASSERT(SlicePitch == 0 || SlicePitch == RowPitch * ZeDstRegion.height,
              UR_RESULT_ERROR_INVALID_IMAGE_SIZE);

    char *ZeHandleDst = nullptr;
    UR_CALL(DstMem->getZeHandle(ZeHandleDst, ur_mem_handle_t_::write_only,
                                Queue->Device));
    ZE2UR_CALL(zeCommandListAppendImageCopyFromMemory,
               (ZeCommandList, ur_cast<ze_image_handle_t>(ZeHandleDst), Src,
                &ZeDstRegion, ZeEvent, WaitList.Length, WaitList.ZeEventList));
  } else if (CommandType == UR_COMMAND_MEM_IMAGE_COPY) {
    _ur_image *SrcImage = ur_cast<_ur_image *>(const_cast<void *>(Src));
    _ur_image *DstImage = ur_cast<_ur_image *>(Dst);

    ze_image_region_t ZeSrcRegion;
    UR_CALL(getImageRegionHelper(SrcImage, SrcOrigin, Region, ZeSrcRegion));
    ze_image_region_t ZeDstRegion;
    UR_CALL(getImageRegionHelper(DstImage, DstOrigin, Region, ZeDstRegion));

    char *ZeHandleSrc = nullptr;
    char *ZeHandleDst = nullptr;
    UR_CALL(SrcImage->getZeHandle(ZeHandleSrc, ur_mem_handle_t_::read_only,
                                  Queue->Device));
    UR_CALL(DstImage->getZeHandle(ZeHandleDst, ur_mem_handle_t_::write_only,
                                  Queue->Device));
    ZE2UR_CALL(zeCommandListAppendImageCopyRegion,
               (ZeCommandList, ur_cast<ze_image_handle_t>(ZeHandleDst),
                ur_cast<ze_image_handle_t>(ZeHandleSrc), &ZeDstRegion,
                &ZeSrcRegion, ZeEvent, 0, nullptr));
  } else {
    urPrint("enqueueMemImageUpdate: unsupported image command type\n");
    return UR_RESULT_ERROR_INVALID_OPERATION;
  }

  UR_CALL(Queue->executeCommandList(CommandList, IsBlocking, OkToBatch));

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueMemBufferRead(
    ur_queue_handle_t Queue, ///< [in] handle of the queue object
    ur_mem_handle_t hBuffer, ///< [in] handle of the buffer object
    bool blockingRead, ///< [in] indicates blocking (true), non-blocking (false)
    size_t offset,     ///< [in] offset in bytes in the buffer object
    size_t size,       ///< [in] size in bytes of data being read
    void *pDst, ///< [in] pointer to host memory where data is to be read into
    uint32_t numEventsInWaitList, ///< [in] size of the event wait list
    const ur_event_handle_t
        *phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)]
                          ///< pointer to a list of events that must be complete
                          ///< before this command can be executed. If nullptr,
                          ///< the numEventsInWaitList must be 0, indicating
                          ///< that this command does not wait on any event to
                          ///< complete.
    ur_event_handle_t
        *phEvent ///< [in,out][optional] return an event object that identifies
                 ///< this particular command instance.
) {
  ur_mem_handle_t_ *Src = ur_cast<ur_mem_handle_t_ *>(hBuffer);

  std::shared_lock<ur_shared_mutex> SrcLock(Src->Mutex, std::defer_lock);
  std::scoped_lock<std::shared_lock<ur_shared_mutex>, ur_shared_mutex> LockAll(
      SrcLock, Queue->Mutex);

  char *ZeHandleSrc = nullptr;
  UR_CALL(Src->getZeHandle(ZeHandleSrc, ur_mem_handle_t_::read_only,
                           Queue->Device));
  return enqueueMemCopyHelper(UR_COMMAND_MEM_BUFFER_READ, Queue, pDst,
                              blockingRead, size, ZeHandleSrc + offset,
                              numEventsInWaitList, phEventWaitList, phEvent,
                              true /* PreferCopyEngine */);
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueMemBufferWrite(
    ur_queue_handle_t Queue, ///< [in] handle of the queue object
    ur_mem_handle_t hBuffer, ///< [in] handle of the buffer object
    bool
        blockingWrite, ///< [in] indicates blocking (true), non-blocking (false)
    size_t offset,     ///< [in] offset in bytes in the buffer object
    size_t size,       ///< [in] size in bytes of data being written
    const void
        *pSrc, ///< [in] pointer to host memory where data is to be written from
    uint32_t numEventsInWaitList, ///< [in] size of the event wait list
    const ur_event_handle_t
        *phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)]
                          ///< pointer to a list of events that must be complete
                          ///< before this command can be executed. If nullptr,
                          ///< the numEventsInWaitList must be 0, indicating
                          ///< that this command does not wait on any event to
                          ///< complete.
    ur_event_handle_t
        *phEvent ///< [in,out][optional] return an event object that identifies
                 ///< this particular command instance.
) {
  ur_mem_handle_t_ *Buffer = ur_cast<ur_mem_handle_t_ *>(hBuffer);

  std::scoped_lock<ur_shared_mutex, ur_shared_mutex> Lock(Queue->Mutex,
                                                          Buffer->Mutex);

  char *ZeHandleDst = nullptr;
  UR_CALL(Buffer->getZeHandle(ZeHandleDst, ur_mem_handle_t_::write_only,
                              Queue->Device));
  return enqueueMemCopyHelper(UR_COMMAND_MEM_BUFFER_WRITE, Queue,
                              ZeHandleDst + offset, // dst
                              blockingWrite, size,
                              pSrc, // src
                              numEventsInWaitList, phEventWaitList, phEvent,
                              true /* PreferCopyEngine */);
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueMemBufferReadRect(
    ur_queue_handle_t Queue, ///< [in] handle of the queue object
    ur_mem_handle_t hBuffer, ///< [in] handle of the buffer object
    bool blockingRead, ///< [in] indicates blocking (true), non-blocking (false)
    ur_rect_offset_t bufferOffset, ///< [in] 3D offset in the buffer
    ur_rect_offset_t hostOffset,   ///< [in] 3D offset in the host region
    ur_rect_region_t
        region, ///< [in] 3D rectangular region descriptor: width, height, depth
    size_t bufferRowPitch,   ///< [in] length of each row in bytes in the buffer
                             ///< object
    size_t bufferSlicePitch, ///< [in] length of each 2D slice in bytes in the
                             ///< buffer object being read
    size_t hostRowPitch,     ///< [in] length of each row in bytes in the host
                             ///< memory region pointed by dst
    size_t hostSlicePitch,   ///< [in] length of each 2D slice in bytes in the
                             ///< host memory region pointed by dst
    void *pDst, ///< [in] pointer to host memory where data is to be read into
    uint32_t numEventsInWaitList, ///< [in] size of the event wait list
    const ur_event_handle_t
        *phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)]
                          ///< pointer to a list of events that must be complete
                          ///< before this command can be executed. If nullptr,
                          ///< the numEventsInWaitList must be 0, indicating
                          ///< that this command does not wait on any event to
                          ///< complete.
    ur_event_handle_t
        *phEvent ///< [in,out][optional] return an event object that identifies
                 ///< this particular command instance.
) {
  ur_mem_handle_t_ *Buffer = ur_cast<ur_mem_handle_t_ *>(hBuffer);

  std::shared_lock<ur_shared_mutex> SrcLock(Buffer->Mutex, std::defer_lock);
  std::scoped_lock<std::shared_lock<ur_shared_mutex>, ur_shared_mutex> LockAll(
      SrcLock, Queue->Mutex);

  char *ZeHandleSrc;
  UR_CALL(Buffer->getZeHandle(ZeHandleSrc, ur_mem_handle_t_::read_only,
                              Queue->Device));
  return enqueueMemCopyRectHelper(
      UR_COMMAND_MEM_BUFFER_READ_RECT, Queue, ZeHandleSrc, pDst, bufferOffset,
      hostOffset, region, bufferRowPitch, hostRowPitch, bufferSlicePitch,
      hostSlicePitch, blockingRead, numEventsInWaitList, phEventWaitList,
      phEvent);
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueMemBufferWriteRect(
    ur_queue_handle_t Queue, ///< [in] handle of the queue object
    ur_mem_handle_t hBuffer, ///< [in] handle of the buffer object
    bool
        blockingWrite, ///< [in] indicates blocking (true), non-blocking (false)
    ur_rect_offset_t bufferOffset, ///< [in] 3D offset in the buffer
    ur_rect_offset_t hostOffset,   ///< [in] 3D offset in the host region
    ur_rect_region_t
        region, ///< [in] 3D rectangular region descriptor: width, height, depth
    size_t bufferRowPitch,   ///< [in] length of each row in bytes in the buffer
                             ///< object
    size_t bufferSlicePitch, ///< [in] length of each 2D slice in bytes in the
                             ///< buffer object being written
    size_t hostRowPitch,     ///< [in] length of each row in bytes in the host
                             ///< memory region pointed by src
    size_t hostSlicePitch,   ///< [in] length of each 2D slice in bytes in the
                             ///< host memory region pointed by src
    void
        *pSrc, ///< [in] pointer to host memory where data is to be written from
    uint32_t numEventsInWaitList, ///< [in] size of the event wait list
    const ur_event_handle_t
        *phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)]
                          ///< points to a list of events that must be complete
                          ///< before this command can be executed. If nullptr,
                          ///< the numEventsInWaitList must be 0, indicating
                          ///< that this command does not wait on any event to
                          ///< complete.
    ur_event_handle_t
        *phEvent ///< [in,out][optional] return an event object that identifies
                 ///< this particular command instance.
) {
  ur_mem_handle_t_ *Buffer = ur_cast<ur_mem_handle_t_ *>(hBuffer);

  std::scoped_lock<ur_shared_mutex, ur_shared_mutex> Lock(Queue->Mutex,
                                                          Buffer->Mutex);

  char *ZeHandleDst = nullptr;
  UR_CALL(Buffer->getZeHandle(ZeHandleDst, ur_mem_handle_t_::write_only,
                              Queue->Device));
  return enqueueMemCopyRectHelper(
      UR_COMMAND_MEM_BUFFER_WRITE_RECT, Queue,
      const_cast<char *>(static_cast<const char *>(pSrc)), ZeHandleDst,
      hostOffset, bufferOffset, region, hostRowPitch, bufferRowPitch,
      hostSlicePitch, bufferSlicePitch, blockingWrite, numEventsInWaitList,
      phEventWaitList, phEvent);
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueMemBufferCopy(
    ur_queue_handle_t Queue,   ///< [in] handle of the queue object
    ur_mem_handle_t BufferSrc, ///< [in] handle of the src buffer object
    ur_mem_handle_t BufferDst, ///< [in] handle of the dest buffer object
    size_t SrcOffset, ///< [in] offset into hBufferSrc to begin copying from
    size_t DstOffset, ///< [in] offset info hBufferDst to begin copying into
    size_t Size,      ///< [in] size in bytes of data being copied
    uint32_t NumEventsInWaitList, ///< [in] size of the event wait list
    const ur_event_handle_t
        *EventWaitList, ///< [in][optional][range(0, numEventsInWaitList)]
                        ///< pointer to a list of events that must be complete
                        ///< before this command can be executed. If nullptr,
                        ///< the numEventsInWaitList must be 0, indicating
                        ///< that this command does not wait on any event to
                        ///< complete.
    ur_event_handle_t
        *OutEvent ///< [in,out][optional] return an event object that identifies
                  ///< this particular command instance.
) {
  _ur_buffer *SrcBuffer = ur_cast<_ur_buffer *>(BufferSrc);
  _ur_buffer *DstBuffer = ur_cast<_ur_buffer *>(BufferDst);

  UR_ASSERT(!SrcBuffer->isImage(), UR_RESULT_ERROR_INVALID_MEM_OBJECT);
  UR_ASSERT(!DstBuffer->isImage(), UR_RESULT_ERROR_INVALID_MEM_OBJECT);

  std::shared_lock<ur_shared_mutex> SrcLock(SrcBuffer->Mutex, std::defer_lock);
  std::scoped_lock<std::shared_lock<ur_shared_mutex>, ur_shared_mutex,
                   ur_shared_mutex>
      LockAll(SrcLock, DstBuffer->Mutex, Queue->Mutex);

  // Copy engine is preferred only for host to device transfer.
  // Device to device transfers run faster on compute engines.
  bool PreferCopyEngine = (SrcBuffer->OnHost || DstBuffer->OnHost);

  // Temporary option added to use copy engine for D2D copy
  PreferCopyEngine |= UseCopyEngineForD2DCopy;

  char *ZeHandleSrc = nullptr;
  UR_CALL(SrcBuffer->getZeHandle(ZeHandleSrc, ur_mem_handle_t_::read_only,
                                 Queue->Device));
  char *ZeHandleDst = nullptr;
  UR_CALL(DstBuffer->getZeHandle(ZeHandleDst, ur_mem_handle_t_::write_only,
                                 Queue->Device));

  return enqueueMemCopyHelper(
      UR_COMMAND_MEM_BUFFER_COPY, Queue, ZeHandleDst + DstOffset,
      false, // blocking
      Size, ZeHandleSrc + SrcOffset, NumEventsInWaitList, EventWaitList,
      OutEvent, PreferCopyEngine);
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueMemBufferCopyRect(
    ur_queue_handle_t Queue,    ///< [in] handle of the queue object
    ur_mem_handle_t BufferSrc,  ///< [in] handle of the source buffer object
    ur_mem_handle_t BufferDst,  ///< [in] handle of the dest buffer object
    ur_rect_offset_t SrcOrigin, ///< [in] 3D offset in the source buffer
    ur_rect_offset_t DstOrigin, ///< [in] 3D offset in the destination buffer
    ur_rect_region_t SrcRegion, ///< [in] source 3D rectangular region
                                ///< descriptor: width, height, depth
    size_t SrcRowPitch,   ///< [in] length of each row in bytes in the source
                          ///< buffer object
    size_t SrcSlicePitch, ///< [in] length of each 2D slice in bytes in the
                          ///< source buffer object
    size_t DstRowPitch, ///< [in] length of each row in bytes in the destination
                        ///< buffer object
    size_t DstSlicePitch, ///< [in] length of each 2D slice in bytes in the
                          ///< destination buffer object
    uint32_t NumEventsInWaitList, ///< [in] size of the event wait list
    const ur_event_handle_t
        *EventWaitList, ///< [in][optional][range(0, numEventsInWaitList)]
                        ///< pointer to a list of events that must be complete
                        ///< before this command can be executed. If nullptr,
                        ///< the numEventsInWaitList must be 0, indicating
                        ///< that this command does not wait on any event to
                        ///< complete.
    ur_event_handle_t
        *OutEvent ///< [in,out][optional] return an event object that identifies
                  ///< this particular command instance.
) {
  _ur_buffer *SrcBuffer = ur_cast<_ur_buffer *>(BufferSrc);
  _ur_buffer *DstBuffer = ur_cast<_ur_buffer *>(BufferDst);

  UR_ASSERT(!SrcBuffer->isImage(), UR_RESULT_ERROR_INVALID_MEM_OBJECT);
  UR_ASSERT(!DstBuffer->isImage(), UR_RESULT_ERROR_INVALID_MEM_OBJECT);

  std::shared_lock<ur_shared_mutex> SrcLock(SrcBuffer->Mutex, std::defer_lock);
  std::scoped_lock<std::shared_lock<ur_shared_mutex>, ur_shared_mutex,
                   ur_shared_mutex>
      LockAll(SrcLock, DstBuffer->Mutex, Queue->Mutex);

  // Copy engine is preferred only for host to device transfer.
  // Device to device transfers run faster on compute engines.
  bool PreferCopyEngine = (SrcBuffer->OnHost || DstBuffer->OnHost);

  char *ZeHandleSrc = nullptr;
  UR_CALL(SrcBuffer->getZeHandle(ZeHandleSrc, ur_mem_handle_t_::read_only,
                                 Queue->Device));
  char *ZeHandleDst = nullptr;
  UR_CALL(DstBuffer->getZeHandle(ZeHandleDst, ur_mem_handle_t_::write_only,
                                 Queue->Device));

  return enqueueMemCopyRectHelper(
      UR_COMMAND_MEM_BUFFER_COPY_RECT, Queue, ZeHandleSrc, ZeHandleDst,
      SrcOrigin, DstOrigin, SrcRegion, SrcRowPitch, DstRowPitch, SrcSlicePitch,
      DstSlicePitch,
      false, // blocking
      NumEventsInWaitList, EventWaitList, OutEvent, PreferCopyEngine);
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueMemBufferFill(
    ur_queue_handle_t Queue, ///< [in] handle of the queue object
    ur_mem_handle_t Buffer,  ///< [in] handle of the buffer object
    const void *Pattern,     ///< [in] pointer to the fill pattern
    size_t PatternSize,      ///< [in] size in bytes of the pattern
    size_t Offset,           ///< [in] offset into the buffer
    size_t Size, ///< [in] fill size in bytes, must be a multiple of patternSize
    uint32_t NumEventsInWaitList, ///< [in] size of the event wait list
    const ur_event_handle_t
        *EventWaitList, ///< [in][optional][range(0, numEventsInWaitList)]
                        ///< pointer to a list of events that must be complete
                        ///< before this command can be executed. If nullptr,
                        ///< the numEventsInWaitList must be 0, indicating
                        ///< that this command does not wait on any event to
                        ///< complete.
    ur_event_handle_t
        *OutEvent ///< [in,out][optional] return an event object that identifies
                  ///< this particular command instance.
) {
  std::scoped_lock<ur_shared_mutex, ur_shared_mutex> Lock(Queue->Mutex,
                                                          Buffer->Mutex);

  char *ZeHandleDst = nullptr;
  _ur_buffer *UrBuffer = reinterpret_cast<_ur_buffer *>(Buffer);
  UR_CALL(UrBuffer->getZeHandle(ZeHandleDst, ur_mem_handle_t_::write_only,
                                Queue->Device));
  return enqueueMemFillHelper(
      UR_COMMAND_MEM_BUFFER_FILL, Queue, ZeHandleDst + Offset,
      Pattern,     // It will be interpreted as an 8-bit value,
      PatternSize, // which is indicated with this pattern_size==1
      Size, NumEventsInWaitList, EventWaitList, OutEvent);
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueMemImageRead(
    ur_queue_handle_t Queue, ///< [in] handle of the queue object
    ur_mem_handle_t Image,   ///< [in] handle of the image object
    bool BlockingRead, ///< [in] indicates blocking (true), non-blocking (false)
    ur_rect_offset_t Origin, ///< [in] defines the (x,y,z) offset in pixels in
                             ///< the 1D, 2D, or 3D image
    ur_rect_region_t Region, ///< [in] defines the (width, height, depth) in
                             ///< pixels of the 1D, 2D, or 3D image
    size_t RowPitch,         ///< [in] length of each row in bytes
    size_t SlicePitch,       ///< [in] length of each 2D slice of the 3D image
    void *Dst, ///< [in] pointer to host memory where image is to be read into
    uint32_t NumEventsInWaitList, ///< [in] size of the event wait list
    const ur_event_handle_t
        *EventWaitList, ///< [in][optional][range(0, numEventsInWaitList)]
                        ///< pointer to a list of events that must be complete
                        ///< before this command can be executed. If nullptr,
                        ///< the numEventsInWaitList must be 0, indicating
                        ///< that this command does not wait on any event to
                        ///< complete.
    ur_event_handle_t
        *OutEvent ///< [in,out][optional] return an event object that identifies
                  ///< this particular command instance.
) {
  std::scoped_lock<ur_shared_mutex, ur_shared_mutex> Lock(Queue->Mutex,
                                                          Image->Mutex);
  return enqueueMemImageCommandHelper(
      UR_COMMAND_MEM_IMAGE_READ, Queue, Image, Dst, BlockingRead, &Origin,
      nullptr, &Region, RowPitch, SlicePitch, NumEventsInWaitList,
      EventWaitList, OutEvent);
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueMemImageWrite(
    ur_queue_handle_t Queue, ///< [in] handle of the queue object
    ur_mem_handle_t Image,   ///< [in] handle of the image object
    bool
        BlockingWrite, ///< [in] indicates blocking (true), non-blocking (false)
    ur_rect_offset_t Origin, ///< [in] defines the (x,y,z) offset in pixels in
                             ///< the 1D, 2D, or 3D image
    ur_rect_region_t Region, ///< [in] defines the (width, height, depth) in
                             ///< pixels of the 1D, 2D, or 3D image
    size_t RowPitch,         ///< [in] length of each row in bytes
    size_t SlicePitch,       ///< [in] length of each 2D slice of the 3D image
    void *Src, ///< [in] pointer to host memory where image is to be read into
    uint32_t NumEventsInWaitList, ///< [in] size of the event wait list
    const ur_event_handle_t
        *EventWaitList, ///< [in][optional][range(0, numEventsInWaitList)]
                        ///< pointer to a list of events that must be complete
                        ///< before this command can be executed. If nullptr,
                        ///< the numEventsInWaitList must be 0, indicating
                        ///< that this command does not wait on any event to
                        ///< complete.
    ur_event_handle_t
        *OutEvent ///< [in,out][optional] return an event object that identifies
                  ///< this particular command instance.
) {
  std::scoped_lock<ur_shared_mutex, ur_shared_mutex> Lock(Queue->Mutex,
                                                          Image->Mutex);
  return enqueueMemImageCommandHelper(
      UR_COMMAND_MEM_IMAGE_WRITE, Queue, Src, Image, BlockingWrite, nullptr,
      &Origin, &Region, RowPitch, SlicePitch, NumEventsInWaitList,
      EventWaitList, OutEvent);
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueMemImageCopy(
    ur_queue_handle_t Queue,    ///< [in] handle of the queue object
    ur_mem_handle_t ImageSrc,   ///< [in] handle of the src image object
    ur_mem_handle_t ImageDst,   ///< [in] handle of the dest image object
    ur_rect_offset_t SrcOrigin, ///< [in] defines the (x,y,z) offset in pixels
                                ///< in the source 1D, 2D, or 3D image
    ur_rect_offset_t DstOrigin, ///< [in] defines the (x,y,z) offset in pixels
                                ///< in the destination 1D, 2D, or 3D image
    ur_rect_region_t Region,    ///< [in] defines the (width, height, depth) in
                                ///< pixels of the 1D, 2D, or 3D image
    uint32_t NumEventsInWaitList, ///< [in] size of the event wait list
    const ur_event_handle_t
        *EventWaitList, ///< [in][optional][range(0, numEventsInWaitList)]
                        ///< pointer to a list of events that must be complete
                        ///< before this command can be executed. If nullptr,
                        ///< the numEventsInWaitList must be 0, indicating
                        ///< that this command does not wait on any event to
                        ///< complete.
    ur_event_handle_t
        *OutEvent ///< [in,out][optional] return an event object that identifies
                  ///< this particular command instance.
) {
  std::shared_lock<ur_shared_mutex> SrcLock(ImageSrc->Mutex, std::defer_lock);
  std::scoped_lock<std::shared_lock<ur_shared_mutex>, ur_shared_mutex,
                   ur_shared_mutex>
      LockAll(SrcLock, ImageDst->Mutex, Queue->Mutex);
  // Copy engine is preferred only for host to device transfer.
  // Device to device transfers run faster on compute engines.
  // Images are always allocated on device.
  bool PreferCopyEngine = false;
  return enqueueMemImageCommandHelper(
      UR_COMMAND_MEM_IMAGE_COPY, Queue, ImageSrc, ImageDst,
      false, // is_blocking
      &SrcOrigin, &DstOrigin, &Region,
      0, // row pitch
      0, // slice pitch
      NumEventsInWaitList, EventWaitList, OutEvent, PreferCopyEngine);
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueMemBufferMap(
    ur_queue_handle_t Queue, ///< [in] handle of the queue object
    ur_mem_handle_t Buf,     ///< [in] handle of the buffer object
    bool BlockingMap, ///< [in] indicates blocking (true), non-blocking (false)
    ur_map_flags_t MapFlags, ///< [in] flags for read, write, readwrite mapping
    size_t Offset, ///< [in] offset in bytes of the buffer region being mapped
    size_t Size,   ///< [in] size in bytes of the buffer region being mapped
    uint32_t NumEventsInWaitList, ///< [in] size of the event wait list
    const ur_event_handle_t
        *EventWaitList, ///< [in][optional][range(0, numEventsInWaitList)]
                        ///< pointer to a list of events that must be complete
                        ///< before this command can be executed. If nullptr,
                        ///< the numEventsInWaitList must be 0, indicating
                        ///< that this command does not wait on any event to
                        ///< complete.
    ur_event_handle_t
        *OutEvent, ///< [in,out][optional] return an event object that
                   ///< identifies this particular command instance.
    void **RetMap  ///< [in,out] return mapped pointer.  TODO: move it before
                   ///< numEventsInWaitList?
) {

  auto Buffer = ur_cast<_ur_buffer *>(Buf);

  UR_ASSERT(!Buffer->isImage(), UR_RESULT_ERROR_INVALID_MEM_OBJECT);

  ur_event_handle_t InternalEvent;
  bool IsInternal = OutEvent == nullptr;
  ur_event_handle_t *Event = OutEvent ? OutEvent : &InternalEvent;
  ze_event_handle_t ZeEvent = nullptr;

  bool UseCopyEngine = false;
  {
    // Lock automatically releases when this goes out of scope.
    std::scoped_lock<ur_shared_mutex> lock(Queue->Mutex);

    _ur_ze_event_list_t TmpWaitList;
    UR_CALL(TmpWaitList.createAndRetainUrZeEventList(
        NumEventsInWaitList, EventWaitList, Queue, UseCopyEngine));

    UR_CALL(
        createEventAndAssociateQueue(Queue, Event, UR_COMMAND_MEM_BUFFER_MAP,
                                     Queue->CommandListMap.end(), IsInternal));

    ZeEvent = (*Event)->ZeEvent;
    (*Event)->WaitList = TmpWaitList;
  }

  // Translate the host access mode info.
  ur_mem_handle_t_::access_mode_t AccessMode = ur_mem_handle_t_::unknown;
  if (MapFlags & UR_MAP_FLAG_WRITE_INVALIDATE_REGION)
    AccessMode = ur_mem_handle_t_::write_only;
  else {
    if (MapFlags & UR_MAP_FLAG_READ) {
      AccessMode = ur_mem_handle_t_::read_only;
      if (MapFlags & UR_MAP_FLAG_WRITE)
        AccessMode = ur_mem_handle_t_::read_write;
    } else if (MapFlags & UR_MAP_FLAG_WRITE)
      AccessMode = ur_mem_handle_t_::write_only;
  }

  UR_ASSERT(AccessMode != ur_mem_handle_t_::unknown,
            UR_RESULT_ERROR_INVALID_VALUE);

  // TODO: Level Zero is missing the memory "mapping" capabilities, so we are
  // left to doing new memory allocation and a copy (read) on discrete devices.
  // For integrated devices, we have allocated the buffer in host memory so no
  // actions are needed here except for synchronizing on incoming events.
  // A host-to-host copy is done if a host pointer had been supplied during
  // buffer creation on integrated devices.
  //
  // TODO: for discrete, check if the input buffer is already allocated
  // in shared memory and thus is accessible from the host as is.
  // Can we get SYCL RT to predict/allocate in shared memory
  // from the beginning?

  // For integrated devices the buffer has been allocated in host memory.
  if (Buffer->OnHost) {
    // Wait on incoming events before doing the copy
    if (NumEventsInWaitList > 0)
      UR_CALL(urEventWait(NumEventsInWaitList, EventWaitList));

    if (Queue->isInOrderQueue())
      UR_CALL(urQueueFinish(Queue));

    // Lock automatically releases when this goes out of scope.
    std::scoped_lock<ur_shared_mutex> Guard(Buffer->Mutex);

    char *ZeHandleSrc;
    UR_CALL(Buffer->getZeHandle(ZeHandleSrc, AccessMode, Queue->Device));

    if (Buffer->MapHostPtr) {
      *RetMap = Buffer->MapHostPtr + Offset;
      if (ZeHandleSrc != Buffer->MapHostPtr &&
          AccessMode != ur_mem_handle_t_::write_only) {
        memcpy(*RetMap, ZeHandleSrc + Offset, Size);
      }
    } else {
      *RetMap = ZeHandleSrc + Offset;
    }

    auto Res = Buffer->Mappings.insert({*RetMap, {Offset, Size}});
    // False as the second value in pair means that mapping was not inserted
    // because mapping already exists.
    if (!Res.second) {
      urPrint("urEnqueueMemBufferMap: duplicate mapping detected\n");
      return UR_RESULT_ERROR_INVALID_VALUE;
    }

    // Signal this event
    ZE2UR_CALL(zeEventHostSignal, (ZeEvent));
    (*Event)->Completed = true;
    return UR_RESULT_SUCCESS;
  }

  // Lock automatically releases when this goes out of scope.
  std::scoped_lock<ur_shared_mutex, ur_shared_mutex> Lock(Queue->Mutex,
                                                          Buffer->Mutex);

  if (Buffer->MapHostPtr) {
    *RetMap = Buffer->MapHostPtr + Offset;
  } else {
    // TODO: use USM host allocator here
    // TODO: Do we even need every map to allocate new host memory?
    //       In the case when the buffer is "OnHost" we use single allocation.
    UR_CALL(ZeHostMemAllocHelper(RetMap, Queue->Context, Size));
  }

  // Take a shortcut if the host is not going to read buffer's data.
  if (AccessMode == ur_mem_handle_t_::write_only) {
    (*Event)->Completed = true;
  } else {
    // For discrete devices we need a command list
    ur_command_list_ptr_t CommandList{};
    UR_CALL(Queue->Context->getAvailableCommandList(Queue, CommandList,
                                                    UseCopyEngine));

    // Add the event to the command list.
    CommandList->second.append(reinterpret_cast<ur_event_handle_t>(*Event));
    (*Event)->RefCount.increment();

    const auto &ZeCommandList = CommandList->first;
    const auto &WaitList = (*Event)->WaitList;

    char *ZeHandleSrc;
    UR_CALL(Buffer->getZeHandle(ZeHandleSrc, AccessMode, Queue->Device));

    ZE2UR_CALL(zeCommandListAppendMemoryCopy,
               (ZeCommandList, *RetMap, ZeHandleSrc + Offset, Size, ZeEvent,
                WaitList.Length, WaitList.ZeEventList));

    UR_CALL(Queue->executeCommandList(CommandList, BlockingMap));
  }

  auto Res = Buffer->Mappings.insert({*RetMap, {Offset, Size}});
  // False as the second value in pair means that mapping was not inserted
  // because mapping already exists.
  if (!Res.second) {
    urPrint("urEnqueueMemBufferMap: duplicate mapping detected\n");
    return UR_RESULT_ERROR_INVALID_VALUE;
  }
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueMemUnmap(
    ur_queue_handle_t Queue, ///< [in] handle of the queue object
    ur_mem_handle_t Mem, ///< [in] handle of the memory (buffer or image) object
    void *MappedPtr,     ///< [in] mapped host address
    uint32_t NumEventsInWaitList, ///< [in] size of the event wait list
    const ur_event_handle_t
        *EventWaitList, ///< [in][optional][range(0, numEventsInWaitList)]
                        ///< pointer to a list of events that must be complete
                        ///< before this command can be executed. If nullptr,
                        ///< the numEventsInWaitList must be 0, indicating
                        ///< that this command does not wait on any event to
                        ///< complete.
    ur_event_handle_t
        *OutEvent ///< [in,out][optional] return an event object that identifies
                  ///< this particular command instance.
) {
  UR_ASSERT(!Mem->isImage(), UR_RESULT_ERROR_INVALID_MEM_OBJECT);

  auto Buffer = ur_cast<_ur_buffer *>(Mem);

  bool UseCopyEngine = false;

  ze_event_handle_t ZeEvent = nullptr;
  ur_event_handle_t InternalEvent;
  bool IsInternal = OutEvent == nullptr;
  ur_event_handle_t *Event = OutEvent ? OutEvent : &InternalEvent;
  {
    // Lock automatically releases when this goes out of scope.
    std::scoped_lock<ur_shared_mutex> lock(Queue->Mutex);

    _ur_ze_event_list_t TmpWaitList;
    UR_CALL(TmpWaitList.createAndRetainUrZeEventList(
        NumEventsInWaitList, EventWaitList, Queue, UseCopyEngine));

    UR_CALL(createEventAndAssociateQueue(Queue, Event, UR_COMMAND_MEM_UNMAP,
                                         Queue->CommandListMap.end(),
                                         IsInternal));
    ZeEvent = (*Event)->ZeEvent;
    (*Event)->WaitList = TmpWaitList;
  }

  _ur_buffer::Mapping MapInfo = {};
  {
    // Lock automatically releases when this goes out of scope.
    std::scoped_lock<ur_shared_mutex> Guard(Buffer->Mutex);
    auto It = Buffer->Mappings.find(MappedPtr);
    if (It == Buffer->Mappings.end()) {
      urPrint("urEnqueueMemUnmap: unknown memory mapping\n");
      return UR_RESULT_ERROR_INVALID_VALUE;
    }
    MapInfo = It->second;
    Buffer->Mappings.erase(It);

    // NOTE: we still have to free the host memory allocated/returned by
    // piEnqueueMemBufferMap, but can only do so after the above copy
    // is completed. Instead of waiting for It here (blocking), we shall
    // do so in piEventRelease called for the pi_event tracking the unmap.
    // In the case of an integrated device, the map operation does not allocate
    // any memory, so there is nothing to free. This is indicated by a nullptr.
    (*Event)->CommandData =
        (Buffer->OnHost ? nullptr : (Buffer->MapHostPtr ? nullptr : MappedPtr));
  }

  // For integrated devices the buffer is allocated in host memory.
  if (Buffer->OnHost) {
    // Wait on incoming events before doing the copy
    if (NumEventsInWaitList > 0)
      UR_CALL(urEventWait(NumEventsInWaitList, EventWaitList));

    if (Queue->isInOrderQueue())
      UR_CALL(urQueueFinish(Queue));

    char *ZeHandleDst;
    UR_CALL(Buffer->getZeHandle(ZeHandleDst, ur_mem_handle_t_::write_only,
                                Queue->Device));

    std::scoped_lock<ur_shared_mutex> Guard(Buffer->Mutex);
    if (Buffer->MapHostPtr)
      memcpy(ZeHandleDst + MapInfo.Offset, MappedPtr, MapInfo.Size);

    // Signal this event
    ZE2UR_CALL(zeEventHostSignal, (ZeEvent));
    (*Event)->Completed = true;
    return UR_RESULT_SUCCESS;
  }

  // Lock automatically releases when this goes out of scope.
  std::scoped_lock<ur_shared_mutex, ur_shared_mutex> Lock(Queue->Mutex,
                                                          Buffer->Mutex);

  ur_command_list_ptr_t CommandList{};
  UR_CALL(Queue->Context->getAvailableCommandList(
      reinterpret_cast<ur_queue_handle_t>(Queue), CommandList, UseCopyEngine));

  CommandList->second.append(reinterpret_cast<ur_event_handle_t>(*Event));
  (*Event)->RefCount.increment();

  const auto &ZeCommandList = CommandList->first;

  // TODO: Level Zero is missing the memory "mapping" capabilities, so we are
  // left to doing copy (write back to the device).
  //
  // NOTE: Keep this in sync with the implementation of
  // piEnqueueMemBufferMap.

  char *ZeHandleDst;
  UR_CALL(Buffer->getZeHandle(ZeHandleDst, ur_mem_handle_t_::write_only,
                              Queue->Device));

  ZE2UR_CALL(zeCommandListAppendMemoryCopy,
             (ZeCommandList, ZeHandleDst + MapInfo.Offset, MappedPtr,
              MapInfo.Size, ZeEvent, (*Event)->WaitList.Length,
              (*Event)->WaitList.ZeEventList));

  // Execute command list asynchronously, as the event will be used
  // to track down its completion.
  UR_CALL(Queue->executeCommandList(CommandList));

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueUSMMemset(
    ur_queue_handle_t Queue,      ///< [in] handle of the queue object
    void *Ptr,                    ///< [in] pointer to USM memory object
    int8_t ByteValue,             ///< [in] byte value to fill
    size_t Count,                 ///< [in] size in bytes to be set
    uint32_t NumEventsInWaitList, ///< [in] size of the event wait list
    const ur_event_handle_t
        *EventWaitList, ///< [in][optional][range(0, numEventsInWaitList)]
                        ///< pointer to a list of events that must be complete
                        ///< before this command can be executed. If nullptr,
                        ///< the numEventsInWaitList must be 0, indicating
                        ///< that this command does not wait on any event to
                        ///< complete.
    ur_event_handle_t *Event ///< [in,out][optional] return an event object that
                             ///< identifies this particular command instance.
) {
  std::ignore = Queue;
  std::ignore = Ptr;
  std::ignore = ByteValue;
  std::ignore = Count;
  std::ignore = NumEventsInWaitList;
  std::ignore = EventWaitList;
  std::ignore = Event;
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueUSMMemcpy(
    ur_queue_handle_t Queue, ///< [in] handle of the queue object
    bool Blocking,           ///< [in] blocking or non-blocking copy
    void *Dst,       ///< [in] pointer to the destination USM memory object
    const void *Src, ///< [in] pointer to the source USM memory object
    size_t Size,     ///< [in] size in bytes to be copied
    uint32_t NumEventsInWaitList, ///< [in] size of the event wait list
    const ur_event_handle_t
        *EventWaitList, ///< [in][optional][range(0, numEventsInWaitList)]
                        ///< pointer to a list of events that must be complete
                        ///< before this command can be executed. If nullptr,
                        ///< the numEventsInWaitList must be 0, indicating
                        ///< that this command does not wait on any event to
                        ///< complete.
    ur_event_handle_t
        *OutEvent ///< [in,out][optional] return an event object that identifies
                  ///< this particular command instance.
) {
  std::scoped_lock<ur_shared_mutex> lock(Queue->Mutex);

  // Device to Device copies are found to execute slower on copy engine
  // (versus compute engine).
  bool PreferCopyEngine = !IsDevicePointer(Queue->Context, Src) ||
                          !IsDevicePointer(Queue->Context, Dst);

  // Temporary option added to use copy engine for D2D copy
  PreferCopyEngine |= UseCopyEngineForD2DCopy;

  return enqueueMemCopyHelper( // TODO: do we need a new command type for this?
      UR_COMMAND_MEM_BUFFER_COPY, Queue, Dst, Blocking, Size, Src,
      NumEventsInWaitList, EventWaitList, OutEvent, PreferCopyEngine);
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueUSMPrefetch(
    ur_queue_handle_t Queue,        ///< [in] handle of the queue object
    const void *Mem,                ///< [in] pointer to the USM memory object
    size_t Size,                    ///< [in] size in bytes to be fetched
    ur_usm_migration_flags_t Flags, ///< [in] USM prefetch flags
    uint32_t NumEventsInWaitList,   ///< [in] size of the event wait list
    const ur_event_handle_t
        *EventWaitList, ///< [in][optional][range(0, numEventsInWaitList)]
                        ///< pointer to a list of events that must be complete
                        ///< before this command can be executed. If nullptr,
                        ///< the numEventsInWaitList must be 0, indicating
                        ///< that this command does not wait on any event to
                        ///< complete.
    ur_event_handle_t
        *OutEvent ///< [in,out][optional] return an event object that identifies
                  ///< this particular command instance.
) {
  std::ignore = Flags;
  // Lock automatically releases when this goes out of scope.
  std::scoped_lock<ur_shared_mutex> lock(Queue->Mutex);

  bool UseCopyEngine = false;

  // Please note that the following code should be run before the
  // subsequent getAvailableCommandList() call so that there is no
  // dead-lock from waiting unsubmitted events in an open batch.
  // The createAndRetainUrZeEventList() has the proper side-effect
  // of submitting batches with dependent events.
  //
  _ur_ze_event_list_t TmpWaitList;
  UR_CALL(TmpWaitList.createAndRetainUrZeEventList(
      NumEventsInWaitList, EventWaitList, Queue, UseCopyEngine));

  // Get a new command list to be used on this call
  ur_command_list_ptr_t CommandList{};
  // TODO: Change UseCopyEngine argument to 'true' once L0 backend
  // support is added
  UR_CALL(Queue->Context->getAvailableCommandList(Queue, CommandList,
                                                  UseCopyEngine));

  // TODO: do we need to create a unique command type for this?
  ze_event_handle_t ZeEvent = nullptr;
  ur_event_handle_t InternalEvent;
  bool IsInternal = OutEvent == nullptr;
  ur_event_handle_t *Event = OutEvent ? OutEvent : &InternalEvent;
  UR_CALL(createEventAndAssociateQueue(Queue, Event, UR_COMMAND_USM_PREFETCH,
                                       CommandList, IsInternal));
  ZeEvent = (*Event)->ZeEvent;
  (*Event)->WaitList = TmpWaitList;

  const auto &WaitList = (*Event)->WaitList;
  const auto &ZeCommandList = CommandList->first;
  if (WaitList.Length) {
    ZE2UR_CALL(zeCommandListAppendWaitOnEvents,
               (ZeCommandList, WaitList.Length, WaitList.ZeEventList));
  }
  // TODO: figure out how to translate "flags"
  ZE2UR_CALL(zeCommandListAppendMemoryPrefetch, (ZeCommandList, Mem, Size));

  // TODO: Level Zero does not have a completion "event" with the prefetch API,
  // so manually add command to signal our event.
  ZE2UR_CALL(zeCommandListAppendSignalEvent, (ZeCommandList, ZeEvent));

  UR_CALL(Queue->executeCommandList(CommandList, false));

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueUSMAdvise(
    ur_queue_handle_t Queue,      ///< [in] handle of the queue object
    const void *Mem,              ///< [in] pointer to the USM memory object
    size_t Size,                  ///< [in] size in bytes to be advised
    ur_usm_advice_flags_t Advice, ///< [in] USM memory advice
    ur_event_handle_t
        *OutEvent ///< [in,out][optional] return an event object that identifies
                  ///< this particular command instance.
) {
  // Lock automatically releases when this goes out of scope.
  std::scoped_lock<ur_shared_mutex> lock(Queue->Mutex);

  auto ZeAdvice = ur_cast<ze_memory_advice_t>(Advice);

  bool UseCopyEngine = false;

  _ur_ze_event_list_t TmpWaitList;
  UR_CALL(TmpWaitList.createAndRetainUrZeEventList(0, nullptr, Queue,
                                                   UseCopyEngine));

  // Get a new command list to be used on this call
  ur_command_list_ptr_t CommandList{};
  // UseCopyEngine is set to 'false' here.
  // TODO: Additional analysis is required to check if this operation will
  // run faster on copy engines.
  UR_CALL(Queue->Context->getAvailableCommandList(Queue, CommandList,
                                                  UseCopyEngine));

  // TODO: do we need to create a unique command type for this?
  ze_event_handle_t ZeEvent = nullptr;
  ur_event_handle_t InternalEvent{};
  bool IsInternal = OutEvent == nullptr;
  ur_event_handle_t *Event = OutEvent ? OutEvent : &InternalEvent;
  UR_CALL(createEventAndAssociateQueue(Queue, Event, UR_COMMAND_USM_ADVISE,
                                       CommandList, IsInternal));
  ZeEvent = (*Event)->ZeEvent;
  (*Event)->WaitList = TmpWaitList;

  const auto &ZeCommandList = CommandList->first;
  const auto &WaitList = (*Event)->WaitList;

  if (WaitList.Length) {
    ZE2UR_CALL(zeCommandListAppendWaitOnEvents,
               (ZeCommandList, WaitList.Length, WaitList.ZeEventList));
  }

  ZE2UR_CALL(zeCommandListAppendMemAdvise,
             (ZeCommandList, Queue->Device->ZeDevice, Mem, Size, ZeAdvice));

  // TODO: Level Zero does not have a completion "event" with the advise API,
  // so manually add command to signal our event.
  ZE2UR_CALL(zeCommandListAppendSignalEvent, (ZeCommandList, ZeEvent));

  Queue->executeCommandList(CommandList, false);

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueUSMFill2D(
    ur_queue_handle_t Queue, ///< [in] handle of the queue to submit to.
    void *Mem,               ///< [in] pointer to memory to be filled.
    size_t Pitch, ///< [in] the total width of the destination memory including
                  ///< padding.
    size_t PatternSize,  ///< [in] the size in bytes of the pattern.
    const void *Pattern, ///< [in] pointer with the bytes of the pattern to set.
    size_t Width,        ///< [in] the width in bytes of each row to fill.
    size_t Height,       ///< [in] the height of the columns to fill.
    uint32_t NumEventsInWaitList, ///< [in] size of the event wait list
    const ur_event_handle_t
        *EventWaitList, ///< [in][optional][range(0, numEventsInWaitList)]
                        ///< pointer to a list of events that must be complete
                        ///< before the kernel execution. If nullptr, the
                        ///< numEventsInWaitList must be 0, indicating that no
                        ///< wait event.
    ur_event_handle_t
        *OutEvent ///< [in,out][optional] return an event object that identifies
                  ///< this particular kernel execution instance.
) {
  std::ignore = Queue;
  std::ignore = Mem;
  std::ignore = Pitch;
  std::ignore = PatternSize;
  std::ignore = Pattern;
  std::ignore = Width;
  std::ignore = Height;
  std::ignore = NumEventsInWaitList;
  std::ignore = EventWaitList;
  std::ignore = OutEvent;
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueUSMMemset2D(
    ur_queue_handle_t Queue, ///< [in] handle of the queue to submit to.
    void *Mem,               ///< [in] pointer to memory to be filled.
    size_t Pitch,  ///< [in] the total width of the destination memory including
                   ///< padding.
    int Value,     ///< [in] the value to fill into the region in pMem.
    size_t Width,  ///< [in] the width in bytes of each row to set.
    size_t Height, ///< [in] the height of the columns to set.
    uint32_t NumEventsInWaitList, ///< [in] size of the event wait list
    const ur_event_handle_t
        *EventWaitList, ///< [in][optional][range(0, numEventsInWaitList)]
                        ///< pointer to a list of events that must be complete
                        ///< before the kernel execution. If nullptr, the
                        ///< numEventsInWaitList must be 0, indicating that no
                        ///< wait event.
    ur_event_handle_t
        *OutEvent ///< [in,out][optional] return an event object that identifies
                  ///< this particular kernel execution instance.
) {
  std::ignore = Queue;
  std::ignore = Mem;
  std::ignore = Pitch;
  std::ignore = Value;
  std::ignore = Width;
  std::ignore = Height;
  std::ignore = NumEventsInWaitList;
  std::ignore = EventWaitList;
  std::ignore = OutEvent;
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueUSMMemcpy2D(
    ur_queue_handle_t Queue, ///< [in] handle of the queue to submit to.
    bool Blocking, ///< [in] indicates if this operation should block the host.
    void *Dst,     ///< [in] pointer to memory where data will be copied.
    size_t DstPitch, ///< [in] the total width of the source memory including
                     ///< padding.
    const void *Src, ///< [in] pointer to memory to be copied.
    size_t SrcPitch, ///< [in] the total width of the source memory including
                     ///< padding.
    size_t Width,    ///< [in] the width in bytes of each row to be copied.
    size_t Height,   ///< [in] the height of columns to be copied.
    uint32_t NumEventsInWaitList, ///< [in] size of the event wait list
    const ur_event_handle_t
        *EventWaitList, ///< [in][optional][range(0, numEventsInWaitList)]
                        ///< pointer to a list of events that must be complete
                        ///< before the kernel execution. If nullptr, the
                        ///< numEventsInWaitList must be 0, indicating that no
                        ///< wait event.
    ur_event_handle_t
        *Event ///< [in,out][optional] return an event object that identifies
               ///< this particular kernel execution instance.
) {

  ur_rect_offset_t ZeroOffset{0, 0, 0};
  ur_rect_region_t Region{Width, Height, 0};

  std::scoped_lock<ur_shared_mutex> lock(Queue->Mutex);

  // Device to Device copies are found to execute slower on copy engine
  // (versus compute engine).
  bool PreferCopyEngine = !IsDevicePointer(Queue->Context, Src) ||
                          !IsDevicePointer(Queue->Context, Dst);

  // Temporary option added to use copy engine for D2D copy
  PreferCopyEngine |= UseCopyEngineForD2DCopy;

  return enqueueMemCopyRectHelper( // TODO: do we need a new command type for
                                   // this?
      UR_COMMAND_MEM_BUFFER_COPY_RECT, Queue, Src, Dst, ZeroOffset, ZeroOffset,
      Region, SrcPitch, DstPitch, 0, /*SrcSlicePitch=*/
      0,                             /*DstSlicePitch=*/
      Blocking, NumEventsInWaitList, EventWaitList, Event, PreferCopyEngine);
}

static ur_result_t ur2zeImageDesc(const ur_image_format_t *ImageFormat,
                                  const ur_image_desc_t *ImageDesc,
                                  ZeStruct<ze_image_desc_t> &ZeImageDesc) {

  ze_image_format_type_t ZeImageFormatType;
  size_t ZeImageFormatTypeSize;
  switch (ImageFormat->channelType) {
  case UR_IMAGE_CHANNEL_TYPE_FLOAT: {
    ZeImageFormatType = ZE_IMAGE_FORMAT_TYPE_FLOAT;
    ZeImageFormatTypeSize = 32;
    break;
  }
  case UR_IMAGE_CHANNEL_TYPE_HALF_FLOAT: {
    ZeImageFormatType = ZE_IMAGE_FORMAT_TYPE_FLOAT;
    ZeImageFormatTypeSize = 16;
    break;
  }
  case UR_IMAGE_CHANNEL_TYPE_UNSIGNED_INT32: {
    ZeImageFormatType = ZE_IMAGE_FORMAT_TYPE_UINT;
    ZeImageFormatTypeSize = 32;
    break;
  }
  case UR_IMAGE_CHANNEL_TYPE_UNSIGNED_INT16: {
    ZeImageFormatType = ZE_IMAGE_FORMAT_TYPE_UINT;
    ZeImageFormatTypeSize = 16;
    break;
  }
  case UR_IMAGE_CHANNEL_TYPE_UNSIGNED_INT8: {
    ZeImageFormatType = ZE_IMAGE_FORMAT_TYPE_UINT;
    ZeImageFormatTypeSize = 8;
    break;
  }
  case UR_IMAGE_CHANNEL_TYPE_UNORM_INT16: {
    ZeImageFormatType = ZE_IMAGE_FORMAT_TYPE_UNORM;
    ZeImageFormatTypeSize = 16;
    break;
  }
  case UR_IMAGE_CHANNEL_TYPE_UNORM_INT8: {
    ZeImageFormatType = ZE_IMAGE_FORMAT_TYPE_UNORM;
    ZeImageFormatTypeSize = 8;
    break;
  }
  case UR_IMAGE_CHANNEL_TYPE_SIGNED_INT32: {
    ZeImageFormatType = ZE_IMAGE_FORMAT_TYPE_SINT;
    ZeImageFormatTypeSize = 32;
    break;
  }
  case UR_IMAGE_CHANNEL_TYPE_SIGNED_INT16: {
    ZeImageFormatType = ZE_IMAGE_FORMAT_TYPE_SINT;
    ZeImageFormatTypeSize = 16;
    break;
  }
  case UR_IMAGE_CHANNEL_TYPE_SIGNED_INT8: {
    ZeImageFormatType = ZE_IMAGE_FORMAT_TYPE_SINT;
    ZeImageFormatTypeSize = 8;
    break;
  }
  case UR_IMAGE_CHANNEL_TYPE_SNORM_INT16: {
    ZeImageFormatType = ZE_IMAGE_FORMAT_TYPE_SNORM;
    ZeImageFormatTypeSize = 16;
    break;
  }
  case UR_IMAGE_CHANNEL_TYPE_SNORM_INT8: {
    ZeImageFormatType = ZE_IMAGE_FORMAT_TYPE_SNORM;
    ZeImageFormatTypeSize = 8;
    break;
  }
  default:
    urPrint("urMemImageCreate: unsupported image data type: data type = %d\n",
            ImageFormat->channelType);
    return UR_RESULT_ERROR_INVALID_VALUE;
  }

  // TODO: populate the layout mapping
  ze_image_format_layout_t ZeImageFormatLayout;
  switch (ImageFormat->channelOrder) {
  case UR_IMAGE_CHANNEL_ORDER_RGBA: {
    switch (ZeImageFormatTypeSize) {
    case 8:
      ZeImageFormatLayout = ZE_IMAGE_FORMAT_LAYOUT_8_8_8_8;
      break;
    case 16:
      ZeImageFormatLayout = ZE_IMAGE_FORMAT_LAYOUT_16_16_16_16;
      break;
    case 32:
      ZeImageFormatLayout = ZE_IMAGE_FORMAT_LAYOUT_32_32_32_32;
      break;
    default:
      urPrint("urMemImageCreate: unexpected data type Size\n");
      return UR_RESULT_ERROR_INVALID_VALUE;
    }
    break;
  }
  default:
    urPrint("format layout = %d\n", ImageFormat->channelOrder);
    die("urMemImageCreate: unsupported image format layout\n");
    break;
  }

  ze_image_format_t ZeFormatDesc = {
      ZeImageFormatLayout, ZeImageFormatType,
      // TODO: are swizzles deducted from image_format->image_channel_order?
      ZE_IMAGE_FORMAT_SWIZZLE_R, ZE_IMAGE_FORMAT_SWIZZLE_G,
      ZE_IMAGE_FORMAT_SWIZZLE_B, ZE_IMAGE_FORMAT_SWIZZLE_A};

  ze_image_type_t ZeImageType;
  switch (ImageDesc->type) {
  case UR_MEM_TYPE_IMAGE1D:
    ZeImageType = ZE_IMAGE_TYPE_1D;
    break;
  case UR_MEM_TYPE_IMAGE2D:
    ZeImageType = ZE_IMAGE_TYPE_2D;
    break;
  case UR_MEM_TYPE_IMAGE3D:
    ZeImageType = ZE_IMAGE_TYPE_3D;
    break;
  case UR_MEM_TYPE_IMAGE1D_ARRAY:
    ZeImageType = ZE_IMAGE_TYPE_1DARRAY;
    break;
  case UR_MEM_TYPE_IMAGE2D_ARRAY:
    ZeImageType = ZE_IMAGE_TYPE_2DARRAY;
    break;
  default:
    urPrint("urMemImageCreate: unsupported image type\n");
    return UR_RESULT_ERROR_INVALID_VALUE;
  }

  ZeImageDesc.arraylevels = ZeImageDesc.flags = 0;
  ZeImageDesc.type = ZeImageType;
  ZeImageDesc.format = ZeFormatDesc;
  ZeImageDesc.width = ur_cast<uint64_t>(ImageDesc->width);
  ZeImageDesc.height = ur_cast<uint64_t>(ImageDesc->height);
  ZeImageDesc.depth = ur_cast<uint64_t>(ImageDesc->depth);
  ZeImageDesc.arraylevels = ur_cast<uint32_t>(ImageDesc->arraySize);
  ZeImageDesc.miplevels = ImageDesc->numMipLevel;

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urMemImageCreate(
    ur_context_handle_t Context, ///< [in] handle of the context object
    ur_mem_flags_t Flags, ///< [in] allocation and usage information flags
    const ur_image_format_t
        *ImageFormat, ///< [in] pointer to image format specification
    const ur_image_desc_t *ImageDesc, ///< [in] pointer to image description
    void *Host,                       ///< [in] pointer to the buffer data
    ur_mem_handle_t *Mem ///< [out] pointer to handle of image object created
) {
  // TODO: implement read-only, write-only
  if ((Flags & UR_MEM_FLAG_READ_WRITE) == 0) {
    die("urMemImageCreate: Level-Zero implements only read-write buffer,"
        "no read-only or write-only yet.");
  }

  std::shared_lock<ur_shared_mutex> Lock(Context->Mutex);

  ZeStruct<ze_image_desc_t> ZeImageDesc;
  UR_CALL(ur2zeImageDesc(ImageFormat, ImageDesc, ZeImageDesc));

  // Currently we have the "0" device in context with mutliple root devices to
  // own the image.
  // TODO: Implement explicit copying for acessing the image from other devices
  // in the context.
  ur_device_handle_t Device = Context->SingleRootDevice
                                  ? Context->SingleRootDevice
                                  : Context->Devices[0];
  ze_image_handle_t ZeImage;
  ZE2UR_CALL(zeImageCreate,
             (Context->ZeContext, Device->ZeDevice, &ZeImageDesc, &ZeImage));

  try {
    auto UrImage = new _ur_image(Context, ZeImage, true /*OwnZeMemHandle*/);
    *Mem = reinterpret_cast<ur_mem_handle_t>(UrImage);

#ifndef NDEBUG
    UrImage->ZeImageDesc = ZeImageDesc;
#endif // !NDEBUG

    if ((Flags & UR_MEM_FLAG_USE_HOST_POINTER) != 0 ||
        (Flags & UR_MEM_FLAG_ALLOC_COPY_HOST_POINTER) != 0) {
      // Initialize image synchronously with immediate offload.
      // zeCommandListAppendImageCopyFromMemory must not be called from
      // simultaneous threads with the same command list handle, so we need
      // exclusive lock.
      std::scoped_lock<ur_mutex> Lock(Context->ImmediateCommandListMutex);
      ZE2UR_CALL(zeCommandListAppendImageCopyFromMemory,
                 (Context->ZeCommandListInit, ZeImage, Host, nullptr, nullptr,
                  0, nullptr));
    }
  } catch (const std::bad_alloc &) {
    return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urMemImageCreateWithNativeHandle(
    ur_native_handle_t NativeMem, ///< [in] the native handle to the memory.
    ur_context_handle_t Context,  ///< [in] handle of the context object.
    [[maybe_unused]] const ur_image_format_t
        *ImageFormat, ///< [in] pointer to image format specification.
    [[maybe_unused]] const ur_image_desc_t
        *ImageDesc, ///< [in] pointer to image description.
    const ur_mem_native_properties_t
        *Properties, ///< [in][optional] pointer to native memory creation
                     ///< properties.
    ur_mem_handle_t *Mem) {
  std::shared_lock<ur_shared_mutex> Lock(Context->Mutex);

  ze_image_handle_t ZeHImage = ur_cast<ze_image_handle_t>(NativeMem);

  _ur_image *Image = nullptr;
  try {
    Image = new _ur_image(Context, ZeHImage, Properties->isNativeHandleOwned);
    *Mem = reinterpret_cast<ur_mem_handle_t>(Image);

#ifndef NDEBUG
    ZeStruct<ze_image_desc_t> ZeImageDesc;
    ur_result_t Res = ur2zeImageDesc(ImageFormat, ImageDesc, ZeImageDesc);
    if (Res != UR_RESULT_SUCCESS) {
      delete Image;
      *Mem = nullptr;
      return Res;
    }
    Image->ZeImageDesc = ZeImageDesc;
#else
    std::ignore = ImageFormat;
    std::ignore = ImageDesc;
#endif // !NDEBUG

  } catch (const std::bad_alloc &) {
    return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urMemBufferCreate(
    ur_context_handle_t Context, ///< [in] handle of the context object
    ur_mem_flags_t Flags, ///< [in] allocation and usage information flags
    size_t Size, ///< [in] size in bytes of the memory object to be allocated
    const ur_buffer_properties_t *Properties,
    ur_mem_handle_t
        *RetBuffer ///< [out] pointer to handle of the memory buffer created
) {
  if (Flags & UR_MEM_FLAG_ALLOC_HOST_POINTER) {
    // Having PI_MEM_FLAGS_HOST_PTR_ALLOC for buffer requires allocation of
    // pinned host memory, see:
    // sycl/doc/extensions/supported/sycl_ext_oneapi_use_pinned_host_memory_property.asciidoc
    // We are however missing such functionality in Level Zero, so we just
    // ignore the flag for now.
    //
  }

  void *Host = nullptr;
  if (Properties) {
    Host = Properties->pHost;
  }

  // If USM Import feature is enabled and hostptr is supplied,
  // import the hostptr if not already imported into USM.
  // Data transfer rate is maximized when both source and destination
  // are USM pointers. Promotion of the host pointer to USM thus
  // optimizes data transfer performance.
  bool HostPtrImported = false;
  if (ZeUSMImport.Enabled && Host != nullptr &&
      (Flags & UR_MEM_FLAG_USE_HOST_POINTER) != 0) {
    // Query memory type of the host pointer
    ze_device_handle_t ZeDeviceHandle;
    ZeStruct<ze_memory_allocation_properties_t> ZeMemoryAllocationProperties;
    ZE2UR_CALL(zeMemGetAllocProperties,
               (Context->ZeContext, Host, &ZeMemoryAllocationProperties,
                &ZeDeviceHandle));

    // If not shared of any type, we can import the ptr
    if (ZeMemoryAllocationProperties.type == ZE_MEMORY_TYPE_UNKNOWN) {
      // Promote the host ptr to USM host memory
      ze_driver_handle_t driverHandle = Context->getPlatform()->ZeDriver;
      ZeUSMImport.doZeUSMImport(driverHandle, Host, Size);
      HostPtrImported = true;
    }
  }

  _ur_buffer *Buffer = nullptr;
  auto HostPtrOrNull = (Flags & UR_MEM_FLAG_USE_HOST_POINTER)
                           ? reinterpret_cast<char *>(Host)
                           : nullptr;
  try {
    Buffer = new _ur_buffer(Context, Size, HostPtrOrNull, HostPtrImported);
  } catch (const std::bad_alloc &) {
    return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }

  // Initialize the buffer with user data
  if (Host) {
    if ((Flags & UR_MEM_FLAG_USE_HOST_POINTER) != 0 ||
        (Flags & UR_MEM_FLAG_ALLOC_COPY_HOST_POINTER) != 0) {

      // We don't yet know which device needs this buffer, so make the first
      // device in the context be the master, and hold the initial valid
      // allocation.
      char *ZeHandleDst;
      UR_CALL(Buffer->getZeHandle(ZeHandleDst, ur_mem_handle_t_::write_only,
                                  Context->Devices[0]));
      if (Buffer->OnHost) {
        // Do a host to host copy.
        // For an imported HostPtr the copy is unneeded.
        if (!HostPtrImported)
          memcpy(ZeHandleDst, Host, Size);
      } else {
        // Initialize the buffer synchronously with immediate offload
        // zeCommandListAppendMemoryCopy must not be called from simultaneous
        // threads with the same command list handle, so we need exclusive lock.
        std::scoped_lock<ur_mutex> Lock(Context->ImmediateCommandListMutex);
        ZE2UR_CALL(zeCommandListAppendMemoryCopy,
                   (Context->ZeCommandListInit, ZeHandleDst, Host, Size,
                    nullptr, 0, nullptr));
      }
    } else if (Flags == 0 || (Flags == UR_MEM_FLAG_READ_WRITE)) {
      // Nothing more to do.
    } else
      die("urMemBufferCreate: not implemented");
  }

  *RetBuffer = reinterpret_cast<ur_mem_handle_t>(Buffer);
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urMemRetain(
    ur_mem_handle_t Mem ///< [in] handle of the memory object to get access
) {
  Mem->RefCount.increment();
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urMemRelease(
    ur_mem_handle_t Mem ///< [in] handle of the memory object to release
) {
  if (!Mem->RefCount.decrementAndTest())
    return UR_RESULT_SUCCESS;

  if (Mem->isImage()) {
    char *ZeHandleImage;
    auto Image = static_cast<_ur_image *>(Mem);
    if (Image->OwnNativeHandle) {
      UR_CALL(Mem->getZeHandle(ZeHandleImage, ur_mem_handle_t_::write_only));
      auto ZeResult = ZE_CALL_NOCHECK(
          zeImageDestroy, (ur_cast<ze_image_handle_t>(ZeHandleImage)));
      // Gracefully handle the case that L0 was already unloaded.
      if (ZeResult && ZeResult != ZE_RESULT_ERROR_UNINITIALIZED)
        return ze2urResult(ZeResult);
    }
  } else {
    auto Buffer = reinterpret_cast<_ur_buffer *>(Mem);
    Buffer->free();
  }
  delete Mem;

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urMemBufferPartition(
    ur_mem_handle_t
        Buffer,           ///< [in] handle of the buffer object to allocate from
    ur_mem_flags_t Flags, ///< [in] allocation and usage information flags
    ur_buffer_create_type_t BufferCreateType, ///< [in] buffer creation type
    const ur_buffer_region_t
        *BufferCreateInfo, ///< [in] pointer to buffer create region information
    ur_mem_handle_t
        *RetMem ///< [out] pointer to the handle of sub buffer created
) {
  std::ignore = BufferCreateType;
  UR_ASSERT(Buffer && !Buffer->isImage() &&
                !(static_cast<_ur_buffer *>(Buffer))->isSubBuffer(),
            UR_RESULT_ERROR_INVALID_MEM_OBJECT);

  std::shared_lock<ur_shared_mutex> Guard(Buffer->Mutex);

  if (Flags != UR_MEM_FLAG_READ_WRITE) {
    die("urMemBufferPartition: Level-Zero implements only read-write buffer,"
        "no read-only or write-only yet.");
  }

  try {
    auto partitionedBuffer =
        new _ur_buffer(static_cast<_ur_buffer *>(Buffer),
                       BufferCreateInfo->origin, BufferCreateInfo->size);
    *RetMem = reinterpret_cast<ur_mem_handle_t>(partitionedBuffer);
  } catch (const std::bad_alloc &) {
    return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urMemGetNativeHandle(
    ur_mem_handle_t Mem, ///< [in] handle of the mem.
    ur_native_handle_t
        *NativeMem ///< [out] a pointer to the native handle of the mem.
) {
  std::shared_lock<ur_shared_mutex> Guard(Mem->Mutex);
  char *ZeHandle = nullptr;
  UR_CALL(Mem->getZeHandle(ZeHandle, ur_mem_handle_t_::read_write));
  *NativeMem = ur_cast<ur_native_handle_t>(ZeHandle);

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urMemBufferCreateWithNativeHandle(
    ur_native_handle_t NativeMem, ///< [in] the native handle to the memory.
    ur_context_handle_t Context,  ///< [in] handle of the context object.
    const ur_mem_native_properties_t
        *Properties, ///< [in][optional] pointer to native memory creation
                     ///< properties.
    ur_mem_handle_t
        *Mem ///< [out] pointer to handle of buffer memory object created.
) {
  bool OwnNativeHandle = Properties->isNativeHandleOwned;

  std::shared_lock<ur_shared_mutex> Lock(Context->Mutex);

  // Get base of the allocation
  void *Base = nullptr;
  size_t Size = 0;
  void *Ptr = ur_cast<void *>(NativeMem);
  ZE2UR_CALL(zeMemGetAddressRange, (Context->ZeContext, Ptr, &Base, &Size));
  UR_ASSERT(Ptr == Base, UR_RESULT_ERROR_INVALID_VALUE);

  ZeStruct<ze_memory_allocation_properties_t> ZeMemProps;
  ze_device_handle_t ZeDevice = nullptr;
  ZE2UR_CALL(zeMemGetAllocProperties,
             (Context->ZeContext, Ptr, &ZeMemProps, &ZeDevice));

  // Check type of the allocation
  switch (ZeMemProps.type) {
  case ZE_MEMORY_TYPE_HOST:
  case ZE_MEMORY_TYPE_SHARED:
  case ZE_MEMORY_TYPE_DEVICE:
    break;
  case ZE_MEMORY_TYPE_UNKNOWN:
    // Memory allocation is unrelated to the context
    return UR_RESULT_ERROR_INVALID_CONTEXT;
  default:
    die("Unexpected memory type");
  }

  ur_device_handle_t Device{};
  if (ZeDevice) {
    Device = Context->getPlatform()->getDeviceFromNativeHandle(ZeDevice);
    UR_ASSERT(Context->isValidDevice(Device), UR_RESULT_ERROR_INVALID_CONTEXT);
  }

  _ur_buffer *Buffer = nullptr;
  try {
    Buffer = new _ur_buffer(Context, Size, Device, ur_cast<char *>(NativeMem),
                            OwnNativeHandle);
    *Mem = reinterpret_cast<ur_mem_handle_t>(Buffer);
  } catch (const std::bad_alloc &) {
    return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }

  ur_platform_handle_t Plt = Context->getPlatform();
  std::unique_lock<ur_shared_mutex> ContextsLock(Plt->ContextsMutex,
                                                 std::defer_lock);
  // If we don't own the native handle then we can't control deallocation of
  // that memory so there is no point of keeping track of the memory
  // allocation for deferred memory release in the mode when indirect access
  // tracking is enabled.
  if (IndirectAccessTrackingEnabled && OwnNativeHandle) {
    // We need to keep track of all memory allocations in the context
    ContextsLock.lock();
    // Retain context to be sure that it is released after all memory
    // allocations in this context are released.
    UR_CALL(urContextRetain(Context));

    Context->MemAllocs.emplace(std::piecewise_construct,
                               std::forward_as_tuple(Ptr),
                               std::forward_as_tuple(Context, OwnNativeHandle));
  }

  if (Device) {
    // If this allocation is on a device, then we re-use it for the buffer.
    // Nothing to do.
  } else if (Buffer->OnHost) {
    // If this is host allocation and buffer always stays on host there
    // nothing more to do.
  } else {
    // In all other cases (shared allocation, or host allocation that cannot
    // represent the buffer in this context) copy the data to a newly
    // created device allocation.
    char *ZeHandleDst;
    UR_CALL(
        Buffer->getZeHandle(ZeHandleDst, ur_mem_handle_t_::write_only, Device));

    // zeCommandListAppendMemoryCopy must not be called from simultaneous
    // threads with the same command list handle, so we need exclusive lock.
    std::scoped_lock<ur_mutex> Lock(Context->ImmediateCommandListMutex);
    ZE2UR_CALL(zeCommandListAppendMemoryCopy,
               (Context->ZeCommandListInit, ZeHandleDst, Ptr, Size, nullptr, 0,
                nullptr));
  }

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urMemGetInfo(
    ur_mem_handle_t Memory, ///< [in] handle to the memory object being queried.
    ur_mem_info_t MemInfoType, ///< [in] type of the info to retrieve.
    size_t PropSize, ///< [in] the number of bytes of memory pointed to by
                     ///< pMemInfo.
    void *MemInfo,   ///< [out][optional] array of bytes holding the info.
                     ///< If propSize is less than the real number of bytes
                     ///< needed to return the info then the
                     ///< ::UR_RESULT_ERROR_INVALID_SIZE error is returned and
                     ///< pMemInfo is not used.
    size_t *PropSizeRet ///< [out][optional] pointer to the actual size in
                        ///< bytes of data queried by pMemInfo.
) {
  UR_ASSERT(MemInfoType == UR_MEM_INFO_CONTEXT || !Memory->isImage(),
            UR_RESULT_ERROR_INVALID_VALUE);

  auto Buffer = reinterpret_cast<_ur_buffer *>(Memory);
  std::shared_lock<ur_shared_mutex> Lock(Buffer->Mutex);
  UrReturnHelper ReturnValue(PropSize, MemInfo, PropSizeRet);

  switch (MemInfoType) {
  case UR_MEM_INFO_CONTEXT: {
    return ReturnValue(Buffer->UrContext);
  }
  case UR_MEM_INFO_SIZE: {
    // Get size of the allocation
    return ReturnValue(size_t{Buffer->Size});
  }
  default: {
    die("urMemGetInfo: Parameter is not implemented");
  }
  }

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urMemImageGetInfo(
    ur_mem_handle_t Memory, ///< [in] handle to the image object being queried.
    ur_image_info_t ImgInfoType, ///< [in] type of image info to retrieve.
    size_t PropSize, ///< [in] the number of bytes of memory pointer to by
                     ///< pImgInfo.
    void *ImgInfo,   ///< [out][optional] array of bytes holding the info.
                     ///< If propSize is less than the real number of bytes
                     ///< needed to return the info then the
                     ///< ::UR_RESULT_ERROR_INVALID_SIZE error is returned and
                     ///< pImgInfo is not used.
    size_t *PropSizeRet ///< [out][optional] pointer to the actual size in
                        ///< bytes of data queried by pImgInfo.
) {
  std::ignore = Memory;
  std::ignore = ImgInfoType;
  std::ignore = PropSize;
  std::ignore = ImgInfo;
  std::ignore = PropSizeRet;
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

// If indirect access tracking is enabled then performs reference counting,
// otherwise just calls zeMemAllocDevice.
static ur_result_t ZeDeviceMemAllocHelper(void **ResultPtr,
                                          ur_context_handle_t Context,
                                          ur_device_handle_t Device,
                                          size_t Size) {
  ur_platform_handle_t Plt = Device->Platform;
  std::unique_lock<ur_shared_mutex> ContextsLock(Plt->ContextsMutex,
                                                 std::defer_lock);
  if (IndirectAccessTrackingEnabled) {
    // Lock the mutex which is guarding contexts container in the platform.
    // This prevents new kernels from being submitted in any context while
    // we are in the process of allocating a memory, this is needed to
    // properly capture allocations by kernels with indirect access.
    ContextsLock.lock();
    // We are going to defer memory release if there are kernels with
    // indirect access, that is why explicitly retain context to be sure
    // that it is released after all memory allocations in this context are
    // released.
    UR_CALL(urContextRetain(Context));
  }

  ze_device_mem_alloc_desc_t ZeDesc = {};
  ZeDesc.flags = 0;
  ZeDesc.ordinal = 0;
  ZE2UR_CALL(zeMemAllocDevice, (Context->ZeContext, &ZeDesc, Size, 1,
                                Device->ZeDevice, ResultPtr));

  if (IndirectAccessTrackingEnabled) {
    // Keep track of all memory allocations in the context
    Context->MemAllocs.emplace(std::piecewise_construct,
                               std::forward_as_tuple(*ResultPtr),
                               std::forward_as_tuple(Context));
  }
  return UR_RESULT_SUCCESS;
}

ur_result_t _ur_buffer::getZeHandle(char *&ZeHandle, access_mode_t AccessMode,
                                    ur_device_handle_t Device) {

  // NOTE: There might be no valid allocation at all yet and we get
  // here from piEnqueueKernelLaunch that would be doing the buffer
  // initialization. In this case the Device is not null as kernel
  // launch is always on a specific device.
  if (!Device)
    Device = LastDeviceWithValidAllocation;
  // If the device is still not selected then use the first one in
  // the context of the buffer.
  if (!Device)
    Device = UrContext->Devices[0];

  auto &Allocation = Allocations[Device];

  // Sub-buffers don't maintain own allocations but rely on parent buffer.
  if (isSubBuffer()) {
    UR_CALL(SubBuffer.Parent->getZeHandle(ZeHandle, AccessMode, Device));
    ZeHandle += SubBuffer.Origin;
    // Still store the allocation info in the PI sub-buffer for
    // getZeHandlePtr to work. At least zeKernelSetArgumentValue needs to
    // be given a pointer to the allocation handle rather than its value.
    //
    Allocation.ZeHandle = ZeHandle;
    Allocation.ReleaseAction = allocation_t::keep;
    LastDeviceWithValidAllocation = Device;
    return UR_RESULT_SUCCESS;
  }

  // First handle case where the buffer is represented by only
  // a single host allocation.
  if (OnHost) {
    auto &HostAllocation = Allocations[nullptr];
    // The host allocation may already exists, e.g. with imported
    // host ptr, or in case of interop buffer.
    if (!HostAllocation.ZeHandle) {
      if (DisjointPoolConfigInstance.EnableBuffers) {
        HostAllocation.ReleaseAction = allocation_t::free;
        ur_usm_desc_t USMDesc{};
        USMDesc.align = getAlignment();
        ur_usm_pool_handle_t Pool{};
        UR_CALL(urUSMHostAlloc(UrContext, &USMDesc, Pool, Size,
                               reinterpret_cast<void **>(&ZeHandle)));
      } else {
        HostAllocation.ReleaseAction = allocation_t::free_native;
        UR_CALL(ZeHostMemAllocHelper(reinterpret_cast<void **>(&ZeHandle),
                                     UrContext, Size));
      }
      HostAllocation.ZeHandle = ZeHandle;
      HostAllocation.Valid = true;
    }
    Allocation = HostAllocation;
    Allocation.ReleaseAction = allocation_t::keep;
    ZeHandle = Allocation.ZeHandle;
    LastDeviceWithValidAllocation = Device;
    return UR_RESULT_SUCCESS;
  }
  // Reads user setting on how to deal with buffers in contexts where
  // all devices have the same root-device. Returns "true" if the
  // preference is to have allocate on each [sub-]device and migrate
  // normally (copy) to other sub-devices as needed. Returns "false"
  // if the preference is to have single root-device allocations
  // serve the needs of all [sub-]devices, meaning potentially more
  // cross-tile traffic.
  //
  static const bool SingleRootDeviceBufferMigration = [] {
    const char *UrRet =
        std::getenv("UR_L0_SINGLE_ROOT_DEVICE_BUFFER_MIGRATION");
    const char *PiRet =
        std::getenv("SYCL_PI_LEVEL_ZERO_SINGLE_ROOT_DEVICE_BUFFER_MIGRATION");
    const char *EnvStr = UrRet ? UrRet : (PiRet ? PiRet : nullptr);
    if (EnvStr)
      return (std::stoi(EnvStr) != 0);
    // The default is to migrate normally, which may not always be the
    // best option (depends on buffer access patterns), but is an
    // overall win on the set of the available benchmarks.
    return true;
  }();

  // Peform actual device allocation as needed.
  if (!Allocation.ZeHandle) {
    if (!SingleRootDeviceBufferMigration && UrContext->SingleRootDevice &&
        UrContext->SingleRootDevice != Device) {
      // If all devices in the context are sub-devices of the same device
      // then we reuse root-device allocation by all sub-devices in the
      // context.
      // TODO: we can probably generalize this and share root-device
      //       allocations by its own sub-devices even if not all other
      //       devices in the context have the same root.
      UR_CALL(getZeHandle(ZeHandle, AccessMode, UrContext->SingleRootDevice));
      Allocation.ReleaseAction = allocation_t::keep;
      Allocation.ZeHandle = ZeHandle;
      Allocation.Valid = true;
      return UR_RESULT_SUCCESS;
    } else { // Create device allocation
      if (DisjointPoolConfigInstance.EnableBuffers) {
        Allocation.ReleaseAction = allocation_t::free;
        ur_usm_desc_t USMDesc{};
        USMDesc.align = getAlignment();
        ur_usm_pool_handle_t Pool{};
        UR_CALL(urUSMDeviceAlloc(UrContext, Device, &USMDesc, Pool, Size,
                                 reinterpret_cast<void **>(&ZeHandle)));
      } else {
        Allocation.ReleaseAction = allocation_t::free_native;
        UR_CALL(ZeDeviceMemAllocHelper(reinterpret_cast<void **>(&ZeHandle),
                                       UrContext, Device, Size));
      }
    }
    Allocation.ZeHandle = ZeHandle;
  } else {
    ZeHandle = Allocation.ZeHandle;
  }

  // If some prior access invalidated this allocation then make it valid again.
  if (!Allocation.Valid) {
    // LastDeviceWithValidAllocation should always have valid allocation.
    if (Device == LastDeviceWithValidAllocation)
      die("getZeHandle: last used allocation is not valid");

    // For write-only access the allocation contents is not going to be used.
    // So don't do anything to make it "valid".
    bool NeedCopy = AccessMode != ur_mem_handle_t_::write_only;
    // It's also possible that the buffer doesn't have a valid allocation
    // yet presumably when it is passed to a kernel that will perform
    // it's intialization.
    if (NeedCopy && !LastDeviceWithValidAllocation) {
      NeedCopy = false;
    }
    char *ZeHandleSrc = nullptr;
    if (NeedCopy) {
      UR_CALL(getZeHandle(ZeHandleSrc, ur_mem_handle_t_::read_only,
                          LastDeviceWithValidAllocation));
      // It's possible with the single root-device contexts that
      // the buffer is represented by the single root-device
      // allocation and then skip the copy to itself.
      if (ZeHandleSrc == ZeHandle)
        NeedCopy = false;
    }

    if (NeedCopy) {
      // Copy valid buffer data to this allocation.
      // TODO: see if we should better use peer's device allocation used
      // directly, if that capability is reported with zeDeviceCanAccessPeer,
      // instead of maintaining a separate allocation and performing
      // explciit copies.
      //
      // zeCommandListAppendMemoryCopy must not be called from simultaneous
      // threads with the same command list handle, so we need exclusive lock.
      ze_bool_t P2P = false;
      ZE2UR_CALL(
          zeDeviceCanAccessPeer,
          (Device->ZeDevice, LastDeviceWithValidAllocation->ZeDevice, &P2P));
      if (!P2P) {
        // P2P copy is not possible, so copy through the host.
        auto &HostAllocation = Allocations[nullptr];
        // The host allocation may already exists, e.g. with imported
        // host ptr, or in case of interop buffer.
        if (!HostAllocation.ZeHandle) {
          void *ZeHandleHost;
          if (DisjointPoolConfigInstance.EnableBuffers) {
            HostAllocation.ReleaseAction = allocation_t::free;
            ur_usm_desc_t USMDesc{};
            USMDesc.align = getAlignment();
            ur_usm_pool_handle_t Pool{};
            UR_CALL(
                urUSMHostAlloc(UrContext, &USMDesc, Pool, Size, &ZeHandleHost));
          } else {
            HostAllocation.ReleaseAction = allocation_t::free_native;
            UR_CALL(ZeHostMemAllocHelper(&ZeHandleHost, UrContext, Size));
          }
          HostAllocation.ZeHandle = reinterpret_cast<char *>(ZeHandleHost);
          HostAllocation.Valid = false;
        }
        std::scoped_lock<ur_mutex> Lock(UrContext->ImmediateCommandListMutex);
        if (!HostAllocation.Valid) {
          ZE2UR_CALL(zeCommandListAppendMemoryCopy,
                     (UrContext->ZeCommandListInit, HostAllocation.ZeHandle,
                      ZeHandleSrc, Size, nullptr, 0, nullptr));
          // Mark the host allocation data  as valid so it can be reused.
          // It will be invalidated below if the current access is not
          // read-only.
          HostAllocation.Valid = true;
        }
        ZE2UR_CALL(zeCommandListAppendMemoryCopy,
                   (UrContext->ZeCommandListInit, ZeHandle,
                    HostAllocation.ZeHandle, Size, nullptr, 0, nullptr));
      } else {
        // Perform P2P copy.
        std::scoped_lock<ur_mutex> Lock(UrContext->ImmediateCommandListMutex);
        ZE2UR_CALL(zeCommandListAppendMemoryCopy,
                   (UrContext->ZeCommandListInit, ZeHandle, ZeHandleSrc, Size,
                    nullptr, 0, nullptr));
      }
    }
    Allocation.Valid = true;
    LastDeviceWithValidAllocation = Device;
  }

  // Invalidate other allocations that would become not valid if
  // this access is not read-only.
  if (AccessMode != ur_mem_handle_t_::read_only) {
    for (auto &Alloc : Allocations) {
      if (Alloc.first != LastDeviceWithValidAllocation)
        Alloc.second.Valid = false;
    }
  }

  urPrint("getZeHandle(pi_device{%p}) = %p\n", (void *)Device,
          (void *)Allocation.ZeHandle);
  return UR_RESULT_SUCCESS;
}

ur_result_t _ur_buffer::free() {
  for (auto &Alloc : Allocations) {
    auto &ZeHandle = Alloc.second.ZeHandle;
    // It is possible that the real allocation wasn't made if the buffer
    // wasn't really used in this location.
    if (!ZeHandle)
      continue;

    switch (Alloc.second.ReleaseAction) {
    case allocation_t::keep:
      break;
    case allocation_t::free: {
      ur_platform_handle_t Plt = UrContext->getPlatform();
      std::scoped_lock<ur_shared_mutex> Lock(IndirectAccessTrackingEnabled
                                                 ? Plt->ContextsMutex
                                                 : UrContext->Mutex);

      UR_CALL(USMFreeHelper(reinterpret_cast<ur_context_handle_t>(UrContext),
                            ZeHandle));
      break;
    }
    case allocation_t::free_native:
      UR_CALL(ZeMemFreeHelper(UrContext, ZeHandle));
      break;
    case allocation_t::unimport:
      ZeUSMImport.doZeUSMRelease(UrContext->getPlatform()->ZeDriver, ZeHandle);
      break;
    default:
      die("_ur_buffer::free(): Unhandled release action");
    }
    ZeHandle = nullptr; // don't leave hanging pointers
  }
  return UR_RESULT_SUCCESS;
}

// Buffer constructor
_ur_buffer::_ur_buffer(ur_context_handle_t Context, size_t Size, char *HostPtr,
                       bool ImportedHostPtr = false)
    : ur_mem_handle_t_(Context), Size(Size), SubBuffer{nullptr, 0} {

  // We treat integrated devices (physical memory shared with the CPU)
  // differently from discrete devices (those with distinct memories).
  // For integrated devices, allocating the buffer in the host memory
  // enables automatic access from the device, and makes copying
  // unnecessary in the map/unmap operations. This improves performance.
  OnHost = Context->Devices.size() == 1 &&
           Context->Devices[0]->ZeDeviceProperties->flags &
               ZE_DEVICE_PROPERTY_FLAG_INTEGRATED;

  // Fill the host allocation data.
  if (HostPtr) {
    MapHostPtr = HostPtr;
    // If this host ptr is imported to USM then use this as a host
    // allocation for this buffer.
    if (ImportedHostPtr) {
      Allocations[nullptr].ZeHandle = HostPtr;
      Allocations[nullptr].Valid = true;
      Allocations[nullptr].ReleaseAction = _ur_buffer::allocation_t::unimport;
    }
  }

  // This initialization does not end up with any valid allocation yet.
  LastDeviceWithValidAllocation = nullptr;
}

_ur_buffer::_ur_buffer(ur_context_handle_t Context, ur_device_handle_t Device,
                       size_t Size)
    : ur_mem_handle_t_(Context, Device), Size(Size) {}

// Interop-buffer constructor
_ur_buffer::_ur_buffer(ur_context_handle_t Context, size_t Size,
                       ur_device_handle_t Device, char *ZeMemHandle,
                       bool OwnZeMemHandle)
    : ur_mem_handle_t_(Context, Device), Size(Size), SubBuffer{nullptr, 0} {

  // Device == nullptr means host allocation
  Allocations[Device].ZeHandle = ZeMemHandle;
  Allocations[Device].Valid = true;
  Allocations[Device].ReleaseAction =
      OwnZeMemHandle ? allocation_t::free_native : allocation_t::keep;

  // Check if this buffer can always stay on host
  OnHost = false;
  if (!Device) { // Host allocation
    if (Context->Devices.size() == 1 &&
        Context->Devices[0]->ZeDeviceProperties->flags &
            ZE_DEVICE_PROPERTY_FLAG_INTEGRATED) {
      OnHost = true;
      MapHostPtr = ZeMemHandle; // map to this allocation
    }
  }
  LastDeviceWithValidAllocation = Device;
}

ur_result_t _ur_buffer::getZeHandlePtr(char **&ZeHandlePtr,
                                       access_mode_t AccessMode,
                                       ur_device_handle_t Device) {
  char *ZeHandle;
  UR_CALL(getZeHandle(ZeHandle, AccessMode, Device));
  ZeHandlePtr = &Allocations[Device].ZeHandle;
  return UR_RESULT_SUCCESS;
}

size_t _ur_buffer::getAlignment() const {
  // Choose an alignment that is at most 64 and is the next power of 2
  // for sizes less than 64.
  auto Alignment = Size;
  if (Alignment > 32UL)
    Alignment = 64UL;
  else if (Alignment > 16UL)
    Alignment = 32UL;
  else if (Alignment > 8UL)
    Alignment = 16UL;
  else if (Alignment > 4UL)
    Alignment = 8UL;
  else if (Alignment > 2UL)
    Alignment = 4UL;
  else if (Alignment > 1UL)
    Alignment = 2UL;
  else
    Alignment = 1UL;
  return Alignment;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueUSMFill(
    ur_queue_handle_t Queue, ///< [in] handle of the queue object
    void *Ptr,               ///< [in] pointer to USM memory object
    size_t PatternSize,  ///< [in] the size in bytes of the pattern. Must be a
                         ///< power of 2 and less than or equal to width.
    const void *Pattern, ///< [in] pointer with the bytes of the pattern to set.
    size_t Size, ///< [in] size in bytes to be set. Must be a multiple of
                 ///< patternSize.
    uint32_t NumEventsInWaitList, ///< [in] size of the event wait list
    const ur_event_handle_t *
        EventWaitList, ///< [in][optional][range(0, numEventsInWaitList)]
                       ///< pointer to a list of events that must be complete
                       ///< before this command can be executed. If nullptr, the
                       ///< numEventsInWaitList must be 0, indicating that this
                       ///< command does not wait on any event to complete.
    ur_event_handle_t *Event ///< [out][optional] return an event object that
                             ///< identifies this particular command instance.
) {
  std::scoped_lock<ur_shared_mutex> Lock(Queue->Mutex);

  return enqueueMemFillHelper(
      // TODO: do we need a new command type for USM memset?
      UR_COMMAND_MEM_BUFFER_FILL, Queue, Ptr,
      Pattern,     // It will be interpreted as an 8-bit value,
      PatternSize, // which is indicated with this pattern_size==1
      Size, NumEventsInWaitList, EventWaitList, Event);
}

/// Host Pipes
UR_APIEXPORT ur_result_t UR_APICALL urEnqueueReadHostPipe(
    ur_queue_handle_t hQueue, ur_program_handle_t hProgram,
    const char *pipe_symbol, bool blocking, void *pDst, size_t size,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  std::ignore = hQueue;
  std::ignore = hProgram;
  std::ignore = pipe_symbol;
  std::ignore = blocking;
  std::ignore = pDst;
  std::ignore = size;
  std::ignore = numEventsInWaitList;
  std::ignore = phEventWaitList;
  std::ignore = phEvent;
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueWriteHostPipe(
    ur_queue_handle_t hQueue, ur_program_handle_t hProgram,
    const char *pipe_symbol, bool blocking, void *pSrc, size_t size,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  std::ignore = hQueue;
  std::ignore = hProgram;
  std::ignore = pipe_symbol;
  std::ignore = blocking;
  std::ignore = pSrc;
  std::ignore = size;
  std::ignore = numEventsInWaitList;
  std::ignore = phEventWaitList;
  std::ignore = phEvent;
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}
