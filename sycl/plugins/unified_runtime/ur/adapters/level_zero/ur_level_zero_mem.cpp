//===--------- ur_level_zero_mem.cpp - Level Zero Adapter -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//

#include "ur_level_zero_mem.hpp"

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueMemBufferRead(
    ur_queue_handle_t Queue, ///< [in] handle of the queue object
    ur_mem_handle_t Buffer,  ///< [in] handle of the buffer object
    bool BlockingRead, ///< [in] indicates blocking (true), non-blocking (false)
    size_t Offset,     ///< [in] offset in bytes in the buffer object
    size_t Size,       ///< [in] size in bytes of data being read
    void *Dst, ///< [in] pointer to host memory where data is to be read into
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
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueMemBufferWrite(
    ur_queue_handle_t Queue, ///< [in] handle of the queue object
    ur_mem_handle_t Buffer,  ///< [in] handle of the buffer object
    bool
        BlockingWrite, ///< [in] indicates blocking (true), non-blocking (false)
    size_t Offset,     ///< [in] offset in bytes in the buffer object
    size_t Size,       ///< [in] size in bytes of data being written
    const void
        *Src, ///< [in] pointer to host memory where data is to be written from
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
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueMemBufferReadRect(
    ur_queue_handle_t Queue, ///< [in] handle of the queue object
    ur_mem_handle_t Buffer,  ///< [in] handle of the buffer object
    bool BlockingRead, ///< [in] indicates blocking (true), non-blocking (false)
    ur_rect_offset_t BufferOffset, ///< [in] 3D offset in the buffer
    ur_rect_offset_t HostOffset,   ///< [in] 3D offset in the host region
    ur_rect_region_t
        Region, ///< [in] 3D rectangular region descriptor: width, height, depth
    size_t BufferRowPitch,   ///< [in] length of each row in bytes in the buffer
                             ///< object
    size_t BufferSlicePitch, ///< [in] length of each 2D slice in bytes in the
                             ///< buffer object being read
    size_t HostRowPitch,     ///< [in] length of each row in bytes in the host
                             ///< memory region pointed by dst
    size_t HostSlicePitch,   ///< [in] length of each 2D slice in bytes in the
                             ///< host memory region pointed by dst
    void *Dst, ///< [in] pointer to host memory where data is to be read into
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
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueMemBufferWriteRect(
    ur_queue_handle_t Queue, ///< [in] handle of the queue object
    ur_mem_handle_t Buffer,  ///< [in] handle of the buffer object
    bool
        BlockingWrite, ///< [in] indicates blocking (true), non-blocking (false)
    ur_rect_offset_t BufferOffset, ///< [in] 3D offset in the buffer
    ur_rect_offset_t HostOffset,   ///< [in] 3D offset in the host region
    ur_rect_region_t
        Region, ///< [in] 3D rectangular region descriptor: width, height, depth
    size_t BufferRowPitch,   ///< [in] length of each row in bytes in the buffer
                             ///< object
    size_t BufferSlicePitch, ///< [in] length of each 2D slice in bytes in the
                             ///< buffer object being written
    size_t HostRowPitch,     ///< [in] length of each row in bytes in the host
                             ///< memory region pointed by src
    size_t HostSlicePitch,   ///< [in] length of each 2D slice in bytes in the
                             ///< host memory region pointed by src
    void *Src, ///< [in] pointer to host memory where data is to be written from
    uint32_t NumEventsInWaitList, ///< [in] size of the event wait list
    const ur_event_handle_t
        *EventWaitList, ///< [in][optional][range(0, numEventsInWaitList)]
                        ///< points to a list of events that must be complete
                        ///< before this command can be executed. If nullptr,
                        ///< the numEventsInWaitList must be 0, indicating
                        ///< that this command does not wait on any event to
                        ///< complete.
    ur_event_handle_t *Event ///< [in,out][optional] return an event object that
                             ///< identifies this particular command instance.
) {
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
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
    ur_event_handle_t *Event ///< [in,out][optional] return an event object that
                             ///< identifies this particular command instance.
) {
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
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
    ur_event_handle_t *Event ///< [in,out][optional] return an event object that
                             ///< identifies this particular command instance.
) {
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
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
    ur_event_handle_t *Event ///< [in,out][optional] return an event object that
                             ///< identifies this particular command instance.
) {
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
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
    ur_event_handle_t *Event ///< [in,out][optional] return an event object that
                             ///< identifies this particular command instance.
) {
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
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
    size_t InputRowPitch,    ///< [in] length of each row in bytes
    size_t InputSlicePitch,  ///< [in] length of each 2D slice of the 3D image
    void *Src, ///< [in] pointer to host memory where image is to be read into
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
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
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
    ur_event_handle_t *Event ///< [in,out][optional] return an event object that
                             ///< identifies this particular command instance.
) {
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueMemBufferMap(
    ur_queue_handle_t Queue, ///< [in] handle of the queue object
    ur_mem_handle_t Buffer,  ///< [in] handle of the buffer object
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
        *Event,   ///< [in,out][optional] return an event object that identifies
                  ///< this particular command instance.
    void **RetMap ///< [in,out] return mapped pointer.  TODO: move it before
                  ///< numEventsInWaitList?
) {
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
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
    ur_event_handle_t *Event ///< [in,out][optional] return an event object that
                             ///< identifies this particular command instance.
) {
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
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
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueUSMMemcpy(
    ur_queue_handle_t Queue, ///< [in] handle of the queue object
    bool Blocking,           ///< [in] blocking or non-blocking copy
    void *Dst,       ///< [in] pointer to the destination USM memory object
    const void *Src, ///< [in] pointer to the source USM memory object
    size_t size,     ///< [in] size in bytes to be copied
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
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueUSMPrefetch(
    ur_queue_handle_t Queue,        ///< [in] handle of the queue object
    const void *Mem,                ///< [in] pointer to the USM memory object
    size_t size,                    ///< [in] size in bytes to be fetched
    ur_usm_migration_flags_t Flags, ///< [in] USM prefetch flags
    uint32_t NumEventsInWaitList,   ///< [in] size of the event wait list
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
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueUSMMemAdvise(
    ur_queue_handle_t Queue, ///< [in] handle of the queue object
    const void *Mem,         ///< [in] pointer to the USM memory object
    size_t Size,             ///< [in] size in bytes to be advised
    ur_mem_advice_t Advice,  ///< [in] USM memory advice
    ur_event_handle_t *Event ///< [in,out][optional] return an event object that
                             ///< identifies this particular command instance.
) {
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
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
        *Event ///< [in,out][optional] return an event object that identifies
               ///< this particular kernel execution instance.
) {
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueUSMMemset2D(
    ur_queue_handle_t Queue, ///< [in] handle of the queue to submit to.
    void *pMem,              ///< [in] pointer to memory to be filled.
    size_t pitch,  ///< [in] the total width of the destination memory including
                   ///< padding.
    int value,     ///< [in] the value to fill into the region in pMem.
    size_t width,  ///< [in] the width in bytes of each row to set.
    size_t height, ///< [in] the height of the columns to set.
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
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
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
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urMemBufferCreate(
    ur_context_handle_t Context, ///< [in] handle of the context object
    ur_mem_flags_t Flags, ///< [in] allocation and usage information flags
    size_t Size, ///< [in] size in bytes of the memory object to be allocated
    void *Host,  ///< [in][optional] pointer to the buffer data
    ur_mem_handle_t
        *Buffer ///< [out] pointer to handle of the memory buffer created
) {
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urMemRetain(
    ur_mem_handle_t Mem ///< [in] handle of the memory object to get access
) {
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urMemRelease(
    ur_mem_handle_t Mem ///< [in] handle of the memory object to release
) {
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urMemBufferPartition(
    ur_mem_handle_t
        Buffer,           ///< [in] handle of the buffer object to allocate from
    ur_mem_flags_t Flags, ///< [in] allocation and usage information flags
    ur_buffer_create_type_t BufferCreateType, ///< [in] buffer creation type
    ur_buffer_region_t
        *BufferCreateInfo, ///< [in] pointer to buffer create region information
    ur_mem_handle_t *Mem ///< [out] pointer to the handle of sub buffer created
) {
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urMemGetNativeHandle(
    ur_mem_handle_t Mem, ///< [in] handle of the mem.
    ur_native_handle_t
        *NativeMem ///< [out] a pointer to the native handle of the mem.
) {
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urMemCreateWithNativeHandle(
    ur_native_handle_t NativeMem, ///< [in] the native handle of the mem.
    ur_context_handle_t Context,  ///< [in] handle of the context object
    ur_mem_handle_t
        *Mem ///< [out] pointer to the handle of the mem object created.
) {
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
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
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
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
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urUSMHostAlloc(
    ur_context_handle_t Context, ///< [in] handle of the context object
    ur_usm_desc_t *USMDesc, ///< [in][optional] USM memory allocation descriptor
    ur_usm_pool_handle_t Pool, ///< [in][optional] Pointer to a pool created
                               ///< using urUSMPoolCreate
    size_t
        Size, ///< [in] size in bytes of the USM memory object to be allocated
    uint32_t Align, ///< [in] alignment of the USM memory object
    void **RetMem   ///< [out] pointer to USM host memory object
) {
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urUSMDeviceAlloc(
    ur_context_handle_t Context, ///< [in] handle of the context object
    ur_device_handle_t Device,   ///< [in] handle of the device object
    ur_usm_desc_t *USMDesc, ///< [in][optional] USM memory allocation descriptor
    ur_usm_pool_handle_t Pool, ///< [in][optional] Pointer to a pool created
                               ///< using urUSMPoolCreate
    size_t
        Size, ///< [in] size in bytes of the USM memory object to be allocated
    uint32_t Alignment, ///< [in] alignment of the USM memory object
    void **RetMem       ///< [out] pointer to USM device memory object
) {
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urUSMSharedAlloc(
    ur_context_handle_t Context, ///< [in] handle of the context object
    ur_device_handle_t Device,   ///< [in] handle of the device object
    ur_usm_desc_t *USMDesc, ///< [in][optional] USM memory allocation descriptor
    ur_usm_pool_handle_t Pool, ///< [in][optional] Pointer to a pool created
                               ///< using urUSMPoolCreate
    size_t
        Size, ///< [in] size in bytes of the USM memory object to be allocated
    uint32_t Alignment, ///< [in] alignment of the USM memory object
    void **RetMem       ///< [out] pointer to USM shared memory object
) {
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urUSMFree(
    ur_context_handle_t Context, ///< [in] handle of the context object
    void *Mem                    ///< [in] pointer to USM memory object
) {
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urUSMGetMemAllocInfo(
    ur_context_handle_t Context, ///< [in] handle of the context object
    const void *Mem,             ///< [in] pointer to USM memory object
    ur_usm_alloc_info_t
        PropName, ///< [in] the name of the USM allocation property to query
    size_t PropValueSize, ///< [in] size in bytes of the USM allocation property
                          ///< value
    void *PropValue, ///< [out][optional] value of the USM allocation property
    size_t *PropValueSizeRet ///< [out][optional] bytes returned in USM
                             ///< allocation property
) {
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urUSMPoolCreate(
    ur_context_handle_t Context, ///< [in] handle of the context object
    ur_usm_pool_desc_t
        *PoolDesc, ///< [in] pointer to USM pool descriptor. Can be chained with
                   ///< ::ur_usm_pool_limits_desc_t
    ur_usm_pool_handle_t *Pool ///< [out] pointer to USM memory pool
) {
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urUSMPoolDestroy(
    ur_context_handle_t Context, ///< [in] handle of the context object
    ur_usm_pool_handle_t Pool    ///< [in] pointer to USM memory pool
) {
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
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
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}