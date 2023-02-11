//===--------- ur_level_zero_event.cpp - Level Zero Adapter -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//

#include "ur_level_zero_event.hpp"

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueEventsWait(
    ur_queue_handle_t Queue,      ///< [in] handle of the queue object
    uint32_t NumEventsInWaitList, ///< [in] size of the event wait list
    const ur_event_handle_t
        *EventWaitList, ///< [in][optional][range(0, numEventsInWaitList)]
                        ///< pointer to a list of events that must be complete
                        ///< before this command can be executed. If nullptr,
                        ///< the numEventsInWaitList must be 0, indicating that
                        ///< all previously enqueued commands must be complete.
    ur_event_handle_t *Event ///< [in,out][optional] return an event object that
                             ///< identifies this particular command instance.
) {
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueEventsWaitWithBarrier(
    ur_queue_handle_t Queue,      ///< [in] handle of the queue object
    uint32_t NumEventsInWaitList, ///< [in] size of the event wait list
    const ur_event_handle_t
        *EventWaitList, ///< [in][optional][range(0, numEventsInWaitList)]
                        ///< pointer to a list of events that must be complete
                        ///< before this command can be executed. If nullptr,
                        ///< the numEventsInWaitList must be 0, indicating that
                        ///< all previously enqueued commands must be complete.
    ur_event_handle_t *Event ///< [in,out][optional] return an event object that
                             ///< identifies this particular command instance.
) {
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urEventGetInfo(
    ur_event_handle_t Event,  ///< [in] handle of the event object
    ur_event_info_t PropName, ///< [in] the name of the event property to query
    size_t PropValueSize, ///< [in] size in bytes of the event property value
    void *PropValue,      ///< [out][optional] value of the event property
    size_t
        *PropValueSizeRet ///< [out][optional] bytes returned in event property
) {
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urEventGetProfilingInfo(
    ur_event_handle_t Event, ///< [in] handle of the event object
    ur_profiling_info_t
        PropName, ///< [in] the name of the profiling property to query
    size_t
        PropValueSize, ///< [in] size in bytes of the profiling property value
    void *PropValue,   ///< [out][optional] value of the profiling property
    size_t *PropValueSizeRet ///< [out][optional] pointer to the actual size in
                             ///< bytes returned in propValue
) {
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urEventWait(
    uint32_t NumEvents, ///< [in] number of events in the event list
    const ur_event_handle_t
        *EventWaitList ///< [in][range(0, numEvents)] pointer to a list of
                       ///< events to wait for completion
) {
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urEventRetain(
    ur_event_handle_t Event ///< [in] handle of the event object
) {
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urEventRelease(
    ur_event_handle_t Event ///< [in] handle of the event object
) {
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urEventGetNativeHandle(
    ur_event_handle_t Event, ///< [in] handle of the event.
    ur_native_handle_t
        *NativeEvent ///< [out] a pointer to the native handle of the event.
) {
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urEventCreateWithNativeHandle(
    ur_native_handle_t NativeEvent, ///< [in] the native handle of the event.
    ur_context_handle_t Context,    ///< [in] handle of the context object
    ur_event_handle_t
        *Event ///< [out] pointer to the handle of the event object created.
) {
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urEventSetCallback(
    ur_event_handle_t Event,        ///< [in] handle of the event object
    ur_execution_info_t ExecStatus, ///< [in] execution status of the event
    ur_event_callback_t Notify,     ///< [in] execution status of the event
    void *UserData ///< [in][out][optional] pointer to data to be passed to
                   ///< callback.
) {
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}
