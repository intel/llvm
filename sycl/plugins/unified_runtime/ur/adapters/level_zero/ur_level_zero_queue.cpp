//===--------- ur_level_zero_queue.cpp - Level Zero Adapter -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//

#include "ur_level_zero_queue.hpp"

UR_APIEXPORT ur_result_t UR_APICALL urQueueGetInfo(
    ur_queue_handle_t Queue,  ///< [in] handle of the queue object
    ur_queue_info_t PropName, ///< [in] name of the queue property to query
    size_t PropValueSize, ///< [in] size in bytes of the queue property value
                          ///< provided
    void *PropValue,      ///< [out] value of the queue property
    size_t
        *PropSizeRet ///< [out] size in bytes returned in queue property value
) {
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urQueueCreate(
    ur_context_handle_t Context, ///< [in] handle of the context object
    ur_device_handle_t Device,   ///< [in] handle of the device object
    const ur_queue_property_t
        *Props, ///< [in] specifies a list of queue properties and their
                ///< corresponding values. Each property name is immediately
                ///< followed by the corresponding desired value. The list is
                ///< terminated with a 0. If a property value is not specified,
                ///< then its default value will be used.
    ur_queue_handle_t
        *Queue ///< [out] pointer to handle of queue object created
) {
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urQueueRetain(
    ur_queue_handle_t Queue ///< [in] handle of the queue object to get access
) {
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urQueueRelease(
    ur_queue_handle_t Queue ///< [in] handle of the queue object to release
) {
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urQueueGetNativeHandle(
    ur_queue_handle_t Queue, ///< [in] handle of the queue.
    ur_native_handle_t
        *NativeQueue ///< [out] a pointer to the native handle of the queue.
) {
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urQueueCreateWithNativeHandle(
    ur_native_handle_t NativeQueue, ///< [in] the native handle of the queue.
    ur_context_handle_t Context,    ///< [in] handle of the context object
    ur_queue_handle_t
        *Queue ///< [out] pointer to the handle of the queue object created.
) {
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urQueueFinish(
    ur_queue_handle_t Queue ///< [in] handle of the queue to be finished.
) {
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urQueueFlush(
    ur_queue_handle_t Queue ///< [in] handle of the queue to be flushed.
) {
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}
