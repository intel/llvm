//===--------- api.cpp - Level Zero Adapter ------------------------------===//
//
// Copyright (C) 2024 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <mutex>
#include <ur_api.h>
#include <ze_api.h>

#include "../common.hpp"
#include "logger/ur_logger.hpp"

std::mutex ZeCall::GlobalLock;

namespace ur::level_zero {

ur_result_t
urContextSetExtendedDeleter(ur_context_handle_t hContext,
                            ur_context_extended_deleter_t pfnDeleter,
                            void *pUserData) {
  UR_LOG(ERR, "{} function not implemented!", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t urKernelSetSpecializationConstants(
    ur_kernel_handle_t hKernel, uint32_t count,
    const ur_specialization_constant_info_t *pSpecConstants) {
  UR_LOG(ERR, "{} function not implemented!", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t urEventSetCallback(ur_event_handle_t hEvent,
                               ur_execution_info_t execStatus,
                               ur_event_callback_t pfnNotify, void *pUserData) {
  UR_LOG(ERR, "{} function not implemented!", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t UR_APICALL urUSMPoolCreateExp(ur_context_handle_t hContext,
                                          ur_device_handle_t hDevice,
                                          ur_usm_pool_desc_t *PoolDesc,
                                          ur_usm_pool_handle_t *pPool) {
  UR_LOG(ERR, "{} function not implemented!", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t UR_APICALL urUSMPoolDestroyExp(ur_context_handle_t hContext,
                                           ur_device_handle_t hDevice,
                                           ur_usm_pool_handle_t hPool) {
  UR_LOG(ERR, "{} function not implemented!", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t UR_APICALL urUSMPoolSetInfoExp(ur_usm_pool_handle_t hPool,
                                           ur_usm_pool_info_t propName,
                                           void *pPropValue, size_t propSize) {
  UR_LOG(ERR, "{} function not implemented!", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t UR_APICALL urUSMPoolGetDefaultDevicePoolExp(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    ur_usm_pool_handle_t *pPool) {
  UR_LOG(ERR, "{} function not implemented!", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t UR_APICALL urUSMPoolGetInfoExp(ur_usm_pool_handle_t hPool,
                                           ur_usm_pool_info_t propName,
                                           void *pPropValue,
                                           size_t *pPropSizeRet) {
  UR_LOG(ERR, "{} function not implemented!", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t UR_APICALL urUSMPoolGetDevicePoolExp(ur_context_handle_t hContext,
                                                 ur_device_handle_t hDevice,
                                                 ur_usm_pool_handle_t *pPool) {
  UR_LOG(ERR, "{} function not implemented!", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t UR_APICALL urUSMPoolSetDevicePoolExp(ur_context_handle_t hContext,
                                                 ur_device_handle_t hDevice,
                                                 ur_usm_pool_handle_t hPool) {
  UR_LOG(ERR, "{} function not implemented!", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t UR_APICALL urUSMPoolTrimToExp(ur_context_handle_t hContext,
                                          ur_device_handle_t hDevice,
                                          ur_usm_pool_handle_t hPool,
                                          size_t minBytesToKeep) {
  UR_LOG(ERR, "{} function not implemented!", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

} // namespace ur::level_zero
