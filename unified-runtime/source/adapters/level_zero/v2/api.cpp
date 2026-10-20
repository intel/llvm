//===--------- api.cpp - Level Zero Adapter ------------------------------===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <mutex>
#include <unified-runtime/ur_api.h>
#include <ze_api.h>

#include "../common.hpp"
#include "logger/ur_logger.hpp"

namespace ur::level_zero::v2 {

ur_result_t
urContextSetExtendedDeleter(::ur_context_handle_t /*hContextOpque*/,
                            ur_context_extended_deleter_t /*pfnDeleter*/,
                            void * /*pUserData*/) {
  UR_LOG(ERR, "{} function not implemented!", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t urKernelSetSpecializationConstants(
    ::ur_kernel_handle_t /*hKernelOpque*/, uint32_t /*count*/,
    const ur_specialization_constant_info_t * /*pSpecConstants*/) {
  UR_LOG(ERR, "{} function not implemented!", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t urEventSetCallback(::ur_event_handle_t /*hEventOpque*/,
                               ur_execution_info_t /*execStatus*/,
                               ur_event_callback_t /*pfnNotify*/,
                               void * /*pUserData*/) {
  UR_LOG(ERR, "{} function not implemented!", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t UR_APICALL
urUSMPoolGetDevicePoolExp(::ur_context_handle_t /*hContextOpque*/,
                          ::ur_device_handle_t /*hDeviceOpque*/,
                          ::ur_usm_pool_handle_t * /*pPoolOpque*/) {
  UR_LOG(ERR, "{} function not implemented!", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t UR_APICALL
urUSMPoolSetDevicePoolExp(::ur_context_handle_t /*hContextOpque*/,
                          ::ur_device_handle_t /*hDeviceOpque*/,
                          ::ur_usm_pool_handle_t /*hPoolOpque*/) {
  UR_LOG(ERR, "{} function not implemented!", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t UR_APICALL urUSMPoolTrimToExp(
    ::ur_context_handle_t /*hContextOpque*/,
    ::ur_device_handle_t /*hDeviceOpque*/,
    ::ur_usm_pool_handle_t /*hPoolOpque*/, size_t /*minBytesToKeep*/) {
  UR_LOG(ERR, "{} function not implemented!", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

} // namespace ur::level_zero::v2
