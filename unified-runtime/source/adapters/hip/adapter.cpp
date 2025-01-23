//===--------- adapter.cpp - HIP Adapter ----------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "adapter.hpp"
#include "common.hpp"

#include <ur_api.h>

#include <memory>

namespace ur::hip {
ur_adapter_handle_t adapter;
}

class ur_legacy_sink : public logger::Sink {
public:
  ur_legacy_sink(std::string logger_name = "", bool skip_prefix = true)
      : Sink(std::move(logger_name), skip_prefix) {
    this->ostream = &std::cerr;
  }

  virtual void print([[maybe_unused]] ur_logger_level_t level,
                     const std::string &msg) override {
    std::cerr << msg << std::endl;
  }

  ~ur_legacy_sink() = default;
};

// FIXME: Remove the default log level when querying logging info is supported
// through UR entry points.
// https://github.com/oneapi-src/unified-runtime/issues/1330
ur_adapter_handle_t_::ur_adapter_handle_t_()
    : handle_base(),
      logger(logger::get_logger("hip",
                                /*default_log_level*/ UR_LOGGER_LEVEL_ERROR)) {
  Platform = std::make_unique<ur_platform_handle_t_>();
  if (std::getenv("UR_LOG_HIP") != nullptr)
    return;

  if (std::getenv("SYCL_PI_SUPPRESS_ERROR_MESSAGE") != nullptr ||
      std::getenv("UR_SUPPRESS_ERROR_MESSAGE") != nullptr) {
    logger.setLegacySink(std::make_unique<ur_legacy_sink>());
  }
}

UR_APIEXPORT ur_result_t UR_APICALL urAdapterGet(
    uint32_t, ur_adapter_handle_t *phAdapters, uint32_t *pNumAdapters) {
  if (phAdapters) {
    static std::once_flag InitFlag;
    std::call_once(InitFlag,
                   [=]() { ur::hip::adapter = new ur_adapter_handle_t_; });

    ur::hip::adapter->RefCount++;
    *phAdapters = ur::hip::adapter;
  }
  if (pNumAdapters) {
    *pNumAdapters = 1;
  }

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urAdapterRelease(ur_adapter_handle_t) {
  if (--ur::hip::adapter->RefCount == 0) {
    delete ur::hip::adapter;
  }

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urAdapterRetain(ur_adapter_handle_t) {
  ur::hip::adapter->RefCount++;
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urAdapterGetLastError(
    ur_adapter_handle_t, const char **ppMessage, int32_t *pError) {
  *ppMessage = ErrorMessage;
  *pError = ErrorMessageCode;
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urAdapterGetInfo(ur_adapter_handle_t,
                                                     ur_adapter_info_t propName,
                                                     size_t propSize,
                                                     void *pPropValue,
                                                     size_t *pPropSizeRet) {
  UrReturnHelper ReturnValue(propSize, pPropValue, pPropSizeRet);

  switch (propName) {
  case UR_ADAPTER_INFO_BACKEND:
    return ReturnValue(UR_ADAPTER_BACKEND_HIP);
  case UR_ADAPTER_INFO_REFERENCE_COUNT:
    return ReturnValue(ur::hip::adapter->RefCount.load());
  case UR_ADAPTER_INFO_VERSION:
    return ReturnValue(uint32_t{1});
  default:
    return UR_RESULT_ERROR_INVALID_ENUMERATION;
  }

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urAdapterSetLoggerCallback(
    ur_adapter_handle_t, ur_logger_callback_t pfnLoggerCallback,
    void *pUserData, ur_logger_level_t level = UR_LOGGER_LEVEL_QUIET) {

  ur::hip::adapter->logger.setCallbackSink(pfnLoggerCallback, pUserData, level);

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urAdapterSetLoggerCallbackLevel(ur_adapter_handle_t, ur_logger_level_t level) {

  ur::hip::adapter->logger.setCallbackLevel(level);

  return UR_RESULT_SUCCESS;
}
