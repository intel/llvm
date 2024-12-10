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
#include "logger/ur_logger.hpp"

#include <atomic>
#include <ur_api.h>

struct ur_adapter_handle_t_ {
  std::atomic<uint32_t> RefCount = 0;
  logger::Logger &logger;
  ur_adapter_handle_t_();
};

class ur_legacy_sink : public logger::Sink {
public:
  ur_legacy_sink(std::string logger_name = "", bool skip_prefix = true)
      : Sink(std::move(logger_name), skip_prefix) {
    this->ostream = &std::cerr;
  }

  virtual void print([[maybe_unused]] logger::Level level,
                     const std::string &msg) override {
    std::cerr << msg << std::endl;
  }

  ~ur_legacy_sink() = default;
};

// FIXME: Remove the default log level when querying logging info is supported
// through UR entry points. See #1330.
ur_adapter_handle_t_::ur_adapter_handle_t_()
    : logger(
          logger::get_logger("hip", /*default_log_level*/ logger::Level::ERR)) {

  if (std::getenv("UR_LOG_HIP") != nullptr)
    return;

  if (std::getenv("SYCL_PI_SUPPRESS_ERROR_MESSAGE") != nullptr ||
      std::getenv("UR_SUPPRESS_ERROR_MESSAGE") != nullptr) {
    logger.setLegacySink(std::make_unique<ur_legacy_sink>());
  }
}

ur_adapter_handle_t_ adapter{};

UR_APIEXPORT ur_result_t UR_APICALL urAdapterGet(
    uint32_t, ur_adapter_handle_t *phAdapters, uint32_t *pNumAdapters) {
  if (phAdapters) {
    adapter.RefCount++;
    *phAdapters = &adapter;
  }
  if (pNumAdapters) {
    *pNumAdapters = 1;
  }

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urAdapterRelease(ur_adapter_handle_t) {
  // No state to clean up so we don't need to check for 0 references
  adapter.RefCount--;
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urAdapterRetain(ur_adapter_handle_t) {
  adapter.RefCount++;
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
    return ReturnValue(adapter.RefCount.load());
  case UR_ADAPTER_INFO_VERSION:
    return ReturnValue(uint32_t{1});
  default:
    return UR_RESULT_ERROR_INVALID_ENUMERATION;
  }

  return UR_RESULT_SUCCESS;
}
