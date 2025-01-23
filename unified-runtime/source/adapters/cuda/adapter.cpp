//===--------- adapter.cpp - CUDA Adapter ---------------------------------===//
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
#include "platform.hpp"
#include "tracing.hpp"

#include <memory>

namespace ur::cuda {
ur_adapter_handle_t adapter;
} // namespace ur::cuda

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
      logger(logger::get_logger("cuda",
                                /*default_log_level*/ UR_LOGGER_LEVEL_ERROR)) {
  Platform = std::make_unique<ur_platform_handle_t_>();

  if (std::getenv("UR_LOG_CUDA") == nullptr &&
      (std::getenv("SYCL_PI_SUPPRESS_ERROR_MESSAGE") != nullptr ||
       std::getenv("UR_SUPPRESS_ERROR_MESSAGE") != nullptr)) {
    logger.setLegacySink(std::make_unique<ur_legacy_sink>());
  }

  TracingCtx = createCUDATracingContext();
  enableCUDATracing(TracingCtx);
}

ur_adapter_handle_t_::~ur_adapter_handle_t_() {
  disableCUDATracing(TracingCtx);
  freeCUDATracingContext(TracingCtx);
}

UR_APIEXPORT ur_result_t UR_APICALL
urAdapterGet(uint32_t NumEntries, ur_adapter_handle_t *phAdapters,
             uint32_t *pNumAdapters) {
  if (NumEntries > 0 && phAdapters) {
    static std::once_flag InitFlag;
    std::call_once(InitFlag,
                   [=]() { ur::cuda::adapter = new ur_adapter_handle_t_; });

    ur::cuda::adapter->RefCount++;
    *phAdapters = ur::cuda::adapter;
  }

  if (pNumAdapters) {
    *pNumAdapters = 1;
  }

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urAdapterRetain(ur_adapter_handle_t) {
  ur::cuda::adapter->RefCount++;

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urAdapterRelease(ur_adapter_handle_t) {
  if (--ur::cuda::adapter->RefCount == 0) {
    delete ur::cuda::adapter;
  }
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
    return ReturnValue(UR_ADAPTER_BACKEND_CUDA);
  case UR_ADAPTER_INFO_REFERENCE_COUNT:
    return ReturnValue(ur::cuda::adapter->RefCount.load());
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

  ur::cuda::adapter->logger.setCallbackSink(pfnLoggerCallback, pUserData,
                                            level);

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urAdapterSetLoggerCallbackLevel(
    ur_adapter_handle_t, ur_logger_level_t level = UR_LOGGER_LEVEL_QUIET) {

  ur::cuda::adapter->logger.setCallbackLevel(level);

  return UR_RESULT_SUCCESS;
}
