//===-------------- adapter.cpp - OpenCL Adapter ---------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "common.hpp"

struct ur_adapter_handle_t_ {
  std::atomic<uint32_t> RefCount = 0;
  std::mutex Mutex;
};

ur_adapter_handle_t_ adapter{};

UR_APIEXPORT ur_result_t UR_APICALL urInit(ur_device_init_flags_t,
                                           ur_loader_config_handle_t) {
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urTearDown(void *) {
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urAdapterGet(uint32_t NumEntries, ur_adapter_handle_t *phAdapters,
             uint32_t *pNumAdapters) {
  if (NumEntries > 0 && phAdapters) {
    std::lock_guard<std::mutex> Lock{adapter.Mutex};
    if (adapter.RefCount++ == 0) {
      cl_ext::ExtFuncPtrCache = std::make_unique<cl_ext::ExtFuncPtrCacheT>();
    }

    *phAdapters = &adapter;
  }

  if (pNumAdapters) {
    *pNumAdapters = 1;
  }

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urAdapterRetain(ur_adapter_handle_t) {
  ++adapter.RefCount;
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urAdapterRelease(ur_adapter_handle_t) {
  std::lock_guard<std::mutex> Lock{adapter.Mutex};
  if (--adapter.RefCount == 0) {
    cl_ext::ExtFuncPtrCache.reset();
  }
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urAdapterGetLastError(
    ur_adapter_handle_t, const char **ppMessage, int32_t *pError) {
  *ppMessage = cl_adapter::ErrorMessage;
  *pError = cl_adapter::ErrorMessageCode;

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
    return ReturnValue(adapter.RefCount.load());
  default:
    return UR_RESULT_ERROR_INVALID_ENUMERATION;
  }

  return UR_RESULT_SUCCESS;
}
