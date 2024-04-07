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
#include "logger/ur_logger.hpp"

struct ur_adapter_handle_t_ {
  std::atomic<uint32_t> RefCount = 0;
  std::mutex Mutex;
  logger::Logger &log = logger::get_logger("opencl");
};

static ur_adapter_handle_t_ *adapter = nullptr;

static void globalAdapterShutdown() {
  if (cl_ext::ExtFuncPtrCache) {
    delete cl_ext::ExtFuncPtrCache;
    cl_ext::ExtFuncPtrCache = nullptr;
  }
  if (adapter) {
    delete adapter;
    adapter = nullptr;
  }
}

UR_APIEXPORT ur_result_t UR_APICALL
urAdapterGet(uint32_t NumEntries, ur_adapter_handle_t *phAdapters,
             uint32_t *pNumAdapters) {
  if (NumEntries > 0 && phAdapters) {
    // Sometimes urAdaterGet may be called after the library already been torn
    // down, we also need to create a temporary handle for it.
    if (!adapter) {
      adapter = new ur_adapter_handle_t_();
      atexit(globalAdapterShutdown);
    }

    std::lock_guard<std::mutex> Lock{adapter->Mutex};
    if (adapter->RefCount++ == 0) {
      cl_ext::ExtFuncPtrCache = new cl_ext::ExtFuncPtrCacheT();
    }

    *phAdapters = adapter;
  }

  if (pNumAdapters) {
    *pNumAdapters = 1;
  }

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urAdapterRetain(ur_adapter_handle_t) {
  ++adapter->RefCount;
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urAdapterRelease(ur_adapter_handle_t) {
  // Check first if the adapter is valid pointer
  if (adapter) {
    std::lock_guard<std::mutex> Lock{adapter->Mutex};
    if (--adapter->RefCount == 0) {
      if (cl_ext::ExtFuncPtrCache) {
        delete cl_ext::ExtFuncPtrCache;
        cl_ext::ExtFuncPtrCache = nullptr;
      }
    }
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
    return ReturnValue(UR_ADAPTER_BACKEND_OPENCL);
  case UR_ADAPTER_INFO_REFERENCE_COUNT:
    return ReturnValue(adapter->RefCount.load());
  default:
    return UR_RESULT_ERROR_INVALID_ENUMERATION;
  }

  return UR_RESULT_SUCCESS;
}
