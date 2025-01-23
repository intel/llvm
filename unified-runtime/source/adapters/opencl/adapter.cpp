//===-------------- adapter.cpp - OpenCL Adapter ---------------------===//
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
#include "ur/ur.hpp"

#ifdef _MSC_VER
#include <Windows.h>
#else
#include <dlfcn.h>
#endif

// There can only be one OpenCL adapter alive at a time.
// If it is alive (more get/retains than releases called), this is a pointer to
// it.
static ur_adapter_handle_t liveAdapter = nullptr;

ur_adapter_handle_t_::ur_adapter_handle_t_() : handle_base() {
#ifdef _MSC_VER

  // Loading OpenCL.dll increments the libraries internal reference count.
  auto handle = LoadLibraryA("OpenCL.dll");

#define CL_CORE_FUNCTION(FUNC)                                                 \
  FUNC = reinterpret_cast<decltype(::FUNC) *>(GetProcAddress(handle, #FUNC));
#include "core_functions.def"
#undef CL_CORE_FUNCTION

  // So we can safely decrement it here wihtout actually unloading OpenCL.dll.
  FreeLibrary(handle);

#else // _MSC_VER

  // Use the default shared object search order (RTLD_DEFAULT) since the
  // OpenCL-ICD-Loader has already been loaded into the process.
#define CL_CORE_FUNCTION(FUNC)                                                 \
  FUNC = reinterpret_cast<decltype(::FUNC) *>(dlsym(RTLD_DEFAULT, #FUNC));
#include "core_functions.def"
#undef CL_CORE_FUNCTION

#endif // _MSC_VER
  assert(!liveAdapter);
  liveAdapter = this;
}

ur_adapter_handle_t_::~ur_adapter_handle_t_() {
  assert(liveAdapter == this);
  liveAdapter = nullptr;
}

ur_adapter_handle_t ur::cl::getAdapter() {
  if (!liveAdapter) {
    die("OpenCL adapter used before initalization or after destruction");
  }
  return liveAdapter;
}

UR_APIEXPORT ur_result_t UR_APICALL
urAdapterGet(uint32_t NumEntries, ur_adapter_handle_t *phAdapters,
             uint32_t *pNumAdapters) {
  static std::mutex AdapterConstructionMutex{};

  if (NumEntries > 0 && phAdapters) {
    std::lock_guard<std::mutex> Lock{AdapterConstructionMutex};

    if (!liveAdapter) {
      *phAdapters = new ur_adapter_handle_t_();
    } else {
      *phAdapters = liveAdapter;
    }

    auto &adapter = *phAdapters;
    adapter->RefCount++;
  }

  if (pNumAdapters) {
    *pNumAdapters = 1;
  }

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urAdapterRetain(ur_adapter_handle_t hAdapter) {
  ++hAdapter->RefCount;
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urAdapterRelease(ur_adapter_handle_t hAdapter) {
  if (--hAdapter->RefCount == 0) {
    delete hAdapter;
  }
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urAdapterGetLastError(
    ur_adapter_handle_t, const char **ppMessage, int32_t *pError) {
  *ppMessage = cl_adapter::ErrorMessage;
  *pError = cl_adapter::ErrorMessageCode;

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urAdapterGetInfo(ur_adapter_handle_t hAdapter, ur_adapter_info_t propName,
                 size_t propSize, void *pPropValue, size_t *pPropSizeRet) {
  UrReturnHelper ReturnValue(propSize, pPropValue, pPropSizeRet);

  switch (propName) {
  case UR_ADAPTER_INFO_BACKEND:
    return ReturnValue(UR_ADAPTER_BACKEND_OPENCL);
  case UR_ADAPTER_INFO_REFERENCE_COUNT:
    return ReturnValue(hAdapter->RefCount.load());
  case UR_ADAPTER_INFO_VERSION:
    return ReturnValue(uint32_t{1});
  default:
    return UR_RESULT_ERROR_INVALID_ENUMERATION;
  }

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urAdapterSetLoggerCallback(
    ur_adapter_handle_t hAdapter, ur_logger_callback_t pfnLoggerCallback,
    void *pUserData, ur_logger_level_t level = UR_LOGGER_LEVEL_QUIET) {

  if (hAdapter) {
    hAdapter->log.setCallbackSink(pfnLoggerCallback, pUserData, level);
  }

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urAdapterSetLoggerCallbackLevel(
    ur_adapter_handle_t hAdapter, ur_logger_level_t level) {

  if (hAdapter) {
    hAdapter->log.setCallbackLevel(level);
  }

  return UR_RESULT_SUCCESS;
}
