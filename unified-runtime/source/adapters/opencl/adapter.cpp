//===-------------- adapter.cpp - OpenCL Adapter ---------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifdef UR_STATIC_ADAPTER_OPENCL
// Define the OpenCL dispatch-pointer table here, before common.hpp activates
// the redirect macros.  Defining here (rather than in a separate TU) ensures
// the symbols land in adapter.cpp.o, which is always pulled into
// libur_loader.so when the static adapter is embedded.
//
// CL_ENABLE_BETA_EXTENSIONS must be defined before including cl_ext.h so that
// extension types such as cl_command_buffer_khr are visible.  common.hpp sets
// this macro before its own <CL/cl_ext.h> include; we must mirror that here
// because once cl_ext.h is included (with its include guard) common.hpp's
// re-include will be a no-op.
#ifndef CL_ENABLE_BETA_EXTENSIONS
#define CL_ENABLE_BETA_EXTENSIONS
#endif
#include <CL/cl.h>
#include <CL/cl_ext.h>
namespace cl_dispatch {
#define CL_FUNCTION(FUNC) decltype(::FUNC) *FUNC##_ptr = nullptr;
#include "all_functions.def"
#undef CL_FUNCTION
} // namespace cl_dispatch
#endif // UR_STATIC_ADAPTER_OPENCL

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
#ifdef UR_STATIC_ADAPTER_OPENCL
  // Load the OpenCL ICD loader dynamically so that the static adapter
  // has no link-time dependency on it.
#ifdef _MSC_VER
  openclLibHandle = LoadLibraryA("OpenCL.dll");
  if (!openclLibHandle) {
    die("Failed to load OpenCL.dll");
  }
#define CL_FUNCTION(FUNC)                                                      \
  cl_dispatch::FUNC##_ptr =                                                    \
      reinterpret_cast<decltype(cl_dispatch::FUNC##_ptr)>(                     \
          GetProcAddress(static_cast<HMODULE>(openclLibHandle), #FUNC));
#include "all_functions.def"
#undef CL_FUNCTION
#define CL_CORE_FUNCTION(FUNC)                                                 \
  FUNC = reinterpret_cast<decltype(::FUNC) *>(                                 \
      GetProcAddress(static_cast<HMODULE>(openclLibHandle), #FUNC));
#include "core_functions.def"
#undef CL_CORE_FUNCTION
#else // !_MSC_VER
  openclLibHandle = dlopen("libOpenCL.so.1", RTLD_LAZY | RTLD_GLOBAL);
  if (!openclLibHandle) {
    openclLibHandle = dlopen("libOpenCL.so", RTLD_LAZY | RTLD_GLOBAL);
  }
  if (!openclLibHandle) {
    die("Failed to load OpenCL ICD loader");
  }
#define CL_FUNCTION(FUNC)                                                      \
  cl_dispatch::FUNC##_ptr =                                                    \
      reinterpret_cast<decltype(cl_dispatch::FUNC##_ptr)>(                     \
          dlsym(openclLibHandle, #FUNC));
#include "all_functions.def"
#undef CL_FUNCTION
#define CL_CORE_FUNCTION(FUNC)                                                 \
  FUNC = reinterpret_cast<decltype(::FUNC) *>(dlsym(openclLibHandle, #FUNC));
#include "core_functions.def"
#undef CL_CORE_FUNCTION
#endif // _MSC_VER
#else  // !UR_STATIC_ADAPTER_OPENCL
#ifdef _MSC_VER

  // Retrieving handle of an already linked OpenCL.dll library doesn't increase
  // the reference count.
  auto handle = GetModuleHandleA("OpenCL.dll");
  assert(handle);

#define CL_CORE_FUNCTION(FUNC)                                                 \
  FUNC = reinterpret_cast<decltype(::FUNC) *>(GetProcAddress(handle, #FUNC));
#include "core_functions.def"
#undef CL_CORE_FUNCTION
#else // _MSC_VER

  // Use the default shared object search order (RTLD_DEFAULT) since the
  // OpenCL-ICD-Loader has already been loaded into the process.
#define CL_CORE_FUNCTION(FUNC)                                                 \
  FUNC = reinterpret_cast<decltype(::FUNC) *>(dlsym(RTLD_DEFAULT, #FUNC));
#include "core_functions.def"

namespace ur::opencl {

#undef CL_CORE_FUNCTION

#endif // _MSC_VER
#endif // UR_STATIC_ADAPTER_OPENCL
  assert(!liveAdapter);
  liveAdapter = this;
}

ur_adapter_handle_t_::~ur_adapter_handle_t_() {
  assert(liveAdapter == this);
  liveAdapter = nullptr;
#ifdef UR_STATIC_ADAPTER_OPENCL
  if (openclLibHandle) {
#ifdef _MSC_VER
    FreeLibrary(static_cast<HMODULE>(openclLibHandle));
#else
    dlclose(openclLibHandle);
#endif
    openclLibHandle = nullptr;
  }
#endif
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
    adapter->RefCount.retain();
  }

  if (pNumAdapters) {
    *pNumAdapters = 1;
  }

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urAdapterRetain(ur_adapter_handle_t hAdapter) {
  hAdapter->RefCount.retain();
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urAdapterRelease(ur_adapter_handle_t hAdapter) {
  if (hAdapter->RefCount.release()) {
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
    return ReturnValue(UR_BACKEND_OPENCL);
  case UR_ADAPTER_INFO_REFERENCE_COUNT:
    return ReturnValue(hAdapter->RefCount.getCount());
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

} // namespace ur::opencl
