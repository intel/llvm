//===------- ocl_dynamic_lib.cpp - OpenCL Dynamic Loading -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//

#ifdef UR_STATIC_ADAPTER_OPENCL

// Define this before including the header to prevent macro redefinitions
#define OCL_DYNAMIC_LIB_IMPL
#include "ocl_dynamic_lib.hpp"

#include "logger/ur_logger.hpp"

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <mutex>

namespace ocl {

// Define storage for all function pointers using X-macros
#define OCL_FUNC(name, required) decltype(::name) *name##_ptr = nullptr;
#include "ocl_functions.def"
#undef OCL_FUNC

static void *OCLLibHandle = nullptr;
static std::mutex OCLLoadMutex;
static bool OCLLoaded = false;
static bool OCLLoadSuccess = false;

template <typename T>
static bool getSymbolAddr(void *handle, const char *name, T *funcPtr) {
#ifdef _WIN32
  *funcPtr = reinterpret_cast<T>(GetProcAddress((HMODULE)handle, name));
#else
  *funcPtr = reinterpret_cast<T>(dlsym(handle, name));
#endif
  return *funcPtr != nullptr;
}

static void loadOCLLibraryImpl() {
#ifdef _WIN32
  OCLLibHandle =
      LoadLibraryExA("OpenCL.dll", NULL, LOAD_LIBRARY_SEARCH_SYSTEM32);
  if (!OCLLibHandle) {
    DWORD error = GetLastError();
    UR_LOG(ERR,
           "Failed to load OpenCL.dll from system directory (error code: {})",
           error);
    return;
  }
  UR_LOG(DEBUG, "Successfully loaded OpenCL.dll");
#else
  OCLLibHandle = dlopen("libOpenCL.so.1", RTLD_NOW | RTLD_LOCAL);
  if (!OCLLibHandle) {
    const char *error1 = dlerror();
    UR_LOG(DEBUG, "Failed to load libOpenCL.so.1: {}",
           error1 ? error1 : "unknown error");

    OCLLibHandle = dlopen("libOpenCL.so", RTLD_NOW | RTLD_LOCAL);
    if (!OCLLibHandle) {
      const char *error2 = dlerror();
      UR_LOG(ERR,
             "Failed to load OpenCL library. Tried libOpenCL.so.1 and "
             "libOpenCL.so: {}",
             error2 ? error2 : "unknown error");
      return;
    }
    UR_LOG(DEBUG, "Successfully loaded libOpenCL.so");
  } else {
    UR_LOG(DEBUG, "Successfully loaded libOpenCL.so.1");
  }
#endif

  bool success = true;
  int required_missing = 0;
  int optional_missing = 0;

#define OCL_FUNC(name, required)                                               \
  do {                                                                         \
    if (!getSymbolAddr(OCLLibHandle, #name, &name##_ptr)) {                    \
      if (required) {                                                          \
        UR_LOG(ERR, "Required OpenCL function not found: {}", #name);          \
        required_missing++;                                                    \
        success = false;                                                       \
      } else {                                                                 \
        UR_LOG(DEBUG, "Optional OpenCL function not found: {}", #name);        \
        optional_missing++;                                                    \
      }                                                                        \
    }                                                                          \
  } while (0);

#include "ocl_functions.def"
#undef OCL_FUNC

  if (required_missing > 0) {
    UR_LOG(ERR, "Failed to load {} required OpenCL function(s)",
           required_missing);
  }

  if (optional_missing > 0) {
    UR_LOG(DEBUG,
           "{} optional OpenCL function(s) not available (normal for older "
           "OpenCL versions)",
           optional_missing);
  }

  if (!success) {
    // Required symbols missing — close the handle we opened to avoid a leak.
#ifdef _WIN32
    FreeLibrary((HMODULE)OCLLibHandle);
#else
    dlclose(OCLLibHandle);
#endif
    OCLLibHandle = nullptr;
#define OCL_FUNC(name, required) name##_ptr = nullptr;
#include "ocl_functions.def"
#undef OCL_FUNC
    return;
  }

  OCLLoadSuccess = true;
}

bool loadOCLLibrary() {
  std::lock_guard<std::mutex> Lock{OCLLoadMutex};
  if (!OCLLoaded) {
    loadOCLLibraryImpl();
    OCLLoaded = true;
  }
  return OCLLoadSuccess;
}

void unloadOCLLibrary() {
  std::lock_guard<std::mutex> Lock{OCLLoadMutex};
  if (OCLLibHandle) {
#ifdef _WIN32
    FreeLibrary((HMODULE)OCLLibHandle);
#else
    dlclose(OCLLibHandle);
#endif
    OCLLibHandle = nullptr;

#define OCL_FUNC(name, required) name##_ptr = nullptr;
#include "ocl_functions.def"
#undef OCL_FUNC
  }
  OCLLoaded = false;
  OCLLoadSuccess = false;
}

} // namespace ocl

#endif // UR_STATIC_ADAPTER_OPENCL
