//===-- library_utils.cpp - Dynamic library utilities ----------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/detail/os_util.hpp>

#ifdef __SYCL_RT_OS_LINUX
#include <dlfcn.h>
#endif

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

std::string getPluginDirectory() {
#ifdef __SYCL_SECURE_DLL_LOAD
  // TODO this funciton must be moved to support library
  return OSUtil::getCurrentDSODir();
#else
  return "";
#endif
}

void *loadOsLibrary(const std::string &PluginPath) {
#if defined(__SYCL_RT_OS_LINUX)
  // TODO: Check if the option RTLD_NOW is correct. Explore using
  // RTLD_DEEPBIND option when there are multiple plugins.
  return dlopen(PluginPath.c_str(), RTLD_NOW);
#elif defined(__SYCL_RT_OS_WINDOWS)
#ifdef __SYCL_SECURE_DLL_LOAD
  // Exclude current directory from DLL search paths according to Microsoft
  // guidelines.
  SetDllDirectory("");
#endif
  // Tells the system to not display the critical-error-handler message box.
  // Instead, the system sends the error to the calling process.
  // This is crucial for graceful handling of plugins that couldn't be
  // loaded, e.g. due to missing native run-times.
  // TODO: add reporting in case of an error.
  // NOTE: we restore the old mode to not affect user app behavior.
  //
  UINT SavedMode = SetErrorMode(SEM_FAILCRITICALERRORS);
  auto Result = (void *)LoadLibraryA(PluginPath.c_str());
  (void)SetErrorMode(SavedMode);

  return Result;
#endif
}

int unloadOsLibrary(void *Library) {
#if defined(__SYCL_RT_OS_LINUX)
  return dlclose(Library);
#elif defined(__SYCL_RT_OS_WINDOWS)
  return (int)FreeLibrary((HMODULE)Library);
#endif
}

void *getOsLibraryFuncAddress(void *Library, const std::string &FunctionName) {
#if defined(__SYCL_RT_OS_LINUX)
  return dlsym(Library, FunctionName.c_str());
#elif defined(__SYCL_RT_OS_WINDOWS)
  return reinterpret_cast<void *>(
      GetProcAddress((HMODULE)Library, FunctionName.c_str()));
#endif
}

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
