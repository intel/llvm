//==---------------- windows_pi.cpp ----------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/detail/defines.hpp>

#include <string>
#include <windows.h>
#include <winreg.h>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {
namespace pi {

void *loadOsLibrary(const std::string &PluginPath) {
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
}

int unloadOsLibrary(void *Library) {
  return (int)FreeLibrary((HMODULE)Library);
}

void *getOsLibraryFuncAddress(void *Library, const std::string &FunctionName) {
  return reinterpret_cast<void *>(
      GetProcAddress((HMODULE)Library, FunctionName.c_str()));
}

} // namespace pi
} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
