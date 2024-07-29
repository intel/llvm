//==---------------- windows_ur.cpp ----------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/backend.hpp>
#include <sycl/detail/defines.hpp>

#include <cassert>
#include <string>
#include <windows.h>
#include <winreg.h>

#include "detail/windows_os_utils.hpp"
#include "ur_win_proxy_loader.hpp"

namespace sycl {
inline namespace _V1 {
namespace detail {
namespace ur {

void *loadOsLibrary(const std::string &LibraryPath) {
  // Tells the system to not display the critical-error-handler message box.
  // Instead, the system sends the error to the calling process.
  // This is crucial for graceful handling of shared libs that can't be
  // loaded, e.g. due to missing native run-times.

  UINT SavedMode = SetErrorMode(SEM_FAILCRITICALERRORS);
  // Exclude current directory from DLL search path
  if (!SetDllDirectoryA("")) {
    assert(false && "Failed to update DLL search path");
  }

  auto Result = (void *)LoadLibraryExA(LibraryPath.c_str(), NULL, NULL);
  (void)SetErrorMode(SavedMode);
  if (!SetDllDirectoryA(nullptr)) {
    assert(false && "Failed to restore DLL search path");
  }

  return Result;
}

int unloadOsLibrary(void *Library) {
  return (int)FreeLibrary((HMODULE)Library);
}

void *getOsLibraryFuncAddress(void *Library, const std::string &FunctionName) {
  return reinterpret_cast<void *>(
      GetProcAddress((HMODULE)Library, FunctionName.c_str()));
}

static std::filesystem::path getCurrentDSODirPath() {
  wchar_t Path[MAX_PATH];
  auto Handle =
      getOSModuleHandle(reinterpret_cast<void *>(&getCurrentDSODirPath));
  DWORD Ret = GetModuleFileName(
      reinterpret_cast<HMODULE>(ExeModuleHandle == Handle ? 0 : Handle), Path,
      MAX_PATH);
  assert(Ret < MAX_PATH && "Path is longer than MAX_PATH?");
  assert(Ret > 0 && "GetModuleFileName failed");
  (void)Ret;

  BOOL RetCode = PathRemoveFileSpec(Path);
  assert(RetCode && "PathRemoveFileSpec failed");
  (void)RetCode;

  return std::filesystem::path(Path);
}

} // namespace ur
} // namespace detail
} // namespace _V1
} // namespace sycl
