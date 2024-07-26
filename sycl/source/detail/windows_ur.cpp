//==---------------- windows_ur.cpp ----------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/dlopen_utils.hpp>

#include <sycl/backend.hpp>
#include <sycl/detail/defines.hpp>

#include <cassert>
#include <string>
#include <windows.h>

#include "detail/windows_os_utils.hpp"
#include "ur_win_proxy_loader.hpp"

namespace sycl {
inline namespace _V1 {
namespace detail::ur {

void *loadOsPluginLibrary(const std::string &PluginPath) {
  // We fetch the preloaded plugin from the pi_win_proxy_loader.
  // The proxy_loader handles any required error suppression.
  auto Result = getPreloadedPlugin(PluginPath);

  return Result;
}

int unloadOsPluginLibrary(void *Library) {
  // The mock plugin does not have an associated library, so we allow nullptr
  // here to avoid it trying to free a non-existent library.
  if (!Library)
    return 1;
  return (int)FreeLibrary((HMODULE)Library);
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

} // namespace detail::ur
} // namespace _V1
} // namespace sycl
