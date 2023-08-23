//===--- windows_os_utils.cpp - OS utilities implementation for Windows ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <Windows.h>
#include <direct.h>
#include <filesystem>
#include <malloc.h>
#include <shlwapi.h>
#include <sycl/detail/windows_os_utils.hpp>
#include <cassert>

namespace sycl {
inline namespace _V1 {
namespace detail {


OSModuleHandle getOSModuleHandle(const void *VirtAddr) {
  HMODULE PhModule;
  DWORD Flag = GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
               GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT;
  auto LpModuleAddr = reinterpret_cast<LPCSTR>(VirtAddr);
  if (!GetModuleHandleExA(Flag, LpModuleAddr, &PhModule)) {
    // Expect the caller to check for zero and take
    // necessary action
    return 0;
  }
  if (PhModule == GetModuleHandleA(nullptr))
    return ExeModuleHandle;
  return reinterpret_cast<OSModuleHandle>(PhModule);
}

std::filesystem::path getCurrentDSODirPath() {
  wchar_t Path[MAX_PATH];
  auto Handle = getOSModuleHandle(reinterpret_cast<void *>(&getCurrentDSODirPath));
  DWORD Ret = GetModuleFileName(
      reinterpret_cast<HMODULE>(ExeModuleHandle == Handle ? 0 : Handle),
      reinterpret_cast<LPWSTR>(&Path), sizeof(Path));
  assert(Ret < sizeof(Path) && "Path is longer than PATH_MAX?");
  assert(Ret > 0 && "GetModuleFileName failed");
  (void)Ret;

  BOOL RetCode = PathRemoveFileSpec(reinterpret_cast<LPWSTR>(&Path));
  assert(RetCode && "PathRemoveFileSpec failed");
  (void)RetCode;

  return std::filesystem::path(Path);
}


} // namespace detail
} // namespace _V1
} // namespace sycl
