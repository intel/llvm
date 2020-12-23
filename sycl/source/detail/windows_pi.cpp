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
  return (void *)LoadLibraryA(PluginPath.c_str());
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
