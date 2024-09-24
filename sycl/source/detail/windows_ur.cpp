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

void *GetWinProcAddress(void *module, const char *funcName) {
  return (void *)GetProcAddress((HMODULE)module, funcName);
}

namespace ur {

void *getURLoaderLibrary() { return getPreloadedURLib(); }

} // namespace ur
} // namespace detail
} // namespace _V1
} // namespace sycl
