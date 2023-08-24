//===---- windows_os_utils.hpp - OS utilities for Windows header file  ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <Windows.h>
#include <direct.h>
#include <filesystem>
#include <malloc.h>
#include <shlwapi.h>

namespace sycl {
inline namespace _V1 {
namespace detail {

std::filesystem::path getCurrentDSODirPath();

using OSModuleHandle = intptr_t;
static constexpr OSModuleHandle ExeModuleHandle = -1;

OSModuleHandle getOSModuleHandle(const void *VirtAddr);

} // namespace detail
} // namespace _V1
} // namespace sycl
