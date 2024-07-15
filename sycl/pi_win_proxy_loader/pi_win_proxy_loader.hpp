//==------------ pi_win_proxy_loader.hpp - SYCL standard header file ------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===--------------------------------------------------------------------===//

#pragma once

#ifdef _WIN32
#include <filesystem>
#include <string>

__declspec(dllexport) void *getPreloadedPlugin(
    const std::filesystem::path &PluginPath);
#endif
