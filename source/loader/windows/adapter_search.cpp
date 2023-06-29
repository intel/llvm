/*
 *
 * Copyright (C) 2023 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* This include needs to be before Libloaderapi.h */
#include <windows.h>

#include <Libloaderapi.h>

#include "ur_filesystem_resolved.hpp"
#include "ur_loader.hpp"

#define MAX_PATH_LEN_WIN 32767

namespace fs = filesystem;

namespace ur_loader {

std::optional<fs::path> getLoaderLibPath() {
    char pathStr[MAX_PATH_LEN_WIN];
    if (GetModuleFileNameA(nullptr, pathStr, MAX_PATH_LEN_WIN)) {
        auto libPath = fs::path(pathStr);
        if (fs::exists(libPath)) {
            return fs::absolute(libPath).parent_path();
        }
    }

    return std::nullopt;
}

std::optional<fs::path> getAdapterNameAsPath(std::string adapterName) {
    return std::nullopt;
}

} // namespace ur_loader
