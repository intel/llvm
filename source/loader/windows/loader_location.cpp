/*
 *
 * Copyright (C) 2023 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 *
 */

/* This include needs to be before Libloaderapi.h */
#include <windows.h>

#include <Libloaderapi.h>

#if __has_include(<filesystem>)
#include <filesystem>
namespace filesystem = std::filesystem;
#else
#include <experimental/filesystem>
namespace filesystem = std::experimental::filesystem;
#endif #include < filesystem>

#include "ur_loader.hpp"

#define MAX_PATH_LEN_WIN 32767

namespace fs = std::filesystem;

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

} // namespace ur_loader
