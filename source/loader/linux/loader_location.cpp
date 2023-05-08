/*
 *
 * Copyright (C) 2023 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 *
 */

#include <dlfcn.h>
#include <optional>

#if __has_include(<filesystem>)
#include <filesystem>
namespace filesystem = std::filesystem;
#else
#include <experimental/filesystem>
namespace filesystem = std::experimental::filesystem;
#endif

#include "ur_loader.hpp"

namespace fs = std::filesystem;

namespace ur_loader {

std::optional<fs::path> getLoaderLibPath() {
    Dl_info info;
    if (dladdr((void *)getLoaderLibPath, &info)) {
        auto libPath = fs::path(info.dli_fname);
        if (fs::exists(libPath)) {
            return fs::absolute(libPath).parent_path();
        }
    }

    return std::nullopt;
}

} // namespace ur_loader
