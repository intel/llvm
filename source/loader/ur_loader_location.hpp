/*
 *
 * Copyright (C) 2023 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 *
 */

#ifndef UR_LOADER_LOCATION_HPP
#define UR_LOADER_LOCATION_HPP 1

#if __has_include(<filesystem>)
#include <filesystem>
namespace filesystem = std::filesystem;
#else
#include <experimental/filesystem>
namespace filesystem = std::experimental::filesystem;
#endif

namespace fs = std::filesystem;

namespace ur_loader {

std::optional<fs::path> getLoaderLibPath();

} // namespace ur_loader

#endif /* UR_LOADER_LOCATION_HPP */
