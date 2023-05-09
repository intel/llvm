// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT

#ifndef UR_FILESYSTEM_RESOLVED_HPP
#define UR_FILESYSTEM_RESOLVED_HPP 1

#if __has_include(<filesystem>)
#include <filesystem>
namespace filesystem = std::filesystem;
#else
#include <experimental/filesystem>
namespace filesystem = std::experimental::filesystem;
#endif

#endif /* UR_FILESYSTEM_RESOLVED_HPP */
