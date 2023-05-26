/*
 *
 * Copyright (C) 2023 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 *
 */

#ifndef UR_LOADER_LOCATION_HPP
#define UR_LOADER_LOCATION_HPP 1

#include "ur_filesystem_resolved.hpp"

namespace fs = filesystem;

namespace ur_loader {

std::optional<fs::path> getLoaderLibPath();

} // namespace ur_loader

#endif /* UR_LOADER_LOCATION_HPP */
