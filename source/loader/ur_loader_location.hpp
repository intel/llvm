/*
 *
 * Copyright (C) 2023 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
