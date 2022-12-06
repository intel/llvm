/*
 *
 * Copyright (C) 2022 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 *
 */

#ifndef UR_LOADER_PLATFORM_DISCOVERY_H
#define UR_LOADER_PLATFORM_DISCOVERY_H 1

#include <string>
#include <vector>

namespace loader {

using PlatformLibraryPath = std::string;

std::vector<PlatformLibraryPath> discoverEnabledPlatforms();

} // namespace loader

#endif /* UR_LOADER_PLATFORM_DISCOVERY_H */
