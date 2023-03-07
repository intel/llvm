/*
 *
 * Copyright (C) 2022-2023 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 *
 */

#include "platform_discovery.hpp"
#include "ur_util.hpp"

#include <array>

namespace loader {

static constexpr std::array<const char *, 1> knownPlatformNames{
    MAKE_LIBRARY_NAME("ur_adapter_level_zero", "0")};

static auto getKnownPlatforms() {
    std::vector<PlatformLibraryPath> enabledPlatforms;

    for (auto path : knownPlatformNames) {
        enabledPlatforms.emplace_back(path);
    }
    return enabledPlatforms;
}

std::vector<PlatformLibraryPath> discoverEnabledPlatforms() {
    std::optional<std::string> altPlatforms;

    // UR_ADAPTERS_FORCE_LOAD  is for development/debug only
    try {
        altPlatforms = ur_getenv("UR_ADAPTERS_FORCE_LOAD");
    } catch (const std::invalid_argument &e) {
        std::cerr << e.what();
        return getKnownPlatforms();
    }
    if (!altPlatforms) {
        return getKnownPlatforms();
    }

    std::vector<PlatformLibraryPath> enabledPlatforms;
    std::stringstream ss(*altPlatforms);
    while (ss.good()) {
        std::string substr;
        getline(ss, substr, ',');
        enabledPlatforms.emplace_back(substr);
    }
    return enabledPlatforms;
}

} // namespace loader
