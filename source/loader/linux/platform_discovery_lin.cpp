/*
 *
 * Copyright (C) 2022-2023 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 *
 */

#include "platform_discovery.hpp"

#include "ur_util.hpp"
#include <iostream>
#include <sstream>
#include <string>

namespace loader {

static const char *knownPlatformNames[] = {
    MAKE_LIBRARY_NAME("ur_null", UR_VERSION),
};

std::vector<PlatformLibraryPath> discoverEnabledPlatforms() {
    std::vector<PlatformLibraryPath> enabledPlatforms;
    const char *altPlatforms = nullptr;

    // UR_ADAPTERS_FORCE_LOAD  is for development/debug only
    altPlatforms = getenv("UR_ADAPTERS_FORCE_LOAD");
    if (altPlatforms == nullptr) {
        for (auto path : knownPlatformNames) {
            enabledPlatforms.emplace_back(path);
        }
    } else {
        std::stringstream ss(altPlatforms);
        while (ss.good()) {
            std::string substr;
            getline(ss, substr, ',');
            enabledPlatforms.emplace_back(substr);
        }
    }
    return enabledPlatforms;
}

} // namespace loader
