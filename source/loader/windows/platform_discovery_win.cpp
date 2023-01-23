/*
 *
 * Copyright (C) 2022-2023 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 *
 */

#include "platform_discovery.hpp"
#include "ur_util.hpp"

#include <Windows.h>

#include <cassert>
#include <cfgmgr32.h>
#include <devguid.h>
#include <devpkey.h>
#include <iostream>
#include <sstream>
#include <string>

namespace loader {

static const char *knownAdaptersNames[] = {
    MAKE_LIBRARY_NAME("ur_null", UR_VERSION),
};

std::vector<PlatformLibraryPath> discoverEnabledPlatforms() {
    std::vector<PlatformLibraryPath> enabledPlatforms;

    // UR_ADAPTERS_FORCE_LOAD  is for development/debug only
    const char *altPlatforms = getenv("UR_ADAPTERS_FORCE_LOAD");

    if (altPlatforms == nullptr) {
        for (auto libName : knownAdaptersNames) {
            enabledPlatforms.emplace_back(libName);
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
