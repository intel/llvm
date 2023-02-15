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

#include <array>
#include <cassert>
#include <cfgmgr32.h>
#include <devguid.h>
#include <devpkey.h>
#include <iostream>
#include <sstream>
#include <string>

namespace loader {

static constexpr std::array<char *, 0> knownAdaptersNames{};

std::vector<PlatformLibraryPath> discoverEnabledPlatforms() {
    std::vector<PlatformLibraryPath> enabledPlatforms;

    // UR_ADAPTERS_FORCE_LOAD  is for development/debug only
    char *altPlatforms = nullptr;
    _dupenv_s(&altPlatforms, NULL, "UR_ADAPTERS_FORCE_LOAD");

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
        free(altPlatforms);
    }
    return enabledPlatforms;
}

} // namespace loader
