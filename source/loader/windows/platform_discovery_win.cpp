/*
 *
 * Copyright (C) 2022 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 *
 */

#include "platform_discovery.h"
#include "ur_util.h"

#include <Windows.h>

#include <cassert>
#include <cfgmgr32.h>
#include <devpkey.h>
#include <devguid.h>
#include <iostream>
#include <sstream>
#include <string>

namespace loader {

static const char *knownAdaptersNames[] = {
  MAKE_LIBRARY_NAME("ur_null", UR_VERSION),
};

std::vector<PlatformLibraryPath> discoverEnabledPlatforms() {
    std::vector<PlatformLibraryPath> enabledPlatforms;

    for (auto libName : knownAdaptersNames) {
      enabledPlatforms.emplace_back(libName);
    }

    return enabledPlatforms;
}
} // namespace loader
