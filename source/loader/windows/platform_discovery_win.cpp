/*
 *
 * Copyright (C) 2022 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 *
 */

#include "platform_discovery.h"

#include <Windows.h>

#include <cassert>
#include <cfgmgr32.h>
#include <devpkey.h>
#include <devguid.h>
#include <iostream>
#include <sstream>
#include <string>

namespace loader {

std::vector<PlatformLibraryPath> discoverEnabledPlatforms() {
    //TODO:Enable windows driver discovery
    std::vector<PlatformLibraryPath> enabledPlatforms;
    return enabledPlatforms;
}
} // namespace loader
