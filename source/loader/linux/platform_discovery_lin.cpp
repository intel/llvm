/*
 *
 * Copyright (C) 2022 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 *
 */

#include "platform_discovery.h"

#include "ur_util.h"
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

  // UR_ENABLE_ALT_DRIVERS is for development/debug only
  altPlatforms = getenv("UR_ENABLE_ALT_DRIVERS");
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
