//==---------- test.cpp - Test piapi library -------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <pi/plugin.hpp>

#include <iostream>

// Configuration hooks
namespace pi {
namespace config {
TraceLevel trace_level_mask() { return {TraceLevel::PI_TRACE_ALL}; }
pi::backend *backend() { return nullptr; }
pi::device_filter_list *device_filter_list() { return nullptr; }
} // namespace config
} // namespace pi

std::string platform_info_string(pi::plugin &p, pi_platform plt,
                                 pi_platform_info param) {
  std::size_t paramSize = 0;
  p.call<PiApiKind::piPlatformGetInfo>(plt, param, 0, nullptr, &paramSize);

  if (paramSize == 0) {
    std::cout << "Empty platform name" << std::endl;
    return "";
  }

  std::vector<char> platformNameBuffer;
  platformNameBuffer.resize(paramSize);

  p.call<PiApiKind::piPlatformGetInfo>(plt, param, paramSize,
                                       platformNameBuffer.data(), nullptr);

  return platformNameBuffer.data();
}

int main() {
  auto plugins = pi::initialize();

  std::cout << "Num plugins: " << plugins.size() << std::endl;

  for (auto &p : plugins) {
    std::cout << "=== plugin" << std::endl;

    pi_uint32 numPlatforms = 0;
    p.call<PiApiKind::piPlatformsGet>(0, nullptr, &numPlatforms);

    std::cout << "Num platforms: " << numPlatforms << std::endl;

    if (numPlatforms == 0) {
      continue;
    }

    std::vector<pi_platform> platforms;
    platforms.resize(numPlatforms);
    p.call<PiApiKind::piPlatformsGet>(numPlatforms, platforms.data(), nullptr);

    for (auto &plt : platforms) {
      std::cout << "Platform name: "
                << platform_info_string(p, plt, PI_PLATFORM_INFO_NAME)
                << std::endl;

      std::cout << "Platform version: "
                << platform_info_string(p, plt, PI_PLATFORM_INFO_VERSION)
                << std::endl;
    }
  }

  return 0;
}
