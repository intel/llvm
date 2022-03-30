// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <CL/sycl/platform.hpp>

#include <algorithm>
#include <functional>
#include <vector>

namespace pi {
inline std::vector<cl::sycl::platform> getPlatformsWithName(const char *name) {
  std::vector<cl::sycl::platform> platforms =
      cl::sycl::platform::get_platforms();

  // Remove platforms that have no devices or doesn't contain the name
  auto end =
      std::remove_if(platforms.begin(), platforms.end(),
                     [=](const cl::sycl::platform &platform) -> bool {
                       const std::string platformName =
                           platform.get_info<cl::sycl::info::platform::name>();
                       return platformName.find(name) == std::string::npos ||
                              platform.get_devices().size() == 0;
                     });
  platforms.erase(end, platforms.end());

  return platforms;
}
} // namespace pi