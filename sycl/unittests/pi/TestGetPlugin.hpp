// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "BackendString.hpp"
#include <algorithm>
#include <detail/plugin.hpp>
#include <functional>
#include <optional>

namespace pi {
inline std::optional<sycl::detail::plugin>
initializeAndGet(sycl::backend backend) {
  auto plugins = sycl::detail::pi::initialize();
  auto it = std::find_if(plugins.begin(), plugins.end(),
                         [=](sycl::detail::plugin p) -> bool {
                           return p.getBackend() == backend;
                         });
  if (it == plugins.end()) {
    std::string msg = GetBackendString(backend);
    msg += " PI plugin not found!";
    std::cerr << "Warning: " << msg << " Tests using it will be skipped.\n";
    return std::nullopt;
  }
  return std::optional<sycl::detail::plugin>(*it);
}

inline std::vector<sycl::detail::plugin> initializeAndRemoveInvalid() {
  auto plugins = sycl::detail::pi::initialize();

  auto end = std::remove_if(
      plugins.begin(), plugins.end(),
      [](const sycl::detail::plugin &plugin) -> bool {
        pi_uint32 num = 0;
        plugin.call_nocheck<sycl::detail::PiApiKind::piPlatformsGet>(0, nullptr,
                                                                     &num);

        bool removePlugin = num <= 0;

        if (removePlugin) {
          std::cerr
              << "Warning: " << GetBackendString(plugin.getBackend())
              << " PI API plugin returned no platforms via piPlatformsGet. "
                 "This plugin will be removed from testing.\n";
        }

        return removePlugin;
      });

  plugins.erase(end, plugins.end());

  return plugins;
}
} // namespace pi
