// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "BackendString.hpp"
#include <algorithm>
#include <detail/plugin.hpp>
#include <functional>

namespace pi {
inline __sycl_internal::__v1::detail::plugin initializeAndGet(__sycl_internal::__v1::backend backend) {
  auto plugins = __sycl_internal::__v1::detail::pi::initialize();
  auto it = std::find_if(plugins.begin(), plugins.end(),
                         [=](__sycl_internal::__v1::detail::plugin p) -> bool {
                           return p.getBackend() == backend;
                         });
  if (it == plugins.end()) {
    std::string msg = GetBackendString(backend);
    msg += " PI plugin not found!";
    throw std::runtime_error(msg);
  }
  return *it;
}

inline std::vector<__sycl_internal::__v1::detail::plugin> initializeAndRemoveInvalid() {
  auto plugins = __sycl_internal::__v1::detail::pi::initialize();

  auto end = std::remove_if(
      plugins.begin(), plugins.end(),
      [](const __sycl_internal::__v1::detail::plugin &plugin) -> bool {
        pi_uint32 num = 0;
        plugin.call_nocheck<__sycl_internal::__v1::detail::PiApiKind::piPlatformsGet>(
            0, nullptr, &num);

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