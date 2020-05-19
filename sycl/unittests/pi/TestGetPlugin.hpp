// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "BackendString.hpp"
#include <algorithm>
#include <detail/plugin.hpp>
#include <functional>

namespace pi {
inline cl::sycl::detail::plugin initializeAndGet(cl::sycl::backend backend) {
  auto plugins = cl::sycl::detail::pi::initialize();
  auto it = std::find_if(plugins.begin(), plugins.end(),
                         [=](cl::sycl::detail::plugin p) -> bool {
                           return p.getBackend() == backend;
                         });
  if (it == plugins.end()) {
    std::string msg = GetBackendString(backend);
    msg += " PI plugin not found!";
    throw std::runtime_error(msg);
  }
  return *it;
}

inline std::vector<cl::sycl::detail::plugin> initializeAndRemoveInvalid() {
  auto plugins = cl::sycl::detail::pi::initialize();

  auto end = std::remove_if(
      plugins.begin(), plugins.end(),
      [](const cl::sycl::detail::plugin &plugin) -> bool {
        pi_uint32 num = 0;
        auto result =
            plugin.call_nocheck<cl::sycl::detail::PiApiKind::piPlatformsGet>(
                0, nullptr, &num);
        return num <= 0 || result != PI_SUCCESS;
      });

  plugins.erase(end, plugins.end());

  return plugins;
}
} // namespace pi