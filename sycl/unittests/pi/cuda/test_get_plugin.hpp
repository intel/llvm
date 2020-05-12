// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <detail/plugin.hpp>

using namespace cl::sycl;

namespace pi {
inline detail::plugin initializeAndGetCuda() {
  auto plugins = detail::pi::initialize();
  auto it = std::find_if(
      plugins.begin(), plugins.end(),
      [](detail::plugin p) -> bool { return p.getBackend() == backend::cuda; });
  if (it == plugins.end()) {
    throw std::runtime_error("PI CUDA plugin not found.");
  }
  return *it;
}
} // namespace pi