// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <detail/plugin.hpp>

namespace pi {
inline cl::sycl::detail::plugin initializeAndGetCuda() {
  auto plugins = cl::sycl::detail::pi::initialize();
  auto it = std::find_if(
      plugins.begin(), plugins.end(),
      [](cl::sycl::detail::plugin p) -> bool { return p.getBackend() == cl::sycl::backend::cuda; });
  if (it == plugins.end()) {
    throw std::runtime_error("PI CUDA plugin not found.");
  }
  return *it;
}
} // namespace pi