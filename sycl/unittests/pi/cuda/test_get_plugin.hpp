#pragma once

#include <CL/sycl.hpp>
#include <CL/sycl/detail/pi.hpp>
#include <detail/plugin.hpp>
#include <algorithm>

using namespace cl::sycl;

namespace pi {
inline detail::plugin initializeAndGetCuda() {
  auto plugins = detail::pi::initialize();
  auto it = std::find_if(plugins.begin(), plugins.end(),
             [](detail::plugin p) -> bool { return p.getBackend() == backend::cuda; });
  if(it == plugins.end()) {
      throw std::runtime_error("PI CUDA plugin not found.");
  }
  return *it;
}
}