#pragma once

#include <algorithm>

#include <sycl/detail/core.hpp>

#include <sycl/atomic_ref.hpp>

using namespace sycl;

bool is_supported(std::vector<memory_order> capabilities,
                  memory_order mem_order) {
  return std::find(capabilities.begin(), capabilities.end(), mem_order) !=
         capabilities.end();
}
