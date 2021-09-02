#pragma once

#include <CL/sycl.hpp>
#include <algorithm>

using namespace sycl;

bool is_supported(std::vector<memory_order> capabilities,
                  memory_order mem_order) {
  return std::find(capabilities.begin(), capabilities.end(), mem_order) !=
         capabilities.end();
}
