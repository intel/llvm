#pragma once

#include <algorithm>
#include <sycl/sycl.hpp>

using namespace sycl;

bool is_supported_order(const std::vector<memory_order> &capabilities,
                        memory_order mem_order) {
  return std::find(capabilities.begin(), capabilities.end(), mem_order) !=
         capabilities.end();
}

bool is_supported_scope(const std::vector<memory_scope> &capabilities,
                        memory_scope mem_scope) {
  return std::find(capabilities.begin(), capabilities.end(), mem_scope) !=
         capabilities.end();
}
