// RUN: %clang -D__INTEL_PREVIEW_BREAKING_CHANGES -fsycl -o - %s
// RUN: %clang -fsycl -o - %s

#include <cassert>
#include <sycl/functional.hpp>
#include <type_traits>

int main() {
  const auto logicalAnd = sycl::logical_and<int>();
  const auto logicalOr = sycl::logical_or<int>();
#ifdef __INTEL_PREVIEW_BREAKING_CHANGES
  static_assert(std::is_same_v<decltype(logicalAnd(1, 2)), bool> == true);
  static_assert(std::is_same_v<decltype(logicalOr(1, 2)), bool> == true);
#else
  static_assert(std::is_same_v<decltype(logicalAnd(1, 2)), int> == true);
  static_assert(std::is_same_v<decltype(logicalOr(1, 2)), int> == true);
#endif
  return 0;
}
