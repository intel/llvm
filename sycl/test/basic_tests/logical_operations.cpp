// RUN: %clang -fpreview-breaking-changes -fsycl -o - %s
// RUN: %clang -fsycl -o - %s

#include <cassert>
#include <sycl/functional.hpp>
#include <type_traits>

int main() {
  const auto logicalAnd = sycl::logical_and<int>();
  const auto logicalOr = sycl::logical_or<int>();
#ifdef __INTEL_PREVIEW_BREAKING_CHANGES
  static_assert(std::is_same_v<decltype(logicalAnd(1, 2)), bool>);
  static_assert(std::is_same_v<decltype(logicalOr(1, 2)), bool>);
#else
  static_assert(std::is_same_v<decltype(logicalAnd(1, 2)), int>);
  static_assert(std::is_same_v<decltype(logicalOr(1, 2)), int>);
#endif
  return 0;
}
