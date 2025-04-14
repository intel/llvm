// RUN: %clang -fpreview-breaking-changes -fsycl -fsyntax-only %s
// RUN: %clang -fsycl -fsyntax-only %s

#include <cassert>
#include <sycl/functional.hpp>
#include <type_traits>

int main() {
  const auto logicalAnd = sycl::logical_and<int>();
  const auto logicalOr = sycl::logical_or<int>();
  const auto logicalAndVoid = sycl::logical_and<void>();
  const auto logicalOrVoid = sycl::logical_or<void>();
#ifdef __INTEL_PREVIEW_BREAKING_CHANGES
  static_assert(std::is_same_v<decltype(logicalAnd(1, 2)), bool>);
  static_assert(std::is_same_v<decltype(logicalOr(1, 2)), bool>);
  static_assert(std::is_same_v<decltype(logicalAndVoid(1, 2)), bool>);
  static_assert(std::is_same_v<decltype(logicalOrVoid(1, 2)), bool>);
#else
  static_assert(std::is_same_v<decltype(logicalAnd(1, 2)), int>);
  static_assert(std::is_same_v<decltype(logicalOr(1, 2)), int>);
  static_assert(std::is_same_v<decltype(logicalAndVoid(1, 2)), bool>);
  static_assert(std::is_same_v<decltype(logicalOrVoid(1, 2)), bool>);
#endif
  return 0;
}
