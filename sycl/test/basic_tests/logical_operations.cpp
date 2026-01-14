// RUN: %clang -fsycl -fsyntax-only %s

#include <cassert>
#include <sycl/functional.hpp>
#include <type_traits>

int main() {
  const auto logicalAnd = sycl::logical_and<int>();
  const auto logicalOr = sycl::logical_or<int>();
  const auto logicalAndVoid = sycl::logical_and<void>();
  const auto logicalOrVoid = sycl::logical_or<void>();
  static_assert(std::is_same_v<decltype(logicalAnd(1, 2)), bool>);
  static_assert(std::is_same_v<decltype(logicalOr(1, 2)), bool>);
  static_assert(std::is_same_v<decltype(logicalAndVoid(1, 2)), bool>);
  static_assert(std::is_same_v<decltype(logicalOrVoid(1, 2)), bool>);
  return 0;
}
