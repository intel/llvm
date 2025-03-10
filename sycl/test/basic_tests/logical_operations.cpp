// RUN: %clang -fpreview-breaking-changes -fsycl -o - %s
// RUN: not %clang -fpreview-breaking-changes -fsycl -DTEST_VOID_TYPES -o - %s
// RUN: %clang -fsycl -o - %s
// RUN: not %clang -fsycl -DTEST_VOID_TYPES -o - %s

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
  const auto logicalAndVoid = sycl::logical_and<void>();
  const auto logicalOrVoid = sycl::logical_or<void>();

  static_assert(std::is_same_v<decltype(logicalAndVoid(1, 2)), bool> == true);
  static_assert(std::is_same_v<decltype(logicalOrVoid(1, 2)), bool> == true);

#ifdef TEST_VOID_TYPES
  static_assert(std::is_same_v<decltype(logicalAndVoid(static_cast<void>(1),
                                                       static_cast<void>(2))),
                               bool> == true);
  static_assert(std::is_same_v<decltype(logicalOrVoid(static_cast<void>(1),
                                                      static_cast<void>(2))),
                               bool> == true);
#endif
  return 0;
}
