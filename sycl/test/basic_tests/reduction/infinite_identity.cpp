/*
    -ffast-math implies -fno-honor-infinities. In this case, the known_identity
   for sycl::minimum<T> cannot be inifinity (which will be 0), but must instead
    be the max<T> .  Otherwise, reducing sycl::minimum<T>
    over {4.0f, 1.0f, 3.0f, 2.0f}  will return 0.0 instead of 1.0.
*/

// RUN: %clangxx -fsycl -fsyntax-only %s
// RUN: %clangxx -fsycl -fsyntax-only %s -ffast-math
// RUN: %clangxx -fsycl -fsyntax-only %s -fno-fast-math
// RUN: %clangxx -fsycl -fsyntax-only %s -fno-fast-math -fno-honor-infinities -fno-honor-nans
// RUN: %clangxx -fsycl -fsyntax-only %s -ffast-math -fhonor-infinities -fhonor-nans

#include <sycl/ext/oneapi/bfloat16.hpp>
#include <sycl/sycl.hpp>

#include <cassert>
#include <limits>

template <typename OperandT> void test_known_identity_min() {
  constexpr OperandT identity =
      sycl::known_identity_v<sycl::minimum<OperandT>, OperandT>;
#if defined(__FINITE_MATH_ONLY__) && (__FINITE_MATH_ONLY__ == 1)
  constexpr OperandT expected = std::numeric_limits<OperandT>::max();
#else
  constexpr OperandT expected = std::numeric_limits<OperandT>::infinity();
#endif

  static_assert(identity == expected,
                "Known identity for sycl::minimum<T> is incorrect");
}

template <typename OperandT> void test_known_identity_max() {
  constexpr OperandT identity =
      sycl::known_identity_v<sycl::maximum<OperandT>, OperandT>;
#if defined(__FINITE_MATH_ONLY__) && (__FINITE_MATH_ONLY__ == 1)
  constexpr OperandT expected = std::numeric_limits<OperandT>::lowest();
#else
  // negative infinity
  constexpr OperandT expected = -std::numeric_limits<OperandT>::infinity();
#endif

  static_assert(identity == expected,
                "Known identity for sycl::maximum<T> is incorrect");
}

int main() {
  test_known_identity_min<float>();
  test_known_identity_min<double>();
  test_known_identity_min<sycl::half>();

  test_known_identity_max<float>();
  test_known_identity_max<double>();
  test_known_identity_max<sycl::half>();

  // bfloat16 seems to be missing constexpr == which is used above.
  // commenting out until fixed.
  // test_known_identity_min<sycl::ext::oneapi::bfloat16>();
  // test_known_identity_max<sycl::ext::oneapi::bfloat16>();

  return 0;
}
