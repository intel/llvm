// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify %s
// expected-no-diagnostics

// This test checks that compiler can apply unary operators to constant simd,
// simd_mask and simd_view objects, as well as to non-constant ones.

#include <CL/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <type_traits>

using namespace sycl::ext::intel::esimd;

template <class T, int N> struct S {
  S() : val(T(0)) {}
  const simd<T, N> val;
};

template <int N> struct Smask {
  Smask() : val(0) {}
  const simd_mask<N> val;
};

template <class T, int N>
SYCL_ESIMD_FUNCTION auto test_simd_unary_plus_minus(const simd<T, N> &x,
                                                    const S<T, N> &y) {
  auto z0 = -x;     // negate constant simd parameter
  auto z1 = -y.val; // negate field of a constant simd parameter
  const auto x0 = x;
  auto z2 = +x0; // unary '+' on constant local simd object
  return z0 + z1 + z2;
}

template <class T, int N>
SYCL_ESIMD_FUNCTION auto test_simd_unary_ops(const simd<T, N> &x,
                                             const S<T, N> &y) {
  auto z0 = !x;     // logically negate constant simd parameter
  auto z1 = !y.val; // logically negate field of a constant simd parameter
  const auto x0 = x;
  auto z2 = !x0; // logically negate constant local simd object

  auto z3 = ~x;     // bitwise invert constant simd parameter
  auto z4 = ~y.val; // bitwise invert field of a constant simd parameter
  const auto x1 = x;
  auto z5 = ~x1; // bitwise invert constant local simd object

  return (z0 | z1 | z2) | ((z3 | z4 | z5) == 0);
}

template <int N>
SYCL_ESIMD_FUNCTION auto test_simd_mask_unary_ops(const simd_mask<N> &x,
                                                  const Smask<N> &y) {
  auto z0 = !x;     // logically negate constant simd_mask parameter
  auto z1 = !y.val; // logically negate field of a constant simd_mask parameter
  const auto x0 = x;
  auto z2 = !x0; // logically negate constant local simd_mask object

  auto z3 = ~x;     // bitwise invert constant simd_mask parameter
  auto z4 = ~y.val; // bitwise invert field of a constant simd_mask parameter
  const auto x1 = x;
  auto z5 = ~x1; // bitwise invert constant local simd_mask object

  return (z0 | z1 | z2) | ((z3 | z4 | z5) == 0);
}

void foo() {
  {
    simd<float, 32> x;
    S<float, 32> y;
    test_simd_unary_plus_minus(x, y);
  }
  {
    simd<char, 8> x;
    S<char, 8> y;
    test_simd_unary_plus_minus(x, y);
  }
  {
    simd<char, 8> x;
    S<char, 8> y;
    test_simd_unary_ops(x, y);
  }
  {
    simd<unsigned int, 32> x;
    S<unsigned int, 32> y;
    test_simd_unary_ops(x, y);
  }
  {
    simd_mask<32> x;
    Smask<32> y;
    test_simd_mask_unary_ops(x, y);
  }
}
