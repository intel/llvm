// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify %s
// expected-no-diagnostics

// Check that is_esimd_arithmetic_type works as expected on simd and simd_view
// types.

#include <sycl/ext/intel/esimd.hpp>

using namespace sycl::ext::intel::esimd;

constexpr bool foo0() {
  return __ESIMD_DNS::is_esimd_arithmetic_type_v<simd<int, 8>>;
}

constexpr bool foo1() {
  return __ESIMD_DNS::is_esimd_arithmetic_type_v<
      simd_view<simd<int, 8>, region1d_t<int, 8, 1>>>;
}

SYCL_EXTERNAL void foo() SYCL_ESIMD_FUNCTION {
  static_assert(!foo0() && !foo1(), "");
}

// This models original user code where compilation failure occurred.
// Similar code exists in operators.cpp, but somehow presence of '^='
// operation (commented out below) hid the problem and compiler did not fail.
SYCL_EXTERNAL auto foo2(simd<char, 8> x, simd<char, 8> x1,
                        simd<char, 8> y) SYCL_ESIMD_FUNCTION {
  //{ auto k = x1 ^= x1; }
  { auto v = x ^ y; }
}
