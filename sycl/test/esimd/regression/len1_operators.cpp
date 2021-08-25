// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify %s
// expected-no-diagnostics

// This test checks that compiler does not report 'ambiguous operator' error
// when compiling simd or simd_view operations with lenth = 1.

#include <sycl/ext/intel/experimental/esimd.hpp>

#include <cstdint>

using namespace sycl::ext::intel::experimental::esimd;

template <typename T1, typename T2>
void test_esimd_ops(simd<T1, 1> a, T2 b, T2 w) SYCL_ESIMD_FUNCTION {
  T2 c1 = a[0] * w + b;
  T2 c2 = a[0] * T2{2} + b;
  T2 c3 = T2{2} * a[0] + b;
  T2 d1 = a[0] ^ w;
  T2 d2 = a[0] ^ T2 { 2 };
  T2 d3 = T2{2} ^ a[0];
  auto e1 = a[0] < w;
  auto e2 = a[0] < T2{2};
  auto e3 = T2{2} < a[0];
  simd<T1, 1> z{4};
  auto f1 = a[0] ^ z;
  auto f2 = z ^ a[0];
  auto f3 = a[0] < z;
  auto f4 = z < a[0];
}

void foo() SYCL_ESIMD_FUNCTION {
  test_esimd_ops(simd<uint32_t, 1>(3), (int)1, (int)9);
  test_esimd_ops(simd<int, 1>(3), (uint32_t)1, (uint32_t)9);
  test_esimd_ops(simd<uint16_t, 1>(3), 1, 9);
  test_esimd_ops(simd<int16_t, 1>(3), 1, 9);
}
