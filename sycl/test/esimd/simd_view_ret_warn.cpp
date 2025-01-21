// RUN: %clangxx -fsycl -fsycl-device-only -fsyntax-only -Xclang -verify %s

#include <sycl/ext/intel/esimd.hpp>
using namespace sycl::ext::intel::esimd;

// Explicitly returning simd_view for local objects is wrong,
// and it should be programmers fault, similar to string_view.
// However, sometimes we could return simd_view from a function
// implicitly. This test checks that users will see a warning in such situation.
simd_view<simd<float, 4>, region1d_t<float, 1, 1>> f1(simd<float, 4> x) {
  // expected-warning@+1 {{address of stack memory associated with parameter 'x' returned}}
  return x[0];
}

simd_view<simd<float, 4>, region1d_t<float, 2, 1>> f2(simd<float, 4> x) {
  // expected-warning@+1 {{address of stack memory associated with parameter 'x' returned}}
  return x.select<2, 1>(0);
}

auto f3(simd<float, 4> x) {
  // expected-warning@+1 {{address of stack memory associated with parameter 'x' returned}}
  return x.bit_cast_view<int>();
}

auto f4(simd<float, 4> x) {
  // expected-warning@+1 {{address of stack memory associated with parameter 'x' returned}}
  return x.bit_cast_view<int, 2, 2>();
}
