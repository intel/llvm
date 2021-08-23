// RUN: %clangxx -fsycl -fsycl-device-only -fsyntax-only -Xclang -verify %s

#include <sycl/ext/intel/experimental/esimd.hpp>
#include <limits>
#include <utility>

using namespace sycl::ext::intel::experimental::esimd;

SYCL_ESIMD_FUNCTION bool test_simd_view_bin_ops() {
  simd<int, 16> v0 = 1;
  simd<int, 16> v1 = 2;
  auto ref0 = v0.select<8, 2>(0);
  auto ref1 = v1.select<8, 2>(0);
  ref0 += ref1;
  ref0 += 2;
  ref0 %= ref1;
  ref0 -= ref1;
  ref0 -= 2;
  ref0 *= ref1;
  ref0 *= 2;
  ref0 /= ref1;
  ref0 /= 2;
  return v0[0] == 1;
}

SYCL_ESIMD_FUNCTION bool test_simd_view_unary_ops() {
  simd<int, 16> v0 = 1;
  simd<int, 16> v1 = 2;
  auto ref0 = v0.select<8, 2>(0);
  auto ref1 = v1.select<8, 2>(0);
  ref0 <<= ref1;
  ref1 = -ref0;
  ref0 = ~ref1;
  auto mask = !ref0;
  return v1[0] == 1;
}

SYCL_ESIMD_FUNCTION bool test_simd_view_assign1() {
  simd<int, 32> v0(0, 1);
  simd<int, 16> v1(0, 1);
  v0.select<8, 1>(0) = v1.select<8, 1>(0) + v1.select<8, 1>(1);
  return v0[8] == 8 + 9;
}

SYCL_ESIMD_FUNCTION bool test_simd_view_assign2() {
  simd<int, 32> v0 = 0;
  simd<int, 16> v1 = 1;
  v0.select<8, 1>(0) = v1.select<8, 1>(0);
  return v0[0] == 1;
}

SYCL_ESIMD_FUNCTION bool test_simd_view_assign3() {
  simd<int, 64> v0 = 0;
  simd<int, 64> v1 = 1;
  auto mask = (v0.select<16, 1>(0) > v1.select<16, 1>(0));
  auto mask2 = (v0 > v1);
  simd<ushort, 64> s = 0;
  auto g4 = s.bit_cast_view<ushort, 4, 16>();
  simd<ushort, 16> val = (g4.row(2) & mask);
  simd<ushort, 16> val1 =
      (g4.row(2) & mask2.bit_cast_view<ushort, 4, 16>().row(0));
  return val[0] == 0 && val1[0] == 0;
}

// copy constructor creates the same view of the underlying data.
SYCL_ESIMD_FUNCTION void test_simd_view_copy_ctor() {
  simd<int, 16> v0 = 1;
  auto v0_view = v0.select<8, 1>(0);
  auto v0_view_copy(v0_view);
}

// move constructor transfers the same view of the underlying data.
SYCL_ESIMD_FUNCTION void test_simd_view_move_ctor() {
  simd<int, 16> v0 = 1;
  auto v0_view = v0.select<8, 1>(0);
  auto v0_view_move(std::move(v0_view));
}

// assignment operator copies the underlying data.
SYCL_ESIMD_FUNCTION void test_simd_view_copy_assign() {
  simd<int, 16> v0 = 0;
  simd<int, 16> v1 = 1;
  auto v0_view = v0.select<8, 1>(0);
  auto v1_view = v1.select<8, 1>(0);
  v0_view = v1_view;
}

// move assignment operator copies the underlying data.
SYCL_ESIMD_FUNCTION void test_simd_view_move_assign() {
  simd<int, 16> v0 = 0;
  simd<int, 16> v1 = 1;
  auto v0_view = v0.select<8, 1>(0);
  auto v1_view = v1.select<8, 1>(0);
  v0_view = std::move(v1_view);
}

// Check that the same holds for specialization with lenght==1.
SYCL_ESIMD_FUNCTION void test_simd_view_ctors_length1() {
  simd<int, 16> v = 1;
  auto v0_view = v[0];
  auto v0_view_copy(v0_view);            // copy ctor
  auto v0_view_move(std::move(v0_view)); // move ctor
  simd<int, 16> vv = 2;
  auto vv0_view = vv[0];
  v0_view_copy = vv0_view;            // copy assign
  v0_view_move = std::move(vv0_view); // move assign
}

// Check that simd_view can be passed as a by-value argument
template <class BaseTy, class RegionTy>
SYCL_ESIMD_FUNCTION void foo(simd_view<BaseTy, RegionTy> view) {}

SYCL_ESIMD_FUNCTION void bar() {
  simd<int, 16> v0 = 0;
  auto v0_view = v0.select<8, 1>(0);
  foo(v0_view);            // lvalue
  foo(v0.select<8, 1>(0)); // rvalue
}

// This test checks that simd_view_impl APIs return objects of derived class.
// I.e. they never return objects of simd_view_impl class.
void test_simd_view_impl_api_ret_types() SYCL_ESIMD_FUNCTION {
  simd<float, 4> x = 0;
  auto v1 =
      x.select<2, 1>(0); // simd_view<simd<float, 4>, region1d_t<float, 2, 1>>
  static_assert(detail::is_simd_view_v<decltype(v1)>::value, "");
  auto v2 = v1.select<1, 1>(
      0); // simd_view<simd<float, 4>, std::pair<region_base<false, float, 1, 0,
          // 1, 1>, region_base<false, float, 1, 0, 2, 1>>>
  static_assert(detail::is_simd_view_v<decltype(v1)>::value, "");

  auto v2_int = v2.bit_cast_view<int>();
  static_assert(detail::is_simd_view_v<decltype(v2_int)>::value, "");
  auto v2_int_2D = v2.bit_cast_view<int, 1, 1>();
  static_assert(detail::is_simd_view_v<decltype(v2_int_2D)>::value, "");

  auto v3 = x.select<2, 1>(2);
  auto &v4 = (v1 += v3);
  static_assert(detail::is_simd_view_v<decltype(v4)>::value, "");
  static_assert(detail::is_simd_view_v<decltype(++v4)>::value, "");
}

void test_simd_view_subscript() SYCL_ESIMD_FUNCTION {
  simd<int, 4> v = 1;
  auto vv = v.select<2, 1>(0);

  int x = vv[1];
  // expected-warning@+2 2 {{deprecated}}
  // expected-note@sycl/ext/intel/experimental/esimd/detail/simd_view_impl.hpp:* 2 {{has been explicitly marked deprecated here}}
  int y = vv(1);
}
