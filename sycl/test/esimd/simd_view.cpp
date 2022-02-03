// RUN: %clangxx -fsycl -fsycl-device-only -fsyntax-only -Xclang -verify %s

#include <sycl/ext/intel/experimental/esimd.hpp>
#include <limits>
#include <utility>

using namespace sycl::ext::intel::experimental::esimd;

SYCL_ESIMD_FUNCTION auto test_simd_view_bin_ops() {
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
  if (v0[0] == 1)
    return ref0 + (short)3;
  else
    return ref0 + ref1;
}

SYCL_ESIMD_FUNCTION auto test_simd_view_bitwise_ops() {
  simd<int, 16> v0 = 1;
  simd<int, 16> v1 = 2;
  auto ref0 = v0.select<8, 2>(0);
  auto ref1 = v1.select<8, 2>(0);
  simd<int, 8> v2 = (ref0 | ref1) & (ref0 | 3);
  ref0 |= 3;
  ref0 |= ref1;
  simd<int, 8> v3 = (ref0 ^ ref1) & (ref0 ^ 3);
  ref0 ^= 3;
  ref0 ^= ref1;
  simd<int, 8> v4 = (ref0 & ref1) | (ref0 & 3);
  ref0 &= 3;
  ref0 &= ref1;
  return ref0;
}

SYCL_ESIMD_FUNCTION auto test_simd_mask_view_bitwise_ops() {
  simd_mask<16> v0 = 1;
  simd_mask<16> v1 = 2;
  auto ref0 = v0.select<8, 2>(0);
  auto ref1 = v1.select<8, 2>(0);
  simd_mask<8> v2 = (ref0 | ref1) & (ref0 | 3);
  ref0 |= 3;
  ref0 |= ref1;
  simd_mask<8> v3 = (ref0 ^ ref1) & (ref0 ^ 3);
  ref0 ^= 3;
  ref0 ^= ref1;
  simd_mask<8> v4 = (ref0 & ref1) | (ref0 & 3);
  ref0 &= 3;
  ref0 &= ref1;
  return ref0;
}

SYCL_ESIMD_FUNCTION bool test_simd_view_unary_ops() {
  simd<int, 16> v0 = 1;
  simd<int, 16> v1 = 2;
  auto ref0 = v0.select<8, 2>(0);
  auto ref1 = v1.select<8, 2>(0);
  ref0 <<= ref1;
  ref1 = -ref0;
  ref0 = ~ref1;
  auto mask = !(ref0 < ref1);
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
  simd_mask<64> s = 0;
  auto g4 = s.bit_cast_view<ushort, 4, 16>();
  simd_mask<16> val = (g4.row(2) & mask);
  simd_mask<16> val1 =
      (g4.row(2) & mask2.bit_cast_view<ushort, 4, 16>().row(0));
  return val[0] == 0 && val1[0] == 0;
}

// copy constructor creates the same view of the underlying data.
SYCL_ESIMD_FUNCTION void test_simd_view_copy_ctor() {
  simd<int, 16> v0 = 1;
  auto v0_view = v0.select<8, 1>(0);
  auto v0_view_copy(v0_view);
}

// test construction from vector.
SYCL_ESIMD_FUNCTION void test_simd_view_from_vector() {
  simd<int, 16> v16 = 0;
  simd_view sv16a = v16;
  simd_view sv16b(v16);
  // expected-error@+5 {{no matching constructor for initialization of 'simd_view}}
  // expected-note@sycl/ext/intel/experimental/esimd/simd_view.hpp:* 3 {{candidate }}
  // expected-note@sycl/ext/intel/experimental/esimd/simd.hpp:* {{candidate }}
  // expected-note@sycl/ext/intel/experimental/esimd/detail/simd_obj_impl.hpp:* {{candidate }}
  // expected-note@sycl/ext/intel/experimental/esimd/simd_view.hpp:* 2 {{candidate }}
  simd_view<simd<int, 16>, region_base<false, int, 1, 1, 16, 1>> sv16c(
      (simd<int, 16>()));

  simd<int, 1> v1 = 0;
  simd_view sv1a = v1;
  simd_view sv1b(v1);
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
  static_assert(detail::is_simd_view_type_v<decltype(v1)>, "");
  auto v2 = v1.select<1, 1>(
      0); // simd_view<simd<float, 4>, std::pair<region_base<false, float, 1, 0,
          // 1, 1>, region_base<false, float, 1, 0, 2, 1>>>
  static_assert(detail::is_simd_view_type_v<decltype(v1)>, "");

  auto v2_int = v2.bit_cast_view<int>();
  static_assert(detail::is_simd_view_type_v<decltype(v2_int)>, "");
  auto v2_int_2D = v2.bit_cast_view<int, 1, 1>();
  static_assert(detail::is_simd_view_type_v<decltype(v2_int_2D)>, "");

  auto v3 = x.select<2, 1>(2);
  auto &v4 = (v1 += v3);
  static_assert(detail::is_simd_view_type_v<decltype(v4)>, "");
  static_assert(detail::is_simd_view_type_v<decltype(++v4)>, "");
}

void test_simd_view_subscript() SYCL_ESIMD_FUNCTION {
  simd<int, 4> v = 1;
  const auto vv = v.select<2, 1>(0);

  int x = vv[1];
}

void test_simd_view_writeable_subscript() SYCL_ESIMD_FUNCTION {
  simd<int, 4> v = 1;
  auto vv1 = v.select<2, 1>(0);
  auto vv2 = v.select<2, 1>(0);
  auto x = vv1 == vv2; // test relational operations
  vv1[1] = 0;          // returns writeable simd_view
  int y = vv1[1];      // nested simd_view -> int
}

// In this test `g.row(1)` return simd_view and `(g.row(1))[0]` returns
// simd_view of size 1. To avoid two implicit conversions (which is not
// allowed), simd_view_impl needs explicit definition of BINOP with a scalar.
void test_simd_view_binop_with_conv_to_scalar() SYCL_ESIMD_FUNCTION {
  simd<ushort, 64> s = 0;
  auto g = s.bit_cast_view<ushort, 4, 16>();
  auto x1 = g.row(1) - (g.row(1))[0]; // binary op
  auto x2 = (g.row(1))[0] - g.row(1); // binary op
  auto y1 = g.row(1) & (g.row(1))[0]; // bitwise op
  auto y2 = (g.row(1))[0] & g.row(1); // bitwise op
  auto z1 = g.row(1) < (g.row(1))[0]; // relational op
  auto z2 = (g.row(1))[0] < g.row(1); // relational op
}

// This code is OK. The result of bit_cast_view should be mapped
// to specialization of simd_view with length 1, so conversion
// to scalar is allowed.
void test_simd_view_len1_bitcast() SYCL_ESIMD_FUNCTION {
  simd<int, 1> s = 0;
  float f = s.bit_cast_view<float>();
}

// This test checks the same thing as previous one but for the
// nested simd_view specialization with length 1.
void test_nested_simd_view_len1_bitcast() SYCL_ESIMD_FUNCTION {
  simd<int, 2> s = 0;
  auto v = s.bit_cast_view<float>(); // generic simd_view
  float f = v[0]; // result of v[0] is a specialized nested simd_view
                  // with length 1, which then converts to a scalar.

  // checking nested views with several bitcasts.
  simd<double, 4> s2;
  ((s2[0].bit_cast_view<float>())[1].bit_cast_view<int>())[0] = 1;
}

void test_simd_view_len1_binop() SYCL_ESIMD_FUNCTION {
  simd<int, 4> s = 0;
  auto v1 = s[0];
  auto v2 = s.select<2, 1>(0);
  auto x = v1 * v2;
}

void test_simd_view_assign_op() SYCL_ESIMD_FUNCTION {
  // multiple elements
  {
#define N 4
    // simd - assign views of different element type
    simd<float, 32> v1 = 0;
    simd<short, 16> v2 = 0;
    // - region is a region type (top-level region)
    v1.select<N, 2>(0) = v2.select<N, 2>(0);
    v2.select<N, 2>(0) = v1.select<N, 2>(0);
    // - region is a std::pair (nested region)
    v1.select<8, 2>(0).select<N, 1>(1) = v2.select<8, 2>(0).select<N, 1>(1);
    v2.select<8, 2>(0).select<N, 1>(1) = v1.select<8, 2>(0).select<N, 1>(1);
    // - first region is top-level, second - nested
    v1.select<4, 2>(0) = v2.select<8, 2>(0).select<4, 1>(1);
    // - first region is nested, second - top-level
    v2.select<8, 2>(0).select<4, 1>(1) = v1.select<4, 2>(0);

    // simd_mask
    simd_mask<32> m1 = 0;
    simd_mask<16> m2 = 0;
    // - region is a region type (top-level region)
    m1.select<4, 2>(0) = m2.select<4, 2>(0);
    m2.select<4, 2>(0) = m1.select<4, 2>(0);
    // - region is a std::pair (nested region)
    m1.select<8, 2>(0).select<N, 1>(1) = m2.select<8, 2>(0).select<N, 1>(1);
    m2.select<8, 2>(0).select<N, 1>(1) = m1.select<8, 2>(0).select<N, 1>(1);
    // - first region is top-level, second - nested
    m1.select<4, 2>(0) = m2.select<8, 2>(0).select<4, 1>(1);
    // - first region is nested, second - top-level
    m2.select<8, 2>(0).select<4, 1>(1) = m1.select<4, 2>(0);
#undef N
  }
  // single element
  {
#define N 1
    // simd - assign views of different element type
    simd<float, 16> v1 = 0;
    simd<short, 8> v2 = 0;
    // - region is a region type (top-level region)
    v1.select<N, 1>(0) = v2.select<N, 1>(0);
    v2[0] = v1[0];
    v2[1] = v1.select<N, 1>(1);
    // - region is a std::pair (nested region)
    v1.select<4, 2>(0).select<N, 1>(1) = v2.select<4, 2>(0).select<N, 1>(1);
    v2.select<4, 2>(0).select<N, 1>(1) = v1.select<4, 2>(0).select<N, 1>(1);

    // simd_mask
    simd_mask<16> m1 = 0;
    simd_mask<8> m2 = 0;
    // - region is a region type (top-level region)
    m1.select<N, 1>(0) = m2.select<N, 1>(0);
    m2[0] = m1[0];
    m2[1] = m1.select<N, 1>(1);
    // - region is a std::pair (nested region)
    m1.select<4, 2>(0).select<N, 1>(1) = m2.select<4, 2>(0).select<N, 1>(1);
    m2.select<4, 2>(0)[1] = m1.select<4, 2>(0)[1];
    m2.select<4, 2>(0)[2] = m1.select<4, 2>(0).select<N, 1>(2);
#undef N
  }
}
