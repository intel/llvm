// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify %s
// expected-no-diagnostics

#include <sycl/ext/intel/experimental/esimd.hpp>
#include <limits>
#include <utility>

using namespace sycl::ext::intel::experimental::esimd;

bool test_simd_ctors() SYCL_ESIMD_FUNCTION {
  simd<int, 16> v0 = 1;
  simd<int, 16> v1(v0);
  simd<int, 16> v2(simd<int, 16>(0, 1));
  const simd<int, 16> v3{0, 2, 4, 6, 1, 3, 5, 7};
  return v0[0] + v1[1] + v2[2] + v3[3] == 1 + 1 + 2 + 6;
}

void test_simd_class_traits() SYCL_ESIMD_FUNCTION {
  static_assert(std::is_default_constructible<simd<int, 4>>::value,
                "type trait mismatch");
  static_assert(std::is_trivially_default_constructible<simd<int, 4>>::value,
                "type trait mismatch");
  static_assert(std::is_copy_constructible<simd<int, 4>>::value,
                "type trait mismatch");
  static_assert(!std::is_trivially_copy_constructible<simd<int, 4>>::value,
                "type trait mismatch");
  static_assert(std::is_move_constructible<simd<int, 4>>::value,
                "type trait mismatch");
  static_assert(!std::is_trivially_move_constructible<simd<int, 4>>::value,
                "type trait mismatch");
  static_assert(std::is_copy_assignable<simd<int, 4>>::value,
                "type trait mismatch");
  static_assert(std::is_trivially_copy_assignable<simd<int, 4>>::value,
                "type trait mismatch");
  static_assert(std::is_move_assignable<simd<int, 4>>::value,
                "type trait mismatch");
  static_assert(std::is_trivially_move_assignable<simd<int, 4>>::value,
                "type trait mismatch");
}

void test_conversion() SYCL_ESIMD_FUNCTION {
  simd<int, 32> v = 3;
  simd<float, 32> f = v;
  simd<char, 32> c = f;
  simd<char, 16> c1 = f.select<16, 1>(0);
  f = v + static_cast<simd<int, 32>>(c);
}

bool test_1d_select() SYCL_ESIMD_FUNCTION {
  simd<int, 32> v = 0;
  v.select<8, 1>(0) = 1;
  v.select<8, 1>(8) = 2;
  v.select<8, 1>(16) = 3;
  v.select<8, 1>(24) = 4;
  return v[0] + v[8] + v[16] + v[24] == (1 + 2 + 3 + 4);
}

bool test_simd_format() SYCL_ESIMD_FUNCTION {
  simd<int, 16> v{0, 1, 2, 3, 4, 5, 6, 7};
  auto ref1 = v.bit_cast_view<short>();
  auto ref2 = v.bit_cast_view<double>();
  auto ref3 = v.bit_cast_view<short, 8, 4>();
  return (decltype(ref1)::length == 32) && (decltype(ref2)::length == 8) &&
         (decltype(ref3)::getSizeX() == 4) && (decltype(ref3)::getSizeY() == 8);
}

bool test_simd_select() SYCL_ESIMD_FUNCTION {
  simd<int, 16> v(0, 1);
  auto ref0 = v.select<4, 2>(1);     // r{1, 3, 5, 7}
  auto ref1 = v.bit_cast_view<int, 4, 4>(); // 0,1,2,3;
                                            // 4,5,6,7;
                                            // 8,9,10,11;
                                            // 12,13,14,15
  auto ref2 = ref1.select<2, 1, 2, 2>(0, 1);
  return ref0[0] == 1 && decltype(ref2)::getSizeX() == 2 &&
         decltype(ref2)::getStrideY() == 1;
}

bool test_2d_offset() SYCL_ESIMD_FUNCTION {
  simd<int, 16> v = 0;
  auto ref = v.bit_cast_view<short, 8, 4>();
  return ref.select<2, 2, 2, 2>(2, 1).getOffsetX() == 1 &&
         ref.select<2, 2, 2, 2>(2, 1).getOffsetY() == 2;
}

bool test_simd_bin_op_promotion() SYCL_ESIMD_FUNCTION {
  simd<short, 8> v0 = std::numeric_limits<short>::max();
  simd<short, 8> v1 = 1;
  simd<int, 8> v2 = v0 + v1;
  return v2[0] == 32768;
}

bool test_simd_bin_ops() SYCL_ESIMD_FUNCTION {
  simd<int, 8> v0 = 1;
  simd<int, 8> v1 = 2;
  v0 += v1;
  v0 %= v1;
  v0 = 2 - v0;
  v0 -= v1;
  v0 -= 2;
  v0 *= v1;
  v0 *= 2;
  v0 /= v1;
  v0 /= 2;
  return v0[0] == 1;
}

bool test_simd_unary_ops() SYCL_ESIMD_FUNCTION {
  simd<int, 8> v0 = 1;
  simd<int, 8> v1 = 2;
  v0 <<= v1;
  v1 = -v0;
  v0 = ~v1;
  v1 = !v0;
  return v1[0] == 1;
}

bool test_nested_1d_select() SYCL_ESIMD_FUNCTION {
  simd<int, 8> r0(0, 1);

  auto r1 = r0.select<4, 2>(0);
  auto r2 = r1.select<2, 2>(0);
  auto r3 = r2.select<1, 0>(1);
  r3 = 37;

  return r0[4] == 37;
}

bool test_format_1d_read() SYCL_ESIMD_FUNCTION {
  simd<int, 8> r = 0x0FF00F0F;
  auto rl = r.bit_cast_view<short>();
  auto rl2 = rl.select<8, 2>(0); // 0F0F
  auto rh = r.bit_cast_view<short>();
  auto rh2 = rh.select<8, 2>(1); // 0FF0
  return rl2[0] == 0x0F0F && rh2[0] == 0x0FF0;
}

bool test_format_1d_write() SYCL_ESIMD_FUNCTION {
  simd<int, 8> r;
  auto rl = r.bit_cast_view<short>();
  auto rl2 = rl.select<8, 2>(0);
  auto rh = r.bit_cast_view<short>();
  auto rh2 = rh.select<8, 2>(1);
  rh2 = 0x0F, rl2 = 0xF0;
  return r[0] == 0x0FF0;
}

bool test_format_1d_read_write_nested() SYCL_ESIMD_FUNCTION {
  simd<int, 8> v = 0;
  auto r1 = v.bit_cast_view<short>();
  auto r11 = r1.select<8, 1>(0);
  auto r12 = r11.bit_cast_view<int>();
  auto r2 = v.bit_cast_view<short>();
  auto r21 = r2.select<8, 1>(8);
  auto r22 = r21.bit_cast_view<int>();
  r12 += 1, r22 += 2;
  return v[0] == 1 && v[4] == 2;
}

bool test_format_2d_read() SYCL_ESIMD_FUNCTION {
  simd<int, 8> v0(0, 1);
  auto r1 = v0.bit_cast_view<int, 2, 4>();
  simd<int, 4> v1 = r1.select<1, 0, 4, 1>(1, 0).read(); // second row
  return v1[0] == 4;
}

bool test_format_2d_write() SYCL_ESIMD_FUNCTION {
  simd<int, 8> v0(0, 1);
  auto r1 = v0.bit_cast_view<int, 2, 4>();
  r1.select<1, 0, 4, 1>(1, 0) = 37;
  return v0[4] == 37;
}

bool test_select_rvalue() SYCL_ESIMD_FUNCTION {
  simd<int, 8> v0(0, 1);
  v0.select<4, 2>(1).select<2, 2>(0) = 37;
  return v0[5] == 37;
}

auto test_format_1d_write_rvalue() SYCL_ESIMD_FUNCTION {
  simd<int, 8> v0 = 0x0F0F0F0F;
  v0.bit_cast_view<short>().select<8, 2>(0) = 0x0E0E;
  return v0[2] == 0x0E0E0E0E;
}

bool test_format_2d_write_rvalue() SYCL_ESIMD_FUNCTION {
  simd<int, 8> v0(0, 1);
  v0.bit_cast_view<int, 2, 4>().select<1, 0, 4, 1>(0, 0) = 37;
  return v0[3] == 37;
}

auto test_format_2d_read_rvalue() SYCL_ESIMD_FUNCTION {
  simd<int, 8> v0(0, 1);
  auto r1 = v0.bit_cast_view<int, 2, 4>()
                .select<1, 0, 4, 1>(1, 0)
                .bit_cast_view<int>()
                .select<2, 2>(1);
  return r1[0] == 5;
}

bool test_row_read_write() SYCL_ESIMD_FUNCTION {
  simd<int, 16> v0(0, 1);
  auto m = v0.bit_cast_view<int, 4, 4>();

  auto r0 = m.row(0); // 0 1 2 3
  auto r1 = m.row(1); // 4 5 6 7
  auto r2 = m.row(2); // 8 9 10 11
  auto r3 = m.row(3); // 12 13 14 15

  r0 += r2; // 8 10 12 14
  r1 += r3; // 16 18 20 22

  return r0[0] == 8 && r1[0] == 16;
}

bool test_column_read_write() SYCL_ESIMD_FUNCTION {
  simd<int, 4> v0(0, 1);
  auto m = v0.bit_cast_view<int, 2, 2>();

  auto c0 = m.column(0); // 0 2
  auto c1 = m.column(1); // 1 3

  c0 += 1; // 1 3
  c1 += 1; // 2 4

  return v0[0] == 1 && v0[3] == 4;
}

bool test_replicate() SYCL_ESIMD_FUNCTION {
  simd<int, 8> v0(0, 1);
  auto v0_rep = v0.replicate<1>();

  return v0[0] == v0_rep[0] && v0[7] == v0_rep[7];
}

bool test_replicate1() SYCL_ESIMD_FUNCTION {
  simd<int, 8> v0(0, 1);
  auto v0_rep = v0.replicate_w<4, 2>(2);

  return v0[2] == v0_rep[2] && v0[3] == v0_rep[5];
}

bool test_replicate2() SYCL_ESIMD_FUNCTION {
  simd<int, 8> v0(0, 1);
  auto v0_rep = v0.replicate_vs_w<2, 4, 2>(1);

  return v0_rep[0] == v0[1] && v0_rep[1] == v0[2] && v0_rep[2] == v0[5];
}

bool test_replicate3() SYCL_ESIMD_FUNCTION {
  simd<int, 8> v0(0, 1);
  auto v0_rep = v0.replicate_vs_w_hs<2, 4, 2, 2>(1);

  return v0_rep[0] == v0[1] && v0_rep[1] == v0[3] && v0_rep[2] == v0[5];
}

bool test_simd_iselect() SYCL_ESIMD_FUNCTION {
  simd<int, 16> v(0, 1);
  simd<ushort, 8> a(0, 2);
  auto data = v.iselect(a);
  data += 16;
  v.iupdate(a, data, 1);
  auto ref = v.select<8, 2>(0);
  return ref[0] == 16 && ref[14] == 32;
}

void test_simd_binop_honor_int_promo() SYCL_ESIMD_FUNCTION {
  simd<short, 32> a;
  simd<unsigned short, 32> b;
  simd<char, 32> c;
  simd<unsigned char, 32> d;
  static_assert(std::is_same<decltype(a + a), simd<int, 32>>::value, "");
  static_assert(std::is_same<decltype(b + b), simd<int, 32>>::value, "");
  static_assert(std::is_same<decltype(c + c), simd<int, 32>>::value, "");
  static_assert(std::is_same<decltype(d + d), simd<int, 32>>::value, "");
}
