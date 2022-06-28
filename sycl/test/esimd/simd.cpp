// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify %s
// expected-no-diagnostics

#include <limits>
#include <sycl/ext/intel/esimd.hpp>
#include <utility>

using namespace sycl::ext::intel::esimd;

template <class T> bool test_simd_ctors() SYCL_ESIMD_FUNCTION {
  simd<T, 16> v0 = 1;
  simd<T, 16> v1(v0);
  simd<T, 16> v2(simd<T, 16>(0, 1));
  const simd<T, 16> v3({0, 2, 4, 6, 1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7});
  return v0[0] + v1[1] + v2[2] + v3[3] == 1 + 1 + 2 + 6;
}

template bool test_simd_ctors<int>() SYCL_ESIMD_FUNCTION;
template bool test_simd_ctors<sycl::half>() SYCL_ESIMD_FUNCTION;

template <class T> void test_simd_class_traits() SYCL_ESIMD_FUNCTION {
  static_assert(std::is_default_constructible<simd<T, 4>>::value,
                "type trait mismatch");
  static_assert(std::is_trivially_default_constructible<simd<T, 4>>::value,
                "type trait mismatch");
  static_assert(std::is_copy_constructible<simd<T, 4>>::value,
                "type trait mismatch");
  static_assert(!std::is_trivially_copy_constructible<simd<T, 4>>::value,
                "type trait mismatch");
  static_assert(std::is_move_constructible<simd<T, 4>>::value,
                "type trait mismatch");
  static_assert(!std::is_trivially_move_constructible<simd<T, 4>>::value,
                "type trait mismatch");
  static_assert(std::is_copy_assignable<simd<T, 4>>::value,
                "type trait mismatch");
  static_assert(std::is_trivially_copy_assignable<simd<T, 4>>::value,
                "type trait mismatch");
  static_assert(std::is_move_assignable<simd<T, 4>>::value,
                "type trait mismatch");
  static_assert(std::is_trivially_move_assignable<simd<T, 4>>::value,
                "type trait mismatch");
}

template void test_simd_class_traits<int>() SYCL_ESIMD_FUNCTION;
template void test_simd_class_traits<sycl::half>() SYCL_ESIMD_FUNCTION;

void test_conversion() SYCL_ESIMD_FUNCTION {
  simd<int, 32> v = 3;
  simd<float, 32> f = v;
  simd<char, 32> c = f;
  simd<sycl::half, 32> h = c;
  simd<char, 16> c1 = h.template select<16, 1>(0);
  c.template select<32, 1>(0) = f;
  h.template select<7, 1>(3) =
      v.template select<22, 1>(0).template select<7, 3>(1);
  f = v + static_cast<simd<int, 32>>(c);
}

template <class T> bool test_1d_select() SYCL_ESIMD_FUNCTION {
  simd<T, 32> v = 0;
  v.template select<8, 1>(0) = 1;
  v.template select<8, 1>(8) = 2;
  v.template select<8, 1>(16) = 3;
  v.template select<8, 1>(24) = 4;
  return v[0] + v[8] + v[16] + v[24] == (1 + 2 + 3 + 4);
}

template bool test_1d_select<int>() SYCL_ESIMD_FUNCTION;
template bool test_1d_select<sycl::half>() SYCL_ESIMD_FUNCTION;

template <class T1, class T2, class T3>
bool test_simd_format() SYCL_ESIMD_FUNCTION {
  simd<T1, 16> v({0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7});
  auto ref1 = v.template bit_cast_view<T2>();
  auto ref2 = v.template bit_cast_view<T3>();
  auto ref3 = v.template bit_cast_view<T2, 8, 4>();
  return (decltype(ref1)::length == 32) && (decltype(ref2)::length == 8) &&
         (decltype(ref3)::getSizeX() == 4) && (decltype(ref3)::getSizeY() == 8);
}

template bool test_simd_format<int, short, double>() SYCL_ESIMD_FUNCTION;
template bool
test_simd_format<uint32_t, sycl::half, uint64_t>() SYCL_ESIMD_FUNCTION;

template <class T1, class T2> bool test_simd_select(T1 a) SYCL_ESIMD_FUNCTION {
  {
    simd<T1, 32> f = a;
    simd<T2, 32> c1 = 2;
    c1.template select<16, 1>(0) = f.template select<16, 1>(0);
    c1.template select<16, 1>(0).template select<16, 1>(0) =
        f.template select<16, 1>(0).template select<16, 1>(0);
  }
  {
    simd<T1, 16> v(0, 1);
    auto ref0 = v.template select<4, 2>(1);           // r{1, 3, 5, 7}
    auto ref1 = v.template bit_cast_view<T1, 4, 4>(); // 0,1,2,3;
                                                      // 4,5,6,7;
                                                      // 8,9,10,11;
                                                      // 12,13,14,15
    auto ref2 = ref1.template select<2, 1, 2, 2>(0, 1);
    return (ref0[0] == 1) && (decltype(ref2)::getSizeX() == 2) &&
           (decltype(ref2)::getStrideY() == 1);
  }
  return false;
}

template bool test_simd_select<float, char>(float) SYCL_ESIMD_FUNCTION;
template bool
    test_simd_select<uint64_t, sycl::half>(uint64_t) SYCL_ESIMD_FUNCTION;

template <class T1, class T2> bool test_2d_offset() SYCL_ESIMD_FUNCTION {
  simd<T1, 16> v = 0;
  auto ref = v.template bit_cast_view<T2, 8, 4>();
  return ref.template select<2, 2, 2, 2>(2, 1).getOffsetX() == 1 &&
         ref.template select<2, 2, 2, 2>(2, 1).getOffsetY() == 2;
}

template bool test_2d_offset<int, short>() SYCL_ESIMD_FUNCTION;
template bool test_2d_offset<sycl::half, uint8_t>() SYCL_ESIMD_FUNCTION;

template <class T1, class T2>
bool test_simd_bin_op_promotion() SYCL_ESIMD_FUNCTION {
  simd<T2, 8> v0 = std::numeric_limits<T2>::max();
  simd<T2, 8> v1 = 1;
  simd<T1, 8> v2 = v0 + v1;
  return v2[0] == 32768;
}

template bool test_simd_bin_op_promotion<int, short>() SYCL_ESIMD_FUNCTION;
template bool
test_simd_bin_op_promotion<sycl::half, uint64_t>() SYCL_ESIMD_FUNCTION;

template <class T> bool test_simd_bin_ops() SYCL_ESIMD_FUNCTION {
  simd<T, 8> v0 = 1;
  simd<T, 8> v1 = 2;
  v0 += v1;
  if constexpr (std::is_integral_v<T>)
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

template bool test_simd_bin_ops<int>() SYCL_ESIMD_FUNCTION;
template bool test_simd_bin_ops<sycl::half>() SYCL_ESIMD_FUNCTION;

template <class T> bool test_simd_unary_ops() SYCL_ESIMD_FUNCTION {
  simd<T, 8> v0 = 1;
  simd<T, 8> v1 = 2;
  if constexpr (std::is_integral_v<T>)
    v0 <<= v1;
  v1 = -v0;
  if constexpr (std::is_integral_v<T>)
    v0 = ~v1;
  return v1[0] == 1;
}

template bool test_simd_unary_ops<int>() SYCL_ESIMD_FUNCTION;
template bool test_simd_unary_ops<sycl::half>() SYCL_ESIMD_FUNCTION;

template <class T> bool test_nested_1d_select() SYCL_ESIMD_FUNCTION {
  simd<T, 8> r0(0, 1);

  auto r1 = r0.template select<4, 2>(0);
  auto r2 = r1.template select<2, 2>(0);
  auto r3 = r2.template select<1, 0>(1);
  r3 = 37;

  return r0[4] == 37;
}

template bool test_nested_1d_select<int>() SYCL_ESIMD_FUNCTION;
template bool test_nested_1d_select<sycl::half>() SYCL_ESIMD_FUNCTION;

template <class T1, class T2> bool test_format_1d_read() SYCL_ESIMD_FUNCTION {
  simd<T1, 8> r = 0x0FF00F0F;
  auto rl = r.template bit_cast_view<T2>();
  auto rl2 = rl.template select<8, 2>(0); // 0F0F
  auto rh = r.template bit_cast_view<T2>();
  auto rh2 = rh.template select<8, 2>(1); // 0FF0
  return rl2[0] == 0x0F0F && rh2[0] == 0x0FF0;
}

template bool test_format_1d_read<int, short>() SYCL_ESIMD_FUNCTION;
template bool test_format_1d_read<sycl::half, uint8_t>() SYCL_ESIMD_FUNCTION;

template <class T1, class T2> bool test_format_1d_write() SYCL_ESIMD_FUNCTION {
  simd<T1, 8> r;
  auto rl = r.template bit_cast_view<T2>();
  auto rl2 = rl.template select<8, 2>(0);
  auto rh = r.template bit_cast_view<T2>();
  auto rh2 = rh.template select<8, 2>(1);
  rh2 = 0x0F, rl2 = 0xF0;
  return r[0] == 0x0FF0;
}

template bool test_format_1d_write<int, short>() SYCL_ESIMD_FUNCTION;
template bool test_format_1d_write<sycl::half, uint64_t>() SYCL_ESIMD_FUNCTION;

template <class T1, class T2>
bool test_format_1d_read_write_nested() SYCL_ESIMD_FUNCTION {
  simd<T1, 32> v = 0;
  auto r1 = v.template bit_cast_view<T2>();
  auto r11 = r1.template select<8, 1>(0);
  auto r12 = r11.template bit_cast_view<T1>();
  auto r2 = v.template bit_cast_view<T2>();
  auto r21 = r2.template select<8, 1>(8);
  auto r22 = r21.template bit_cast_view<T1>();
  r12 += 1, r22 += 2;
  return v[0] == 1 && v[4] == 2;
}

template bool
test_format_1d_read_write_nested<int, short>() SYCL_ESIMD_FUNCTION;
template bool
test_format_1d_read_write_nested<sycl::half, uint64_t>() SYCL_ESIMD_FUNCTION;

template <class T> bool test_format_2d_read() SYCL_ESIMD_FUNCTION {
  simd<T, 8> v0(0, 1);
  auto r1 = v0.template bit_cast_view<T, 2, 4>();
  simd<T, 4> v1 = r1.template select<1, 0, 4, 1>(1, 0).read(); // second row
  return v1[0] == 4;
}

template bool test_format_2d_read<int>() SYCL_ESIMD_FUNCTION;
template bool test_format_2d_read<sycl::half>() SYCL_ESIMD_FUNCTION;

template <class T> bool test_format_2d_write() SYCL_ESIMD_FUNCTION {
  simd<T, 8> v0(0, 1);
  auto r1 = v0.template bit_cast_view<T, 2, 4>();
  r1.template select<1, 0, 4, 1>(1, 0) = 37;
  return v0[4] == 37;
}

template bool test_format_2d_write<int>() SYCL_ESIMD_FUNCTION;
template bool test_format_2d_write<sycl::half>() SYCL_ESIMD_FUNCTION;

template <class T> bool test_select_rvalue() SYCL_ESIMD_FUNCTION {
  simd<T, 8> v0(0, 1);
  v0.template select<4, 2>(1).template select<2, 2>(0) = 37;
  return v0[5] == 37;
}

template bool test_select_rvalue<int>() SYCL_ESIMD_FUNCTION;
template bool test_select_rvalue<sycl::half>() SYCL_ESIMD_FUNCTION;

template <class T> bool test_format_1d_write_rvalue() SYCL_ESIMD_FUNCTION {
  simd<T, 8> v0 = 0x0F0F0F0F;
  v0.template bit_cast_view<short>().template select<8, 2>(0) = 0x0E0E;
  return v0[2] == 0x0E0E0E0E;
}

template bool test_format_1d_write_rvalue<int>() SYCL_ESIMD_FUNCTION;
template bool test_format_1d_write_rvalue<sycl::half>() SYCL_ESIMD_FUNCTION;

template <class T> bool test_format_2d_write_rvalue() SYCL_ESIMD_FUNCTION {
  simd<T, 8> v0(0, 1);
  v0.template bit_cast_view<T, 2, 4>().template select<1, 0, 4, 1>(0, 0) = 37;
  return v0[3] == 37;
}

template bool test_format_2d_write_rvalue<int>() SYCL_ESIMD_FUNCTION;
template bool test_format_2d_write_rvalue<sycl::half>() SYCL_ESIMD_FUNCTION;

template <class T> bool test_format_2d_read_rvalue() SYCL_ESIMD_FUNCTION {
  simd<T, 8> v0(0, 1);
  auto r1 = v0.template bit_cast_view<T, 2, 4>()
                .template select<1, 0, 4, 1>(1, 0)
                .template bit_cast_view<T>()
                .template select<2, 2>(1);
  return r1[0] == 5;
}

template bool test_format_2d_read_rvalue<int>() SYCL_ESIMD_FUNCTION;
template bool test_format_2d_read_rvalue<sycl::half>() SYCL_ESIMD_FUNCTION;

template <class T> bool test_row_read_write() SYCL_ESIMD_FUNCTION {
  simd<T, 16> v0(0, 1);
  auto m = v0.template bit_cast_view<T, 4, 4>();

  auto r0 = m.row(0); // 0 1 2 3
  auto r1 = m.row(1); // 4 5 6 7
  auto r2 = m.row(2); // 8 9 10 11
  auto r3 = m.row(3); // 12 13 14 15

  r0 += r2; // 8 10 12 14
  r1 += r3; // 16 18 20 22

  return r0[0] == 8 && r1[0] == 16;
}

template bool test_row_read_write<int>() SYCL_ESIMD_FUNCTION;
template bool test_row_read_write<sycl::half>() SYCL_ESIMD_FUNCTION;

template <class T> bool test_column_read_write() SYCL_ESIMD_FUNCTION {
  simd<T, 4> v0(0, 1);
  auto m = v0.template bit_cast_view<T, 2, 2>();

  auto c0 = m.column(0); // 0 2
  auto c1 = m.column(1); // 1 3

  c0 += 1; // 1 3
  c1 += 1; // 2 4

  return v0[0] == 1 && v0[3] == 4;
}

template bool test_column_read_write<int>() SYCL_ESIMD_FUNCTION;
template bool test_column_read_write<sycl::half>() SYCL_ESIMD_FUNCTION;

template <class T> bool test_replicate() SYCL_ESIMD_FUNCTION {
  simd<T, 8> v0(0, 1);
  auto v0_rep = v0.template replicate<1>();

  return v0[0] == v0_rep[0] && v0[7] == v0_rep[7];
}

template bool test_replicate<int>() SYCL_ESIMD_FUNCTION;
template bool test_replicate<sycl::half>() SYCL_ESIMD_FUNCTION;

template <class T> bool test_replicate1() SYCL_ESIMD_FUNCTION {
  simd<T, 8> v0(0, 1);
  auto v0_rep = v0.template replicate_w<4, 2>(2);

  return v0[2] == v0_rep[2] && v0[3] == v0_rep[5];
}

template bool test_replicate1<int>() SYCL_ESIMD_FUNCTION;
template bool test_replicate1<sycl::half>() SYCL_ESIMD_FUNCTION;

template <class T> bool test_replicate2() SYCL_ESIMD_FUNCTION {
  simd<T, 8> v0(0, 1);
  auto v0_rep = v0.template replicate_vs_w<2, 4, 2>(1);

  return v0_rep[0] == v0[1] && v0_rep[1] == v0[2] && v0_rep[2] == v0[5];
}

template bool test_replicate2<int>() SYCL_ESIMD_FUNCTION;
template bool test_replicate2<sycl::half>() SYCL_ESIMD_FUNCTION;

template <class T> bool test_replicate3() SYCL_ESIMD_FUNCTION {
  simd<T, 8> v0(0, 1);
  auto v0_rep = v0.template replicate_vs_w_hs<2, 4, 2, 2>(1);

  return v0_rep[0] == v0[1] && v0_rep[1] == v0[3] && v0_rep[2] == v0[5];
}

template bool test_replicate3<int>() SYCL_ESIMD_FUNCTION;
template bool test_replicate3<sycl::half>() SYCL_ESIMD_FUNCTION;

template <class T1, class T2> bool test_simd_iselect() SYCL_ESIMD_FUNCTION {
  simd<T1, 16> v(0, 1);
  simd<T2, 8> a(0, 2);
  auto data = v.iselect(a);
  data += 16;
  v.template iupdate(a, data, simd_mask<8>(1));
  auto ref = v.template select<8, 2>(0);
  return ref[0] == 16 && ref[14] == 32;
}

template bool test_simd_iselect<int, ushort>() SYCL_ESIMD_FUNCTION;
template bool test_simd_iselect<sycl::half, ushort>() SYCL_ESIMD_FUNCTION;

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
