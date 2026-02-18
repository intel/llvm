#include <gtest/gtest.h>
#include <sycl/ext/oneapi/experimental/float_8bit/types.hpp>

#include <cmath>
#include <cstdint>
#include <limits>

using namespace sycl::ext::oneapi::experimental;

TEST(FP8E5M2Test, VariadicConstructorHalf) {
  fp8_e5m2<2> a(sycl::half(1.0f), sycl::half(2.0f));

  EXPECT_EQ(sizeof(a.vals), 2u);
  EXPECT_EQ(a.vals[0], 0x3C); // 1.0  -> 0b0_01111_00
  EXPECT_EQ(a.vals[1], 0x40); // 2.0  -> 0b0_10000_00

  fp8_e5m2<1> b(sycl::half(1.1f));
  EXPECT_EQ(sizeof(b.vals), 1u);
  EXPECT_EQ(b.vals[0], 0x3C); // 1.1 rounds to 1.0
}

TEST(FP8E5M2Test, VariadicConstructorBFloat16) {
  fp8_e5m2<2> a(sycl::ext::oneapi::bfloat16(1.0f),
                sycl::ext::oneapi::bfloat16(2.0f));

  EXPECT_EQ(sizeof(a.vals), 2u);
  EXPECT_EQ(a.vals[0], 0x3C);
  EXPECT_EQ(a.vals[1], 0x40);

  fp8_e5m2<1> b(sycl::ext::oneapi::bfloat16(1.1f));
  EXPECT_EQ(sizeof(b.vals), 1u);
  EXPECT_EQ(b.vals[0], 0x3C);
}

TEST(FP8E5M2Test, VariadicConstructorFloat) {
  fp8_e5m2<2> a(1.0f, 2.0f);

  EXPECT_EQ(sizeof(a.vals), 2u);
  EXPECT_EQ(a.vals[0], 0x3C);
  EXPECT_EQ(a.vals[1], 0x40);

  fp8_e5m2<1> b(1.1f);
  EXPECT_EQ(sizeof(b.vals), 1u);
  EXPECT_EQ(b.vals[0], 0x3C);
}

TEST(FP8E5M2Test, VariadicConstructorBoundaryEncodingsFloat) {
  fp8_e5m2<6> a(
      57344.0f,            // max normal  -> S.11110.11
      0.00006103515625f,   // min normal  -> S.00001.00  (2^-14)
      0.0000457763671875f, // max subnorm -> S.00000.11  (0.75 * 2^-14)
      0.0000152587890625f, // min subnorm -> S.00000.01  (2^-16)
      0.0f,                // +0
      -0.0f                // -0
  );

  EXPECT_EQ(sizeof(a.vals), 6u);

  EXPECT_EQ(a.vals[0], 0x7B); // +57344.0 -> 0b0_11110_11
  EXPECT_EQ(a.vals[1], 0x04); // +2^-14  -> 0b0_00001_00
  EXPECT_EQ(a.vals[2], 0x03); // +max subnorm -> 0b0_00000_11
  EXPECT_EQ(a.vals[3], 0x01); // +min subnorm -> 0b0_00000_01
  EXPECT_EQ(a.vals[4], 0x00); // +0 -> 0b0_00000_00
  EXPECT_EQ(a.vals[5], 0x80); // -0 -> 0b1_00000_00
}

TEST(FP8E5M2Test, VariadicConstructorNaNEncodingFloat) {
  fp8_e5m2<2> a(std::numeric_limits<float>::quiet_NaN(),
                -std::numeric_limits<float>::quiet_NaN());

  EXPECT_EQ(sizeof(a.vals), 2u);
  EXPECT_EQ(a.vals[0], 0x7F); // +NaN -> 0b0_11111_11
  EXPECT_EQ(a.vals[1], 0xFF); // -NaN -> 0b1_11111_11
}

TEST(FP8E5M2Test, IntegerConstructorToEvenFiniteAndSize) {
  fp8_e5m2<1> a0(0);
  fp8_e5m2<1> a1(1);
  fp8_e5m2<1> a2(2);
  fp8_e5m2<1> an1(-1);

  EXPECT_EQ(sizeof(a0.vals), 1u);
  EXPECT_EQ(sizeof(a1.vals), 1u);
  EXPECT_EQ(sizeof(a2.vals), 1u);
  EXPECT_EQ(sizeof(an1.vals), 1u);

  EXPECT_EQ(a0.vals[0], 0x00);  // +0
  EXPECT_EQ(a1.vals[0], 0x3C);  // +1.0 -> 0b0_01111_00
  EXPECT_EQ(a2.vals[0], 0x40);  // +2.0 -> 0b0_10000_00
  EXPECT_EQ(an1.vals[0], 0xBC); // -1.0 -> 0b1_01111_00
}

TEST(FP8E5M2Test, AssignmentOperatorToEvenFiniteAndSize) {
  fp8_e5m2<1> a(0.0f);
  EXPECT_EQ(sizeof(a.vals), 1u);
  EXPECT_EQ(a.vals[0], 0x00);

  a = 1.0f;
  EXPECT_EQ(a.vals[0], 0x3C);

  a = -2.0f;
  EXPECT_EQ(a.vals[0], 0xC0); // -2.0 -> 0b1_10000_00

  a = 0.00006103515625f; // min normal
  EXPECT_EQ(a.vals[0], 0x04);
}

TEST(FP8E5M2Test, FloatingPointConversionOperators) {
  // Floating-point operators: convert stored fp8 to the respective type.
  fp8_e5m2<1> one(1.0f);
  fp8_e5m2<1> zero_pos(0.0f);
  fp8_e5m2<1> zero_neg(-0.0f);
  fp8_e5m2<1> min_norm(0.00006103515625f);

  EXPECT_EQ(sizeof(one.vals), 1u);
  EXPECT_EQ(one.vals[0], 0x3C);

  float f1 = static_cast<float>(one);
  float fz = static_cast<float>(zero_pos);
  float fnz = static_cast<float>(zero_neg);
  float fmn = static_cast<float>(min_norm);

  EXPECT_EQ(f1, 1.0f);
  EXPECT_EQ(fz, 0.0f);
  EXPECT_EQ(fnz, 0.0f);
  EXPECT_TRUE(std::signbit(fnz));

  EXPECT_EQ(fmn, 0.00006103515625f);
}

TEST(FP8E5M2Test, IntegerConversionOperatorsTowardZero) {
  // Integer operators: convert using rounding::toward_zero.
  fp8_e5m2<1> p(1.5f);  // 1.5 exactly representable: 0b0_01111_10 (0x3E)
  fp8_e5m2<1> n(-1.5f); // 0xBE

  EXPECT_EQ(sizeof(p.vals), 1u);
  EXPECT_EQ(sizeof(n.vals), 1u);
  EXPECT_EQ(p.vals[0], 0x3E);
  EXPECT_EQ(n.vals[0], 0xBE);

  int ip = static_cast<int>(p);
  int in = static_cast<int>(n);

  EXPECT_EQ(ip, 1);  // toward zero
  EXPECT_EQ(in, -1); // toward zero
}

TEST(FP8E5M2Test, BoolOperatorZeroRules) {
  // bool operator: false iff +0 or -0; otherwise true.
  fp8_e5m2<1> zp(0.0f);
  fp8_e5m2<1> zn(-0.0f);
  fp8_e5m2<1> one(1.0f);
  fp8_e5m2<1> sub(0.0000152587890625f); // min subnormal

  EXPECT_EQ(sizeof(zp.vals), 1u);
  EXPECT_EQ(sizeof(zn.vals), 1u);
  EXPECT_EQ(sizeof(one.vals), 1u);
  EXPECT_EQ(sizeof(sub.vals), 1u);

  EXPECT_FALSE(static_cast<bool>(zp));
  EXPECT_FALSE(static_cast<bool>(zn));
  EXPECT_TRUE(static_cast<bool>(one));
  EXPECT_TRUE(static_cast<bool>(sub));
}

TEST(FP8E5M2Test, VariadicConstructorSaturatesFinite) {
  // Variadic constructors: to_even + finite saturation (CPU).
  fp8_e5m2<4> a(1.0f,
                100000.0f,  // above max normal: clamp to +57344
                -100000.0f, // clamp to -57344
                -0.0f);

  EXPECT_EQ(sizeof(a.vals), 4u);
  EXPECT_EQ(a.vals[0], 0x3C);
  EXPECT_EQ(a.vals[1], 0x7B); // +max normal
  EXPECT_EQ(a.vals[2], 0xFB); // -max normal
  EXPECT_EQ(a.vals[3], 0x80); // -0
}

TEST(FP8E5M2Test, VariadicConstructorToEvenTie) {
  // Tie case: between 1.0 (0x3C) and 1.25 (0x3D) is 1.125 exactly.
  // to_even => choose 1.0 because its LSB (fraction) is even (0).
  // Tie between 1.25 (0x3D) and 1.5 (0x3E) is 1.375 exactly => choose 1.5.
  fp8_e5m2<2> a(1.125f, -1.375f);
  EXPECT_EQ(sizeof(a.vals), 2u);
  EXPECT_EQ(a.vals[0], 0x3C);
  EXPECT_EQ(a.vals[1], 0xBE);
}

TEST(FP8E5M2Test, CArrayConstructorFloatHostToEvenFinite) {
  // Host code supports only rounding::to_even and saturation::finite.
  const float in[5] = {1.0f, 1.1f, 1.125f, 100000.0f, -0.0f};
  fp8_e5m2<5> a(in);

  EXPECT_EQ(sizeof(a.vals), 5u);
  EXPECT_EQ(a.vals[0], 0x3C); // 1.0
  EXPECT_EQ(a.vals[1], 0x3C); // 1.1 -> 1.0
  EXPECT_EQ(a.vals[2], 0x3C); // tie -> to_even => 1.0
  EXPECT_EQ(a.vals[3], 0x7B); // finite saturation => +57344
  EXPECT_EQ(a.vals[4], 0x80); // -0
}

TEST(FP8E5M2Test, CArrayConstructorDoubleToEvenFinite) {
  // Double c-array: to_even + finite saturation.
  const double in[6] = {57344.0,
                        60000.0,
                        0.00006103515625,
                        0.0000457763671875,
                        0.0000152587890625,
                        std::numeric_limits<double>::quiet_NaN()};
  fp8_e5m2<6> a(in);

  EXPECT_EQ(sizeof(a.vals), 6u);
  EXPECT_EQ(a.vals[0], 0x7B); // +57344
  EXPECT_EQ(a.vals[1], 0x7B); // 60000 -> clamp to +57344
  EXPECT_EQ(a.vals[2], 0x04); // min normal
  EXPECT_EQ(a.vals[3], 0x03); // max subnormal
  EXPECT_EQ(a.vals[4], 0x01); // min subnormal
  EXPECT_EQ(a.vals[5], 0x7F); // NaN
}

TEST(FP8E5M2Test, CArrayConstructorHalfHostToEvenFinite) {
  const sycl::half in[4] = {sycl::half(1.0f), sycl::half(2.0f),
                            sycl::half(1.125f), sycl::half(-0.0f)};
  fp8_e5m2<4> a(in);

  EXPECT_EQ(sizeof(a.vals), 4u);
  EXPECT_EQ(a.vals[0], 0x3C);
  EXPECT_EQ(a.vals[1], 0x40);
  EXPECT_EQ(a.vals[2], 0x3C); // tie -> to_even => 1.0
  EXPECT_EQ(a.vals[3], 0x80);
}

TEST(FP8E5M2Test, CArrayConstructorBFloat16HostToEvenFinite) {
  const sycl::ext::oneapi::bfloat16 in[4] = {
      sycl::ext::oneapi::bfloat16(1.0f), sycl::ext::oneapi::bfloat16(2.0f),
      sycl::ext::oneapi::bfloat16(1.125f), sycl::ext::oneapi::bfloat16(-0.0f)};
  fp8_e5m2<4> a(in);

  EXPECT_EQ(sizeof(a.vals), 4u);
  EXPECT_EQ(a.vals[0], 0x3C);
  EXPECT_EQ(a.vals[1], 0x40);
  EXPECT_EQ(a.vals[2], 0x3C); // tie -> to_even => 1.0
  EXPECT_EQ(a.vals[3], 0x80);
}

TEST(FP8E5M2Test, MarrayConstructorAndOperatorsHostAllN) {
  // marray constructors/operators: host supports all N.
  sycl::marray<float, 8> in = {
      1.0f, 2.0f, 0.0f, -0.0f, 57344.0f, 100000.0f, 0.0000152587890625f, -1.5f};
  fp8_e5m2<8> a(in);

  EXPECT_EQ(sizeof(a.vals), 8u);
  EXPECT_EQ(a.vals[0], 0x3C);
  EXPECT_EQ(a.vals[1], 0x40);
  EXPECT_EQ(a.vals[2], 0x00);
  EXPECT_EQ(a.vals[3], 0x80);
  EXPECT_EQ(a.vals[4], 0x7B);
  EXPECT_EQ(a.vals[5], 0x7B); // finite saturation
  EXPECT_EQ(a.vals[6], 0x01);
  EXPECT_EQ(a.vals[7], 0xBE); // -1.5

  sycl::marray<float, 8> out = static_cast<sycl::marray<float, 8>>(a);
  EXPECT_EQ(out[0], 1.0f);
  EXPECT_EQ(out[1], 2.0f);
  EXPECT_EQ(out[2], 0.0f);
  EXPECT_EQ(out[3], 0.0f);
  EXPECT_TRUE(std::signbit(out[3]));
  EXPECT_EQ(out[4], 57344.0f);
  EXPECT_EQ(out[5], 57344.0f);
  EXPECT_EQ(out[6], 0.0000152587890625f);
  EXPECT_EQ(out[7], -1.5f);
}

TEST(FP8E5M2Test, MarrayConstructorHalfBFloat16Double) {
  sycl::marray<sycl::half, 4> hvals = {sycl::half(1.0f), sycl::half(2.0f),
                                       sycl::half(57344.0f), sycl::half(-0.0f)};
  sycl::marray<sycl::ext::oneapi::bfloat16, 4> bvals = {
      sycl::ext::oneapi::bfloat16(1.0f), sycl::ext::oneapi::bfloat16(2.0f),
      sycl::ext::oneapi::bfloat16(0.0000152587890625f),
      sycl::ext::oneapi::bfloat16(-0.0f)};
  sycl::marray<double, 4> dvals = {1.0, 2.0, 57344.0, -0.0};

  fp8_e5m2<4> ah(hvals);
  fp8_e5m2<4> ab(bvals);
  fp8_e5m2<4> ad(dvals);

  EXPECT_EQ(sizeof(ah.vals), 4u);
  EXPECT_EQ(sizeof(ab.vals), 4u);
  EXPECT_EQ(sizeof(ad.vals), 4u);

  EXPECT_EQ(ah.vals[0], 0x3C);
  EXPECT_EQ(ah.vals[1], 0x40);
  EXPECT_EQ(ah.vals[2], 0x7B);
  EXPECT_EQ(ah.vals[3], 0x80);

  EXPECT_EQ(ab.vals[0], 0x3C);
  EXPECT_EQ(ab.vals[1], 0x40);
  EXPECT_EQ(ab.vals[2], 0x01);
  EXPECT_EQ(ab.vals[3], 0x80);

  EXPECT_EQ(ad.vals[0], 0x3C);
  EXPECT_EQ(ad.vals[1], 0x40);
  EXPECT_EQ(ad.vals[2], 0x7B);
  EXPECT_EQ(ad.vals[3], 0x80);
}

TEST(FP8E5M2Test, FloatingPointConversionOperatorsMoreTypes) {
  fp8_e5m2<1> a(1.0f);
  fp8_e5m2<1> b(0.00006103515625f);
  fp8_e5m2<1> nanv(std::numeric_limits<float>::quiet_NaN());

  EXPECT_EQ(sizeof(a.vals), 1u);
  EXPECT_EQ(sizeof(b.vals), 1u);
  EXPECT_EQ(sizeof(nanv.vals), 1u);

  double da = static_cast<double>(a);
  sycl::half ha = static_cast<sycl::half>(a);
  sycl::ext::oneapi::bfloat16 ba = static_cast<sycl::ext::oneapi::bfloat16>(a);

  EXPECT_EQ(da, 1.0);
  EXPECT_EQ(static_cast<float>(ha), 1.0f);
  EXPECT_EQ(static_cast<float>(ba), 1.0f);

  EXPECT_EQ(static_cast<float>(b), 0.00006103515625f);

  float fn = static_cast<float>(nanv);
  EXPECT_TRUE(std::isnan(fn));
}

TEST(FP8E5M2Test, MarrayConversionOperatorsHalfBFloat16) {
  fp8_e5m2<2> a(1.0f, -0.0f);

  EXPECT_EQ(sizeof(a.vals), 2u);
  EXPECT_EQ(a.vals[0], 0x3C);
  EXPECT_EQ(a.vals[1], 0x80);

  sycl::marray<sycl::half, 2> ho = static_cast<sycl::marray<sycl::half, 2>>(a);
  sycl::marray<sycl::ext::oneapi::bfloat16, 2> bo =
      static_cast<sycl::marray<sycl::ext::oneapi::bfloat16, 2>>(a);

  EXPECT_EQ(static_cast<float>(ho[0]), 1.0f);
  EXPECT_EQ(static_cast<float>(ho[1]), 0.0f);
  EXPECT_TRUE(std::signbit(static_cast<float>(ho[1])));

  EXPECT_EQ(static_cast<float>(bo[0]), 1.0f);
  EXPECT_EQ(static_cast<float>(bo[1]), 0.0f);
  EXPECT_TRUE(std::signbit(static_cast<float>(bo[1])));
}

TEST(FP8E5M2Test, IntegerConversionOperatorsMultipleWidthsTowardZero) {
  fp8_e5m2<1> p(1.5f);
  fp8_e5m2<1> n(-1.5f);

  std::int32_t i32p = static_cast<std::int32_t>(p);
  std::int32_t i32n = static_cast<std::int32_t>(n);
  std::int64_t i64p = static_cast<std::int64_t>(p);
  std::int64_t i64n = static_cast<std::int64_t>(n);

  EXPECT_EQ(i32p, 1);
  EXPECT_EQ(i32n, -1);
  EXPECT_EQ(i64p, 1);
  EXPECT_EQ(i64n, -1);
}

TEST(FP8E5M2Test, IntegerConversionOperatorsAllTypesTowardZero) {
  fp8_e5m2<1> p(1.5f);
  fp8_e5m2<1> n(-1.5f);

  EXPECT_EQ(sizeof(p.vals), 1u);
  EXPECT_EQ(sizeof(n.vals), 1u);
  EXPECT_EQ(p.vals[0], 0x3E);
  EXPECT_EQ(n.vals[0], 0xBE);

  EXPECT_EQ(static_cast<char>(p), 1);
  EXPECT_EQ(static_cast<signed char>(n), -1);
  EXPECT_EQ(static_cast<short>(n), -1);
  EXPECT_EQ(static_cast<int>(n), -1);
  EXPECT_EQ(static_cast<long>(n), -1);
  EXPECT_EQ(static_cast<long long>(n), -1);
  EXPECT_EQ(static_cast<unsigned char>(p), 1u);
  EXPECT_EQ(static_cast<unsigned short>(p), 1u);
  EXPECT_EQ(static_cast<unsigned int>(p), 1u);
  EXPECT_EQ(static_cast<unsigned long>(p), 1u);
  EXPECT_EQ(static_cast<unsigned long long>(p), 1u);
}

TEST(FP8E5M2Test, VariadicConstructorHalfBoundaryEncodings) {
  fp8_e5m2<4> a(sycl::half(57344.0f), sycl::half(0.00006103515625f),
                sycl::half(0.0000152587890625f), sycl::half(-0.0f));

  EXPECT_EQ(sizeof(a.vals), 4u);
  EXPECT_EQ(a.vals[0], 0x7B); // +max normal
  EXPECT_EQ(a.vals[1], 0x04); // min normal
  EXPECT_EQ(a.vals[2], 0x01); // min subnormal
  EXPECT_EQ(a.vals[3], 0x80); // -0
}

TEST(FP8E5M2Test, VariadicConstructorBFloat16BoundaryEncodings) {
  fp8_e5m2<4> a(sycl::ext::oneapi::bfloat16(1.0f),
                sycl::ext::oneapi::bfloat16(2.0f),
                sycl::ext::oneapi::bfloat16(0.0000152587890625f),
                sycl::ext::oneapi::bfloat16(-0.0f));

  EXPECT_EQ(sizeof(a.vals), 4u);
  EXPECT_EQ(a.vals[0], 0x3C);
  EXPECT_EQ(a.vals[1], 0x40);
  EXPECT_EQ(a.vals[2], 0x01);
  EXPECT_EQ(a.vals[3], 0x80);
}

TEST(FP8E5M2Test, VariadicConstructorDoubleBoundaryEncodingsAndSaturation) {
  fp8_e5m2<5> a(57344.0, 60000.0, 0.0000457763671875, 0.0000152587890625,
                -100000.0);

  EXPECT_EQ(sizeof(a.vals), 5u);
  EXPECT_EQ(a.vals[0], 0x7B); // +57344
  EXPECT_EQ(a.vals[1], 0x7B); // clamp to +57344 (finite saturation)
  EXPECT_EQ(a.vals[2], 0x03); // max subnormal
  EXPECT_EQ(a.vals[3], 0x01); // min subnormal
  EXPECT_EQ(a.vals[4], 0xFB); // clamp to -57344
}

TEST(FP8E5M2Test, IntegerConstructorsAllTypes) {
  fp8_e5m2<1> s(static_cast<short>(1));
  fp8_e5m2<1> i(static_cast<int>(2));
  fp8_e5m2<1> l(static_cast<long>(3));
  fp8_e5m2<1> ll(static_cast<long long>(-1));
  fp8_e5m2<1> us(static_cast<unsigned short>(1));
  fp8_e5m2<1> ui(static_cast<unsigned int>(2));
  fp8_e5m2<1> ul(static_cast<unsigned long>(3));
  fp8_e5m2<1> ull(static_cast<unsigned long long>(4));

  EXPECT_EQ(sizeof(s.vals), 1u);
  EXPECT_EQ(sizeof(i.vals), 1u);
  EXPECT_EQ(sizeof(l.vals), 1u);
  EXPECT_EQ(sizeof(ll.vals), 1u);
  EXPECT_EQ(sizeof(us.vals), 1u);
  EXPECT_EQ(sizeof(ui.vals), 1u);
  EXPECT_EQ(sizeof(ul.vals), 1u);
  EXPECT_EQ(sizeof(ull.vals), 1u);

  EXPECT_EQ(s.vals[0], 0x3C);
  EXPECT_EQ(i.vals[0], 0x40);
  EXPECT_EQ(l.vals[0], 0x42);  // 3.0 -> 0b0_10000_10
  EXPECT_EQ(ll.vals[0], 0xBC); // -1.0
  EXPECT_EQ(us.vals[0], 0x3C);
  EXPECT_EQ(ui.vals[0], 0x40);
  EXPECT_EQ(ul.vals[0], 0x42);  // 3.0
  EXPECT_EQ(ull.vals[0], 0x44); // 4.0 -> 0b0_10001_00
}

TEST(FP8E5M2Test, AssignmentOperatorsAllTypes) {
  fp8_e5m2<1> a(0.0f);

  EXPECT_EQ(sizeof(a.vals), 1u);
  EXPECT_EQ(a.vals[0], 0x00);

  a = sycl::half(1.0f);
  EXPECT_EQ(a.vals[0], 0x3C);

  a = sycl::ext::oneapi::bfloat16(2.0f);
  EXPECT_EQ(a.vals[0], 0x40);

  a = 3.0f;
  EXPECT_EQ(a.vals[0], 0x42); // 3.0

  a = 4.0;
  EXPECT_EQ(a.vals[0], 0x44); // 4.0

  a = static_cast<short>(-1);
  EXPECT_EQ(a.vals[0], 0xBC);

  a = static_cast<int>(2);
  EXPECT_EQ(a.vals[0], 0x40);

  a = static_cast<long>(1);
  EXPECT_EQ(a.vals[0], 0x3C);

  a = static_cast<long long>(-2);
  EXPECT_EQ(a.vals[0], 0xC0);

  a = static_cast<unsigned short>(1);
  EXPECT_EQ(a.vals[0], 0x3C);

  a = static_cast<unsigned int>(2);
  EXPECT_EQ(a.vals[0], 0x40);

  a = static_cast<unsigned long>(3);
  EXPECT_EQ(a.vals[0], 0x42);

  a = static_cast<unsigned long long>(4);
  EXPECT_EQ(a.vals[0], 0x44);
}

TEST(FP8E5M2Test, BoolOperatorWithNaN) {
  float pz = 0.0f;
  fp8_e5m2<1> zp(pz);
  float zv = -0.0f;
  fp8_e5m2<1> zn(zv);
  float nv = {std::numeric_limits<float>::quiet_NaN()};
  fp8_e5m2<1> nanv(nv);

  EXPECT_EQ(sizeof(zp.vals), 1u);
  EXPECT_EQ(sizeof(zn.vals), 1u);
  EXPECT_EQ(sizeof(nanv.vals), 1u);

  EXPECT_FALSE(static_cast<bool>(zp));
  EXPECT_FALSE(static_cast<bool>(zn));
  EXPECT_TRUE(static_cast<bool>(nanv)); // not +0 or -0
  EXPECT_EQ(nanv.vals[0], 0x7F);        // NaN encoding remains S.11111.11
}
