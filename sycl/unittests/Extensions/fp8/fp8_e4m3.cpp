#include <gtest/gtest.h>
#include <sycl/ext/oneapi/experimental/float_8bit/types.hpp>

#include <cmath>
#include <cstdint>
#include <limits>

using namespace sycl::ext::oneapi::experimental;

TEST(FP8E4M3Test, VariadicConstructorHalf) {
  fp8_e4m3<2> a(sycl::half(1.0f), sycl::half(2.0f));

  EXPECT_EQ(sizeof(a.vals), 2u);
  EXPECT_EQ(a.vals[0], 0x38); // 1.0  -> 0b0_0111_000
  EXPECT_EQ(a.vals[1], 0x40); // 2.0  -> 0b0_1000_000

  fp8_e4m3<1> b(sycl::half(1.1f));
  EXPECT_EQ(sizeof(b.vals), 1u);
  EXPECT_EQ(b.vals[0], 0x39); // 1.1 rounds to 1.125 -> frac=1
}

TEST(FP8E4M3Test, VariadicConstructorBFloat16) {
  fp8_e4m3<2> a(sycl::ext::oneapi::bfloat16(1.0f), sycl::ext::oneapi::bfloat16(2.0f));

  EXPECT_EQ(sizeof(a.vals), 2u);
  EXPECT_EQ(a.vals[0], 0x38);
  EXPECT_EQ(a.vals[1], 0x40);

  fp8_e4m3<1> b(sycl::ext::oneapi::bfloat16(1.1f));
  EXPECT_EQ(sizeof(b.vals), 1u);
  EXPECT_EQ(b.vals[0], 0x39);
}

TEST(FP8E4M3Test, VariadicConstructorFloat) {
  fp8_e4m3<2> a(1.0f, 2.0f);

  EXPECT_EQ(sizeof(a.vals), 2u);
  EXPECT_EQ(a.vals[0], 0x38);
  EXPECT_EQ(a.vals[1], 0x40);

  fp8_e4m3<1> b(1.1f);
  EXPECT_EQ(sizeof(b.vals), 1u);
  EXPECT_EQ(b.vals[0], 0x39);
}

TEST(FP8E4M3Test, VariadicBoundaryEncodingsFloat) {
  // CPU host path: variadic constructors use rounding::to_even and saturation::finite.
  fp8_e4m3<6> a(
      448.0f,        // max normal  -> S.1111.110
      0.015625f,     // min normal  -> S.0001.000  (2^-6)
      0.013671875f,  // max subnorm -> S.0000.111  (0.875 * 2^-6)
      0.001953125f,  // min subnorm -> S.0000.001  (2^-9)
      0.0f,          // +0
      -0.0f          // -0
  );

  EXPECT_EQ(sizeof(a.vals), 6u);

  EXPECT_EQ(a.vals[0], 0x7E); // +448.0  -> 0b0_1111_110
  EXPECT_EQ(a.vals[1], 0x08); // +2^-6   -> 0b0_0001_000
  EXPECT_EQ(a.vals[2], 0x07); // +max subnorm -> 0b0_0000_111
  EXPECT_EQ(a.vals[3], 0x01); // +min subnorm -> 0b0_0000_001
  EXPECT_EQ(a.vals[4], 0x00); // +0 -> 0b0_0000_000
  EXPECT_EQ(a.vals[5], 0x80); // -0 -> 0b1_0000_000
}

TEST(FP8E4M3Test, VariadicNaNEncodingFloat) {
  // NaN is encoded as S.1111.111; sign is permitted.
  fp8_e4m3<2> a(std::numeric_limits<float>::quiet_NaN(),
               -std::numeric_limits<float>::quiet_NaN());

  EXPECT_EQ(sizeof(a.vals), 2u);
  EXPECT_EQ(a.vals[0], 0x7F); // +NaN -> 0b0_1111_111
  EXPECT_EQ(a.vals[1], 0xFF); // -NaN -> 0b1_1111_111
}

TEST(FP8E4M3Test, IntegerToEvenFiniteAndSize) {
  // Integer constructors: to_even + finite saturation (CPU).
  fp8_e4m3<1> a0(0);
  fp8_e4m3<1> a1(1);
  fp8_e4m3<1> a2(2);
  fp8_e4m3<1> an1(-1);

  EXPECT_EQ(sizeof(a0.vals), 1u);
  EXPECT_EQ(sizeof(a1.vals), 1u);
  EXPECT_EQ(sizeof(a2.vals), 1u);
  EXPECT_EQ(sizeof(an1.vals), 1u);

  EXPECT_EQ(a0.vals[0], 0x00); // +0
  EXPECT_EQ(a1.vals[0], 0x38); // +1.0 -> 0b0_0111_000
  EXPECT_EQ(a2.vals[0], 0x40); // +2.0 -> 0b0_1000_000
  EXPECT_EQ(an1.vals[0], 0xB8); // -1.0 -> sign set: 0b1_0111_000
}

TEST(FP8E4M3Test, AssignmentOperatorToEvenFiniteAndSize) {
  // operator= from scalar: to_even + finite saturation (CPU).
  fp8_e4m3<1> a(0.0f);
  EXPECT_EQ(sizeof(a.vals), 1u);
  EXPECT_EQ(a.vals[0], 0x00);

  a = 1.0f;
  EXPECT_EQ(a.vals[0], 0x38);

  a = -2.0f;
  EXPECT_EQ(a.vals[0], 0xC0); // -2.0 -> 0b1_1000_000

  a = 0.015625f; // min normal
  EXPECT_EQ(a.vals[0], 0x08);
}

TEST(FP8E4M3Test, FloatingPointConversionOperators) {
  // Floating-point operators: convert stored fp8 to the respective type.
  fp8_e4m3<1> one(1.0f);
  fp8_e4m3<1> zero_pos(0.0f);
  fp8_e4m3<1> zero_neg(-0.0f);
  fp8_e4m3<1> min_norm(0.015625f);

  EXPECT_EQ(sizeof(one.vals), 1u);
  EXPECT_EQ(one.vals[0], 0x38);

  float f1 = static_cast<float>(one);
  float fz = static_cast<float>(zero_pos);
  float fnz = static_cast<float>(zero_neg);
  float fmn = static_cast<float>(min_norm);

  EXPECT_EQ(f1, 1.0f);
  EXPECT_EQ(fz, 0.0f);
  // -0.0 compares equal to +0.0; check signbit to validate negative zero survives.
  EXPECT_EQ(fnz, 0.0f);
  EXPECT_TRUE(std::signbit(fnz));

  EXPECT_EQ(fmn, 0.015625f);
}

TEST(FP8E4M3Test, IntegerConversionOperatorsTowardZero) {
  // Integer operators: convert using rounding::toward_zero.
  fp8_e4m3<1> p(1.5f);   // 1.5 exactly representable: 0b0_0111_100 (0x3C)
  fp8_e4m3<1> n(-1.5f);  // 0xBC

  EXPECT_EQ(sizeof(p.vals), 1u);
  EXPECT_EQ(sizeof(n.vals), 1u);
  EXPECT_EQ(p.vals[0], 0x3C);
  EXPECT_EQ(n.vals[0], 0xBC);

  int ip = static_cast<int>(p);
  int in = static_cast<int>(n);

  EXPECT_EQ(ip, 1);   // toward zero
  EXPECT_EQ(in, -1);  // toward zero
}

TEST(FP8E4M3Test, BoolOperatorZeroRules) {
  // bool operator: false iff +0 or -0; otherwise true.
  fp8_e4m3<1> zp(0.0f);
  fp8_e4m3<1> zn(-0.0f);
  fp8_e4m3<1> one(1.0f);
  fp8_e4m3<1> sub(0.001953125f); // min subnormal

  EXPECT_EQ(sizeof(zp.vals), 1u);
  EXPECT_EQ(sizeof(zn.vals), 1u);
  EXPECT_EQ(sizeof(one.vals), 1u);
  EXPECT_EQ(sizeof(sub.vals), 1u);

  EXPECT_FALSE(static_cast<bool>(zp));
  EXPECT_FALSE(static_cast<bool>(zn));
  EXPECT_TRUE(static_cast<bool>(one));
  EXPECT_TRUE(static_cast<bool>(sub));
}

TEST(FP8E4M3Test, VariadicSaturatesFinite) {
  // Variadic constructors: to_even + finite saturation (CPU).
  fp8_e4m3<4> a(
      1.0f,
      1000.0f,   // above max normal: clamp to +448
      -1000.0f,  // clamp to -448
      -0.0f);

  EXPECT_EQ(sizeof(a.vals), 4u);
  EXPECT_EQ(a.vals[0], 0x38);
  EXPECT_EQ(a.vals[1], 0x7E); // +max normal
  EXPECT_EQ(a.vals[2], 0xFE); // -max normal
  EXPECT_EQ(a.vals[3], 0x80); // -0
}

TEST(FP8E4M3Test, VariadicToEvenTie) {
  // Tie case: between 1.0 (0x38) and 1.125 (0x39) is 1.0625 exactly.
  // to_even => choose 1.0 because its LSB (fraction) is even (0).
  fp8_e4m3<2> a(1.0625f, -1.0625f);
  EXPECT_EQ(sizeof(a.vals), 2u);
  EXPECT_EQ(a.vals[0], 0x38);
  EXPECT_EQ(a.vals[1], 0xB8);
}

TEST(FP8E4M3Test, CArrayFloatHostToEvenFinite) {
  // Host code supports only rounding::to_even and saturation::finite.
  const float in[5] = {1.0f, 1.1f, 1.0625f, 1000.0f, -0.0f};
  fp8_e4m3<5> a(in);

  EXPECT_EQ(sizeof(a.vals), 5u);
  EXPECT_EQ(a.vals[0], 0x38); // 1.0
  EXPECT_EQ(a.vals[1], 0x39); // 1.1 -> 1.125
  EXPECT_EQ(a.vals[2], 0x38); // tie -> to_even => 1.0
  EXPECT_EQ(a.vals[3], 0x7E); // finite saturation => +448
  EXPECT_EQ(a.vals[4], 0x80); // -0
}

TEST(FP8E4M3Test, CArrayDoubleToEvenFinite) {
  // Double c-array: to_even + finite saturation.
  const double in[6] = {448.0, 449.0, 0.015625, 0.013671875, 0.001953125, std::numeric_limits<double>::quiet_NaN()};
  fp8_e4m3<6> a(in);

  EXPECT_EQ(sizeof(a.vals), 6u);
  EXPECT_EQ(a.vals[0], 0x7E); // +448
  EXPECT_EQ(a.vals[1], 0x7E); // 449 -> clamp to +448
  EXPECT_EQ(a.vals[2], 0x08); // min normal
  EXPECT_EQ(a.vals[3], 0x07); // max subnormal
  EXPECT_EQ(a.vals[4], 0x01); // min subnormal
  EXPECT_EQ(a.vals[5], 0x7F); // NaN
}

TEST(FP8E4M3Test, CArrayHalfHostToEvenFinite) {
  // Host code supports only rounding::to_even and saturation::finite.
  const sycl::half in[6] = {sycl::half(448.0f),       sycl::half(449.0f),
                            sycl::half(0.015625f),    sycl::half(0.013671875f),
                            sycl::half(0.001953125f), sycl::half(-0.0f)};
  fp8_e4m3<6> a(in);

  EXPECT_EQ(sizeof(a.vals), 6u);
  EXPECT_EQ(a.vals[0], 0x7E); // +448
  EXPECT_EQ(a.vals[1], 0x7E); // 449 -> clamp to +448
  EXPECT_EQ(a.vals[2], 0x08); // min normal
  EXPECT_EQ(a.vals[3], 0x07); // max subnormal
  EXPECT_EQ(a.vals[4], 0x01); // min subnormal
  EXPECT_EQ(a.vals[5], 0x80); // -0
}

TEST(FP8E4M3Test, CArrayBFloat16HostToEvenFinite) {
  // Host code supports only rounding::to_even and saturation::finite.
  const sycl::ext::oneapi::bfloat16 in[6] = {
      sycl::ext::oneapi::bfloat16(448.0f),
      sycl::ext::oneapi::bfloat16(449.0f),
      sycl::ext::oneapi::bfloat16(0.015625f),
      sycl::ext::oneapi::bfloat16(0.013671875f),
      sycl::ext::oneapi::bfloat16(0.001953125f),
      sycl::ext::oneapi::bfloat16(-0.0f)};
  fp8_e4m3<6> a(in);

  EXPECT_EQ(sizeof(a.vals), 6u);
  EXPECT_EQ(a.vals[0], 0x7E); // +448
  EXPECT_EQ(a.vals[1], 0x7E); // 449 -> clamp to +448
  EXPECT_EQ(a.vals[2], 0x08); // min normal
  EXPECT_EQ(a.vals[3], 0x07); // max subnormal
  EXPECT_EQ(a.vals[4], 0x01); // min subnormal
  EXPECT_EQ(a.vals[5], 0x80); // -0
}

TEST(FP8E4M3Test, MarrayAndOperatorsHostAllN) {
  // marray constructors/operators: host supports all N.
  sycl::marray<float, 8> in = {1.0f, 2.0f, 0.0f, -0.0f, 448.0f, 1000.0f, 0.001953125f, -1.5f};
  fp8_e4m3<8> a(in);

  EXPECT_EQ(sizeof(a.vals), 8u);
  EXPECT_EQ(a.vals[0], 0x38);
  EXPECT_EQ(a.vals[1], 0x40);
  EXPECT_EQ(a.vals[2], 0x00);
  EXPECT_EQ(a.vals[3], 0x80);
  EXPECT_EQ(a.vals[4], 0x7E);
  EXPECT_EQ(a.vals[5], 0x7E); // finite saturation
  EXPECT_EQ(a.vals[6], 0x01);
  EXPECT_EQ(a.vals[7], 0xBC); // -1.5

  // marray operator: convert fp8 vector back to marray<float, N>.
  sycl::marray<float, 8> out = static_cast<sycl::marray<float, 8>>(a);
  EXPECT_EQ(out[0], 1.0f);
  EXPECT_EQ(out[1], 2.0f);
  EXPECT_EQ(out[2], 0.0f);
  EXPECT_EQ(out[3], 0.0f);
  EXPECT_TRUE(std::signbit(out[3])); // preserve -0
  EXPECT_EQ(out[4], 448.0f);
  EXPECT_EQ(out[5], 448.0f);
  EXPECT_EQ(out[6], 0.001953125f);
  EXPECT_EQ(out[7], -1.5f);
}

TEST(FP8E4M3Test, FloatingPointConversionOperatorsMoreTypes) {
  fp8_e4m3<1> a(1.0f);
  fp8_e4m3<1> b(0.015625f);
  fp8_e4m3<1> nanv(std::numeric_limits<float>::quiet_NaN());

  EXPECT_EQ(sizeof(a.vals), 1u);
  EXPECT_EQ(sizeof(b.vals), 1u);
  EXPECT_EQ(sizeof(nanv.vals), 1u);

  double da = static_cast<double>(a);
  sycl::half ha = static_cast<sycl::half>(a);
  sycl::ext::oneapi::bfloat16 ba = static_cast<sycl::ext::oneapi::bfloat16>(a);

  EXPECT_EQ(da, 1.0);
  EXPECT_EQ(static_cast<float>(ha), 1.0f);
  EXPECT_EQ(static_cast<float>(ba), 1.0f);

  EXPECT_EQ(static_cast<float>(b), 0.015625f);

  float fn = static_cast<float>(nanv);
  EXPECT_TRUE(std::isnan(fn));
}

TEST(FP8E4M3Test, IntegerConversionOperatorsMultipleWidthsTowardZero) {
  fp8_e4m3<1> p(1.5f);
  fp8_e4m3<1> n(-1.5f);

  std::int32_t i32p = static_cast<std::int32_t>(p);
  std::int32_t i32n = static_cast<std::int32_t>(n);
  std::int64_t i64p = static_cast<std::int64_t>(p);
  std::int64_t i64n = static_cast<std::int64_t>(n);

  EXPECT_EQ(i32p, 1);
  EXPECT_EQ(i32n, -1);
  EXPECT_EQ(i64p, 1);
  EXPECT_EQ(i64n, -1);
}

TEST(FP8E4M3Test, VariadicHalfBoundaryEncodings) {
  fp8_e4m3<4> a(sycl::half(448.0f), sycl::half(0.015625f), sycl::half(0.001953125f),
                sycl::half(-0.0f));

  EXPECT_EQ(sizeof(a.vals), 4u);
  EXPECT_EQ(a.vals[0], 0x7E); // +max normal
  EXPECT_EQ(a.vals[1], 0x08); // min normal
  EXPECT_EQ(a.vals[2], 0x01); // min subnormal
  EXPECT_EQ(a.vals[3], 0x80); // -0
}

TEST(FP8E4M3Test, VariadicBFloat16BoundaryEncodings) {
  fp8_e4m3<4> a(sycl::ext::oneapi::bfloat16(1.0f),
                sycl::ext::oneapi::bfloat16(2.0f),
                sycl::ext::oneapi::bfloat16(0.001953125f),
                sycl::ext::oneapi::bfloat16(-0.0f));

  EXPECT_EQ(sizeof(a.vals), 4u);
  EXPECT_EQ(a.vals[0], 0x38);
  EXPECT_EQ(a.vals[1], 0x40);
  EXPECT_EQ(a.vals[2], 0x01);
  EXPECT_EQ(a.vals[3], 0x80);
}

TEST(FP8E4M3Test, VariadicDoubleBoundaryEncodingsAndSaturation) {
  fp8_e4m3<5> a(448.0, 449.0, 0.013671875, 0.001953125, -1000.0);

  EXPECT_EQ(sizeof(a.vals), 5u);
  EXPECT_EQ(a.vals[0], 0x7E); // +448
  EXPECT_EQ(a.vals[1], 0x7E); // clamp to +448 (finite saturation)
  EXPECT_EQ(a.vals[2], 0x07); // max subnormal
  EXPECT_EQ(a.vals[3], 0x01); // min subnormal
  EXPECT_EQ(a.vals[4], 0xFE); // clamp to -448
}

TEST(FP8E4M3Test, BoolOperatorWithNaN) {
  float pz = 0.0f;
  fp8_e4m3<1> zp(pz);
  float zv = -0.0f;
  fp8_e4m3<1> zn(zv);
  float nv = {std::numeric_limits<float>::quiet_NaN()};
  fp8_e4m3<1> nanv(nv);

  EXPECT_EQ(sizeof(zp.vals), 1u);
  EXPECT_EQ(sizeof(zn.vals), 1u);
  EXPECT_EQ(sizeof(nanv.vals), 1u);

  EXPECT_FALSE(static_cast<bool>(zp));
  EXPECT_FALSE(static_cast<bool>(zn));
  EXPECT_TRUE(static_cast<bool>(nanv)); // not +0 or -0
  EXPECT_EQ(nanv.vals[0], 0x7F);        // NaN encoding remains S.1111.111
}

TEST(FP8E4M3Test, CArrayFloatRoundingToEven) {
  const float in[3] = {0.012f, 1.0625f, 1000.0f};
  fp8_e4m3<3> a(in, rounding::to_even);

  EXPECT_EQ(a.vals[0], 0x06);
  EXPECT_EQ(a.vals[1], 0x38);
  EXPECT_EQ(a.vals[2], 0x7E);
}

TEST(FP8E4M3Test, CArrayFloatRoundingUpward) {
  const float in[3] = {0.012f, 1.0625f, 1000.0f};
  fp8_e4m3<3> a(in, rounding::upward);

  EXPECT_EQ(a.vals[0], 0x07);
  EXPECT_EQ(a.vals[1], 0x39);
  EXPECT_EQ(a.vals[2], 0x7E);
}

TEST(FP8E4M3Test, CArrayFloatRoundingDownward) {
  const float in[3] = {0.012f, 1.0625f, 1000.0f};
  fp8_e4m3<3> a(in, rounding::downward);

  EXPECT_EQ(a.vals[0], 0x06);
  EXPECT_EQ(a.vals[1], 0x38);
  EXPECT_EQ(a.vals[2], 0x7E);
}

TEST(FP8E4M3Test, CArrayFloatRoundingTowardZero) {
  const float in[3] = {0.012f, 1.0625f, 1000.0f};
  fp8_e4m3<3> a(in, rounding::toward_zero);

  EXPECT_EQ(a.vals[0], 0x06);
  EXPECT_EQ(a.vals[1], 0x38);
  EXPECT_EQ(a.vals[2], 0x7E);
}

TEST(FP8E4M3Test, CArrayFloatRoundingToAway) {
  const float in[3] = {0.012f, 1.0625f, 1000.0f};
  fp8_e4m3<3> a(in, rounding::to_away);

  EXPECT_EQ(a.vals[0], 0x06);
  EXPECT_EQ(a.vals[1], 0x38);
  EXPECT_EQ(a.vals[2], 0x7E);
}

TEST(FP8E4M3Test, CArrayHalfRoundingToEven) {
  const sycl::half in[3] = {sycl::half(0.012f), sycl::half(1.0625f),
                            sycl::half(1000.0f)};
  fp8_e4m3<3> a(in, rounding::to_even);

  EXPECT_EQ(a.vals[0], 0x06);
  EXPECT_EQ(a.vals[1], 0x38);
  EXPECT_EQ(a.vals[2], 0x7E);
}

TEST(FP8E4M3Test, CArrayHalfRoundingUpward) {
  const sycl::half in[3] = {sycl::half(0.012f), sycl::half(1.0625f),
                            sycl::half(1000.0f)};
  fp8_e4m3<3> a(in, rounding::upward);

  EXPECT_EQ(a.vals[0], 0x07);
  EXPECT_EQ(a.vals[1], 0x39);
  EXPECT_EQ(a.vals[2], 0x7E);
}

TEST(FP8E4M3Test, CArrayHalfRoundingDownward) {
  const sycl::half in[3] = {sycl::half(0.012f), sycl::half(1.0625f),
                            sycl::half(1000.0f)};
  fp8_e4m3<3> a(in, rounding::downward);

  EXPECT_EQ(a.vals[0], 0x06);
  EXPECT_EQ(a.vals[1], 0x38);
  EXPECT_EQ(a.vals[2], 0x7E);
}

TEST(FP8E4M3Test, CArrayHalfRoundingTowardZero) {
  const sycl::half in[3] = {sycl::half(0.012f), sycl::half(1.0625f),
                            sycl::half(1000.0f)};
  fp8_e4m3<3> a(in, rounding::toward_zero);

  EXPECT_EQ(a.vals[0], 0x06);
  EXPECT_EQ(a.vals[1], 0x38);
  EXPECT_EQ(a.vals[2], 0x7E);
}

TEST(FP8E4M3Test, CArrayHalfRoundingToAway) {
  const sycl::half in[3] = {sycl::half(0.012f), sycl::half(1.0625f),
                            sycl::half(1000.0f)};
  fp8_e4m3<3> a(in, rounding::to_away);

  EXPECT_EQ(a.vals[0], 0x06);
  EXPECT_EQ(a.vals[1], 0x38);
  EXPECT_EQ(a.vals[2], 0x7E);
}

TEST(FP8E4M3Test, CArrayBFloat16RoundingToEven) {
  const sycl::ext::oneapi::bfloat16 in[3] = {
      sycl::ext::oneapi::bfloat16(0.012f),
      sycl::ext::oneapi::bfloat16(1.0625f),
      sycl::ext::oneapi::bfloat16(1000.0f)};
  fp8_e4m3<3> a(in, rounding::to_even);

  EXPECT_EQ(a.vals[0], 0x06);
  EXPECT_EQ(a.vals[1], 0x38);
  EXPECT_EQ(a.vals[2], 0x7E);
}

TEST(FP8E4M3Test, CArrayBFloat16RoundingUpward) {
  const sycl::ext::oneapi::bfloat16 in[3] = {
      sycl::ext::oneapi::bfloat16(0.012f),
      sycl::ext::oneapi::bfloat16(1.0625f),
      sycl::ext::oneapi::bfloat16(1000.0f)};
  fp8_e4m3<3> a(in, rounding::upward);

  EXPECT_EQ(a.vals[0], 0x07);
  EXPECT_EQ(a.vals[1], 0x39);
  EXPECT_EQ(a.vals[2], 0x7E);
}

TEST(FP8E4M3Test, CArrayBFloat16Downward) {
  const sycl::ext::oneapi::bfloat16 in[3] = {
      sycl::ext::oneapi::bfloat16(0.012f),
      sycl::ext::oneapi::bfloat16(1.0625f),
      sycl::ext::oneapi::bfloat16(1000.0f)};
  fp8_e4m3<3> a(in, rounding::downward);

  EXPECT_EQ(a.vals[0], 0x06);
  EXPECT_EQ(a.vals[1], 0x38);
  EXPECT_EQ(a.vals[2], 0x7E);
}

TEST(FP8E4M3Test, CArrayBFloat16TowardZero) {
  const sycl::ext::oneapi::bfloat16 in[3] = {
      sycl::ext::oneapi::bfloat16(0.012f),
      sycl::ext::oneapi::bfloat16(1.0625f),
      sycl::ext::oneapi::bfloat16(1000.0f)};
  fp8_e4m3<3> a(in, rounding::toward_zero);

  EXPECT_EQ(a.vals[0], 0x06);
  EXPECT_EQ(a.vals[1], 0x38);
  EXPECT_EQ(a.vals[2], 0x7E);
}

TEST(FP8E4M3Test, CArrayBFloat16ToAway) {
  const sycl::ext::oneapi::bfloat16 in[3] = {
      sycl::ext::oneapi::bfloat16(0.012f),
      sycl::ext::oneapi::bfloat16(1.0625f),
      sycl::ext::oneapi::bfloat16(1000.0f)};
  fp8_e4m3<3> a(in, rounding::to_away);

  EXPECT_EQ(a.vals[0], 0x06);
  EXPECT_EQ(a.vals[1], 0x38);
  EXPECT_EQ(a.vals[2], 0x7E);
}

TEST(FP8E4M3Test, MarrayHalfRoundingToEven) {
  const sycl::marray<sycl::half, 3> in = {sycl::half(0.012f),
                                          sycl::half(1.0625f),
                                          sycl::half(1000.0f)};
  fp8_e4m3<3> a(in, rounding::to_even);

  EXPECT_EQ(a.vals[0], 0x06);
  EXPECT_EQ(a.vals[1], 0x38);
  EXPECT_EQ(a.vals[2], 0x7E);
}

TEST(FP8E4M3Test, MarrayHalfRoundingUpward) {
  const sycl::marray<sycl::half, 3> in = {sycl::half(0.012f),
                                          sycl::half(1.0625f),
                                          sycl::half(1000.0f)};
  fp8_e4m3<3> a(in, rounding::upward);

  EXPECT_EQ(a.vals[0], 0x07);
  EXPECT_EQ(a.vals[1], 0x39);
  EXPECT_EQ(a.vals[2], 0x7E);
}

TEST(FP8E4M3Test, MarrayHalfRoundingDownward) {
  const sycl::marray<sycl::half, 3> in = {sycl::half(0.012f),
                                          sycl::half(1.0625f),
                                          sycl::half(1000.0f)};
  fp8_e4m3<3> a(in, rounding::downward);

  EXPECT_EQ(a.vals[0], 0x06);
  EXPECT_EQ(a.vals[1], 0x38);
  EXPECT_EQ(a.vals[2], 0x7E);
}

TEST(FP8E4M3Test, MarrayHalfRoundingTowardZero) {
  const sycl::marray<sycl::half, 3> in = {sycl::half(0.012f),
                                          sycl::half(1.0625f),
                                          sycl::half(1000.0f)};
  fp8_e4m3<3> a(in, rounding::toward_zero);

  EXPECT_EQ(a.vals[0], 0x06);
  EXPECT_EQ(a.vals[1], 0x38);
  EXPECT_EQ(a.vals[2], 0x7E);
}

TEST(FP8E4M3Test, MarrayHalfRoundingToAway) {
  const sycl::marray<sycl::half, 3> in = {sycl::half(0.012f),
                                          sycl::half(1.0625f),
                                          sycl::half(1000.0f)};
  fp8_e4m3<3> a(in, rounding::to_away);

  EXPECT_EQ(a.vals[0], 0x06);
  EXPECT_EQ(a.vals[1], 0x38);
  EXPECT_EQ(a.vals[2], 0x7E);
}

TEST(FP8E4M3Test, MarrayBFloat16RoundingToEven) {
  const sycl::marray<sycl::ext::oneapi::bfloat16, 3> in = {
      sycl::ext::oneapi::bfloat16(0.012f),
      sycl::ext::oneapi::bfloat16(1.0625f),
      sycl::ext::oneapi::bfloat16(1000.0f)};
  fp8_e4m3<3> a(in, rounding::to_even);

  EXPECT_EQ(a.vals[0], 0x06);
  EXPECT_EQ(a.vals[1], 0x38);
  EXPECT_EQ(a.vals[2], 0x7E);
}

TEST(FP8E4M3Test, MarrayBFloat16RoundingUpward) {
  const sycl::marray<sycl::ext::oneapi::bfloat16, 3> in = {
      sycl::ext::oneapi::bfloat16(0.012f),
      sycl::ext::oneapi::bfloat16(1.0625f),
      sycl::ext::oneapi::bfloat16(1000.0f)};
  fp8_e4m3<3> a(in, rounding::upward);

  EXPECT_EQ(a.vals[0], 0x07);
  EXPECT_EQ(a.vals[1], 0x39);
  EXPECT_EQ(a.vals[2], 0x7E);
}

TEST(FP8E4M3Test, MarrayBFloat16RoundingDownward) {
  const sycl::marray<sycl::ext::oneapi::bfloat16, 3> in = {
      sycl::ext::oneapi::bfloat16(0.012f),
      sycl::ext::oneapi::bfloat16(1.0625f),
      sycl::ext::oneapi::bfloat16(1000.0f)};
  fp8_e4m3<3> a(in, rounding::downward);

  EXPECT_EQ(a.vals[0], 0x06);
  EXPECT_EQ(a.vals[1], 0x38);
  EXPECT_EQ(a.vals[2], 0x7E);
}

TEST(FP8E4M3Test, MarrayBFloat16RoundingTowardZero) {
  const sycl::marray<sycl::ext::oneapi::bfloat16, 3> in = {
      sycl::ext::oneapi::bfloat16(0.012f),
      sycl::ext::oneapi::bfloat16(1.0625f),
      sycl::ext::oneapi::bfloat16(1000.0f)};
  fp8_e4m3<3> a(in, rounding::toward_zero);

  EXPECT_EQ(a.vals[0], 0x06);
  EXPECT_EQ(a.vals[1], 0x38);
  EXPECT_EQ(a.vals[2], 0x7E);
}

TEST(FP8E4M3Test, MarrayBFloat16RoundingToAway) {
  const sycl::marray<sycl::ext::oneapi::bfloat16, 3> in = {
      sycl::ext::oneapi::bfloat16(0.012f),
      sycl::ext::oneapi::bfloat16(1.0625f),
      sycl::ext::oneapi::bfloat16(1000.0f)};
  fp8_e4m3<3> a(in, rounding::to_away);

  EXPECT_EQ(a.vals[0], 0x06);
  EXPECT_EQ(a.vals[1], 0x38);
  EXPECT_EQ(a.vals[2], 0x7E);
}

TEST(FP8E4M3Test, MarrayFloatRoundingToEven) {
  const sycl::marray<float, 3> in = {0.012f, 1.0625f, 1000.0f};
  fp8_e4m3<3> a(in, rounding::to_even);

  EXPECT_EQ(a.vals[0], 0x06);
  EXPECT_EQ(a.vals[1], 0x38);
  EXPECT_EQ(a.vals[2], 0x7E);
}

TEST(FP8E4M3Test, MarrayFloatRoundingUpward) {
  const sycl::marray<float, 3> in = {0.012f, 1.0625f, 1000.0f};
  fp8_e4m3<3> a(in, rounding::upward);

  EXPECT_EQ(a.vals[0], 0x07);
  EXPECT_EQ(a.vals[1], 0x39);
  EXPECT_EQ(a.vals[2], 0x7E);
}

TEST(FP8E4M3Test, MarrayFloatRoundingDownward) {
  const sycl::marray<float, 3> in = {0.012f, 1.0625f, 1000.0f};
  fp8_e4m3<3> a(in, rounding::downward);

  EXPECT_EQ(a.vals[0], 0x06);
  EXPECT_EQ(a.vals[1], 0x38);
  EXPECT_EQ(a.vals[2], 0x7E);
}

TEST(FP8E4M3Test, MarrayFloatRoundingTowardZero) {
  const sycl::marray<float, 3> in = {0.012f, 1.0625f, 1000.0f};
  fp8_e4m3<3> a(in, rounding::toward_zero);

  EXPECT_EQ(a.vals[0], 0x06);
  EXPECT_EQ(a.vals[1], 0x38);
  EXPECT_EQ(a.vals[2], 0x7E);
}

TEST(FP8E4M3Test, MarrayFloatRoundingToAway) {
  const sycl::marray<float, 3> in = {0.012f, 1.0625f, 1000.0f};
  fp8_e4m3<3> a(in, rounding::to_away);

  EXPECT_EQ(a.vals[0], 0x06);
  EXPECT_EQ(a.vals[1], 0x38);
  EXPECT_EQ(a.vals[2], 0x7E);
}


TEST(FP8E4M3Test, MarrayDoubleToEven) {
  const sycl::marray<double, 3> in = {0.012, 1.0625, 1000.0};
  fp8_e4m3<3> a(in);

  EXPECT_EQ(a.vals[0], 0x06);
  EXPECT_EQ(a.vals[1], 0x38);
  EXPECT_EQ(a.vals[2], 0x7E);
}
