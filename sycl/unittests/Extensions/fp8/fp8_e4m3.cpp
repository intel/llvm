#include <gtest/gtest.h>
#include <sycl/ext/oneapi/experimental/float_8bit/types.hpp>

#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

/*
Unit tests check only CPU versions. Most of the constraints related to device
code thus unit tests check only API
*/

using namespace sycl::ext::oneapi::experimental;

TEST(FP8E4M3Test, VariadicHalf) {
  fp8_e4m3_x2 a(sycl::half(1.0f), sycl::half(2.0f));

  EXPECT_EQ(sizeof(a.vals), 2u);
  EXPECT_EQ(a.vals[0], 0x38); // 1.0  -> 0b0_0111_000
  EXPECT_EQ(a.vals[1], 0x40); // 2.0  -> 0b0_1000_000

  fp8_e4m3 b(sycl::half(1.1f));
  EXPECT_EQ(sizeof(b.vals), 1u);
  EXPECT_EQ(b.vals[0], 0x39); // 1.1 rounds to 1.125 -> frac=1
}

TEST(FP8E4M3Test, VariadicBFloat16) {
  fp8_e4m3_x2 a(sycl::ext::oneapi::bfloat16(1.0f),
                sycl::ext::oneapi::bfloat16(2.0f));

  EXPECT_EQ(sizeof(a.vals), 2u);
  EXPECT_EQ(a.vals[0], 0x38);
  EXPECT_EQ(a.vals[1], 0x40);

  fp8_e4m3 b(sycl::ext::oneapi::bfloat16(1.1f));
  EXPECT_EQ(sizeof(b.vals), 1u);
  EXPECT_EQ(b.vals[0], 0x39);
}

TEST(FP8E4M3Test, VariadicFloat) {
  fp8_e4m3_x2 a(1.0f, 2.0f);

  EXPECT_EQ(sizeof(a.vals), 2u);
  EXPECT_EQ(a.vals[0], 0x38);
  EXPECT_EQ(a.vals[1], 0x40);

  fp8_e4m3 b(1.1f);
  EXPECT_EQ(sizeof(b.vals), 1u);
  EXPECT_EQ(b.vals[0], 0x39);
}

TEST(FP8E4M3Test, VariadicBoundaryEncodingsFloat) {
  // CPU host path: variadic constructors use rounding::to_even and
  // saturation::finite.
  fp8_e4m3_x2 a(448.0f,   // max normal  -> S.1111.110
                0.015625f // min normal  -> S.0001.000  (2^-6)
  );

  fp8_e4m3_x2 b(0.013671875f, // max subnorm -> S.0000.111  (0.875 * 2^-6)
                0.001953125f  // min subnorm -> S.0000.001  (2^-9)
  );

  fp8_e4m3_x2 c(0.0f, // +0
                -0.0f // -0
  );

  EXPECT_EQ(sizeof(a.vals), 2u);
  EXPECT_EQ(sizeof(b.vals), 2u);
  EXPECT_EQ(sizeof(c.vals), 2u);
  EXPECT_EQ(a.vals[0], 0x7E); // +448.0  -> 0b0_1111_110
  EXPECT_EQ(a.vals[1], 0x08); // +2^-6   -> 0b0_0001_000
  EXPECT_EQ(b.vals[0], 0x07); // +max subnorm -> 0b0_0000_111
  EXPECT_EQ(b.vals[1], 0x01); // +min subnorm -> 0b0_0000_001
  EXPECT_EQ(c.vals[0], 0x00); // +0 -> 0b0_0000_000
  EXPECT_EQ(c.vals[1], 0x80); // -0 -> 0b1_0000_000
}

TEST(FP8E4M3Test, VariadicNaNEncodingFloat) {
  // NaN is encoded as S.1111.111; sign is permitted.
  fp8_e4m3_x2 a(std::numeric_limits<float>::quiet_NaN(),
                -std::numeric_limits<float>::quiet_NaN());

  EXPECT_EQ(a.vals[0], 0x7F); // +NaN -> 0b0_1111_111
  EXPECT_EQ(a.vals[1], 0xFF); // -NaN -> 0b1_1111_111
}

TEST(FP8E4M3Test, IntegerToEvenFiniteAndSize) {
  // Integer constructors: to_even + finite saturation (CPU).
  fp8_e4m3 a0(0);
  fp8_e4m3 a1(1);
  fp8_e4m3 a2(2);
  fp8_e4m3 an1(-1);

  EXPECT_EQ(sizeof(a0.vals), 1u);
  EXPECT_EQ(sizeof(a1.vals), 1u);
  EXPECT_EQ(sizeof(a2.vals), 1u);
  EXPECT_EQ(sizeof(an1.vals), 1u);

  EXPECT_EQ(a0.vals[0], 0x00);  // +0
  EXPECT_EQ(a1.vals[0], 0x38);  // +1.0 -> 0b0_0111_000
  EXPECT_EQ(a2.vals[0], 0x40);  // +2.0 -> 0b0_1000_000
  EXPECT_EQ(an1.vals[0], 0xB8); // -1.0 -> sign set: 0b1_0111_000
}

TEST(FP8E4M3Test, AssignmentOperatorToEvenFiniteAndSize) {
  // operator= from scalar: to_even + finite saturation (CPU).
  fp8_e4m3 a(0.0f);
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
  fp8_e4m3 one(1.0f);
  fp8_e4m3 zero_pos(0.0f);
  fp8_e4m3 zero_neg(-0.0f);
  fp8_e4m3 min_norm(0.015625f);

  EXPECT_EQ(sizeof(one.vals), 1u);
  EXPECT_EQ(one.vals[0], 0x38);

  float f1 = static_cast<float>(one);
  float fz = static_cast<float>(zero_pos);
  float fnz = static_cast<float>(zero_neg);
  float fmn = static_cast<float>(min_norm);

  EXPECT_EQ(f1, 1.0f);
  EXPECT_EQ(fz, 0.0f);
  // -0.0 compares equal to +0.0; check signbit to validate negative zero
  // survives.
  EXPECT_EQ(fnz, 0.0f);
  EXPECT_TRUE(std::signbit(fnz));

  EXPECT_EQ(fmn, 0.015625f);
}

TEST(FP8E4M3Test, IntegerConversionOperatorsTowardZero) {
  // Integer operators: convert using rounding::toward_zero.
  fp8_e4m3 p(1.5f);  // 1.5 exactly representable: 0b0_0111_100 (0x3C)
  fp8_e4m3 n(-1.5f); // 0xBC

  EXPECT_EQ(sizeof(p.vals), 1u);
  EXPECT_EQ(sizeof(n.vals), 1u);
  EXPECT_EQ(p.vals[0], 0x3C);
  EXPECT_EQ(n.vals[0], 0xBC);

  int ip = static_cast<int>(p);
  int in = static_cast<int>(n);

  EXPECT_EQ(ip, 1);  // toward zero
  EXPECT_EQ(in, -1); // toward zero
}

TEST(FP8E4M3Test, BoolOperatorZeroRules) {
  // bool operator: false iff +0 or -0; otherwise true.
  fp8_e4m3 zp(0.0f);
  fp8_e4m3 zn(-0.0f);
  fp8_e4m3 one(1.0f);
  fp8_e4m3 sub(0.001953125f); // min subnormal

  EXPECT_EQ(sizeof(zp.vals), 1u);
  EXPECT_EQ(sizeof(zn.vals), 1u);
  EXPECT_EQ(sizeof(one.vals), 1u);
  EXPECT_EQ(sizeof(sub.vals), 1u);

  EXPECT_FALSE(static_cast<bool>(zp));
  EXPECT_FALSE(static_cast<bool>(zn));
  EXPECT_TRUE(static_cast<bool>(one));
  EXPECT_TRUE(static_cast<bool>(sub));
}

TEST(FP8E4M3Test, CArrayFloatHostToEvenFinite) {
  // Host code supports only rounding::to_even and saturation::finite.
  const float in[2] = {1.0f, 1.1f};
  const float in1[2] = {1.0625f, 1000.0f};
  const float in2[2] = {-0.0f, 0.0f};
  fp8_e4m3_x2 a(in);
  fp8_e4m3_x2 a1(in1);
  fp8_e4m3_x2 a2(in2);

  EXPECT_EQ(sizeof(a.vals), 2u);
  EXPECT_EQ(sizeof(a1.vals), 2u);
  EXPECT_EQ(sizeof(a2.vals), 2u);
  EXPECT_EQ(a.vals[0], 0x38);  // 1.0
  EXPECT_EQ(a.vals[1], 0x39);  // 1.1 -> 1.125
  EXPECT_EQ(a1.vals[0], 0x38); // tie -> to_even => 1.0
  EXPECT_EQ(a1.vals[1], 0x7E); // finite saturation => +448
  EXPECT_EQ(a2.vals[0], 0x80); // -0
  EXPECT_EQ(a2.vals[1], 0x00); // 0
}

TEST(FP8E4M3Test, CArrayDoubleToEvenFinite) {
  // Double c-array: to_even + finite saturation.
  const double in[2] = {448.0, 449.0};
  const double in1[2] = {0.015625, 0.013671875};
  const double in2[2] = {0.001953125, std::numeric_limits<double>::quiet_NaN()};
  fp8_e4m3_x2 a(in);
  fp8_e4m3_x2 a1(in1);
  fp8_e4m3_x2 a2(in2);

  EXPECT_EQ(sizeof(a.vals), 2u);
  EXPECT_EQ(sizeof(a1.vals), 2u);
  EXPECT_EQ(sizeof(a2.vals), 2u);
  EXPECT_EQ(a.vals[0], 0x7E);  // +448
  EXPECT_EQ(a.vals[1], 0x7E);  // 449 -> clamp to +448
  EXPECT_EQ(a1.vals[0], 0x08); // min normal
  EXPECT_EQ(a1.vals[1], 0x07); // max subnormal
  EXPECT_EQ(a2.vals[0], 0x01); // min subnormal
  EXPECT_EQ(a2.vals[1], 0x7F); // NaN
}

TEST(FP8E4M3Test, CArrayHalfHostToEvenFinite) {
  // Host code supports only rounding::to_even and saturation::finite.
  const sycl::half in[2] = {sycl::half(448.0f), sycl::half(449.0f)};
  const sycl::half in1[2] = {sycl::half(0.015625f), sycl::half(0.013671875f)};
  const sycl::half in2[2] = {sycl::half(0.001953125f), sycl::half(-0.0f)};

  fp8_e4m3_x2 a(in);
  fp8_e4m3_x2 a1(in1);
  fp8_e4m3_x2 a2(in2);

  EXPECT_EQ(sizeof(a.vals), 2u);
  EXPECT_EQ(sizeof(a1.vals), 2u);
  EXPECT_EQ(sizeof(a2.vals), 2u);
  EXPECT_EQ(a.vals[0], 0x7E);  // +448
  EXPECT_EQ(a.vals[1], 0x7E);  // 449 -> clamp to +448
  EXPECT_EQ(a1.vals[0], 0x08); // min normal
  EXPECT_EQ(a1.vals[1], 0x07); // max subnormal
  EXPECT_EQ(a2.vals[0], 0x01); // min subnormal
  EXPECT_EQ(a2.vals[1], 0x80); // -0
}

TEST(FP8E4M3Test, CArrayBFloat16HostToEvenFinite) {
  // Host code supports only rounding::to_even and saturation::finite.
  const sycl::ext::oneapi::bfloat16 in[2] = {
      sycl::ext::oneapi::bfloat16(448.0f), sycl::ext::oneapi::bfloat16(449.0f)};
  const sycl::ext::oneapi::bfloat16 in1[2] = {
      sycl::ext::oneapi::bfloat16(0.015625f),
      sycl::ext::oneapi::bfloat16(0.013671875f)};
  const sycl::ext::oneapi::bfloat16 in2[2] = {
      sycl::ext::oneapi::bfloat16(0.001953125f),
      sycl::ext::oneapi::bfloat16(-0.0f)};

  fp8_e4m3_x2 a(in);
  fp8_e4m3_x2 a1(in1);
  fp8_e4m3_x2 a2(in2);

  EXPECT_EQ(sizeof(a.vals), 2u);
  EXPECT_EQ(sizeof(a1.vals), 2u);
  EXPECT_EQ(sizeof(a2.vals), 2u);
  EXPECT_EQ(a.vals[0], 0x7E);  // +448
  EXPECT_EQ(a.vals[1], 0x7E);  // 449 -> clamp to +448
  EXPECT_EQ(a1.vals[0], 0x08); // min normal
  EXPECT_EQ(a1.vals[1], 0x07); // max subnormal
  EXPECT_EQ(a2.vals[0], 0x01); // min subnormal
  EXPECT_EQ(a2.vals[1], 0x80); // -0
}

TEST(FP8E4M3Test, MarrayAndOperatorsHostAllN) {
  // marray constructors/operators: host supports all N.
  sycl::marray<float, 2> in = {1.0f, 2.0f};
  sycl::marray<float, 2> in1 = {0.0f, -0.0f};
  sycl::marray<float, 2> in2 = {448.0f, 1000.0f};
  sycl::marray<float, 2> in3 = {0.001953125f, -1.5f};

  fp8_e4m3_x2 a(in);
  fp8_e4m3_x2 a1(in1);
  fp8_e4m3_x2 a2(in2);
  fp8_e4m3_x2 a3(in3);

  EXPECT_EQ(sizeof(a.vals), 2u);
  EXPECT_EQ(sizeof(a1.vals), 2u);
  EXPECT_EQ(sizeof(a2.vals), 2u);
  EXPECT_EQ(sizeof(a3.vals), 2u);

  EXPECT_EQ(a.vals[0], 0x38);
  EXPECT_EQ(a.vals[1], 0x40);
  EXPECT_EQ(a1.vals[0], 0x00);
  EXPECT_EQ(a1.vals[1], 0x80);
  EXPECT_EQ(a2.vals[0], 0x7E);
  EXPECT_EQ(a2.vals[1], 0x7E); // finite saturation
  EXPECT_EQ(a3.vals[0], 0x01);
  EXPECT_EQ(a3.vals[1], 0xBC); // -1.5

  // marray operator: convert fp8 vector back to marray<float, N>.
  sycl::marray<float, 2> out = static_cast<sycl::marray<float, 2>>(a);
  sycl::marray<float, 2> out1 = static_cast<sycl::marray<float, 2>>(a1);
  sycl::marray<float, 2> out2 = static_cast<sycl::marray<float, 2>>(a2);
  sycl::marray<float, 2> out3 = static_cast<sycl::marray<float, 2>>(a3);
  EXPECT_EQ(out[0], 1.0f);
  EXPECT_EQ(out[1], 2.0f);
  EXPECT_EQ(out1[0], 0.0f);
  EXPECT_EQ(out1[1], 0.0f);
  EXPECT_TRUE(std::signbit(out1[1])); // preserve -0
  EXPECT_EQ(out2[0], 448.0f);
  EXPECT_EQ(out2[1], 448.0f);
  EXPECT_EQ(out3[0], 0.001953125f);
  EXPECT_EQ(out3[1], -1.5f);
}

TEST(FP8E4M3Test, FloatingPointConversionOperatorsMoreTypes) {
  fp8_e4m3 a(1.0f);
  fp8_e4m3 b(0.015625f);
  fp8_e4m3 nanv(std::numeric_limits<float>::quiet_NaN());

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
  fp8_e4m3 p(1.5f);
  fp8_e4m3 n(-1.5f);

  int i = static_cast<int>(p);
  short s = static_cast<short>(n);
  long l = static_cast<long>(p);
  long long ll = static_cast<long long>(n);

  EXPECT_EQ(i, 1);
  EXPECT_EQ(s, -1);
  EXPECT_EQ(l, 1);
  EXPECT_EQ(ll, -1);
}

TEST(FP8E4M3Test, CArrayFloatRoundingToEven) {
  const float in[2] = {0.012f, 1000.0f};
  fp8_e4m3_x2 a(in, rounding::to_even);

  EXPECT_EQ(a.vals[0], 0x06);
  EXPECT_EQ(a.vals[1], 0x7E);
}

TEST(FP8E4M3Test, CArrayHalfRoundingToEven) {
  const sycl::half in[2] = {sycl::half(0.012f), sycl::half(1000.0f)};
  fp8_e4m3_x2 a(in, rounding::to_even);

  EXPECT_EQ(a.vals[0], 0x06);
  EXPECT_EQ(a.vals[1], 0x7E);
}

TEST(FP8E4M3Test, CArrayBFloat16RoundingToEven) {
  const sycl::ext::oneapi::bfloat16 in[2] = {
      sycl::ext::oneapi::bfloat16(0.012f),
      sycl::ext::oneapi::bfloat16(1000.0f)};
  fp8_e4m3_x2 a(in, rounding::to_even);

  EXPECT_EQ(a.vals[0], 0x06);
  EXPECT_EQ(a.vals[1], 0x7E);
}

TEST(FP8E4M3Test, MarrayHalfRoundingToEven) {
  const sycl::marray<sycl::half, 2> in = {sycl::half(0.012f),
                                          sycl::half(1.0625f)};
  fp8_e4m3_x2 a(in, rounding::to_even);

  EXPECT_EQ(a.vals[0], 0x06);
  EXPECT_EQ(a.vals[1], 0x38);
}

TEST(FP8E4M3Test, MarrayBFloat16RoundingToEven) {
  const sycl::marray<sycl::ext::oneapi::bfloat16, 2> in = {
      sycl::ext::oneapi::bfloat16(0.012f),
      sycl::ext::oneapi::bfloat16(1.0625f)};
  fp8_e4m3_x2 a(in, rounding::to_even);

  EXPECT_EQ(a.vals[0], 0x06);
  EXPECT_EQ(a.vals[1], 0x38);
}

TEST(FP8E4M3Test, MarrayFloatRoundingToEven) {
  const sycl::marray<float, 2> in = {0.012f, 1.0625f};
  fp8_e4m3_x2 a(in, rounding::to_even);

  EXPECT_EQ(a.vals[0], 0x06);
  EXPECT_EQ(a.vals[1], 0x38);
}

TEST(FP8E4M3Test, MarrayDoubleToEven) {
  const sycl::marray<double, 2> in = {0.012, 1.0625};
  fp8_e4m3_x2 a(in);

  EXPECT_EQ(a.vals[0], 0x06);
  EXPECT_EQ(a.vals[1], 0x38);
}

TEST(FP8E4M3Test, VariadicRejectsMixedTypes) {
  EXPECT_FALSE((std::is_constructible_v<fp8_e4m3_x2, float, sycl::half>));
  EXPECT_FALSE((std::is_constructible_v<fp8_e4m3_x2, sycl::half, float>));
}

TEST(FP8E4M3Test, X2NotConstructibleFromSingleShort) {
  EXPECT_FALSE((std::is_constructible_v<fp8_e4m3_x2, short>));
}

TEST(FP8E4M3Test, X2NotConstructibleFromSingleInt) {
  EXPECT_FALSE((std::is_constructible_v<fp8_e4m3_x2, int>));
}

TEST(FP8E4M3Test, X2NotConstructibleFromSingleLong) {
  EXPECT_FALSE((std::is_constructible_v<fp8_e4m3_x2, long>));
}

TEST(FP8E4M3Test, X2NotConstructibleFromSingleLL) {
  EXPECT_FALSE((std::is_constructible_v<fp8_e4m3_x2, long long>));
}

TEST(FP8E4M3Test, X2NotConstructibleFromSingleUShort) {
  EXPECT_FALSE((std::is_constructible_v<fp8_e4m3_x2, unsigned short>));
}

TEST(FP8E4M3Test, X2NotConstructibleFromSingleUInt) {
  EXPECT_FALSE((std::is_constructible_v<fp8_e4m3_x2, unsigned int>));
}

TEST(FP8E4M3Test, X2NotConstructibleFromSingleUL) {
  EXPECT_FALSE((std::is_constructible_v<fp8_e4m3_x2, unsigned long>));
}

TEST(FP8E4M3Test, X2NotConstructibleFromSingleULL) {
  EXPECT_FALSE((std::is_constructible_v<fp8_e4m3_x2, unsigned long long>));
}

TEST(FP8E4M3Test, X2NotConstructibleFromSingleFloat) {
  EXPECT_FALSE((std::is_constructible_v<fp8_e4m3_x2, float>));
}

TEST(FP8E4M3Test, X2NotConstructibleFromSingleDouble) {
  EXPECT_FALSE((std::is_constructible_v<fp8_e4m3_x2, double>));
}

TEST(FP8E4M3Test, X2NotConstructibleFromSingleBFloat16) {
  EXPECT_FALSE(
      (std::is_constructible_v<fp8_e4m3_x2, sycl::ext::oneapi::bfloat16>));
}

TEST(FP8E4M3Test, X2NotConstructibleFromSingleHalf) {
  EXPECT_FALSE((std::is_constructible_v<fp8_e4m3_x2, sycl::half>));
}

TEST(FP8E4M3Test, X2NotConstructibleFromSingleChar) {
  EXPECT_FALSE((std::is_constructible_v<fp8_e4m3_x2, char>));
}

TEST(FP8E4M3Test, X2NotConstructibleFromSingleUChar) {
  EXPECT_FALSE((std::is_constructible_v<fp8_e4m3_x2, unsigned char>));
}

TEST(FP8E4M3Test, X2NotAssignableFromSingleHalf) {
  EXPECT_FALSE((std::is_assignable_v<fp8_e4m3_x2 &, sycl::half>));
}

TEST(FP8E4M3Test, X2NotAssignableFromSingleBFloat16) {
  EXPECT_FALSE(
      (std::is_assignable_v<fp8_e4m3_x2 &, sycl::ext::oneapi::bfloat16>));
}

TEST(FP8E4M3Test, X2NotAssignableFromSingleFloat) {
  EXPECT_FALSE((std::is_assignable_v<fp8_e4m3_x2 &, float>));
}

TEST(FP8E4M3Test, X2NotAssignableFromSingleDouble) {
  EXPECT_FALSE((std::is_assignable_v<fp8_e4m3_x2 &, double>));
}

TEST(FP8E4M3Test, X2NotAssignableFromSingleChar) {
  EXPECT_FALSE((std::is_assignable_v<fp8_e4m3_x2 &, char>));
}

TEST(FP8E4M3Test, X2NotAssignableFromSingleSignedChar) {
  EXPECT_FALSE((std::is_assignable_v<fp8_e4m3_x2 &, signed char>));
}

TEST(FP8E4M3Test, X2NotAssignableFromSingleUChar) {
  EXPECT_FALSE((std::is_assignable_v<fp8_e4m3_x2 &, unsigned char>));
}

TEST(FP8E4M3Test, X2NotAssignableFromSingleShort) {
  EXPECT_FALSE((std::is_assignable_v<fp8_e4m3_x2 &, short>));
}

TEST(FP8E4M3Test, X2NotAssignableFromSingleInt) {
  EXPECT_FALSE((std::is_assignable_v<fp8_e4m3_x2 &, int>));
}

TEST(FP8E4M3Test, X2NotAssignableFromSingleLong) {
  EXPECT_FALSE((std::is_assignable_v<fp8_e4m3_x2 &, long>));
}

TEST(FP8E4M3Test, X2NotAssignableFromSingleLL) {
  EXPECT_FALSE((std::is_assignable_v<fp8_e4m3_x2 &, long long>));
}

TEST(FP8E4M3Test, X2NotAssignableFromSingleUShort) {
  EXPECT_FALSE((std::is_assignable_v<fp8_e4m3_x2 &, unsigned short>));
}

TEST(FP8E4M3Test, X2NotAssignableFromSingleUInt) {
  EXPECT_FALSE((std::is_assignable_v<fp8_e4m3_x2 &, unsigned int>));
}

TEST(FP8E4M3Test, X2NotAssignableFromSingleUL) {
  EXPECT_FALSE((std::is_assignable_v<fp8_e4m3_x2 &, unsigned long>));
}

TEST(FP8E4M3Test, X2NotAssignableFromSingleULL) {
  EXPECT_FALSE((std::is_assignable_v<fp8_e4m3_x2 &, unsigned long long>));
}

TEST(FP8E4M3Test, CArrayHalfRejectsUpwardRounding) {
  const sycl::half in[2] = {sycl::half(1.0f), sycl::half(2.0f)};
  EXPECT_DEATH(
      {
        fp8_e4m3_x2 value(in, rounding::upward);
        (void)value;
      },
      "fp8_e4m3_x: only rounding::to_even is supported");
}

TEST(FP8E4M3Test, CArrayHalfRejectsTowardZeroRounding) {
  const sycl::half in[2] = {sycl::half(1.0f), sycl::half(2.0f)};
  EXPECT_DEATH(
      {
        fp8_e4m3_x2 value(in, rounding::toward_zero);
        (void)value;
      },
      "fp8_e4m3_x: only rounding::to_even is supported");
}

TEST(FP8E4M3Test, CArrayBFloat16RejectsUpwardRounding) {
  const sycl::ext::oneapi::bfloat16 in[2] = {sycl::ext::oneapi::bfloat16(1.0f),
                                             sycl::ext::oneapi::bfloat16(2.0f)};
  EXPECT_DEATH(
      {
        fp8_e4m3_x2 value(in, rounding::upward);
        (void)value;
      },
      "fp8_e4m3_x: only rounding::to_even is supported");
}

TEST(FP8E4M3Test, CArrayBFloat16RejectsTowardZeroRounding) {
  const sycl::ext::oneapi::bfloat16 in[2] = {sycl::ext::oneapi::bfloat16(1.0f),
                                             sycl::ext::oneapi::bfloat16(2.0f)};
  EXPECT_DEATH(
      {
        fp8_e4m3_x2 value(in, rounding::toward_zero);
        (void)value;
      },
      "fp8_e4m3_x: only rounding::to_even is supported");
}

TEST(FP8E4M3Test, CArrayFloatRejectsUpwardRounding) {
  const float in[2] = {1.0f, 2.0f};
  EXPECT_DEATH(
      {
        fp8_e4m3_x2 value(in, rounding::upward);
        (void)value;
      },
      "fp8_e4m3_x: only rounding::to_even is supported");
}

TEST(FP8E4M3Test, CArrayFloatRejectsTowardZeroRounding) {
  const float in[2] = {1.0f, 2.0f};
  EXPECT_DEATH(
      {
        fp8_e4m3_x2 value(in, rounding::toward_zero);
        (void)value;
      },
      "fp8_e4m3_x: only rounding::to_even is supported");
}

TEST(FP8E4M3Test, MarrayHalfRejectsUpwardRounding) {
  const sycl::marray<sycl::half, 2> in = {sycl::half(1.0f), sycl::half(2.0f)};
  EXPECT_DEATH(
      {
        fp8_e4m3_x2 value(in, rounding::upward);
        (void)value;
      },
      "fp8_e4m3_x: only rounding::to_even is supported");
}

TEST(FP8E4M3Test, MarrayHalfRejectsTowardZeroRounding) {
  const sycl::marray<sycl::half, 2> in = {sycl::half(1.0f), sycl::half(2.0f)};
  EXPECT_DEATH(
      {
        fp8_e4m3_x2 value(in, rounding::toward_zero);
        (void)value;
      },
      "fp8_e4m3_x: only rounding::to_even is supported");
}

TEST(FP8E4M3Test, MarrayBFloat16RejectsUpwardRounding) {
  const sycl::marray<sycl::ext::oneapi::bfloat16, 2> in = {
      sycl::ext::oneapi::bfloat16(1.0f), sycl::ext::oneapi::bfloat16(2.0f)};
  EXPECT_DEATH(
      {
        fp8_e4m3_x2 value(in, rounding::upward);
        (void)value;
      },
      "fp8_e4m3_x: only rounding::to_even is supported");
}

TEST(FP8E4M3Test, MarrayBFloat16RejectsTowardZeroRounding) {
  const sycl::marray<sycl::ext::oneapi::bfloat16, 2> in = {
      sycl::ext::oneapi::bfloat16(1.0f), sycl::ext::oneapi::bfloat16(2.0f)};
  EXPECT_DEATH(
      {
        fp8_e4m3_x2 value(in, rounding::toward_zero);
        (void)value;
      },
      "fp8_e4m3_x: only rounding::to_even is supported");
}

TEST(FP8E4M3Test, MarrayFloatRejectsUpwardRounding) {
  const sycl::marray<float, 2> in = {1.0f, 2.0f};
  EXPECT_DEATH(
      {
        fp8_e4m3_x2 value(in, rounding::upward);
        (void)value;
      },
      "fp8_e4m3_x: only rounding::to_even is supported");
}

TEST(FP8E4M3Test, MarrayFloatRejectsTowardZeroRounding) {
  const sycl::marray<float, 2> in = {1.0f, 2.0f};
  EXPECT_DEATH(
      {
        fp8_e4m3_x2 value(in, rounding::toward_zero);
        (void)value;
      },
      "fp8_e4m3_x: only rounding::to_even is supported");
}
