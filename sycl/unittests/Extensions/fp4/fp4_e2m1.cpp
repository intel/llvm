#include <gtest/gtest.h>
#include <sycl/ext/oneapi/experimental/float_4bit/types.hpp>

#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

/*
Unit tests check only CPU versions. Most of the constraints related to device
code thus unit tests check only API
*/

using namespace sycl::ext::oneapi::experimental;

// E2M1 encoding (S.EE.M, bias=1, no Inf/NaN):
//   0x0 = +0,   0x1 = +0.5, 0x2 = +1.0, 0x3 = +1.5,
//   0x4 = +2.0, 0x5 = +3.0, 0x6 = +4.0, 0x7 = +6.0 (max finite),
//   negatives: bit 3 set.
//
// Packed x2 storage (one byte): element 0 in low 4 bits, element 1 in high 4
// bits.

TEST(FP4E2M1Test, DeductionGuide) {
  fp4_e2m1_x one(1.0f);
  fp4_e2m1_x pair(1.0f, 2.0f);

  EXPECT_TRUE((std::is_same_v<decltype(one), fp4_e2m1>));
  EXPECT_TRUE((std::is_same_v<decltype(pair), fp4_e2m1_x2>));
}

TEST(FP4E2M1Test, TrivialSpecialMembers) {
  EXPECT_TRUE((std::is_trivially_default_constructible_v<fp4_e2m1>));
  EXPECT_TRUE((std::is_trivially_copy_constructible_v<fp4_e2m1>));
  EXPECT_TRUE((std::is_trivially_destructible_v<fp4_e2m1>));
  EXPECT_TRUE((std::is_trivially_copy_assignable_v<fp4_e2m1>));

  EXPECT_TRUE((std::is_trivially_default_constructible_v<fp4_e2m1_x2>));
  EXPECT_TRUE((std::is_trivially_copy_constructible_v<fp4_e2m1_x2>));
  EXPECT_TRUE((std::is_trivially_destructible_v<fp4_e2m1_x2>));
  EXPECT_TRUE((std::is_trivially_copy_assignable_v<fp4_e2m1_x2>));

  fp4_e2m1 source(1.0f);
  fp4_e2m1 copy(source);
  fp4_e2m1 assigned;
  assigned = source;

  EXPECT_EQ(copy.vals[0], source.vals[0]);
  EXPECT_EQ(assigned.vals[0], source.vals[0]);
}

TEST(FP4E2M1Test, StorageSize) {
  EXPECT_EQ(sizeof(fp4_e2m1{}.vals), 1u);
  EXPECT_EQ(sizeof(fp4_e2m1_x2{}.vals), 1u);
}

TEST(FP4E2M1Test, VariadicHalf) {
  fp4_e2m1_x2 a(sycl::half(1.0f), sycl::half(2.0f));
  // element 0 -> low nibble = 0x2; element 1 -> high nibble = 0x4
  EXPECT_EQ(a.vals[0], 0x42);

  fp4_e2m1 b(sycl::half(1.5f));
  EXPECT_EQ(b.vals[0], 0x3);
}

TEST(FP4E2M1Test, VariadicBFloat16) {
  fp4_e2m1_x2 a(sycl::ext::oneapi::bfloat16(1.0f),
                sycl::ext::oneapi::bfloat16(2.0f));
  EXPECT_EQ(a.vals[0], 0x42);

  fp4_e2m1 b(sycl::ext::oneapi::bfloat16(1.5f));
  EXPECT_EQ(b.vals[0], 0x3);
}

TEST(FP4E2M1Test, VariadicFloat) {
  fp4_e2m1_x2 a(1.0f, 2.0f);
  EXPECT_EQ(a.vals[0], 0x42);

  fp4_e2m1 b(1.5f);
  EXPECT_EQ(b.vals[0], 0x3);
}

TEST(FP4E2M1Test, VariadicBoundaryEncodingsFloat) {
  // Boundaries: max normal, min normal, max/min subnormal, +/-0.
  fp4_e2m1_x2 a(6.0f,  // max finite -> 0x7
                1.0f); // min positive normal -> 0x2
  fp4_e2m1_x2 b(0.5f,  // only positive subnormal -> 0x1
                0.0f); // +0 -> 0x0
  fp4_e2m1_x2 c(0.0f,  // +0
                -0.0f);

  EXPECT_EQ(a.vals[0], static_cast<uint8_t>(0x7 | (0x2 << 4)));
  EXPECT_EQ(b.vals[0], static_cast<uint8_t>(0x1 | (0x0 << 4)));
  EXPECT_EQ(c.vals[0], static_cast<uint8_t>(0x0 | (0x8 << 4)));
}

TEST(FP4E2M1Test, ScalarInfinityClampsToMaxNormalPreservingSign) {
  // Spec: non-stochastic conversion clamps Infinity to max normal preserving
  // sign (E2M1 has no Inf representation).
  fp4_e2m1 pos(std::numeric_limits<float>::infinity());
  fp4_e2m1 neg(-std::numeric_limits<float>::infinity());

  EXPECT_EQ(pos.vals[0], 0x7); // +6.0
  EXPECT_EQ(neg.vals[0], 0xF); // -6.0

  EXPECT_EQ(static_cast<float>(pos), 6.0f);
  EXPECT_EQ(static_cast<float>(neg), -6.0f);
}

TEST(FP4E2M1Test, X2InfinityClampsToMaxNormalPreservingSign) {
  const float in[2] = {std::numeric_limits<float>::infinity(),
                       -std::numeric_limits<float>::infinity()};
  fp4_e2m1_x2 value(in);

  EXPECT_EQ(value.vals[0], static_cast<uint8_t>(0x7 | (0xF << 4)));

  sycl::marray<float, 2> out = static_cast<sycl::marray<float, 2>>(value);
  EXPECT_EQ(out[0], 6.0f);
  EXPECT_EQ(out[1], -6.0f);
}

TEST(FP4E2M1Test, ScalarFiniteOverflowClampsToMaxNormalPreservingSign) {
  fp4_e2m1 pos(1000.0f);
  fp4_e2m1 neg(-1000.0f);

  EXPECT_EQ(pos.vals[0], 0x7);
  EXPECT_EQ(neg.vals[0], 0xF);

  EXPECT_EQ(static_cast<float>(pos), 6.0f);
  EXPECT_EQ(static_cast<float>(neg), -6.0f);
}

TEST(FP4E2M1Test, X2FiniteOverflowClampsToMaxNormalPreservingSign) {
  const float in[2] = {1000.0f, -1000.0f};
  fp4_e2m1_x2 value(in);

  EXPECT_EQ(value.vals[0], static_cast<uint8_t>(0x7 | (0xF << 4)));

  sycl::marray<float, 2> out = static_cast<sycl::marray<float, 2>>(value);
  EXPECT_EQ(out[0], 6.0f);
  EXPECT_EQ(out[1], -6.0f);
}

TEST(FP4E2M1Test, NaNClampsToMaxNormalPreservingSign) {
  // E2M1 has no NaN representation; non-stochastic conversion clamps.
  float pos_nan = std::numeric_limits<float>::quiet_NaN();
  float neg_nan = std::copysign(pos_nan, -1.0f);

  fp4_e2m1_x2 a(pos_nan, neg_nan);
  EXPECT_EQ(a.vals[0], static_cast<uint8_t>(0x7 | (0xF << 4)));
}

TEST(FP4E2M1Test, IntegerToEvenAndSize) {
  // Integer constructors: to_even (CPU host).
  fp4_e2m1 a0(0);
  fp4_e2m1 a1(1);
  fp4_e2m1 a2(2);
  fp4_e2m1 an1(-1);
  fp4_e2m1 an2(-2);

  EXPECT_EQ(a0.vals[0], 0x0);  // +0
  EXPECT_EQ(a1.vals[0], 0x2);  // +1.0
  EXPECT_EQ(a2.vals[0], 0x4);  // +2.0
  EXPECT_EQ(an1.vals[0], 0xA); // -1.0
  EXPECT_EQ(an2.vals[0], 0xC); // -2.0
}

TEST(FP4E2M1Test, IntegerOverflowClampsToMaxNormal) {
  fp4_e2m1 big(100);
  fp4_e2m1 nbig(-100);

  EXPECT_EQ(big.vals[0], 0x7);
  EXPECT_EQ(nbig.vals[0], 0xF);
}

TEST(FP4E2M1Test, AssignmentOperatorToEvenAndSize) {
  fp4_e2m1 a(0.0f);
  EXPECT_EQ(a.vals[0], 0x0);

  a = 1.0f;
  EXPECT_EQ(a.vals[0], 0x2);

  a = -2.0f;
  EXPECT_EQ(a.vals[0], 0xC);

  a = 0.5f; // only positive subnormal
  EXPECT_EQ(a.vals[0], 0x1);
}

TEST(FP4E2M1Test, AssignmentOperatorsAllScalarWidths) {
  fp4_e2m1 value(1.0f);

  EXPECT_EQ(&(value = sycl::half(1.0f)), &value);
  EXPECT_EQ(static_cast<float>(value), 1.0f);

  EXPECT_EQ(&(value = sycl::ext::oneapi::bfloat16(-3.0f)), &value);
  EXPECT_EQ(static_cast<float>(value), -3.0f);

  EXPECT_EQ(&(value = 4.0f), &value);
  EXPECT_EQ(static_cast<float>(value), 4.0f);

  EXPECT_EQ(&(value = static_cast<short>(2)), &value);
  EXPECT_EQ(static_cast<float>(value), 2.0f);

  EXPECT_EQ(&(value = -3), &value);
  EXPECT_EQ(static_cast<float>(value), -3.0f);

  EXPECT_EQ(&(value = 4L), &value);
  EXPECT_EQ(static_cast<float>(value), 4.0f);

  EXPECT_EQ(&(value = -2LL), &value);
  EXPECT_EQ(static_cast<float>(value), -2.0f);

  EXPECT_EQ(&(value = static_cast<unsigned short>(3)), &value);
  EXPECT_EQ(static_cast<float>(value), 3.0f);

  EXPECT_EQ(&(value = 2U), &value);
  EXPECT_EQ(static_cast<float>(value), 2.0f);

  EXPECT_EQ(&(value = 4UL), &value);
  EXPECT_EQ(static_cast<float>(value), 4.0f);

  EXPECT_EQ(&(value = 1ULL), &value);
  EXPECT_EQ(static_cast<float>(value), 1.0f);
}

TEST(FP4E2M1Test, FloatingPointConversionOperators) {
  fp4_e2m1 one(1.0f);
  fp4_e2m1 zero_pos(0.0f);
  fp4_e2m1 zero_neg(-0.0f);
  fp4_e2m1 sub(0.5f); // only positive subnormal

  EXPECT_EQ(one.vals[0], 0x2);
  EXPECT_EQ(zero_pos.vals[0], 0x0);
  EXPECT_EQ(zero_neg.vals[0], 0x8);
  EXPECT_EQ(sub.vals[0], 0x1);

  EXPECT_EQ(static_cast<float>(one), 1.0f);
  EXPECT_EQ(static_cast<float>(zero_pos), 0.0f);

  float fnz = static_cast<float>(zero_neg);
  EXPECT_EQ(fnz, 0.0f);
  EXPECT_TRUE(std::signbit(fnz));

  EXPECT_EQ(static_cast<float>(sub), 0.5f);
}

TEST(FP4E2M1Test, IntegerConversionOperatorsTowardZero) {
  fp4_e2m1 p(1.5f);
  fp4_e2m1 n(-1.5f);

  EXPECT_EQ(p.vals[0], 0x3);
  EXPECT_EQ(n.vals[0], 0xB);

  int ip = static_cast<int>(p);
  int in = static_cast<int>(n);

  EXPECT_EQ(ip, 1);
  EXPECT_EQ(in, -1);
}

TEST(FP4E2M1Test, BoolOperatorZeroRules) {
  fp4_e2m1 zp(0.0f);
  fp4_e2m1 zn(-0.0f);
  fp4_e2m1 one(1.0f);
  fp4_e2m1 sub(0.5f);

  EXPECT_FALSE(static_cast<bool>(zp));
  EXPECT_FALSE(static_cast<bool>(zn));
  EXPECT_TRUE(static_cast<bool>(one));
  EXPECT_TRUE(static_cast<bool>(sub));
}

TEST(FP4E2M1Test, CArrayFloatHostToEvenSaturating) {
  const float in[2] = {1.0f, 1.25f};
  const float in1[2] = {1.0625f, 1000.0f};
  const float in2[2] = {-0.0f, 0.0f};
  fp4_e2m1_x2 a(in);
  fp4_e2m1_x2 a1(in1);
  fp4_e2m1_x2 a2(in2);

  // 1.25 is exactly between 1.0 (frac=0, even) and 1.5 (frac=1, odd).
  // round-to-even -> 1.0 -> 0x2.
  EXPECT_EQ(a.vals[0], static_cast<uint8_t>(0x2 | (0x2 << 4)));

  // 1.0625 closer to 1.0 -> 0x2; 1000.0 -> +6.0 -> 0x7.
  EXPECT_EQ(a1.vals[0], static_cast<uint8_t>(0x2 | (0x7 << 4)));

  // -0, +0
  EXPECT_EQ(a2.vals[0], static_cast<uint8_t>(0x8 | (0x0 << 4)));
}

TEST(FP4E2M1Test, CArrayHalfHostToEvenSaturating) {
  const sycl::half in[2] = {sycl::half(6.0f), sycl::half(7.0f)};
  const sycl::half in1[2] = {sycl::half(1.0f), sycl::half(0.5f)};
  const sycl::half in2[2] = {sycl::half(-0.0f), sycl::half(0.0f)};

  fp4_e2m1_x2 a(in);
  fp4_e2m1_x2 a1(in1);
  fp4_e2m1_x2 a2(in2);

  EXPECT_EQ(a.vals[0], static_cast<uint8_t>(0x7 | (0x7 << 4)));
  EXPECT_EQ(a1.vals[0], static_cast<uint8_t>(0x2 | (0x1 << 4)));
  EXPECT_EQ(a2.vals[0], static_cast<uint8_t>(0x8 | (0x0 << 4)));
}

TEST(FP4E2M1Test, CArrayBFloat16HostToEvenSaturating) {
  const sycl::ext::oneapi::bfloat16 in[2] = {
      sycl::ext::oneapi::bfloat16(6.0f), sycl::ext::oneapi::bfloat16(7.0f)};
  const sycl::ext::oneapi::bfloat16 in1[2] = {
      sycl::ext::oneapi::bfloat16(1.0f), sycl::ext::oneapi::bfloat16(0.5f)};
  const sycl::ext::oneapi::bfloat16 in2[2] = {
      sycl::ext::oneapi::bfloat16(-0.0f), sycl::ext::oneapi::bfloat16(0.0f)};

  fp4_e2m1_x2 a(in);
  fp4_e2m1_x2 a1(in1);
  fp4_e2m1_x2 a2(in2);

  EXPECT_EQ(a.vals[0], static_cast<uint8_t>(0x7 | (0x7 << 4)));
  EXPECT_EQ(a1.vals[0], static_cast<uint8_t>(0x2 | (0x1 << 4)));
  EXPECT_EQ(a2.vals[0], static_cast<uint8_t>(0x8 | (0x0 << 4)));
}

TEST(FP4E2M1Test, MarrayAndOperatorsHostAllN) {
  sycl::marray<float, 2> in = {1.0f, 2.0f};
  sycl::marray<float, 2> in1 = {0.0f, -0.0f};
  sycl::marray<float, 2> in2 = {6.0f, 1000.0f};
  sycl::marray<float, 2> in3 = {0.5f, -1.5f};

  fp4_e2m1_x2 a(in);
  fp4_e2m1_x2 a1(in1);
  fp4_e2m1_x2 a2(in2);
  fp4_e2m1_x2 a3(in3);

  EXPECT_EQ(a.vals[0], static_cast<uint8_t>(0x2 | (0x4 << 4)));
  EXPECT_EQ(a1.vals[0], static_cast<uint8_t>(0x0 | (0x8 << 4)));
  EXPECT_EQ(a2.vals[0], static_cast<uint8_t>(0x7 | (0x7 << 4)));
  EXPECT_EQ(a3.vals[0], static_cast<uint8_t>(0x1 | (0xB << 4)));

  sycl::marray<float, 2> out = static_cast<sycl::marray<float, 2>>(a);
  sycl::marray<float, 2> out1 = static_cast<sycl::marray<float, 2>>(a1);
  sycl::marray<float, 2> out2 = static_cast<sycl::marray<float, 2>>(a2);
  sycl::marray<float, 2> out3 = static_cast<sycl::marray<float, 2>>(a3);
  EXPECT_EQ(out[0], 1.0f);
  EXPECT_EQ(out[1], 2.0f);
  EXPECT_EQ(out1[0], 0.0f);
  EXPECT_EQ(out1[1], 0.0f);
  EXPECT_TRUE(std::signbit(out1[1]));
  EXPECT_EQ(out2[0], 6.0f);
  EXPECT_EQ(out2[1], 6.0f);
  EXPECT_EQ(out3[0], 0.5f);
  EXPECT_EQ(out3[1], -1.5f);
}

TEST(FP4E2M1Test, FloatingPointConversionOperatorsMoreTypes) {
  fp4_e2m1 a(1.0f);
  fp4_e2m1 b(0.5f);

  sycl::half ha = static_cast<sycl::half>(a);
  sycl::ext::oneapi::bfloat16 ba = static_cast<sycl::ext::oneapi::bfloat16>(a);

  EXPECT_EQ(static_cast<float>(ha), 1.0f);
  EXPECT_EQ(static_cast<float>(ba), 1.0f);
  EXPECT_EQ(static_cast<float>(b), 0.5f);
}

TEST(FP4E2M1Test, MarrayConversionOperatorsHalfNumericValues) {
  fp4_e2m1_x2 a(4.0f, -3.0f);

  sycl::marray<sycl::half, 2> out = static_cast<sycl::marray<sycl::half, 2>>(a);

  EXPECT_EQ(static_cast<float>(out[0]), 4.0f);
  EXPECT_EQ(static_cast<float>(out[1]), -3.0f);
}

TEST(FP4E2M1Test, MarrayConversionOperatorsBFloat16NumericValues) {
  fp4_e2m1_x2 a(-2.0f, 1.5f);

  sycl::marray<sycl::ext::oneapi::bfloat16, 2> out =
      static_cast<sycl::marray<sycl::ext::oneapi::bfloat16, 2>>(a);

  EXPECT_EQ(static_cast<float>(out[0]), -2.0f);
  EXPECT_EQ(static_cast<float>(out[1]), 1.5f);
}

TEST(FP4E2M1Test, IntegerConversionOperatorsMultipleWidthsTowardZero) {
  fp4_e2m1 p(1.5f);
  fp4_e2m1 n(-1.5f);

  int i = static_cast<int>(p);
  short s = static_cast<short>(n);
  long l = static_cast<long>(p);
  long long ll = static_cast<long long>(n);

  EXPECT_EQ(i, 1);
  EXPECT_EQ(s, -1);
  EXPECT_EQ(l, 1);
  EXPECT_EQ(ll, -1);
}

TEST(FP4E2M1Test, IntegerConversionOperatorsRemainingWidthsTowardZero) {
  fp4_e2m1 pos_char(3.0f);
  fp4_e2m1 neg_schar(-2.0f);
  fp4_e2m1 pos_uchar(4.0f);
  fp4_e2m1 pos_ushort(6.0f);
  fp4_e2m1 pos_uint(2.0f);
  fp4_e2m1 pos_ulong(4.0f);
  fp4_e2m1 pos_ull(6.0f);

  char c = static_cast<char>(pos_char);
  signed char sc = static_cast<signed char>(neg_schar);
  unsigned char uc = static_cast<unsigned char>(pos_uchar);
  unsigned short us = static_cast<unsigned short>(pos_ushort);
  unsigned int ui = static_cast<unsigned int>(pos_uint);
  unsigned long ul = static_cast<unsigned long>(pos_ulong);
  unsigned long long ull = static_cast<unsigned long long>(pos_ull);

  EXPECT_EQ(c, static_cast<char>(3));
  EXPECT_EQ(sc, static_cast<signed char>(-2));
  EXPECT_EQ(uc, static_cast<unsigned char>(4));
  EXPECT_EQ(us, static_cast<unsigned short>(6));
  EXPECT_EQ(ui, 2u);
  EXPECT_EQ(ul, 4ul);
  EXPECT_EQ(ull, 6ull);
}

TEST(FP4E2M1Test, CArrayFloatRoundingToEven) {
  // 1.25 ties between 1.0 (even) and 1.5 (odd) -> to_even = 1.0.
  // 1000 saturates to +6.0.
  const float in[2] = {1.25f, 1000.0f};
  fp4_e2m1_x2 a(in, rounding::to_even);

  EXPECT_EQ(a.vals[0], static_cast<uint8_t>(0x2 | (0x7 << 4)));
}

TEST(FP4E2M1Test, CArrayHalfRoundingToEven) {
  const sycl::half in[2] = {sycl::half(1.25f), sycl::half(1000.0f)};
  fp4_e2m1_x2 a(in, rounding::to_even);

  EXPECT_EQ(a.vals[0], static_cast<uint8_t>(0x2 | (0x7 << 4)));
}

TEST(FP4E2M1Test, CArrayBFloat16RoundingToEven) {
  const sycl::ext::oneapi::bfloat16 in[2] = {
      sycl::ext::oneapi::bfloat16(1.25f),
      sycl::ext::oneapi::bfloat16(1000.0f)};
  fp4_e2m1_x2 a(in, rounding::to_even);

  EXPECT_EQ(a.vals[0], static_cast<uint8_t>(0x2 | (0x7 << 4)));
}

TEST(FP4E2M1Test, MarrayHalfRoundingToEven) {
  const sycl::marray<sycl::half, 2> in = {sycl::half(1.25f), sycl::half(2.5f)};
  fp4_e2m1_x2 a(in, rounding::to_even);

  // 2.5 ties between 2.0 (even) and 3.0 (odd) -> to_even = 2.0 -> 0x4.
  EXPECT_EQ(a.vals[0], static_cast<uint8_t>(0x2 | (0x4 << 4)));
}

TEST(FP4E2M1Test, MarrayBFloat16RoundingToEven) {
  const sycl::marray<sycl::ext::oneapi::bfloat16, 2> in = {
      sycl::ext::oneapi::bfloat16(1.25f), sycl::ext::oneapi::bfloat16(2.5f)};
  fp4_e2m1_x2 a(in, rounding::to_even);

  EXPECT_EQ(a.vals[0], static_cast<uint8_t>(0x2 | (0x4 << 4)));
}

TEST(FP4E2M1Test, MarrayFloatRoundingToEven) {
  const sycl::marray<float, 2> in = {1.25f, 2.5f};
  fp4_e2m1_x2 a(in, rounding::to_even);

  EXPECT_EQ(a.vals[0], static_cast<uint8_t>(0x2 | (0x4 << 4)));
}

TEST(FP4E2M1Test, VariadicRejectsMixedTypes) {
  EXPECT_FALSE((std::is_constructible_v<fp4_e2m1_x2, float, sycl::half>));
  EXPECT_FALSE((std::is_constructible_v<fp4_e2m1_x2, sycl::half, float>));
}

TEST(FP4E2M1Test, X2NotConstructibleFromSingleShort) {
  EXPECT_FALSE((std::is_constructible_v<fp4_e2m1_x2, short>));
}

TEST(FP4E2M1Test, X2NotConstructibleFromSingleInt) {
  EXPECT_FALSE((std::is_constructible_v<fp4_e2m1_x2, int>));
}

TEST(FP4E2M1Test, X2NotConstructibleFromSingleLong) {
  EXPECT_FALSE((std::is_constructible_v<fp4_e2m1_x2, long>));
}

TEST(FP4E2M1Test, X2NotConstructibleFromSingleLL) {
  EXPECT_FALSE((std::is_constructible_v<fp4_e2m1_x2, long long>));
}

TEST(FP4E2M1Test, X2NotConstructibleFromSingleUShort) {
  EXPECT_FALSE((std::is_constructible_v<fp4_e2m1_x2, unsigned short>));
}

TEST(FP4E2M1Test, X2NotConstructibleFromSingleUInt) {
  EXPECT_FALSE((std::is_constructible_v<fp4_e2m1_x2, unsigned int>));
}

TEST(FP4E2M1Test, X2NotConstructibleFromSingleUL) {
  EXPECT_FALSE((std::is_constructible_v<fp4_e2m1_x2, unsigned long>));
}

TEST(FP4E2M1Test, X2NotConstructibleFromSingleULL) {
  EXPECT_FALSE((std::is_constructible_v<fp4_e2m1_x2, unsigned long long>));
}

TEST(FP4E2M1Test, X2NotConstructibleFromSingleFloat) {
  EXPECT_FALSE((std::is_constructible_v<fp4_e2m1_x2, float>));
}

TEST(FP4E2M1Test, X2NotConstructibleFromSingleBFloat16) {
  EXPECT_FALSE(
      (std::is_constructible_v<fp4_e2m1_x2, sycl::ext::oneapi::bfloat16>));
}

TEST(FP4E2M1Test, X2NotConstructibleFromSingleHalf) {
  EXPECT_FALSE((std::is_constructible_v<fp4_e2m1_x2, sycl::half>));
}

TEST(FP4E2M1Test, X2NotConstructibleFromSingleChar) {
  EXPECT_FALSE((std::is_constructible_v<fp4_e2m1_x2, char>));
}

TEST(FP4E2M1Test, X2NotConstructibleFromSingleUChar) {
  EXPECT_FALSE((std::is_constructible_v<fp4_e2m1_x2, unsigned char>));
}

TEST(FP4E2M1Test, X2NotAssignableFromSingleHalf) {
  EXPECT_FALSE((std::is_assignable_v<fp4_e2m1_x2 &, sycl::half>));
}

TEST(FP4E2M1Test, X2NotAssignableFromSingleBFloat16) {
  EXPECT_FALSE(
      (std::is_assignable_v<fp4_e2m1_x2 &, sycl::ext::oneapi::bfloat16>));
}

TEST(FP4E2M1Test, X2NotAssignableFromSingleFloat) {
  EXPECT_FALSE((std::is_assignable_v<fp4_e2m1_x2 &, float>));
}

TEST(FP4E2M1Test, X2NotAssignableFromSingleChar) {
  EXPECT_FALSE((std::is_assignable_v<fp4_e2m1_x2 &, char>));
}

TEST(FP4E2M1Test, X2NotAssignableFromSingleSignedChar) {
  EXPECT_FALSE((std::is_assignable_v<fp4_e2m1_x2 &, signed char>));
}

TEST(FP4E2M1Test, X2NotAssignableFromSingleUChar) {
  EXPECT_FALSE((std::is_assignable_v<fp4_e2m1_x2 &, unsigned char>));
}

TEST(FP4E2M1Test, X2NotAssignableFromSingleShort) {
  EXPECT_FALSE((std::is_assignable_v<fp4_e2m1_x2 &, short>));
}

TEST(FP4E2M1Test, X2NotAssignableFromSingleInt) {
  EXPECT_FALSE((std::is_assignable_v<fp4_e2m1_x2 &, int>));
}

TEST(FP4E2M1Test, X2NotAssignableFromSingleLong) {
  EXPECT_FALSE((std::is_assignable_v<fp4_e2m1_x2 &, long>));
}

TEST(FP4E2M1Test, X2NotAssignableFromSingleLL) {
  EXPECT_FALSE((std::is_assignable_v<fp4_e2m1_x2 &, long long>));
}

TEST(FP4E2M1Test, X2NotAssignableFromSingleUShort) {
  EXPECT_FALSE((std::is_assignable_v<fp4_e2m1_x2 &, unsigned short>));
}

TEST(FP4E2M1Test, X2NotAssignableFromSingleUInt) {
  EXPECT_FALSE((std::is_assignable_v<fp4_e2m1_x2 &, unsigned int>));
}

TEST(FP4E2M1Test, X2NotAssignableFromSingleUL) {
  EXPECT_FALSE((std::is_assignable_v<fp4_e2m1_x2 &, unsigned long>));
}

TEST(FP4E2M1Test, X2NotAssignableFromSingleULL) {
  EXPECT_FALSE((std::is_assignable_v<fp4_e2m1_x2 &, unsigned long long>));
}

#if LLVM_ENABLE_ASSERTIONS
TEST(FP4E2M1Test, CArrayHalfRejectsTowardZeroRounding) {
  const sycl::half in[2] = {sycl::half(1.0f), sycl::half(2.0f)};
  EXPECT_DEATH(
      {
        fp4_e2m1_x2 value(in, rounding::toward_zero);
        (void)value;
      },
      "fp4_e2m1_x: only rounding::to_even is supported");
}

TEST(FP4E2M1Test, CArrayBFloat16RejectsTowardZeroRounding) {
  const sycl::ext::oneapi::bfloat16 in[2] = {sycl::ext::oneapi::bfloat16(1.0f),
                                             sycl::ext::oneapi::bfloat16(2.0f)};
  EXPECT_DEATH(
      {
        fp4_e2m1_x2 value(in, rounding::toward_zero);
        (void)value;
      },
      "fp4_e2m1_x: only rounding::to_even is supported");
}

TEST(FP4E2M1Test, CArrayFloatRejectsTowardZeroRounding) {
  const float in[2] = {1.0f, 2.0f};
  EXPECT_DEATH(
      {
        fp4_e2m1_x2 value(in, rounding::toward_zero);
        (void)value;
      },
      "fp4_e2m1_x: only rounding::to_even is supported");
}

TEST(FP4E2M1Test, MarrayHalfRejectsTowardZeroRounding) {
  const sycl::marray<sycl::half, 2> in = {sycl::half(1.0f), sycl::half(2.0f)};
  EXPECT_DEATH(
      {
        fp4_e2m1_x2 value(in, rounding::toward_zero);
        (void)value;
      },
      "fp4_e2m1_x: only rounding::to_even is supported");
}

TEST(FP4E2M1Test, MarrayBFloat16RejectsTowardZeroRounding) {
  const sycl::marray<sycl::ext::oneapi::bfloat16, 2> in = {
      sycl::ext::oneapi::bfloat16(1.0f), sycl::ext::oneapi::bfloat16(2.0f)};
  EXPECT_DEATH(
      {
        fp4_e2m1_x2 value(in, rounding::toward_zero);
        (void)value;
      },
      "fp4_e2m1_x: only rounding::to_even is supported");
}

TEST(FP4E2M1Test, MarrayFloatRejectsTowardZeroRounding) {
  const sycl::marray<float, 2> in = {1.0f, 2.0f};
  EXPECT_DEATH(
      {
        fp4_e2m1_x2 value(in, rounding::toward_zero);
        (void)value;
      },
      "fp4_e2m1_x: only rounding::to_even is supported");
}
#endif // LLVM_ENABLE_ASSERTIONS

TEST(FP4E2M1Test, VariadicFloatReferences) {
  float x = 1.0f;
  float y = 2.0f;
  float &xf = x;
  float &yf = y;

  fp4_e2m1_x2 a(xf, yf);

  EXPECT_EQ(a.vals[0], static_cast<uint8_t>(0x2 | (0x4 << 4)));
}
