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

namespace {

constexpr const char *UnsupportedRoundingAssertRegex =
    "fp8_e8m0_x: only rounding::upward and rounding::toward_zero are "
    "\" \"supported";

bool checkCode(float Input, rounding Mode, uint8_t Expected) {
  const float Values[1] = {Input};
  const fp8_e8m0 Encoded(Values, Mode);
  return Encoded.vals[0] == Expected;
}

} // namespace

TEST(FP8E8M0Test, VariadicFloat) {
  fp8_e8m0_x2 a(1.0f, 2.0f);
  fp8_e8m0_x2 a1(1.1f, 0.0f);

  EXPECT_EQ(sizeof(a.vals), 2u);
  EXPECT_EQ(sizeof(a1.vals), 2u);
  EXPECT_EQ(a.vals[0], 0x7F);  // 1.0 -> exp=127
  EXPECT_EQ(a.vals[1], 0x80);  // 2.0 -> exp=128
  EXPECT_EQ(a1.vals[0], 0x80); // 1.1 -> upward to 2.0
  EXPECT_EQ(a1.vals[1], 0x00); // 0.0 -> min normal
}

TEST(FP8E8M0Test, VariadicHalf) {
  fp8_e8m0_x2 a(sycl::half(1.0f), sycl::half(3.0f));

  EXPECT_EQ(sizeof(a.vals), 2u);
  EXPECT_EQ(a.vals[0], 0x7F);
  EXPECT_EQ(a.vals[1], 0x81); // 3.0 -> upward to 4.0
}

TEST(FP8E8M0Test, VariadicBFloat16) {
  fp8_e8m0_x2 a(sycl::ext::oneapi::bfloat16(1.0f),
                sycl::ext::oneapi::bfloat16(2.0f));

  EXPECT_EQ(sizeof(a.vals), 2u);
  EXPECT_EQ(a.vals[0], 0x7F);
  EXPECT_EQ(a.vals[1], 0x80);
}

TEST(FP8E8M0Test, VariadicDouble) {
  fp8_e8m0_x2 a(1.0, 3.0);

  EXPECT_EQ(sizeof(a.vals), 2u);
  EXPECT_EQ(a.vals[0], 0x7F);
  EXPECT_EQ(a.vals[1], 0x81);
}

TEST(FP8E8M0Test, VariadicBoundaryEncodings) {
  fp8_e8m0_x2 a(std::ldexp(1.0f, -127),
                std::numeric_limits<float>::quiet_NaN());

  EXPECT_EQ(sizeof(a.vals), 2u);
  EXPECT_EQ(a.vals[0], 0x00); // min normal
  EXPECT_EQ(a.vals[1], 0xFF); // NaN
}

TEST(FP8E8M0Test, CArrayFloatHostUpwardFinite) {
  const float in[2] = {1.0f, 1.1f};
  const float in1[2] = {3.0f, 1000.0f};
  fp8_e8m0_x2 a(in, rounding::upward);
  fp8_e8m0_x2 a1(in1, rounding::upward);

  EXPECT_EQ(sizeof(a.vals), 2u);
  EXPECT_EQ(sizeof(a1.vals), 2u);
  EXPECT_EQ(a.vals[0], 0x7F);
  EXPECT_EQ(a.vals[1], 0x80);  // upward to 2.0
  EXPECT_EQ(a1.vals[0], 0x81); // upward to 4.0
  EXPECT_EQ(a1.vals[1], 0x89); // upward to 2^10 = 1024
}

TEST(FP8E8M0Test, CArrayFloatRoundingModes) {
  EXPECT_TRUE(checkCode(3.0f, rounding::upward, 0x81));
  EXPECT_TRUE(checkCode(3.0f, rounding::toward_zero, 0x80));

  // E8M0 drops sign per the extension specification, so negative inputs are
  // rounded using their magnitude.
  EXPECT_TRUE(checkCode(-3.0f, rounding::upward, 0x81));
  EXPECT_TRUE(checkCode(-3.0f, rounding::toward_zero, 0x80));
  EXPECT_TRUE(checkCode(-1.5f, rounding::upward, 0x80));
  EXPECT_TRUE(checkCode(-1.5f, rounding::toward_zero, 0x7F));
  EXPECT_TRUE(checkCode(-0.5f, rounding::upward, 0x7E));
  EXPECT_TRUE(checkCode(-0.5f, rounding::toward_zero, 0x7E));

  EXPECT_TRUE(checkCode(1.0f, rounding::upward, 0x7F));
  EXPECT_TRUE(checkCode(0.5f, rounding::upward, 0x7E));
  EXPECT_TRUE(checkCode(0.5f, rounding::toward_zero, 0x7E));
  EXPECT_TRUE(checkCode(0.0f, rounding::toward_zero, 0x00));
  EXPECT_TRUE(checkCode(std::numeric_limits<float>::quiet_NaN(),
                        rounding::upward, 0xFF));
}

TEST(FP8E8M0Test, RoundClipZeroFractionNegativeAndTieCases) {
  EXPECT_EQ(detail::RoundClip(0.25f, 0, rounding::upward, 0u), 1u);
  EXPECT_EQ(detail::RoundClip(0.25f, 0, rounding::upward, 1u), 0u);
  EXPECT_EQ(detail::RoundClip(0.5f, 0, rounding::to_even, 0u), 0u);
  EXPECT_EQ(detail::RoundClip(0.75f, 0, rounding::to_even, 0u), 1u);
}

TEST(FP8E8M0Test, CArrayHalfHostUpwardFinite) {
  const sycl::half in[2] = {sycl::half(1.0f), sycl::half(1.1f)};
  const sycl::half in1[2] = {sycl::half(3.0f), sycl::half(0.0f)};

  fp8_e8m0_x2 a(in, rounding::upward);
  fp8_e8m0_x2 a1(in1, rounding::upward);

  EXPECT_EQ(sizeof(a.vals), 2u);
  EXPECT_EQ(sizeof(a1.vals), 2u);
  EXPECT_EQ(a.vals[0], 0x7F);
  EXPECT_EQ(a.vals[1], 0x80);
  EXPECT_EQ(a1.vals[0], 0x81);
  EXPECT_EQ(a1.vals[1], 0x00);
}

TEST(FP8E8M0Test, CArrayBFloat16HostUpwardFinite) {
  const sycl::ext::oneapi::bfloat16 in[2] = {sycl::ext::oneapi::bfloat16(1.0f),
                                             sycl::ext::oneapi::bfloat16(2.0f)};
  fp8_e8m0_x2 a(in, rounding::upward);

  EXPECT_EQ(sizeof(a.vals), 2u);
  EXPECT_EQ(a.vals[0], 0x7F);
  EXPECT_EQ(a.vals[1], 0x80);
}

TEST(FP8E8M0Test, CArrayDoubleDefaultUpwardFinite) {
  const double in[2] = {1.0, 3.0};
  fp8_e8m0_x2 a(in);

  EXPECT_EQ(sizeof(a.vals), 2u);
  EXPECT_EQ(a.vals[0], 0x7F);
  EXPECT_EQ(a.vals[1], 0x81);
}

TEST(FP8E8M0Test, MarrayAndOperatorsFloat) {
  sycl::marray<float, 2> in = {1.0f, 2.0f};
  sycl::marray<float, 2> in1 = {3.0f, 0.0f};

  fp8_e8m0_x2 a(in, rounding::upward);
  fp8_e8m0_x2 a1(in1, rounding::upward);

  EXPECT_EQ(sizeof(a.vals), 2u);
  EXPECT_EQ(sizeof(a1.vals), 2u);
  EXPECT_EQ(a.vals[0], 0x7F);
  EXPECT_EQ(a.vals[1], 0x80);
  EXPECT_EQ(a1.vals[0], 0x81);
  EXPECT_EQ(a1.vals[1], 0x00);

  sycl::marray<float, 2> out = static_cast<sycl::marray<float, 2>>(a);
  sycl::marray<float, 2> out1 = static_cast<sycl::marray<float, 2>>(a1);
  EXPECT_EQ(out[0], 1.0f);
  EXPECT_EQ(out[1], 2.0f);
  EXPECT_EQ(out1[0], 4.0f);
  EXPECT_EQ(out1[1], std::ldexp(1.0f, -127));
}

TEST(FP8E8M0Test, MarrayHalfBFloat16Double) {
  sycl::marray<sycl::half, 2> hvals = {sycl::half(1.0f), sycl::half(3.0f)};
  sycl::marray<sycl::ext::oneapi::bfloat16, 2> bvals = {
      sycl::ext::oneapi::bfloat16(1.0f), sycl::ext::oneapi::bfloat16(2.0f)};
  sycl::marray<double, 2> dvals = {1.0, 3.0};

  fp8_e8m0_x2 ah(hvals, rounding::upward);
  fp8_e8m0_x2 ab(bvals, rounding::upward);
  fp8_e8m0_x2 ad(dvals);

  EXPECT_EQ(sizeof(ah.vals), 2u);
  EXPECT_EQ(sizeof(ab.vals), 2u);
  EXPECT_EQ(sizeof(ad.vals), 2u);

  EXPECT_EQ(ah.vals[0], 0x7F);
  EXPECT_EQ(ah.vals[1], 0x81);
  EXPECT_EQ(ab.vals[0], 0x7F);
  EXPECT_EQ(ab.vals[1], 0x80);
  EXPECT_EQ(ad.vals[0], 0x7F);
  EXPECT_EQ(ad.vals[1], 0x81);
}

TEST(FP8E8M0Test, IntegerConstructorsAllTypes) {
  fp8_e8m0 s(static_cast<short>(1));
  fp8_e8m0 i(static_cast<int>(2));
  fp8_e8m0 l(static_cast<long>(3));
  fp8_e8m0 ll(static_cast<long long>(4));
  fp8_e8m0 us(static_cast<unsigned short>(1));
  fp8_e8m0 ui(static_cast<unsigned int>(2));
  fp8_e8m0 ul(static_cast<unsigned long>(3));
  fp8_e8m0 ull(static_cast<unsigned long long>(4));

  EXPECT_EQ(sizeof(s.vals), 1u);
  EXPECT_EQ(sizeof(i.vals), 1u);
  EXPECT_EQ(sizeof(l.vals), 1u);
  EXPECT_EQ(sizeof(ll.vals), 1u);
  EXPECT_EQ(sizeof(us.vals), 1u);
  EXPECT_EQ(sizeof(ui.vals), 1u);
  EXPECT_EQ(sizeof(ul.vals), 1u);
  EXPECT_EQ(sizeof(ull.vals), 1u);

  EXPECT_EQ(s.vals[0], 0x7F);  // 1.0
  EXPECT_EQ(i.vals[0], 0x80);  // 2.0
  EXPECT_EQ(l.vals[0], 0x81);  // 3.0 -> upward to 4.0
  EXPECT_EQ(ll.vals[0], 0x81); // 4.0
  EXPECT_EQ(us.vals[0], 0x7F);
  EXPECT_EQ(ui.vals[0], 0x80);
  EXPECT_EQ(ul.vals[0], 0x81);
  EXPECT_EQ(ull.vals[0], 0x81);
}

TEST(FP8E8M0Test, AssignmentOperatorsAllTypes) {
  fp8_e8m0 a(1.0f);
  EXPECT_EQ(sizeof(a.vals), 1u);

  a = sycl::half(1.0f);
  EXPECT_EQ(a.vals[0], 0x7F);

  a = sycl::ext::oneapi::bfloat16(2.0f);
  EXPECT_EQ(a.vals[0], 0x80);

  a = 3.0f;
  EXPECT_EQ(a.vals[0], 0x81);

  a = 4.0;
  EXPECT_EQ(a.vals[0], 0x81);

  a = static_cast<short>(1);
  EXPECT_EQ(a.vals[0], 0x7F);

  a = static_cast<int>(2);
  EXPECT_EQ(a.vals[0], 0x80);

  a = static_cast<long>(3);
  EXPECT_EQ(a.vals[0], 0x81);

  a = static_cast<long long>(4);
  EXPECT_EQ(a.vals[0], 0x81);

  a = static_cast<unsigned short>(1);
  EXPECT_EQ(a.vals[0], 0x7F);

  a = static_cast<unsigned int>(2);
  EXPECT_EQ(a.vals[0], 0x80);

  a = static_cast<unsigned long>(3);
  EXPECT_EQ(a.vals[0], 0x81);

  a = static_cast<unsigned long long>(4);
  EXPECT_EQ(a.vals[0], 0x81);
}

TEST(FP8E8M0Test, FloatingPointConversionOperators) {
  fp8_e8m0 one(1.0f);
  fp8_e8m0 max(std::ldexp(1.0f, 127));
  fp8_e8m0 min(std::ldexp(1.0f, -127));

  EXPECT_EQ(sizeof(one.vals), 1u);
  EXPECT_EQ(one.vals[0], 0x7F);
  EXPECT_EQ(max.vals[0], 0xFE);
  EXPECT_EQ(min.vals[0], 0x00);

  float fo = static_cast<float>(one);
  double doo = static_cast<double>(one);
  sycl::half ho = static_cast<sycl::half>(one);
  sycl::ext::oneapi::bfloat16 bo =
      static_cast<sycl::ext::oneapi::bfloat16>(one);

  EXPECT_EQ(fo, 1.0f);
  EXPECT_EQ(doo, 1.0);
  EXPECT_EQ(static_cast<float>(ho), 1.0f);
  EXPECT_EQ(static_cast<float>(bo), 1.0f);

  sycl::half hmax = static_cast<sycl::half>(max);
  EXPECT_TRUE(std::isinf(static_cast<float>(hmax)));
  EXPECT_FALSE(std::signbit(static_cast<float>(hmax)));

  EXPECT_EQ(static_cast<float>(min), std::ldexp(1.0f, -127));
}

TEST(FP8E8M0Test, BoolOperatorAlwaysTrue) {
  fp8_e8m0 min(std::ldexp(1.0f, -127));
  fp8_e8m0 nanv(std::numeric_limits<float>::quiet_NaN());

  EXPECT_TRUE(static_cast<bool>(min));
  EXPECT_TRUE(static_cast<bool>(nanv));
}

TEST(FP8E8M0Test, MarrayConversionOperators) {
  fp8_e8m0_x2 a(1.0f, 3.0f);

  sycl::marray<sycl::half, 2> ho = static_cast<sycl::marray<sycl::half, 2>>(a);
  sycl::marray<sycl::ext::oneapi::bfloat16, 2> bo =
      static_cast<sycl::marray<sycl::ext::oneapi::bfloat16, 2>>(a);
  sycl::marray<float, 2> fo = static_cast<sycl::marray<float, 2>>(a);

  EXPECT_EQ(static_cast<float>(ho[0]), 1.0f);
  EXPECT_EQ(static_cast<float>(ho[1]), 4.0f);

  EXPECT_EQ(static_cast<float>(bo[0]), 1.0f);
  EXPECT_EQ(static_cast<float>(bo[1]), 4.0f);

  EXPECT_EQ(fo[0], 1.0f);
  EXPECT_EQ(fo[1], 4.0f);
}

TEST(FP8E8M0Test, VariadicMixedTypes) {
  EXPECT_FALSE((std::is_constructible_v<fp8_e8m0_x2, float, sycl::half>));
  EXPECT_FALSE((std::is_constructible_v<fp8_e8m0_x2,
                                        sycl::ext::oneapi::bfloat16, double>));
}

TEST(FP8E8M0Test, X2NotConstructibleFromSingleShort) {
  EXPECT_FALSE((std::is_constructible_v<fp8_e8m0_x2, short>));
}

TEST(FP8E8M0Test, X2NotConstructibleFromSingleInt) {
  EXPECT_FALSE((std::is_constructible_v<fp8_e8m0_x2, int>));
}

TEST(FP8E8M0Test, X2NotConstructibleFromSingleLong) {
  EXPECT_FALSE((std::is_constructible_v<fp8_e8m0_x2, long>));
}

TEST(FP8E8M0Test, X2NotConstructibleFromSingleLL) {
  EXPECT_FALSE((std::is_constructible_v<fp8_e8m0_x2, long long>));
}

TEST(FP8E8M0Test, X2NotConstructibleFromSingleUShort) {
  EXPECT_FALSE((std::is_constructible_v<fp8_e8m0_x2, unsigned short>));
}

TEST(FP8E8M0Test, X2NotConstructibleFromSingleUInt) {
  EXPECT_FALSE((std::is_constructible_v<fp8_e8m0_x2, unsigned int>));
}

TEST(FP8E8M0Test, X2NotConstructibleFromSingleUL) {
  EXPECT_FALSE((std::is_constructible_v<fp8_e8m0_x2, unsigned long>));
}

TEST(FP8E8M0Test, X2NotConstructibleFromSingleULL) {
  EXPECT_FALSE((std::is_constructible_v<fp8_e8m0_x2, unsigned long long>));
}

TEST(FP8E8M0Test, X2NotConstructibleFromSingleFloat) {
  EXPECT_FALSE((std::is_constructible_v<fp8_e8m0_x2, float>));
}

TEST(FP8E8M0Test, X2NotConstructibleFromSingleDouble) {
  EXPECT_FALSE((std::is_constructible_v<fp8_e8m0_x2, double>));
}

TEST(FP8E8M0Test, X2NotConstructibleFromSingleBFloat16) {
  EXPECT_FALSE(
      (std::is_constructible_v<fp8_e8m0_x2, sycl::ext::oneapi::bfloat16>));
}

TEST(FP8E8M0Test, X2NotConstructibleFromSingleHalf) {
  EXPECT_FALSE((std::is_constructible_v<fp8_e8m0_x2, sycl::half>));
}

TEST(FP8E8M0Test, X2NotConstructibleFromSingleChar) {
  EXPECT_FALSE((std::is_constructible_v<fp8_e8m0_x2, char>));
}

TEST(FP8E8M0Test, X2NotConstructibleFromSingleUChar) {
  EXPECT_FALSE((std::is_constructible_v<fp8_e8m0_x2, unsigned char>));
}

TEST(FP8E8M0Test, X2NotAssignableFromSingleHalf) {
  EXPECT_FALSE((std::is_assignable_v<fp8_e8m0_x2 &, sycl::half>));
}

TEST(FP8E8M0Test, X2NotAssignableFromSingleBFloat16) {
  EXPECT_FALSE(
      (std::is_assignable_v<fp8_e8m0_x2 &, sycl::ext::oneapi::bfloat16>));
}

TEST(FP8E8M0Test, X2NotAssignableFromSingleFloat) {
  EXPECT_FALSE((std::is_assignable_v<fp8_e8m0_x2 &, float>));
}

TEST(FP8E8M0Test, X2NotAssignableFromSingleDouble) {
  EXPECT_FALSE((std::is_assignable_v<fp8_e8m0_x2 &, double>));
}

TEST(FP8E8M0Test, X2NotAssignableFromSingleChar) {
  EXPECT_FALSE((std::is_assignable_v<fp8_e8m0_x2 &, char>));
}

TEST(FP8E8M0Test, X2NotAssignableFromSingleSignedChar) {
  EXPECT_FALSE((std::is_assignable_v<fp8_e8m0_x2 &, signed char>));
}

TEST(FP8E8M0Test, X2NotAssignableFromSingleUChar) {
  EXPECT_FALSE((std::is_assignable_v<fp8_e8m0_x2 &, unsigned char>));
}

TEST(FP8E8M0Test, X2NotAssignableFromSingleShort) {
  EXPECT_FALSE((std::is_assignable_v<fp8_e8m0_x2 &, short>));
}

TEST(FP8E8M0Test, X2NotAssignableFromSingleInt) {
  EXPECT_FALSE((std::is_assignable_v<fp8_e8m0_x2 &, int>));
}

TEST(FP8E8M0Test, X2NotAssignableFromSingleLong) {
  EXPECT_FALSE((std::is_assignable_v<fp8_e8m0_x2 &, long>));
}

TEST(FP8E8M0Test, X2NotAssignableFromSingleLL) {
  EXPECT_FALSE((std::is_assignable_v<fp8_e8m0_x2 &, long long>));
}

TEST(FP8E8M0Test, X2NotAssignableFromSingleUShort) {
  EXPECT_FALSE((std::is_assignable_v<fp8_e8m0_x2 &, unsigned short>));
}

TEST(FP8E8M0Test, X2NotAssignableFromSingleUInt) {
  EXPECT_FALSE((std::is_assignable_v<fp8_e8m0_x2 &, unsigned int>));
}

TEST(FP8E8M0Test, X2NotAssignableFromSingleUL) {
  EXPECT_FALSE((std::is_assignable_v<fp8_e8m0_x2 &, unsigned long>));
}

TEST(FP8E8M0Test, X2NotAssignableFromSingleULL) {
  EXPECT_FALSE((std::is_assignable_v<fp8_e8m0_x2 &, unsigned long long>));
}

TEST(FP8E8M0Test, CArrayHalfToEvenRounding) {
  const sycl::half in[2] = {sycl::half(1.0f), sycl::half(2.0f)};
  EXPECT_DEATH(
      {
        fp8_e8m0_x2 value(in, rounding::to_even);
        (void)value;
      },
      UnsupportedRoundingAssertRegex);
}

TEST(FP8E8M0Test, CArrayBFloat16ToEvenRounding) {
  const sycl::ext::oneapi::bfloat16 in[2] = {sycl::ext::oneapi::bfloat16(1.0f),
                                             sycl::ext::oneapi::bfloat16(2.0f)};
  EXPECT_DEATH(
      {
        fp8_e8m0_x2 value(in, rounding::to_even);
        (void)value;
      },
      UnsupportedRoundingAssertRegex);
}

TEST(FP8E8M0Test, CArrayFloatToEvenRounding) {
  const float in[2] = {1.0f, 2.0f};
  EXPECT_DEATH(
      {
        fp8_e8m0_x2 value(in, rounding::to_even);
        (void)value;
      },
      UnsupportedRoundingAssertRegex);
}

TEST(FP8E8M0Test, MarrayHalfToEvenRounding) {
  const sycl::marray<sycl::half, 2> in = {sycl::half(1.0f), sycl::half(2.0f)};
  EXPECT_DEATH(
      {
        fp8_e8m0_x2 value(in, rounding::to_even);
        (void)value;
      },
      UnsupportedRoundingAssertRegex);
}

TEST(FP8E8M0Test, MarrayBFloat16ToEvenRounding) {
  const sycl::marray<sycl::ext::oneapi::bfloat16, 2> in = {
      sycl::ext::oneapi::bfloat16(1.0f), sycl::ext::oneapi::bfloat16(2.0f)};
  EXPECT_DEATH(
      {
        fp8_e8m0_x2 value(in, rounding::to_even);
        (void)value;
      },
      UnsupportedRoundingAssertRegex);
}

TEST(FP8E8M0Test, MarrayFloatToEvenRounding) {
  const sycl::marray<float, 2> in = {1.0f, 2.0f};
  EXPECT_DEATH(
      {
        fp8_e8m0_x2 value(in, rounding::to_even);
        (void)value;
      },
      UnsupportedRoundingAssertRegex);
}
