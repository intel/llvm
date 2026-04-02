#include <sycl/ext/oneapi/bfloat16.hpp>
#include <sycl/sycl.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <string>

namespace {
// Helper to convert the expected bits to float value to compare with the result.
typedef union {
  float Value;
  struct {
    uint32_t Mantissa : 23;
    uint32_t Exponent : 8;
    uint32_t Sign : 1;
  } RawData;
} floatConvHelper;

float bitsToFloatConv(std::string Bits) {
  floatConvHelper Helper;
  Helper.RawData.Sign = static_cast<uint32_t>(Bits[0] - '0');
  uint32_t Exponent = 0;
  for (size_t I = 1; I != 9; ++I)
    Exponent = Exponent + static_cast<uint32_t>(Bits[I] - '0') * std::pow(2, 8 - I);
  Helper.RawData.Exponent = Exponent;
  uint32_t Mantissa = 0;
  for (size_t I = 9; I != 32; ++I)
    Mantissa = Mantissa + static_cast<uint32_t>(Bits[I] - '0') * std::pow(2, 31 - I);
  Helper.RawData.Mantissa = Mantissa;
  return Helper.Value;
}
} // namespace

TEST(BFloat16, BF16FromFloat) {
  sycl::ext::oneapi::bfloat16 B = 0.0f;
  auto Result = sycl::bit_cast<uint16_t>(B);
  ASSERT_EQ(Result, std::stoi("0000000000000000", nullptr, 2));

  B = 42.0f;
  Result = sycl::bit_cast<uint16_t>(B);
  ASSERT_EQ(Result, std::stoi("100001000101000", nullptr, 2));

  B = std::numeric_limits<float>::min();
  Result = sycl::bit_cast<uint16_t>(B);
  ASSERT_EQ(Result, std::stoi("0000000010000000", nullptr, 2));

  B = std::numeric_limits<float>::max();
  Result = sycl::bit_cast<uint16_t>(B);
  ASSERT_EQ(Result, std::stoi("0111111110000000", nullptr, 2));

  B = std::numeric_limits<float>::quiet_NaN();
  Result = sycl::bit_cast<uint16_t>(B);
  ASSERT_EQ(Result, std::stoi("1111111111000001", nullptr, 2));
}

TEST(BFloat16, BF16ToFloat) {
  // See https://float.exposed/b0xffff
  uint16_t V = 0;
  float Res = sycl::bit_cast<sycl::ext::oneapi::bfloat16>(V);
  ASSERT_EQ(Res,
            bitsToFloatConv(std::string("00000000000000000000000000000000")));

  V = 1;
  Res = sycl::bit_cast<sycl::ext::oneapi::bfloat16>(V);
  ASSERT_EQ(Res,
            bitsToFloatConv(std::string("00000000000000010000000000000000")));

  V = 42;
  Res = sycl::bit_cast<sycl::ext::oneapi::bfloat16>(V);
  ASSERT_EQ(Res,
            bitsToFloatConv(std::string("00000000001010100000000000000000")));

  // std::numeric_limits<uint16_t>::max() - 0xffff is bfloat16 -Nan and
  // -Nan == -Nan check in check_bf16_to_float would fail, so use not Nan:
  V = 65407;
  Res = sycl::bit_cast<sycl::ext::oneapi::bfloat16>(V);
  ASSERT_EQ(Res,
            bitsToFloatConv(std::string("11111111011111110000000000000000")));
}

TEST(BFloat16, BF16Limits) {
  namespace sycl_ext = sycl::ext::oneapi;
  using Limit = std::numeric_limits<sycl_ext::bfloat16>;
  constexpr float Log10_2 = 0.30103f;
  auto constexpr_ceil = [](float Val) constexpr -> int {
    return Val + (float(int(Val)) == Val ? 0.f : 1.f);
  };

  static_assert(Limit::is_specialized);
  static_assert(Limit::is_signed);
  static_assert(!Limit::is_integer);
  static_assert(!Limit::is_exact);
  static_assert(Limit::has_infinity);
  static_assert(Limit::has_quiet_NaN);
  static_assert(Limit::has_signaling_NaN);
  static_assert(Limit::has_denorm == std::float_denorm_style::denorm_present);
  static_assert(!Limit::has_denorm_loss);
  static_assert(!Limit::tinyness_before);
  static_assert(!Limit::traps);
  static_assert(Limit::max_exponent10 == 35);
  static_assert(Limit::max_exponent == 127);
  static_assert(Limit::min_exponent10 == -37);
  static_assert(Limit::min_exponent == -126);
  static_assert(Limit::radix == 2);
  static_assert(Limit::digits == 8);
  static_assert(Limit::max_digits10 ==
                constexpr_ceil(float(Limit::digits) * Log10_2 + 1.0f));
  static_assert(Limit::is_bounded);
  static_assert(Limit::digits10 == int(Limit::digits * Log10_2));
  static_assert(!Limit::is_modulo);
  static_assert(Limit::is_iec559);
  static_assert(Limit::round_style == std::float_round_style::round_to_nearest);

  EXPECT_TRUE(sycl_ext::experimental::isnan(Limit::quiet_NaN()));
  EXPECT_TRUE(sycl_ext::experimental::isnan(Limit::signaling_NaN()));
  // isinf does not exist for bfloat16 currently.
  EXPECT_EQ(Limit::infinity(),
            sycl::bit_cast<sycl_ext::bfloat16>(uint16_t(0xff << 7)));
  EXPECT_EQ(Limit::round_error(), sycl_ext::bfloat16(0.5f));
  EXPECT_GT(sycl_ext::bfloat16{1.0f} + Limit::epsilon(),
            sycl_ext::bfloat16{1.0f});

  for (uint16_t Sign : {0, 1})
    for (uint16_t Exponent = 0; Exponent < 0xff; ++Exponent)
      for (uint16_t Significand = 0; Significand < 0x7f; ++Significand) {
        const auto Value = sycl::bit_cast<sycl_ext::bfloat16>(
            uint16_t((Sign << 15) | (Exponent << 7) | Significand));

        EXPECT_LE(Limit::lowest(), Value);
        EXPECT_GE(Limit::max(), Value);

        // min() is the lowest normal number, so if Value is negative, 0 or a
        // subnormal - the latter two being represented by a 0-exponent - min()
        // must be strictly greater.
        if (Sign || Exponent == 0x0)
          EXPECT_GT(Limit::min(), Value);
        else
          EXPECT_LE(Limit::min(), Value);

        // denorm_min() is the lowest subnormal number, so if Value is negative
        // or 0 denorm_min() must be strictly greater.
        if (Sign || (Exponent == 0x0 && Significand == 0x0))
          EXPECT_GT(Limit::denorm_min(), Value);
        else
          EXPECT_LE(Limit::denorm_min(), Value);
      }
}
