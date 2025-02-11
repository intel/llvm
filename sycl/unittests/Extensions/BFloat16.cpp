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
