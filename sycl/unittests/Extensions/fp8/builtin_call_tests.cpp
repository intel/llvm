#include "builtin_mocks.hpp"
#include <gtest/gtest.h>
#include <sycl/ext/oneapi/experimental/float_8bit/types.hpp>

namespace {

using namespace sycl::ext::oneapi::experimental;

class Fp8BuiltinCallTest : public ::testing::Test {
protected:
  void SetUp() override { fp8_builtin_mock::resetCounters(); }
};

TEST_F(Fp8BuiltinCallTest, E4M3CtorFromHalfCallsClampConvertFP16ToE4M3) {
  fp8_e4m3 Value(static_cast<sycl::half>(1.25f));
  (void)Value;
  EXPECT_EQ(fp8_builtin_mock::getCounters().ClampConvertFP16ToE4M3INTEL, 1);
}

TEST_F(Fp8BuiltinCallTest, E4M3CtorFromBf16CallsClampConvertBF16ToE4M3) {
  fp8_e4m3 Value(static_cast<sycl::ext::oneapi::bfloat16>(1.25f));
  (void)Value;
  EXPECT_EQ(fp8_builtin_mock::getCounters().ClampConvertBF16ToE4M3INTEL, 1);
}

TEST_F(Fp8BuiltinCallTest, E4M3CastToHalfCallsClampConvertE4M3ToFP16) {
  fp8_e4m3 Value(static_cast<sycl::half>(1.0f));
  fp8_builtin_mock::resetCounters();
  (void)static_cast<sycl::half>(Value);
  EXPECT_EQ(fp8_builtin_mock::getCounters().ConvertE4M3ToFP16EXT, 1);
}

TEST_F(Fp8BuiltinCallTest, E4M3CastToBf16CallsConvertE4M3ToBF16) {
  fp8_e4m3 Value(static_cast<sycl::half>(1.0f));
  fp8_builtin_mock::resetCounters();
  (void)static_cast<sycl::ext::oneapi::bfloat16>(Value);
  EXPECT_EQ(fp8_builtin_mock::getCounters().ConvertE4M3ToBF16EXT, 1);
}

TEST_F(Fp8BuiltinCallTest, E4M3CastToBoolCallsConvertE4M3ToFP16) {
  fp8_e4m3 Value(static_cast<sycl::half>(1.0f));
  fp8_builtin_mock::resetCounters();
  (void)static_cast<bool>(Value);
  EXPECT_EQ(fp8_builtin_mock::getCounters().ConvertE4M3ToFP16EXT, 1);
}

TEST_F(Fp8BuiltinCallTest, E5M2CtorFromHalfCallsClampConvertFP16ToE5M2) {
  fp8_e5m2 Value(static_cast<sycl::half>(2.0f));
  (void)Value;
  EXPECT_EQ(fp8_builtin_mock::getCounters().ClampConvertFP16ToE5M2INTEL, 1);
}

TEST_F(Fp8BuiltinCallTest, E5M2CtorFromBf16CallsClampConvertBF16ToE5M2) {
  fp8_e5m2 Value(static_cast<sycl::ext::oneapi::bfloat16>(2.0f));
  (void)Value;
  EXPECT_EQ(fp8_builtin_mock::getCounters().ClampConvertBF16ToE5M2INTEL, 1);
}

TEST_F(Fp8BuiltinCallTest, E5M2CastToHalfCallsConvertE5M2ToFP16) {
  fp8_e5m2 Value(static_cast<sycl::half>(2.0f));
  fp8_builtin_mock::resetCounters();
  (void)static_cast<sycl::half>(Value);
  EXPECT_EQ(fp8_builtin_mock::getCounters().ConvertE5M2ToFP16EXT, 1);
}

TEST_F(Fp8BuiltinCallTest, E5M2CastToBf16CallsConvertE5M2ToBF16) {
  fp8_e5m2 Value(static_cast<sycl::half>(2.0f));
  fp8_builtin_mock::resetCounters();
  (void)static_cast<sycl::ext::oneapi::bfloat16>(Value);
  EXPECT_EQ(fp8_builtin_mock::getCounters().ConvertE5M2ToBF16EXT, 1);
}

TEST_F(Fp8BuiltinCallTest, E5M2CtorFromHalfWithNoSaturationCallsConvertFP16ToE5M2) {
  sycl::half Input[1] = {static_cast<sycl::half>(2.0f)};

  fp8_e5m2 Value(Input, rounding::to_even, saturation::none);
  (void)Value;

  EXPECT_EQ(fp8_builtin_mock::getCounters().ConvertFP16ToE5M2EXT, 1);
}

TEST_F(Fp8BuiltinCallTest, E5M2CtorFromBf16WithNoSaturationCallsConvertBF16ToE5M2) {
  sycl::ext::oneapi::bfloat16 Input[1] = {
      static_cast<sycl::ext::oneapi::bfloat16>(2.0f)};

  fp8_e5m2 Value(Input, rounding::to_even, saturation::none);
  (void)Value;

  EXPECT_EQ(fp8_builtin_mock::getCounters().ConvertBF16ToE5M2EXT, 1);
}

TEST_F(Fp8BuiltinCallTest, E5M2StochasticHalfFiniteCallsClampStochastic) {
  sycl::half Input[1] = {static_cast<sycl::half>(3.0f)};
  uint32_t SeedValue = 10;
  stochastic_seed Seed(&SeedValue);

  fp8_e5m2 Value(Input, Seed, saturation::finite);
  (void)Value;

  EXPECT_EQ(fp8_builtin_mock::getCounters().ClampStochasticRoundFP16ToE5M2INTEL,
            1);
  EXPECT_EQ(SeedValue, 11u);
}

TEST_F(Fp8BuiltinCallTest, E5M2StochasticHalfNoneCallsNonClampStochastic) {
  sycl::half Input[1] = {static_cast<sycl::half>(3.0f)};
  uint32_t SeedValue = 20;
  stochastic_seed Seed(&SeedValue);

  fp8_e5m2 Value(Input, Seed, saturation::none);
  (void)Value;

  EXPECT_EQ(fp8_builtin_mock::getCounters().StochasticRoundFP16ToE5M2INTEL, 1);
  EXPECT_EQ(SeedValue, 21u);
}

TEST_F(Fp8BuiltinCallTest, E5M2StochasticBf16FiniteCallsClampStochastic) {
  sycl::ext::oneapi::bfloat16 Input[1] = {
      static_cast<sycl::ext::oneapi::bfloat16>(3.0f)};
  uint32_t SeedValue = 30;
  stochastic_seed Seed(&SeedValue);

  fp8_e5m2 Value(Input, Seed, saturation::finite);
  (void)Value;

  EXPECT_EQ(fp8_builtin_mock::getCounters().ClampStochasticRoundBF16ToE5M2INTEL,
            1);
}

TEST_F(Fp8BuiltinCallTest, E5M2StochasticBf16NoneCallsNonClampStochastic) {
  sycl::ext::oneapi::bfloat16 Input[1] = {
      static_cast<sycl::ext::oneapi::bfloat16>(3.0f)};
  uint32_t SeedValue = 40;
  stochastic_seed Seed(&SeedValue);

  fp8_e5m2 Value(Input, Seed, saturation::none);
  (void)Value;

  EXPECT_EQ(fp8_builtin_mock::getCounters().StochasticRoundBF16ToE5M2INTEL, 1);
}

} // namespace
