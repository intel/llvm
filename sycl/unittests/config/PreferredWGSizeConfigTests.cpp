//==---- PreferredWGSizeConfigTests.cpp --- SYCL preferred WG size config --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Tests that valid and invalid configuration values for
// SYCL_REDUCTION_PREFERRED_WORKGROUP_SIZE behave as expected.

#include <detail/config.hpp>
#include <gtest/gtest.h>
#include <regex>
#include <sycl/sycl.hpp>

// Sets the SYCL_REDUCTION_PREFERRED_WORKGROUP_SIZE configuration and forces
// a reparse.
void SetConfig(const char *Value) {
#ifdef _WIN32
  _putenv_s("SYCL_REDUCTION_PREFERRED_WORKGROUP_SIZE", Value);
#else
  setenv("SYCL_REDUCTION_PREFERRED_WORKGROUP_SIZE", Value, 1);
#endif
  sycl::detail::SYCLConfig<
      sycl::detail::SYCL_REDUCTION_PREFERRED_WORKGROUP_SIZE>::reset();
}

// Gets the parsed value of the SYCL_REDUCTION_PREFERRED_WORKGROUP_SIZE
// configuration for a given device-type.
size_t GetConfigValue(sycl::info::device_type DevType) {
  return sycl::detail::SYCLConfig<
      sycl::detail::SYCL_REDUCTION_PREFERRED_WORKGROUP_SIZE>::get(DevType);
}

// Sets the SYCL_REDUCTION_PREFERRED_WORKGROUP_SIZE configuration and checks
// the parsed values.
void SetAndCheck(const char *ConfigValue, size_t CPUValue, size_t GPUValue,
                 size_t AccValue) {
  SetConfig(ConfigValue);
  EXPECT_EQ(GetConfigValue(sycl::info::device_type::cpu), CPUValue)
      << "Unexpected value for CPU with '" << ConfigValue << "'.";
  EXPECT_EQ(GetConfigValue(sycl::info::device_type::gpu), GPUValue)
      << "Unexpected value for GPU with '" << ConfigValue << "'.";
  EXPECT_EQ(GetConfigValue(sycl::info::device_type::accelerator), AccValue)
      << "Unexpected value for accelerator with '" << ConfigValue << "'.";
}

// Sets the SYCL_REDUCTION_PREFERRED_WORKGROUP_SIZE configuration and expects
// a sycl::exception to be thrown with the specified error code.
void SetAndExpectException(const char *ConfigValue,
                           sycl::errc ExpectedErrorCode) {
  try {
    SetConfig(ConfigValue);
    EXPECT_TRUE(false) << "Setting the config with '" << ConfigValue
                       << "' unexpectedly succeeded.";
  } catch (sycl::exception &E) {
    EXPECT_EQ(E.code(), sycl::make_error_code(ExpectedErrorCode))
        << "Exception thrown when setting the config with '" << ConfigValue
        << "' does not have the expected error code.";
  } catch (...) {
    EXPECT_TRUE(false) << "Setting the config with '" << ConfigValue
                       << "' throw a non-SYCL exception.";
  }
}

// NOTE: All checks are kept in the same file to avoid potential multi-threading
// from overwriting the program-wide configurations.
TEST(ConfigTests, CheckPreferredWGSizeConfigProcessing) {
  SetAndCheck("cpu:32", 32, 0, 0);
  SetAndCheck("gpu:32", 0, 32, 0);
  SetAndCheck("acc:32", 0, 0, 32);
  SetAndCheck("*:32", 32, 32, 32);

  SetAndCheck("cpu:1,gpu:2", 1, 2, 0);
  SetAndCheck("gpu:2,cpu:1", 1, 2, 0);
  SetAndCheck("cpu:1,acc:3", 1, 0, 3);
  SetAndCheck("acc:3,cpu:1", 1, 0, 3);
  SetAndCheck("gpu:2,acc:3", 0, 2, 3);
  SetAndCheck("acc:3,gpu:2", 0, 2, 3);

  SetAndCheck("cpu:1,gpu:2,acc:3", 1, 2, 3);
  SetAndCheck("cpu:1,acc:3,gpu:2", 1, 2, 3);
  SetAndCheck("acc:3,cpu:1,gpu:2", 1, 2, 3);
  SetAndCheck("acc:3,gpu:2,cpu:1", 1, 2, 3);
  SetAndCheck("gpu:2,acc:3,cpu:1", 1, 2, 3);
  SetAndCheck("gpu:2,cpu:1,acc:3", 1, 2, 3);

  SetAndCheck("cpu:1,cpu:2", 2, 0, 0);
  SetAndCheck("cpu:2,cpu:1", 1, 0, 0);
  SetAndCheck("gpu:1,gpu:2", 0, 2, 0);
  SetAndCheck("gpu:2,gpu:1", 0, 1, 0);
  SetAndCheck("acc:1,acc:2", 0, 0, 2);
  SetAndCheck("acc:2,acc:1", 0, 0, 1);
  SetAndCheck("*:1,*:2", 2, 2, 2);
  SetAndCheck("*:2,*:1", 1, 1, 1);

  SetAndCheck("cpu:1,*:2", 2, 2, 2);
  SetAndCheck("gpu:1,*:2", 2, 2, 2);
  SetAndCheck("acc:1,*:2", 2, 2, 2);
  SetAndCheck("*:2,cpu:1", 1, 2, 2);
  SetAndCheck("*:2,gpu:1", 2, 1, 2);
  SetAndCheck("*:2,acc:1", 2, 2, 1);

  SetAndExpectException("cpu:0", sycl::errc::invalid);
  SetAndExpectException("gpu:0", sycl::errc::invalid);
  SetAndExpectException("acc:0", sycl::errc::invalid);
  SetAndExpectException("*:0", sycl::errc::invalid);
  SetAndExpectException("cpu:-32", sycl::errc::invalid);
  SetAndExpectException("gpu:-32", sycl::errc::invalid);
  SetAndExpectException("acc:-32", sycl::errc::invalid);
  SetAndExpectException("*:-32", sycl::errc::invalid);

  SetAndExpectException("cpu:0,gpu:32", sycl::errc::invalid);
  SetAndExpectException("gpu:32,cpu:0", sycl::errc::invalid);
  SetAndExpectException("cpu:-32,gpu:32", sycl::errc::invalid);
  SetAndExpectException("gpu:32,cpu:-32", sycl::errc::invalid);

  SetAndExpectException("cpu:some invalid value", sycl::errc::invalid);
  SetAndExpectException("gpu:some invalid value", sycl::errc::invalid);
  SetAndExpectException("acc:some invalid value", sycl::errc::invalid);
  SetAndExpectException("*:some invalid value", sycl::errc::invalid);
  
  SetAndExpectException("cpu:some invalid value,gpu:32", sycl::errc::invalid);
  SetAndExpectException("gpu:32,cpu:some invalid value", sycl::errc::invalid);

  SetAndExpectException("invalid_device_type:32", sycl::errc::invalid);
  SetAndExpectException("cpu:32,invalid_device_type:32", sycl::errc::invalid);

  SetAndExpectException("cpu", sycl::errc::invalid);
  SetAndExpectException("cpu,gpu:32", sycl::errc::invalid);
  SetAndExpectException("cpu:32,gpu", sycl::errc::invalid);
}
