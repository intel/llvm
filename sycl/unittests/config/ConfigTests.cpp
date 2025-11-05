//==------- ConfigTests.cpp --- SYCL config processing unit test -----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/config.hpp>

#include <gtest/gtest.h>

#include <fstream>
#include <regex>

TEST(ConfigTests, CheckConfigProcessing) {
#ifdef _WIN32
  _putenv_s("SYCL_CONFIG_FILE_NAME", "conf.txt");
#else
  setenv("SYCL_CONFIG_FILE_NAME", "conf.txt", 1);
#endif

  // Check SPACE at first position
  std::ofstream File;
  File.open("conf.txt");
  if (File.is_open()) {
    File << " a=b" << std::endl;
    File.close();
  }
  try {
    sycl::detail::SYCLConfig<sycl::detail::SYCL_DEVICE_ALLOWLIST>::get();
    throw std::logic_error("sycl::exception didn't throw");
  } catch (sycl::exception &e) {
    EXPECT_EQ(
        std::string(
            "SPACE found at the beginning/end of the line or before/after '='"),
        e.what());
  } catch (...) {
    FAIL() << "Check SPACE at first position failed";
  }

  // Check SPACE at last position
  File.open("conf.txt");
  if (File.is_open()) {
    File << "a=b " << std::endl;
    File.close();
  }
  try {
    sycl::detail::SYCLConfig<sycl::detail::SYCL_DEVICE_ALLOWLIST>::get();
    throw std::logic_error("sycl::exception didn't throw");
  } catch (sycl::exception &e) {
    EXPECT_EQ(
        std::string(
            "SPACE found at the beginning/end of the line or before/after '='"),
        e.what());
  } catch (...) {
    FAIL() << "Check SPACE at last position failed";
  }

  // Check SPACE before assignment
  File.open("conf.txt");
  if (File.is_open()) {
    File << "a =b" << std::endl;
    File.close();
  }
  try {
    sycl::detail::SYCLConfig<sycl::detail::SYCL_DEVICE_ALLOWLIST>::get();
    throw std::logic_error("sycl::exception didn't throw");
  } catch (sycl::exception &e) {
    EXPECT_EQ(
        std::string(
            "SPACE found at the beginning/end of the line or before/after '='"),
        e.what());
  } catch (...) {
    FAIL() << "Check SPACE before assignment failed";
  }

  // Check SPACE after assignment
  File.open("conf.txt");
  if (File.is_open()) {
    File << "a= b" << std::endl;
    File.close();
  }
  try {
    sycl::detail::SYCLConfig<sycl::detail::SYCL_DEVICE_ALLOWLIST>::get();
    throw std::logic_error("sycl::exception didn't throw");
  } catch (sycl::exception &e) {
    EXPECT_EQ(
        std::string(
            "SPACE found at the beginning/end of the line or before/after '='"),
        e.what());
  } catch (...) {
    FAIL() << "Check SPACE after assignment failed";
  }

  // Check variable name bigger than MAX_CONFIG_NAME
  File.open("conf.txt");
  if (File.is_open()) {
    for (int i = 0; i <= sycl::detail::MAX_CONFIG_NAME; i++) {
      File << "a";
    }
    File << "=b" << std::endl;
    File.close();
  }
  try {
    sycl::detail::SYCLConfig<sycl::detail::SYCL_DEVICE_ALLOWLIST>::get();
    throw std::logic_error("sycl::exception didn't throw");
  } catch (sycl::exception &e) {
    EXPECT_TRUE(std::regex_match(
        e.what(),
        std::regex(
            "Variable name is more than ([\\d]+) or less than one character")));
  } catch (...) {
    FAIL() << "Check variable name bigger than MAX_CONFIG_NAME failed";
  }

  // Check variable without name
  File.open("conf.txt");
  if (File.is_open()) {
    File << "=b" << std::endl;
    File.close();
  }
  try {
    sycl::detail::SYCLConfig<sycl::detail::SYCL_DEVICE_ALLOWLIST>::get();
    throw std::logic_error("sycl::exception didn't throw");
  } catch (sycl::exception &e) {
    EXPECT_TRUE(std::regex_match(
        e.what(), std::regex("Variable name is more than ([\\d]+) or less "
                             "than one character")));
  } catch (...) {
    FAIL() << "Check variable without name failed";
  }

  // Check variable value bigger than MAX_CONFIG_VALUE
  File.open("conf.txt");
  if (File.is_open()) {
    File << "a=";
    for (int i = 0; i <= sycl::detail::MAX_CONFIG_VALUE; i++) {
      File << "b";
    }
    File << std::endl;
    File.close();
  }
  try {
    sycl::detail::SYCLConfig<sycl::detail::SYCL_DEVICE_ALLOWLIST>::get();
    throw std::logic_error("sycl::exception didn't throw");
  } catch (sycl::exception &e) {
    EXPECT_TRUE(std::regex_match(
        e.what(), std::regex("The value contains more than ([\\d]+) characters "
                             "or does not contain them at all")));
  } catch (...) {
    FAIL() << "Check variable value bigger than MAX_CONFIG_VALUE failed";
  }

  // Check variable without value
  File.open("conf.txt");
  if (File.is_open()) {
    File << "a=" << std::endl;
    File.close();
  }
  try {
    sycl::detail::SYCLConfig<sycl::detail::SYCL_DEVICE_ALLOWLIST>::get();
    throw std::logic_error("sycl::exception didn't throw");
  } catch (sycl::exception &e) {
    EXPECT_TRUE(std::regex_match(
        e.what(), std::regex("The value contains more than ([\\d]+) characters "
                             "or does not contain them at all")));
  } catch (...) {
    FAIL() << "Check variable without value failed";
  }

  // Check incorrect complex config processing
  File.open("conf.txt");
  if (File.is_open()) {
    File << "#\n   #\r\n#a=b\n\n\na\r\n aaa \r\na=b\r\na=\r" << std::endl;
    File.close();
  }
  try {
    sycl::detail::SYCLConfig<sycl::detail::SYCL_DEVICE_ALLOWLIST>::get();
    throw std::logic_error("sycl::exception didn't throw");
  } catch (sycl::exception &e) {
    EXPECT_TRUE(std::regex_match(
        e.what(), std::regex("The value contains more than ([\\d]+) characters "
                             "or does not contain them at all")));
  } catch (...) {
    FAIL() << "Check incorrect complex config processing failed";
  }

  // Check simple correct config processing
  File.open("conf.txt");
  if (File.is_open()) {
    File << "SYCL_PRINT_EXECUTION_GRAPH=before_addCG" << std::endl;
    File.close();
  }
  sycl::detail::readConfig(true);
  EXPECT_EQ(std::string("before_addCG"),
            sycl::detail::SYCLConfig<
                sycl::detail::SYCL_PRINT_EXECUTION_GRAPH>::get());

  // Check correct config processing
  File.open("conf.txt");
  if (File.is_open()) {
    File << "#a\r\n # b \r\n #a=b\r\n # a = "
            "b\r\n\r\nSYCL_PRINT_EXECUTION_GRAPH=after_addCG\r\n"
         << std::endl;
    File.close();
  }
  sycl::detail::readConfig(true);
  EXPECT_EQ(std::string("after_addCG"),
            sycl::detail::SYCLConfig<
                sycl::detail::SYCL_PRINT_EXECUTION_GRAPH>::get());

  // Check multi assignment
  File.open("conf.txt");
  if (File.is_open()) {
    File << "a=b\r\nb=c\nSYCL_PRINT_EXECUTION_GRAPH=before_addCopyBack\r\nc=d";
    File.close();
  }
  sycl::detail::readConfig(true);
  EXPECT_EQ(std::string("before_addCopyBack"),
            sycl::detail::SYCLConfig<
                sycl::detail::SYCL_PRINT_EXECUTION_GRAPH>::get());

  // Check assignment with comment
  File.open("conf.txt");
  if (File.is_open()) {
    File << "SYCL_PRINT_EXECUTION_GRAPH=after_addCopyBack #comment\r\n";
    File.close();
  }
  sycl::detail::readConfig(true);
  EXPECT_EQ(std::string("after_addCopyBack"),
            sycl::detail::SYCLConfig<
                sycl::detail::SYCL_PRINT_EXECUTION_GRAPH>::get());
}

// SYCL_CACHE_TRACE accepts a bit-mask to control the tracing of
// different SYCL caches. The input value is parsed as an integer and
// the following bit-masks is used to determine the tracing behavior:
// 0x01 - trace disk cache
// 0x02 - trace in-memory cache
// 0x04 - trace kernel_compiler cache
// Any valid combination of the above bit-masks can be used to enable/disable
// tracing of the corresponding caches. If the input value is not null and
// not a valid number, the disk cache tracing will be enabled (depreciated
// behavior). The default value is 0 and no tracing is enabled.
using namespace sycl::detail;
TEST(ConfigTests, CheckSyclCacheTraceTest) {

  // Lambda to test parsing of SYCL_CACHE_TRACE
  auto TestConfig = [](int expectedValue, int expectedDiskCache,
                       int expectedInMemCache, int expectedKernelCompiler) {
    EXPECT_EQ(static_cast<unsigned int>(expectedValue),
              SYCLConfig<SYCL_CACHE_TRACE>::get());

    EXPECT_EQ(
        expectedDiskCache,
        static_cast<int>(
            sycl::detail::SYCLConfig<SYCL_CACHE_TRACE>::isTraceDiskCache()));
    EXPECT_EQ(
        expectedInMemCache,
        static_cast<int>(
            sycl::detail::SYCLConfig<SYCL_CACHE_TRACE>::isTraceInMemCache()));
    EXPECT_EQ(expectedKernelCompiler,
              static_cast<int>(sycl::detail::SYCLConfig<
                               SYCL_CACHE_TRACE>::isTraceKernelCompiler()));
  };

  // Lambda to set SYCL_CACHE_TRACE
  auto SetSyclCacheTraceEnv = [](const char *value) {
#ifdef _WIN32
    _putenv_s("SYCL_CACHE_TRACE", value);
#else
    setenv("SYCL_CACHE_TRACE", value, 1);
#endif
  };

  SetSyclCacheTraceEnv("0");
  sycl::detail::readConfig(true);
  TestConfig(0, 0, 0, 0);

  SetSyclCacheTraceEnv("1");
  sycl::detail::SYCLConfig<SYCL_CACHE_TRACE>::reset();
  TestConfig(1, 1, 0, 0);

  SetSyclCacheTraceEnv("2");
  sycl::detail::SYCLConfig<SYCL_CACHE_TRACE>::reset();
  TestConfig(2, 0, 1, 0);

  SetSyclCacheTraceEnv("3");
  sycl::detail::SYCLConfig<SYCL_CACHE_TRACE>::reset();
  TestConfig(3, 1, 1, 0);

  SetSyclCacheTraceEnv("4");
  sycl::detail::SYCLConfig<SYCL_CACHE_TRACE>::reset();
  TestConfig(4, 0, 0, 1);

  SetSyclCacheTraceEnv("5");
  sycl::detail::SYCLConfig<SYCL_CACHE_TRACE>::reset();
  TestConfig(5, 1, 0, 1);

  SetSyclCacheTraceEnv("6");
  sycl::detail::SYCLConfig<SYCL_CACHE_TRACE>::reset();
  TestConfig(6, 0, 1, 1);

  SetSyclCacheTraceEnv("7");
  sycl::detail::SYCLConfig<SYCL_CACHE_TRACE>::reset();
  TestConfig(7, 1, 1, 1);

  SetSyclCacheTraceEnv("8");
  sycl::detail::SYCLConfig<SYCL_CACHE_TRACE>::reset();
  TestConfig(1, 1, 0, 0);

  // Set random non-null value. It should default to 1.
  SetSyclCacheTraceEnv("random");
  sycl::detail::SYCLConfig<SYCL_CACHE_TRACE>::reset();
  TestConfig(1, 1, 0, 0);

  // When SYCL_CACHE_TRACE is not set, it should default to 0.
#ifdef _WIN32
  _putenv_s("SYCL_CACHE_TRACE", "");
#else
  unsetenv("SYCL_CACHE_TRACE");
#endif
  sycl::detail::SYCLConfig<SYCL_CACHE_TRACE>::reset();
  TestConfig(0, 0, 0, 0);
}

// SYCL_IN_MEM_CACHE_EVICTION_THRESHOLD accepts an integer that specifies
// the maximum size of the in-memory Program cache.
// Cache eviction is performed when the cache size exceeds the threshold.
// The thresholds are specified in bytes.
// The default value is "0" which means that eviction is disabled.
TEST(ConfigTests, CheckSyclCacheEvictionThresholdTest) {

  using InMemEvicType =
      sycl::detail::SYCLConfig<SYCL_IN_MEM_CACHE_EVICTION_THRESHOLD>;

  // Lambda to test parsing of SYCL_IN_MEM_CACHE_EVICTION_THRESHOLD.
  auto TestConfig = [](int expectedProgramCacheSize) {
    EXPECT_EQ(expectedProgramCacheSize, InMemEvicType::getProgramCacheSize());
    EXPECT_EQ(expectedProgramCacheSize > 0,
              InMemEvicType::isProgramCacheEvictionEnabled());
  };

  // Lambda to set SYCL_IN_MEM_CACHE_EVICTION_THRESHOLD.
  auto SetSyclInMemCacheEvictionThresholdEnv = [](const char *value) {
#ifdef _WIN32
    _putenv_s("SYCL_IN_MEM_CACHE_EVICTION_THRESHOLD", value);
#else
    setenv("SYCL_IN_MEM_CACHE_EVICTION_THRESHOLD", value, 1);
#endif
  };

  // Lambda to test invalid inputs. An exception should be thrown
  // when parsing invalid values.
  auto TestInvalidValues = [&](const char *value, const char *errMsg) {
    SetSyclInMemCacheEvictionThresholdEnv(value);
    try {
      InMemEvicType::reset();
      TestConfig(0);
      FAIL() << errMsg;
    } catch (...) {
    }
  };

  // Test eviction threshold with zero.
  SetSyclInMemCacheEvictionThresholdEnv("0");
  sycl::detail::readConfig(true);
  TestConfig(0);

  // Test invalid values.
  TestInvalidValues("-1", "Should throw exception for negative value");
  TestInvalidValues("a", "Should throw exception for non-integer value");

  // Test valid values.
  SetSyclInMemCacheEvictionThresholdEnv("1024");
  InMemEvicType::reset();
  TestConfig(1024);

  // When SYCL_IN_MEM_CACHE_EVICTION_THRESHOLD is not set, it should default to
  // 0:0:0.
#ifdef _WIN32
  _putenv_s("SYCL_IN_MEM_CACHE_EVICTION_THRESHOLD", "");
#else
  unsetenv("SYCL_IN_MEM_CACHE_EVICTION_THRESHOLD");
#endif
  InMemEvicType::reset();
  TestConfig(0);
}

// SYCL_CACHE_MAX_SIZE accepts an integer that specifies
// the maximum size of the persistent Program cache.
// Cache eviction is performed when the cache size exceeds the threshold.
// The thresholds are specified in bytes.
// The default value is "0" which means that eviction is disabled.
TEST(ConfigTests, CheckPersistentCacheEvictionThresholdTest) {

  using OnDiskEvicType = sycl::detail::SYCLConfig<SYCL_CACHE_MAX_SIZE>;

  // Lambda to test parsing of SYCL_CACHE_MAX_SIZE.
  auto TestConfig = [](int expectedProgramCacheSize) {
    EXPECT_EQ(expectedProgramCacheSize, OnDiskEvicType::getProgramCacheSize());
    EXPECT_EQ(expectedProgramCacheSize > 0,
              OnDiskEvicType::isPersistentCacheEvictionEnabled());
  };

  // Lambda to set SYCL_CACHE_MAX_SIZE.
  auto SetSyclDiskCacheEvictionThresholdEnv = [](const char *value) {
#ifdef _WIN32
    _putenv_s("SYCL_CACHE_MAX_SIZE", value);
#else
    setenv("SYCL_CACHE_MAX_SIZE", value, 1);
#endif
  };

  // Lambda to test invalid inputs. An exception should be thrown
  // when parsing invalid values.
  auto TestInvalidValues = [&](const char *value, const char *errMsg) {
    SetSyclDiskCacheEvictionThresholdEnv(value);
    try {
      OnDiskEvicType::reset();
      TestConfig(0);
      FAIL() << errMsg;
    } catch (...) {
    }
  };

  // Test eviction threshold with zero.
  SetSyclDiskCacheEvictionThresholdEnv("0");
  sycl::detail::readConfig(true);
  TestConfig(0);

  // Test invalid values.
  TestInvalidValues("-1", "Should throw exception for negative value");
  TestInvalidValues("a", "Should throw exception for non-integer value");

  // Test valid values.
  SetSyclDiskCacheEvictionThresholdEnv("1024");
  OnDiskEvicType::reset();
  TestConfig(1024);

  // When SYCL_CACHE_MAX_SIZE is not set, it should default to
  // 0:0:0.
#ifdef _WIN32
  _putenv_s("SYCL_CACHE_MAX_SIZE", "");
#else
  unsetenv("SYCL_CACHE_MAX_SIZE");
#endif
  OnDiskEvicType::reset();
  TestConfig(0);
}

// SYCL_PARALLEL_FOR_RANGE_ROUNDING_PARAMS accepts ...
TEST(ConfigTests, CheckParallelForRangeRoundingParams) {

  // Lambda to set SYCL_PARALLEL_FOR_RANGE_ROUNDING_PARAMS.
  auto SetRoundingParams = [](const char *value) {
#ifdef _WIN32
    _putenv_s("SYCL_PARALLEL_FOR_RANGE_ROUNDING_PARAMS", value);
#else
    setenv("SYCL_PARALLEL_FOR_RANGE_ROUNDING_PARAMS", value, 1);
#endif
    sycl::detail::readConfig(true);
  };

  // Lambda to assert test parameters are as expected.
  auto AssertRoundingParams = [](size_t MF, size_t GF, size_t MR,
                                 const char *errMsg, bool ForceUpdate = false) {
    size_t ResultMF = 0, ResultGF = 0, ResultMR = 0;
    SYCLConfig<SYCL_PARALLEL_FOR_RANGE_ROUNDING_PARAMS>::GetSettings(
        ResultMF, ResultGF, ResultMR, ForceUpdate);
    EXPECT_EQ(MF, ResultMF) << errMsg;
    EXPECT_EQ(GF, ResultGF) << errMsg;
    EXPECT_EQ(MR, ResultMR) << errMsg;
  };

  // Lambda to test invalid input -- factors should remain unchanged.
  auto TestBadInput = [&](const char *value, const char *errMsg) {
    // Original factor values are stored as its own variable as size of size_t
    // varies depending on system and architecture:
    constexpr size_t MF = 1, GF = 2, MR = 3;
    size_t TestMF = MF, TestGF = GF, TestMR = MR;
    SetRoundingParams(value);
    SYCLConfig<SYCL_PARALLEL_FOR_RANGE_ROUNDING_PARAMS>::GetSettings(
        TestMF, TestGF, TestMR, true);
    EXPECT_EQ(TestMF, MF) << errMsg;
    EXPECT_EQ(TestGF, GF) << errMsg;
    EXPECT_EQ(TestMR, MR) << errMsg;
  };

  // Test malformed input:
  constexpr char MalformedErr[] =
      "Rounding parameters should be ignored on malformed input";
  TestBadInput("abc", MalformedErr);
  TestBadInput("42", MalformedErr);
  TestBadInput(":7", MalformedErr);
  TestBadInput("7:", MalformedErr);
  TestBadInput("1:2", MalformedErr);
  TestBadInput("1:2:", MalformedErr);
  TestBadInput("1:abc:3", MalformedErr);

  // Test well-formed input, but bad parameters:
  constexpr char BadParamsErr[] = "Rounding parameters should be ignored if "
                                  "parameters provided are invalid";
  TestBadInput("0:1:2", BadParamsErr);
  TestBadInput("1:0:2", BadParamsErr);
  TestBadInput("-1:2:3", BadParamsErr);
  TestBadInput("1:2:31415926535897932384626433832795028841971", BadParamsErr);

  // Test valid values.
  SetRoundingParams("8:16:32");
  AssertRoundingParams(8, 16, 32,
                       "Failed to read rounding parameters properly");
  SetRoundingParams("8:16:0");
  AssertRoundingParams(8, 16, 0, "0 is a valid value for MinRange",
                       /*ForceUpdate =*/true);
}
