//==------- ParseAllowList.cpp --- SYCL_DEVICE_ALLOWLIST unit test ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/allowlist.hpp>
#include <detail/config.hpp> // for getSyclBeMap() and getSyclDeviceTypeMap()

#include <gtest/gtest.h>

TEST(ParseAllowListTests, CheckAllowListIsEmpty) {
  sycl::detail::AllowListParsedT ActualValue = sycl::detail::parseAllowList("");
  sycl::detail::AllowListParsedT ExpectedValue{};
  EXPECT_EQ(ExpectedValue, ActualValue);
}

TEST(ParseAllowListTests, CheckAllowListSingleDeviceDesc) {
  sycl::detail::AllowListParsedT ActualValue = sycl::detail::parseAllowList(
      "BackendName:level_zero,DeviceType:gpu,DeviceVendorId:0x0000");
  sycl::detail::AllowListParsedT ExpectedValue{{{"BackendName", "level_zero"},
                                                {"DeviceType", "gpu"},
                                                {"DeviceVendorId", "0x0000"}}};
  EXPECT_EQ(ExpectedValue, ActualValue);
}

TEST(ParseAllowListTests, CheckAllowListMultipleDeviceDesc) {
  sycl::detail::AllowListParsedT ActualValue = sycl::detail::parseAllowList(
      "BackendName:level_zero,DeviceType:gpu,DeviceVendorId:0x0000|BackendName:"
      "opencl,DeviceType:cpu,DeviceVendorId:0x1234|BackendName:opencl,"
      "DeviceType:acc,DeviceVendorId:0x4321");
  sycl::detail::AllowListParsedT ExpectedValue{{{"BackendName", "level_zero"},
                                                {"DeviceType", "gpu"},
                                                {"DeviceVendorId", "0x0000"}},
                                               {{"BackendName", "opencl"},
                                                {"DeviceType", "cpu"},
                                                {"DeviceVendorId", "0x1234"}},
                                               {{"BackendName", "opencl"},
                                                {"DeviceType", "acc"},
                                                {"DeviceVendorId", "0x4321"}}};
  EXPECT_EQ(ExpectedValue, ActualValue);
}

TEST(ParseAllowListTests, CheckUnsupportedKeyNameIsHandledInSingleDeviceDesc) {
  try {
    sycl::detail::AllowListParsedT ActualValue = sycl::detail::parseAllowList(
        "BackendName:level_zero,SomeUnsupportedKey:gpu");
    FAIL() << "Expected an exception";
  } catch (sycl::exception const &e) {
    EXPECT_EQ(
        std::string("Unrecognized key in SYCL_DEVICE_ALLOWLIST. For "
                    "details, please refer to "
                    "https://github.com/intel/llvm/blob/sycl/sycl/doc/"
                    "EnvironmentVariables.md -30 (PI_ERROR_INVALID_VALUE)"),
        e.what());
  } catch (...) {
    FAIL() << "Expected sycl::exception";
  }
}

TEST(
    ParseAllowListTests,
    CheckUnsupportedKeyNameIsHandledInTwoDeviceDescsFirstContainingRegexValue) {
  try {
    sycl::detail::AllowListParsedT ActualValue = sycl::detail::parseAllowList(
        "DriverVersion:{{value}}|SomeUnsupportedKey:gpu");
    FAIL() << "Expected an exception";
  } catch (sycl::exception const &e) {
    EXPECT_EQ(
        std::string("Unrecognized key in SYCL_DEVICE_ALLOWLIST. For "
                    "details, please refer to "
                    "https://github.com/intel/llvm/blob/sycl/sycl/doc/"
                    "EnvironmentVariables.md -30 (PI_ERROR_INVALID_VALUE)"),
        e.what());
  } catch (...) {
    FAIL() << "Expected sycl::exception";
  }
}

TEST(
    ParseAllowListTests,
    CheckUnsupportedKeyNameIsHandledInTwoDeviceDescsFirstContainingFixedValue) {
  try {
    sycl::detail::AllowListParsedT ActualValue = sycl::detail::parseAllowList(
        "BackendName:level_zero|SomeUnsupportedKey:gpu");
    FAIL() << "Expected an exception";
  } catch (sycl::exception const &e) {
    EXPECT_EQ(
        std::string("Unrecognized key in SYCL_DEVICE_ALLOWLIST. For "
                    "details, please refer to "
                    "https://github.com/intel/llvm/blob/sycl/sycl/doc/"
                    "EnvironmentVariables.md -30 (PI_ERROR_INVALID_VALUE)"),
        e.what());
  } catch (...) {
    FAIL() << "Expected sycl::exception";
  }
}

TEST(ParseAllowListTests,
     CheckUnsupportedKeyNameIsHandledInTwoDeviceDescsBothContainingRegexValue) {
  try {
    sycl::detail::AllowListParsedT ActualValue = sycl::detail::parseAllowList(
        "DriverVersion:{{value1}}|SomeUnsupportedKey:{{value2}}");
    FAIL() << "Expected an exception";
  } catch (sycl::exception const &e) {
    EXPECT_EQ(
        std::string("Unrecognized key in SYCL_DEVICE_ALLOWLIST. For "
                    "details, please refer to "
                    "https://github.com/intel/llvm/blob/sycl/sycl/doc/"
                    "EnvironmentVariables.md -30 (PI_ERROR_INVALID_VALUE)"),
        e.what());
  } catch (...) {
    FAIL() << "Expected sycl::exception";
  }
}

TEST(ParseAllowListTests, CheckRegexIsProcessedCorrectly) {
  sycl::detail::AllowListParsedT ActualValue = sycl::detail::parseAllowList(
      "DeviceName:{{regex1}},DriverVersion:{{regex1|regex2}}|PlatformName:{{"
      "regex3}},PlatformVersion:{{regex4|regex5|regex6}}");
  sycl::detail::AllowListParsedT ExpectedValue{
      {{"DeviceName", "regex1"}, {"DriverVersion", "regex1|regex2"}},
      {{"PlatformName", "regex3"},
       {"PlatformVersion", "regex4|regex5|regex6"}}};
  EXPECT_EQ(ExpectedValue, ActualValue);
}

TEST(ParseAllowListTests, CheckMissingOpenDoubleCurlyBracesAreHandled) {
  try {
    sycl::detail::AllowListParsedT ActualValue = sycl::detail::parseAllowList(
        "DeviceName:regex1}},DriverVersion:{{regex1|regex2}}");
    FAIL() << "Expected an exception";
  } catch (sycl::exception const &e) {
    EXPECT_EQ(std::string("Key DeviceName of SYCL_DEVICE_ALLOWLIST "
                          "should have value which starts with {{ -30 "
                          "(PI_ERROR_INVALID_VALUE)"),
              e.what());
  } catch (...) {
    FAIL() << "Expected sycl::exception";
  }
}

TEST(ParseAllowListTests, CheckMissingClosedDoubleCurlyBracesAreHandled) {
  try {
    sycl::detail::AllowListParsedT ActualValue = sycl::detail::parseAllowList(
        "DeviceName:{{regex1}},DriverVersion:{{regex1|regex2");
    FAIL() << "Expected an exception";
  } catch (sycl::exception const &e) {
    EXPECT_EQ(std::string("Key DriverVersion of SYCL_DEVICE_ALLOWLIST "
                          "should have value which ends with }} -30 "
                          "(PI_ERROR_INVALID_VALUE)"),
              e.what());
  } catch (...) {
    FAIL() << "Expected sycl::exception";
  }
}

TEST(ParseAllowListTests, CheckAllValidBackendNameValuesAreProcessed) {
  std::string AllowList;
  for (const auto &SyclBe : sycl::detail::getSyclBeMap()) {
    if (!AllowList.empty())
      AllowList += "|";
    AllowList += "BackendName:" + SyclBe.first;
  }
  sycl::detail::AllowListParsedT ActualValue =
      sycl::detail::parseAllowList(AllowList);
  sycl::detail::AllowListParsedT ExpectedValue{
      {{"BackendName", "host"}},       {{"BackendName", "opencl"}},
      {{"BackendName", "level_zero"}}, {{"BackendName", "cuda"}},
      {{"BackendName", "hip"}},        {{"BackendName", "native_cpu"}},
      {{"BackendName", "*"}}};
  EXPECT_EQ(ExpectedValue, ActualValue);
}

TEST(ParseAllowListTests, CheckAllValidDeviceTypeValuesAreProcessed) {
  std::string AllowList;
  for (const auto &SyclDeviceType :
       sycl::detail::getSyclDeviceTypeMap<true /*Enable 'acc'*/>()) {
    if (!AllowList.empty())
      AllowList += "|";
    AllowList += "DeviceType:" + SyclDeviceType.first;
  }
  sycl::detail::AllowListParsedT ActualValue =
      sycl::detail::parseAllowList(AllowList);
  sycl::detail::AllowListParsedT ExpectedValue{
      {{"DeviceType", "host"}}, {{"DeviceType", "cpu"}},
      {{"DeviceType", "gpu"}},  {{"DeviceType", "acc"}},
      {{"DeviceType", "fpga"}}, {{"DeviceType", "*"}}};
  EXPECT_EQ(ExpectedValue, ActualValue);
}

TEST(ParseAllowListTests, CheckIncorrectBackendNameValueIsHandled) {
  try {
    sycl::detail::AllowListParsedT ActualValue =
        sycl::detail::parseAllowList("BackendName:blablabla");
    FAIL() << "Expected an exception";
  } catch (sycl::exception const &e) {
    EXPECT_EQ(
        std::string("Value blablabla for key BackendName is not valid in "
                    "SYCL_DEVICE_ALLOWLIST. For details, please refer to "
                    "https://github.com/intel/llvm/blob/sycl/sycl/doc/"
                    "EnvironmentVariables.md -30 (PI_ERROR_INVALID_VALUE)"),
        e.what());
  } catch (...) {
    FAIL() << "Expected sycl::exception";
  }
}

TEST(ParseAllowListTests, CheckIncorrectDeviceTypeValueIsHandled) {
  try {
    sycl::detail::AllowListParsedT ActualValue =
        sycl::detail::parseAllowList("DeviceType:blablabla");
    FAIL() << "Expected an exception";
  } catch (sycl::exception const &e) {
    EXPECT_EQ(
        std::string("Value blablabla for key DeviceType is not valid in "
                    "SYCL_DEVICE_ALLOWLIST. For details, please refer to "
                    "https://github.com/intel/llvm/blob/sycl/sycl/doc/"
                    "EnvironmentVariables.md -30 (PI_ERROR_INVALID_VALUE)"),
        e.what());
  } catch (...) {
    FAIL() << "Expected sycl::exception";
  }
}

TEST(ParseAllowListTests, CheckIncorrectDeviceVendorIdValueIsHandled) {
  try {
    sycl::detail::AllowListParsedT ActualValue =
        sycl::detail::parseAllowList("DeviceVendorId:blablabla");
    FAIL() << "Expected an exception";
  } catch (sycl::exception const &e) {
    EXPECT_EQ(
        std::string("Value blablabla for key DeviceVendorId is not valid in "
                    "SYCL_DEVICE_ALLOWLIST. It should have the hex format. For "
                    "details, please refer to "
                    "https://github.com/intel/llvm/blob/sycl/sycl/doc/"
                    "EnvironmentVariables.md -30 (PI_ERROR_INVALID_VALUE)"),
        e.what());
  } catch (...) {
    FAIL() << "Expected sycl::exception";
  }
}

TEST(ParseAllowListTests, CheckTwoColonsBetweenKeyAndValue) {
  sycl::detail::AllowListParsedT ActualValue =
      sycl::detail::parseAllowList("DeviceVendorId::0x1234");
  sycl::detail::AllowListParsedT ExpectedValue{{{"DeviceVendorId", "0x1234"}}};
  EXPECT_EQ(ExpectedValue, ActualValue);
}

TEST(ParseAllowListTests, CheckMultipleColonsBetweenKeyAndValue) {
  sycl::detail::AllowListParsedT ActualValue =
      sycl::detail::parseAllowList("DeviceVendorId:::::0x1234");
  sycl::detail::AllowListParsedT ExpectedValue{{{"DeviceVendorId", "0x1234"}}};
  EXPECT_EQ(ExpectedValue, ActualValue);
}

TEST(ParseAllowListTests, CheckExceptionIsThrownForValueWOColonDelim) {
  try {
    sycl::detail::AllowListParsedT ActualValue =
        sycl::detail::parseAllowList("SomeValueWOColonDelimiter");
    FAIL() << "Expected an exception";
  } catch (sycl::exception const &e) {
    EXPECT_EQ(
        std::string("SYCL_DEVICE_ALLOWLIST has incorrect format. For "
                    "details, please refer to "
                    "https://github.com/intel/llvm/blob/sycl/sycl/"
                    "doc/EnvironmentVariables.md -30 (PI_ERROR_INVALID_VALUE)"),
        e.what());
  } catch (...) {
    FAIL() << "Expected sycl::exception";
  }
}

TEST(ParseAllowListTests, CheckDeviceNameDeprecationWarning) {
  testing::internal::CaptureStdout();
  sycl::detail::parseAllowList("DeviceName:{{regex}}");
  std::string ActualOutput = testing::internal::GetCapturedStdout();
  EXPECT_EQ("\nWARNING: DeviceName in SYCL_DEVICE_ALLOWLIST is deprecated. "
            "Please use BackendName, DeviceType and DeviceVendorId instead. "
            "For details, please refer to "
            "https://github.com/intel/llvm/blob/sycl/sycl/doc/"
            "EnvironmentVariables.md\n\n",
            ActualOutput);
}

TEST(ParseAllowListTests, CheckPlatformNameDeprecationWarning) {
  testing::internal::CaptureStdout();
  sycl::detail::parseAllowList("PlatformName:{{regex}}");
  std::string ActualOutput = testing::internal::GetCapturedStdout();
  EXPECT_EQ("\nWARNING: PlatformName in SYCL_DEVICE_ALLOWLIST is deprecated. "
            "Please use BackendName, DeviceType and DeviceVendorId instead. "
            "For details, please refer to "
            "https://github.com/intel/llvm/blob/sycl/sycl/doc/"
            "EnvironmentVariables.md\n\n",
            ActualOutput);
}

TEST(ParseAllowListTests, CheckDeviceNameAndPlatformNameDeprecationWarning) {
  testing::internal::CaptureStdout();
  sycl::detail::parseAllowList("DeviceName:{{regex}},PlatformName:{{regex}}");
  std::string ActualOutput = testing::internal::GetCapturedStdout();
  EXPECT_EQ("\nWARNING: DeviceName and PlatformName in SYCL_DEVICE_ALLOWLIST "
            "are deprecated. Please use BackendName, DeviceType and "
            "DeviceVendorId instead. For details, please refer to "
            "https://github.com/intel/llvm/blob/sycl/sycl/doc/"
            "EnvironmentVariables.md\n\n",
            ActualOutput);
}

TEST(ParseAllowListTests, CheckNoDeprecationWarningForNotDeprecatedKeys) {
  testing::internal::CaptureStdout();
  sycl::detail::parseAllowList(
      "BackendName:level_zero,DeviceType:gpu,DeviceVendorId:0x0000,"
      "DriverVersion:{{regex1}},PlatformVersion:{{regex2}}");
  std::string ActualOutput = testing::internal::GetCapturedStdout();
  EXPECT_EQ("", ActualOutput);
}
