//==------- DeviceIsAllowed.cpp --- SYCL_DEVICE_ALLOWLIST unit test --------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/allowlist.hpp>
#include <sycl/platform.hpp>

#include <gtest/gtest.h>

#ifdef _WIN32
#include <windows.h> // SetEnvironmentVariable
#endif

constexpr char SyclDeviceAllowList[] =
    "BackendName:opencl,DeviceType:gpu,DeviceVendorId:0x8086,DriverVersion:{{("
    "19\\.(4[3-9]|[5-9]\\d)\\..*)|([2-9][0-9]\\.\\d+\\..*)|(\\d+\\.\\d+\\."
    "100\\.(737[2-9]|73[8-9]\\d|7[4-9]\\d+|[8-9]\\d+)|\\.\\d+\\.\\d+\\.10[1-9]"
    "\\.\\d+)}}|BackendName:opencl,DeviceType:cpu,DeviceVendorId:0x8086,"
    "DriverVersion:{{(2019\\.[^\\.]+\\.[1-9][1-9]\\..*)|(20[2-9][0-9]\\..*)}}|"
    "BackendName:level_zero,DeviceType:gpu,DeviceVendorId:0x8086,DriverVersion:"
    "{{.*}}";
constexpr char SyclDeviceAllowListOldStyle[] =
    "DeviceName:{{.*Intel.*Graphics.*}},DriverVersion:{{(19\\.(4[3-9]|[5-9]\\d)"
    "\\..*)|([2-9][0-9]\\.\\d+\\..*)|(\\d+\\.\\d+\\.100\\.(737[2-9]|73[8-9]\\d|"
    "7[4-9]\\d+|[8-9]\\d+)|\\.\\d+\\.\\d+\\.10[1-9]\\.\\d+)}}|DeviceName:{{.*"
    "Intel.*(CPU|Processor).*}},DriverVersion:{{(2019\\.[^\\.]+\\.[1-9][1-9]\\."
    ".*)|(20[2-9][0-9]\\..*)}}|PlatformName:{{.*Intel.*Level-Zero."
    "*}},DeviceName:{{.*Intel.*Gen.*}},DriverVersion:{{.*}}";

sycl::detail::DeviceDescT OpenCLGPUDeviceDesc{
    {"BackendName", "opencl"},
    {"DeviceType", "gpu"},
    {"DeviceVendorId", "0x8086"},
    {"DriverVersion", "21.19.19792"},
    {"PlatformVersion", "OpenCL 3.0"},
    {"DeviceName", "Intel(R) HD Graphics 630 [0x5912]"},
    {"PlatformName", "Intel(R) OpenCL HD Graphics"}};

sycl::detail::DeviceDescT OpenCLCPUDeviceDesc{
    {"BackendName", "opencl"},
    {"DeviceType", "cpu"},
    {"DeviceVendorId", "0x8086"},
    {"DriverVersion", "2021.12.5.0.09"},
    {"PlatformVersion", "OpenCL 2.1 LINUX"},
    {"DeviceName", "Intel(R) Core(TM) i7-8700K Processor @ 4.60GHz"},
    {"PlatformName", "Intel(R) OpenCL"}};

sycl::detail::DeviceDescT LevelZeroGPUDeviceDesc{
    {"BackendName", "level_zero"},
    {"DeviceType", "gpu"},
    {"DeviceVendorId", "0x8086"},
    {"DriverVersion", "1.1.19792"},
    {"PlatformVersion", "1.1"},
    {"DeviceName", "Intel(R) Gen9 HD Graphics 630"},
    {"PlatformName", "Intel(R) Level-Zero"}};

TEST(DeviceIsAllowedTests, CheckSupportedOpenCLGPUDeviceIsAllowed) {
  bool Actual = sycl::detail::deviceIsAllowed(
      OpenCLGPUDeviceDesc, sycl::detail::parseAllowList(SyclDeviceAllowList));
  EXPECT_EQ(Actual, true);
}

TEST(DeviceIsAllowedTests, CheckLocalizationDoesNotImpact) {
  // The localization can affect std::stringstream output.
  // We want to make sure that DeviceVenderId doesn't have a comma
  // inserted (ie "0x8,086" ), which will break the platform retrieval.

  if (sycl::platform::get_platforms().empty()) {
    GTEST_SKIP() << "No SYCL platforms found.";
  }

  try {
    auto previous = std::locale::global(std::locale("en_US.UTF-8"));
#ifdef _WIN32
    SetEnvironmentVariableA("SYCL_DEVICE_ALLOWLIST", SyclDeviceAllowList);
#else
    setenv("SYCL_DEVICE_ALLOWLIST", SyclDeviceAllowList, 1);
#endif

    auto post_platforms = sycl::platform::get_platforms();
    std::locale::global(previous);
#ifdef _WIN32
    SetEnvironmentVariableA("SYCL_DEVICE_ALLOWLIST", nullptr);
#else
    unsetenv("SYCL_DEVICE_ALLOWLIST");
#endif

    EXPECT_NE(size_t{0}, post_platforms.size());
  } catch (...) {
    // It is possible that the en_US locale is not available.
    // In this case, we just skip the test.
    GTEST_SKIP() << "Locale en_US.UTF-8 not available.";
  }
}

TEST(DeviceIsAllowedTests, CheckSupportedOpenCLCPUDeviceIsAllowed) {
  bool Actual = sycl::detail::deviceIsAllowed(
      OpenCLCPUDeviceDesc, sycl::detail::parseAllowList(SyclDeviceAllowList));
  EXPECT_EQ(Actual, true);
}

TEST(DeviceIsAllowedTests, CheckSupportedLevelZeroGPUDeviceIsAllowed) {
  bool Actual = sycl::detail::deviceIsAllowed(
      LevelZeroGPUDeviceDesc,
      sycl::detail::parseAllowList(SyclDeviceAllowList));
  EXPECT_EQ(Actual, true);
}

TEST(DeviceIsAllowedTests,
     CheckOpenCLGPUDeviceWithNotSupportedBackendNameIsNotAllowed) {
  auto DeviceDesc = OpenCLGPUDeviceDesc;
  DeviceDesc.at("BackendName") = "cuda";
  bool Actual = sycl::detail::deviceIsAllowed(
      DeviceDesc, sycl::detail::parseAllowList(SyclDeviceAllowList));
  EXPECT_EQ(Actual, false);
}

TEST(DeviceIsAllowedTests,
     CheckOpenCLGPUDeviceWithNotSupportedDeviceTypeIsNotAllowed) {
  auto DeviceDesc = OpenCLGPUDeviceDesc;
  DeviceDesc.at("DeviceType") = "cpu";
  bool Actual = sycl::detail::deviceIsAllowed(
      DeviceDesc, sycl::detail::parseAllowList(SyclDeviceAllowList));
  EXPECT_EQ(Actual, false);
}

TEST(DeviceIsAllowedTests,
     CheckOpenCLGPUDeviceWithNotSupportedDeviceVendorIdIsNotAllowed) {
  auto DeviceDesc = OpenCLGPUDeviceDesc;
  DeviceDesc.at("DeviceVendorId") = "0x0000";
  bool Actual = sycl::detail::deviceIsAllowed(
      DeviceDesc, sycl::detail::parseAllowList(SyclDeviceAllowList));
  EXPECT_EQ(Actual, false);
}

TEST(DeviceIsAllowedTests,
     CheckOpenCLGPUDeviceWithNotSupportedDriverVersionIsNotAllowed) {
  auto DeviceDesc = OpenCLGPUDeviceDesc;
  DeviceDesc.at("DriverVersion") = "0.0.0.0";
  bool Actual = sycl::detail::deviceIsAllowed(
      DeviceDesc, sycl::detail::parseAllowList(SyclDeviceAllowList));
  EXPECT_EQ(Actual, false);
}

TEST(DeviceIsAllowedTests,
     DISABLED_CheckAssertHappensIfIncompleteDeviceDescIsPassedToTheFunc) {
  sycl::detail::DeviceDescT IncompleteDeviceDesc{{"BackendName", "level_zero"}};
  EXPECT_DEATH(sycl::detail::deviceIsAllowed(
                   IncompleteDeviceDesc,
                   sycl::detail::parseAllowList(SyclDeviceAllowList)),
               ".*DeviceDesc map should have all supported keys for.*"
               "SYCL_DEVICE_ALLOWLIST..*");
}

TEST(DeviceIsAllowedTests, CheckSupportedOpenCLGPUDeviceIsAllowedInOldStyle) {
  bool Actual = sycl::detail::deviceIsAllowed(
      OpenCLGPUDeviceDesc,
      sycl::detail::parseAllowList(SyclDeviceAllowListOldStyle));
  EXPECT_EQ(Actual, true);
}

TEST(DeviceIsAllowedTests, CheckSupportedOpenCLCPUDeviceIsAllowedInOldStyle) {
  bool Actual = sycl::detail::deviceIsAllowed(
      OpenCLCPUDeviceDesc,
      sycl::detail::parseAllowList(SyclDeviceAllowListOldStyle));
  EXPECT_EQ(Actual, true);
}

TEST(DeviceIsAllowedTests,
     CheckSupportedLevelZeroGPUDeviceIsAllowedInOldStyle) {
  bool Actual = sycl::detail::deviceIsAllowed(
      LevelZeroGPUDeviceDesc,
      sycl::detail::parseAllowList(SyclDeviceAllowListOldStyle));
  EXPECT_EQ(Actual, true);
}

TEST(DeviceIsAllowedTests,
     CheckLevelZeroGPUDeviceWithNotSupportedDeviceNameIsNotAllowedInOldStyle) {
  auto DeviceDesc = OpenCLGPUDeviceDesc;
  DeviceDesc.at("DeviceName") = "ABCD";
  bool Actual = sycl::detail::deviceIsAllowed(
      DeviceDesc, sycl::detail::parseAllowList(SyclDeviceAllowListOldStyle));
  EXPECT_EQ(Actual, false);
}
