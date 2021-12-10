//==------- DeviceIsAllowed.cpp --- SYCL_DEVICE_ALLOWLIST unit test --------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/allowlist.hpp>

#include <gtest/gtest.h>

constexpr char SyclDeviceAllowList[] =
    "BackendName:opencl,DeviceType:gpu,DeviceVendorId:0x8086,DriverVersion:{{("
    "19\\.(4[3-9]|[5-9]\\d)\\..*)|([2-9][0-9]\\.\\d+\\..*)|(\\d+\\.\\d+\\."
    "100\\.(737[2-9]|73[8-9]\\d|7[4-9]\\d+|[8-9]\\d+)|\\.\\d+\\.\\d+\\.10[1-9]"
    "\\.\\d+)}}|BackendName:opencl,DeviceType:cpu,DeviceVendorId:0x8086,"
    "DriverVersion:{{(2019\\.[^\\.]+\\.[1-9][1-9]\\..*)|(20[2-9][0-9]\\..*)}}|"
    "BackendName:opencl,DeviceType:acc,DeviceVendorId:0x1172,DriverVersion:{{("
    "2019\\.[^\\.]+\\.[1-9][0-9]\\..*)|(20[2-9][0-9]\\..*)}}|BackendName:"
    "opencl,DeviceType:acc,DeviceVendorId:0x1172,PlatformVersion:{{.*Version "
    "(19\\.[3-9][0-9]*|2[0-9]\\.[0-9]+).*}}|BackendName:level_zero,DeviceType:"
    "gpu,DeviceVendorId:0x8086,DriverVersion:{{.*}}";
constexpr char SyclDeviceAllowListOldStyle[] =
    "DeviceName:{{.*Intel.*Graphics.*}},DriverVersion:{{(19\\.(4[3-9]|[5-9]\\d)"
    "\\..*)|([2-9][0-9]\\.\\d+\\..*)|(\\d+\\.\\d+\\.100\\.(737[2-9]|73[8-9]\\d|"
    "7[4-9]\\d+|[8-9]\\d+)|\\.\\d+\\.\\d+\\.10[1-9]\\.\\d+)}}|DeviceName:{{.*"
    "Intel.*(CPU|Processor).*}},DriverVersion:{{(2019\\.[^\\.]+\\.[1-9][1-9]\\."
    ".*)|(20[2-9][0-9]\\..*)}}|DeviceName:{{.*Intel.*FPGA "
    "Emulation.*}},DriverVersion:{{(2019\\.[^\\.]+\\.[1-9][0-9]\\..*)|(20[2-9]["
    "0-9]\\..*)}}|PlatformName:{{.*Intel.*FPGA.*}},PlatformVersion:{{.*Version "
    "(19\\.[3-9][0-9]*|2[0-9]\\.[0-9]+).*}}|PlatformName:{{.*Intel.*Level-Zero."
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

sycl::detail::DeviceDescT OpenCLFPGAEmuDeviceDesc{
    {"BackendName", "opencl"},
    {"DeviceType", "acc"},
    {"DeviceVendorId", "0x1172"},
    {"DriverVersion", "2021.12.5.0.09"},
    {"PlatformVersion",
     "OpenCL 1.2 Intel(R) FPGA SDK for OpenCL(TM), Version 20.3"},
    {"DeviceName", "Intel(R) FPGA Emulation Device"},
    {"PlatformName", "Intel(R) FPGA Emulation Platform for OpenCL(TM)"}};

sycl::detail::DeviceDescT OpenCLFPGABoardDeviceDesc{
    {"BackendName", "opencl"},
    {"DeviceType", "acc"},
    {"DeviceVendorId", "0x1172"},
    {"DriverVersion", "20.3.0.0.00"},
    {"PlatformVersion",
     "OpenCL 1.0 Intel(R) FPGA SDK for OpenCL(TM), Version 20.3"},
    {"DeviceName", "Intel(R) Arria(R) 10 GX FPGA"},
    {"PlatformName", "Intel(R) FPGA SDK for OpenCL(TM)"}};

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

TEST(DeviceIsAllowedTests, CheckSupportedOpenCLCPUDeviceIsAllowed) {
  bool Actual = sycl::detail::deviceIsAllowed(
      OpenCLCPUDeviceDesc, sycl::detail::parseAllowList(SyclDeviceAllowList));
  EXPECT_EQ(Actual, true);
}

TEST(DeviceIsAllowedTests, CheckSupportedOpenCLFPGAEmuDeviceIsAllowed) {
  bool Actual = sycl::detail::deviceIsAllowed(
      OpenCLFPGAEmuDeviceDesc,
      sycl::detail::parseAllowList(SyclDeviceAllowList));
  EXPECT_EQ(Actual, true);
}

TEST(DeviceIsAllowedTests, CheckSupportedOpenCLFPGABoardDeviceIsAllowed) {
  bool Actual = sycl::detail::deviceIsAllowed(
      OpenCLFPGABoardDeviceDesc,
      sycl::detail::parseAllowList(SyclDeviceAllowList));
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
     CheckOpenCLFPGABoardDeviceWithNotSupportedPlatformVersionIsNotAllowed) {
  auto DeviceDesc = OpenCLFPGABoardDeviceDesc;
  DeviceDesc.at("PlatformVersion") = "42";
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
     CheckSupportedOpenCLFPGAEmuDeviceIsAllowedInOldStyle) {
  bool Actual = sycl::detail::deviceIsAllowed(
      OpenCLFPGAEmuDeviceDesc,
      sycl::detail::parseAllowList(SyclDeviceAllowListOldStyle));
  EXPECT_EQ(Actual, true);
}

TEST(DeviceIsAllowedTests,
     CheckSupportedOpenCLFPGABoardDeviceIsAllowedInOldStyle) {
  bool Actual = sycl::detail::deviceIsAllowed(
      OpenCLFPGABoardDeviceDesc,
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

TEST(
    DeviceIsAllowedTests,
    CheckOpenCLFPGABoardDeviceWithNotSupportedPlatformNameIsNotAllowedInOldStyle) {
  auto DeviceDesc = OpenCLFPGABoardDeviceDesc;
  DeviceDesc.at("PlatformName") = "AABBCCDD";
  bool Actual = sycl::detail::deviceIsAllowed(
      DeviceDesc, sycl::detail::parseAllowList(SyclDeviceAllowListOldStyle));
  EXPECT_EQ(Actual, false);
}
