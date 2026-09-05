//==---------- NvidiaDeviceArchitecture.cpp - architecture queries --------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <gtest/gtest.h>
#include <helpers/UrMock.hpp>
#include <sycl/sycl.hpp>

#include <algorithm>
#include <cstring>
#include <string>

namespace {

namespace syclex = sycl::ext::oneapi::experimental;

std::string DeviceVersion;

ur_result_t afterDeviceGetInfo(void *Params) {
  auto Args = *static_cast<ur_device_get_info_params_t *>(Params);
  if (*Args.ppropName != UR_DEVICE_INFO_VERSION)
    return UR_RESULT_SUCCESS;

  const size_t Size = DeviceVersion.size() + 1;
  if (*Args.ppPropValue)
    std::memcpy(*Args.ppPropValue, DeviceVersion.c_str(), Size);
  if (*Args.ppPropSizeRet)
    **Args.ppPropSizeRet = Size;
  return UR_RESULT_SUCCESS;
}

struct ArchitectureCase {
  const char *Version;
  syclex::architecture Expected;
};

class NvidiaDeviceArchitectureTest
    : public ::testing::TestWithParam<ArchitectureCase> {
protected:
  void SetUp() override {
    DeviceVersion = GetParam().Version;
    mock::getCallbacks().set_after_callback("urDeviceGetInfo",
                                            &afterDeviceGetInfo);
  }

  sycl::unittest::UrMock<sycl::backend::ext_oneapi_cuda> Mock;
};

TEST_P(NvidiaDeviceArchitectureTest, MapsComputeCapability) {
  const sycl::device Device = sycl::platform().get_devices()[0];
  EXPECT_EQ(Device.get_info<syclex::info::device::architecture>(),
            GetParam().Expected);
}

TEST_P(NvidiaDeviceArchitectureTest, ProvidesMatrixCombinations) {
  const sycl::device Device = sycl::platform().get_devices()[0];
  const auto Combinations =
      Device.get_info<syclex::info::device::matrix_combinations>();
  const bool IsKnownArchitecture =
      GetParam().Expected != syclex::architecture::unknown;
  const bool HasFp64 = std::any_of(
      Combinations.begin(), Combinations.end(), [](const auto &Combination) {
        using syclex::matrix::matrix_type;
        return Combination.atype == matrix_type::fp64 &&
               Combination.btype == matrix_type::fp64 &&
               Combination.ctype == matrix_type::fp64 &&
               Combination.dtype == matrix_type::fp64;
      });

  EXPECT_EQ(!Combinations.empty(), IsKnownArchitecture);
  EXPECT_EQ(HasFp64, IsKnownArchitecture);
}

INSTANTIATE_TEST_SUITE_P(
    ModernArchitectures, NvidiaDeviceArchitectureTest,
    ::testing::Values(
        ArchitectureCase{"8.8", syclex::architecture::nvidia_gpu_sm_88},
        ArchitectureCase{"10.0", syclex::architecture::nvidia_gpu_sm_100},
        ArchitectureCase{"10.1", syclex::architecture::nvidia_gpu_sm_101},
        ArchitectureCase{"10.3", syclex::architecture::nvidia_gpu_sm_103},
        ArchitectureCase{"11.0", syclex::architecture::nvidia_gpu_sm_110},
        ArchitectureCase{"12.0", syclex::architecture::nvidia_gpu_sm_120},
        ArchitectureCase{"12.1", syclex::architecture::nvidia_gpu_sm_121},
        ArchitectureCase{"12.2", syclex::architecture::unknown}));

} // namespace
