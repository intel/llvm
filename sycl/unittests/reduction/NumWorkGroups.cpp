//==------ NumWorkGroups.cpp --- range reduction work-group count tests ----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <gtest/gtest.h>
#include <helpers/UrMock.hpp>
#include <sycl/sycl.hpp>

#include <array>
#include <climits>

namespace {

uint32_t ComputeUnits;
uint32_t VendorId;
ur_bool_t Integrated;
size_t MaxGroupsX;

ur_result_t redefinedDeviceGetInfoAfter(void *pParams) {
  auto Params = *static_cast<ur_device_get_info_params_t *>(pParams);
  auto SetValue = [&](auto Value) {
    if (*Params.ppPropValue)
      *static_cast<decltype(Value) *>(*Params.ppPropValue) = Value;
    if (*Params.ppPropSizeRet)
      **Params.ppPropSizeRet = sizeof(Value);
  };
  if (*Params.ppropName == UR_DEVICE_INFO_MAX_COMPUTE_UNITS)
    SetValue(ComputeUnits);
  else if (*Params.ppropName == UR_DEVICE_INFO_VENDOR_ID)
    SetValue(VendorId);
  else if (*Params.ppropName == UR_DEVICE_INFO_HOST_UNIFIED_MEMORY)
    SetValue(Integrated);
  else if (*Params.ppropName == UR_DEVICE_INFO_MAX_WORK_GROUPS_3D)
    SetValue(std::array<size_t, 3>{MaxGroupsX, 1, 1});
  return UR_RESULT_SUCCESS;
}

class ReductionNumWorkGroupsTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Discrete Intel GPU with 448 compute units unless a test overrides it.
    ComputeUnits = 448;
    VendorId = 0x8086;
    Integrated = false;
    MaxGroupsX = UINT32_MAX;
    mock::getCallbacks().set_after_callback("urDeviceGetInfo",
                                            &redefinedDeviceGetInfoAfter);
  }

  size_t getMaxNumWorkGroups(size_t NWorkItems, size_t WGSize) {
    size_t Result = 0;
    sycl::queue{}.submit([&](sycl::handler &CGH) {
      Result = sycl::detail::reduGetMaxNumWorkGroupsForRange(
          CGH, NWorkItems, WGSize, /*ElemSize=*/sizeof(double));
    });
    return Result;
  }

  sycl::unittest::UrMock<> Mock;
};

constexpr size_t SmallRange = size_t{1} << 20;
constexpr size_t LargeRange = size_t{1} << 27;

} // namespace

TEST_F(ReductionNumWorkGroupsTest, SmallRangeKeepsBaseLimit) {
  EXPECT_EQ(getMaxNumWorkGroups(SmallRange, 1024), 448u);
}

TEST_F(ReductionNumWorkGroupsTest, LargeRangeIncreasesLimit) {
  // 128 bytes per work-item: 16 doubles, so 2^27 / (1024 * 16) work-groups.
  EXPECT_EQ(getMaxNumWorkGroups(LargeRange, 1024), 8192u);
}

TEST_F(ReductionNumWorkGroupsTest, NonIntelGPUKeepsBaseLimit) {
  VendorId = 0x10de;
  EXPECT_EQ(getMaxNumWorkGroups(LargeRange, 1024), 448u);
}

TEST_F(ReductionNumWorkGroupsTest, IntegratedGPUKeepsHigherBaseLimit) {
  ComputeUnits = 96;
  Integrated = true;
  EXPECT_EQ(getMaxNumWorkGroups(SmallRange, 1024), 96u * 8);
}

TEST_F(ReductionNumWorkGroupsTest, HugeRangeKeepsGlobalRangeWithinInt) {
  EXPECT_EQ(getMaxNumWorkGroups(size_t{1} << 40, 1024), size_t{INT_MAX} / 1024);
}

TEST_F(ReductionNumWorkGroupsTest, DeviceLimitIsRespected) {
  MaxGroupsX = 4096;
  EXPECT_EQ(getMaxNumWorkGroups(LargeRange, 1024), 4096u);
}

TEST_F(ReductionNumWorkGroupsTest, ZeroWorkGroupSize) {
  EXPECT_EQ(getMaxNumWorkGroups(LargeRange, 0), 448u);

  TEST_F(ReductionNumWorkGroupsTest, DeviceLimitBelowBaseLimitIsRespected) {
    MaxGroupsX = 128;
    EXPECT_EQ(getMaxNumWorkGroups(SmallRange, 1024), 128u);
  }
