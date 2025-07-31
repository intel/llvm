//==---- HasExtensionWordBoundary.cpp --- Test word boundary fix ----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This test verifies that has_extension correctly matches full extension names
// and doesn't match partial substrings.
//
//===----------------------------------------------------------------------===//

#include <detail/device_impl.hpp>
#include <gtest/gtest.h>
#include <helpers/UrMock.hpp>
#include <sycl/sycl.hpp>
#include <ur_mock_helpers.hpp>

using namespace sycl;

thread_local std::string MockExtensions = "";

static ur_result_t redefinedDeviceGetInfo(void *pParams) {
  auto Params = *static_cast<ur_device_get_info_params_t *>(pParams);

  if (*Params.ppropName == UR_DEVICE_INFO_EXTENSIONS) {
    // Override extensions query with mock data.
    if (*Params.ppPropValue) {
      size_t Len = MockExtensions.length() + 1;
      if (*Params.ppropSize >= Len)
        std::memcpy(*Params.ppPropValue, MockExtensions.c_str(), Len);
    }
    if (*Params.ppPropSizeRet)
      **Params.ppPropSizeRet = MockExtensions.length() + 1;

    return UR_RESULT_SUCCESS;
  }

  // Delegate to the default mock.
  return sycl::unittest::MockAdapter::mock_urDeviceGetInfo(pParams);
}

class HasExtensionWordBoundaryTest : public ::testing::Test {
public:
  HasExtensionWordBoundaryTest() : Mock{} {}

protected:
  void SetUp() override {
    mock::getCallbacks().set_replace_callback("urDeviceGetInfo",
                                              &redefinedDeviceGetInfo);
  }

  sycl::unittest::UrMock<> Mock;
};

TEST_F(HasExtensionWordBoundaryTest, ExactMatchWorks) {
  MockExtensions = "cl_khr_fp64 cl_intel_subgroups cl_khr_subgroups";

  sycl::platform Plt{sycl::platform()};
  sycl::device Dev = Plt.get_devices()[0];

  EXPECT_TRUE(Dev.has_extension("cl_khr_fp64"));
  EXPECT_TRUE(Dev.has_extension("cl_intel_subgroups"));
  EXPECT_TRUE(Dev.has_extension("cl_khr_subgroups"));
}

TEST_F(HasExtensionWordBoundaryTest, SubstringDoesNotMatch) {
  MockExtensions = "cl_intel_subgroups cl_khr_fp64_extended";

  sycl::platform Plt{sycl::platform()};
  sycl::device Dev = Plt.get_devices()[0];

  EXPECT_FALSE(Dev.has_extension("cl_intel_subgroup"));
  EXPECT_FALSE(Dev.has_extension("cl_khr_fp64"));
  EXPECT_FALSE(Dev.has_extension("subgroups"));
  EXPECT_FALSE(Dev.has_extension("intel_subgroups"));
}

TEST_F(HasExtensionWordBoundaryTest, EmptyExtensions) {
  MockExtensions = "";

  sycl::platform Plt{sycl::platform()};
  sycl::device Dev = Plt.get_devices()[0];

  EXPECT_FALSE(Dev.has_extension("cl_khr_fp64"));
}

TEST_F(HasExtensionWordBoundaryTest, SingleExtension) {
  MockExtensions = "cl_khr_fp64";

  sycl::platform Plt{sycl::platform()};
  sycl::device Dev = Plt.get_devices()[0];
  auto DevImpl = detail::getSyclObjImpl(Dev);

  EXPECT_TRUE(Dev.has_extension("cl_khr_fp64"));
  EXPECT_FALSE(Dev.has_extension("cl_khr_fp6"));
}

TEST_F(HasExtensionWordBoundaryTest, FirstMiddleLastExtensions) {
  MockExtensions = "cl_first_ext cl_middle_ext cl_last_ext";

  sycl::platform Plt{sycl::platform()};
  sycl::device Dev = Plt.get_devices()[0];
  auto DevImpl = detail::getSyclObjImpl(Dev);

  EXPECT_TRUE(Dev.has_extension("cl_first_ext"));
  EXPECT_TRUE(Dev.has_extension("cl_middle_ext"));
  EXPECT_TRUE(Dev.has_extension("cl_last_ext"));
}

TEST_F(HasExtensionWordBoundaryTest, NonUniformGroupExtensions) {
  MockExtensions = "cl_khr_subgroup_non_uniform_vote "
                   "cl_khr_subgroup_ballot "
                   "cl_intel_subgroups "
                   "cl_intel_spirv_subgroups "
                   "cl_intel_subgroup_matrix_multiply_accumulate";

  sycl::platform Plt{sycl::platform()};
  sycl::device Dev = Plt.get_devices()[0];
  auto DevImpl = detail::getSyclObjImpl(Dev);

  EXPECT_TRUE(Dev.has_extension("cl_khr_subgroup_non_uniform_vote"));
  EXPECT_TRUE(Dev.has_extension("cl_khr_subgroup_ballot"));
  EXPECT_TRUE(Dev.has_extension("cl_intel_subgroups"));
  EXPECT_TRUE(Dev.has_extension("cl_intel_spirv_subgroups"));
  EXPECT_TRUE(
      Dev.has_extension("cl_intel_subgroup_matrix_multiply_accumulate"));

  EXPECT_FALSE(Dev.has_extension("cl_khr_subgroup"));
  EXPECT_FALSE(Dev.has_extension("cl_intel_subgroup"));
  EXPECT_FALSE(Dev.has_extension("non_uniform_vote"));
  EXPECT_FALSE(Dev.has_extension("subgroup_matrix_multiply"));
}
