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

static std::string MockExtensions = "";

static ur_result_t redefinedDeviceGetInfo(void *pParams) {
  auto params = *static_cast<ur_device_get_info_params_t *>(pParams);

  if (*params.ppropName == UR_DEVICE_INFO_EXTENSIONS) {
    // override extensions query with mock data
    if (*params.ppPropValue) {
      size_t len = MockExtensions.length() + 1;
      if (*params.ppropSize >= len)
        std::memcpy(*params.ppPropValue, MockExtensions.c_str(), len);
    }
    if (*params.ppPropSizeRet)
      **params.ppPropSizeRet = MockExtensions.length() + 1;

    return UR_RESULT_SUCCESS;
  }

  // delegate to the default mock
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

  void TearDown() override {
    // get default mock by passing nullptr
    mock::getCallbacks().set_replace_callback("urDeviceGetInfo", nullptr);
  }

  sycl::unittest::UrMock<> Mock;
};

TEST_F(HasExtensionWordBoundaryTest, ExactMatchWorks) {
  MockExtensions = "cl_khr_fp64 cl_intel_subgroups cl_khr_subgroups";

  sycl::platform Plt{sycl::platform()};
  sycl::device Dev = Plt.get_devices()[0];
  auto DevImpl = detail::getSyclObjImpl(Dev);

  EXPECT_TRUE(DevImpl->has_extension("cl_khr_fp64"));
  EXPECT_TRUE(DevImpl->has_extension("cl_intel_subgroups"));
  EXPECT_TRUE(DevImpl->has_extension("cl_khr_subgroups"));
}

TEST_F(HasExtensionWordBoundaryTest, SubstringDoesNotMatch) {
  MockExtensions = "cl_intel_subgroups cl_khr_fp64_extended";

  sycl::platform Plt{sycl::platform()};
  sycl::device Dev = Plt.get_devices()[0];
  auto DevImpl = detail::getSyclObjImpl(Dev);

  // these should NOT match because they're substrings
  EXPECT_FALSE(DevImpl->has_extension("cl_intel_subgroup")); // missing 's'

  // would match in old implementation
  EXPECT_FALSE(DevImpl->has_extension("cl_khr_fp64"));
  EXPECT_FALSE(DevImpl->has_extension("subgroups"));       // partial match
  EXPECT_FALSE(DevImpl->has_extension("intel_subgroups")); // partial match
}

TEST_F(HasExtensionWordBoundaryTest, EmptyExtensions) {
  MockExtensions = "";

  sycl::platform Plt{sycl::platform()};
  sycl::device Dev = Plt.get_devices()[0];
  auto DevImpl = detail::getSyclObjImpl(Dev);

  EXPECT_FALSE(DevImpl->has_extension("cl_khr_fp64"));
}

TEST_F(HasExtensionWordBoundaryTest, SingleExtension) {
  MockExtensions = "cl_khr_fp64";

  sycl::platform Plt{sycl::platform()};
  sycl::device Dev = Plt.get_devices()[0];
  auto DevImpl = detail::getSyclObjImpl(Dev);

  EXPECT_TRUE(DevImpl->has_extension("cl_khr_fp64"));
  EXPECT_FALSE(DevImpl->has_extension("cl_khr_fp6")); // a substring
}

TEST_F(HasExtensionWordBoundaryTest, FirstMiddleLastExtensions) {
  MockExtensions = "cl_first_ext cl_middle_ext cl_last_ext";

  sycl::platform Plt{sycl::platform()};
  sycl::device Dev = Plt.get_devices()[0];
  auto DevImpl = detail::getSyclObjImpl(Dev);

  EXPECT_TRUE(DevImpl->has_extension("cl_first_ext"));
  EXPECT_TRUE(DevImpl->has_extension("cl_middle_ext"));
  EXPECT_TRUE(DevImpl->has_extension("cl_last_ext"));
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

  // should match (real extensions)
  EXPECT_TRUE(DevImpl->has_extension("cl_khr_subgroup_non_uniform_vote"));
  EXPECT_TRUE(DevImpl->has_extension("cl_khr_subgroup_ballot"));
  EXPECT_TRUE(DevImpl->has_extension("cl_intel_subgroups"));
  EXPECT_TRUE(DevImpl->has_extension("cl_intel_spirv_subgroups"));
  EXPECT_TRUE(
      DevImpl->has_extension("cl_intel_subgroup_matrix_multiply_accumulate"));

  // next should NOT match (substrings that would match with old impl.)
  EXPECT_FALSE(DevImpl->has_extension("cl_khr_subgroup"));
  EXPECT_FALSE(DevImpl->has_extension("cl_intel_subgroup")); // missing 's'
  EXPECT_FALSE(DevImpl->has_extension("non_uniform_vote"));  // missing prefix
  EXPECT_FALSE(DevImpl->has_extension("subgroup_matrix_multiply"));
}
