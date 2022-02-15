//==---- PlatformTest.cpp --- PI unit tests --------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestGetPlugin.hpp"
#include <CL/sycl.hpp>
#include <CL/sycl/detail/pi.hpp>
#include <detail/plugin.hpp>
#include <gtest/gtest.h>
#include <vector>

namespace {

using namespace cl::sycl;

class PlatformTest : public testing::TestWithParam<detail::plugin> {
protected:
  std::vector<pi_platform> _platforms;
  PlatformTest() : _platforms{} {};

  ~PlatformTest() override = default;

  void SetUp() override {

    detail::plugin plugin = GetParam();

    ASSERT_NO_FATAL_FAILURE(Test::SetUp());

    const static char *platform_count_key = "PiPlatformCount";

    pi_uint32 platform_count = 0u;

    // Initialize the logged number of platforms before the following assertion.
    RecordProperty(platform_count_key, platform_count);

    // TODO: Change the test to check this for all plugins present.
    // Currently, it is only checking for the first plugin attached.
    ASSERT_EQ((plugin.call_nocheck<detail::PiApiKind::piPlatformsGet>(
                  0, nullptr, &platform_count)),
              PI_SUCCESS);

    // Overwrite previous log value with queried number of platforms.
    RecordProperty(platform_count_key, platform_count);

    if (platform_count == 0u) {
      std::cout << "WARNING: piPlatformsGet does not find any PI platforms.\n";

      // Do not call into OpenCL below as a platform count of 0 might fail with
      // OpenCL implementations if the platforms pointer is not `nullptr`.
      return;
    }

    _platforms.resize(platform_count, nullptr);

    ASSERT_EQ((plugin.call_nocheck<detail::PiApiKind::piPlatformsGet>(
                  _platforms.size(), _platforms.data(), nullptr)),
              PI_SUCCESS);
  }
};

INSTANTIATE_TEST_SUITE_P(
    PlatformTestImpl, PlatformTest,
    testing::ValuesIn(pi::initializeAndRemoveInvalid()),
    [](const testing::TestParamInfo<PlatformTest::ParamType> &info) {
      return pi::GetBackendString(info.param.getBackend());
    });

TEST_P(PlatformTest, piPlatformsGet) {
  // The PlatformTest::SetUp method is called to prepare for this test case
  // implicitly tests the calls to `piPlatformsGet`.
}

TEST_P(PlatformTest, piPlatformGetInfo) {

  detail::plugin plugin = GetParam();

  auto get_info_test = [&](pi_platform platform, _pi_platform_info info) {
    size_t reported_string_length = 0;
    EXPECT_EQ((plugin.call_nocheck<detail::PiApiKind::piPlatformGetInfo>(
                  platform, info, 0u, nullptr, &reported_string_length)),
              PI_SUCCESS);

    // Create a larger result string to catch overwrites.
    std::vector<char> param_value(reported_string_length * 2u, '\0');
    EXPECT_EQ(
        (plugin.call_nocheck<detail::PiApiKind::piPlatformGetInfo>(
            platform, info, param_value.size(), param_value.data(), nullptr)),
        PI_SUCCESS)
        << "piPlatformGetInfo for " << detail::pi::platformInfoToString(info)
        << " failed.\n";

    const auto returned_string_length = strlen(param_value.data()) + 1;

    EXPECT_EQ(returned_string_length, reported_string_length)
        << "Returned string length " << returned_string_length
        << " does not equal reported string length " << reported_string_length
        << ".\n";
  };

  for (const auto &platform : _platforms) {
    get_info_test(platform, PI_PLATFORM_INFO_NAME);
    get_info_test(platform, PI_PLATFORM_INFO_VENDOR);
    get_info_test(platform, PI_PLATFORM_INFO_PROFILE);
    get_info_test(platform, PI_PLATFORM_INFO_VERSION);
    get_info_test(platform, PI_PLATFORM_INFO_EXTENSIONS);
  }
}
} // namespace
