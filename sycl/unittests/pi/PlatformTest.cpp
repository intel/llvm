//==---- PlatformTest.cpp --- PI unit tests --------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>
#include <CL/sycl/detail/pi.hpp>
#include <gtest/gtest.h>
#include <memory>

using namespace cl::sycl;

namespace pi {
class PlatformTest : public ::testing::Test {
protected:

constexpr static size_t out_string_size =
    8192u; // Using values from OpenCL CTS clGetPlatforms test

  PlatformTest() { detail::pi::initialize(); }

  ~PlatformTest() = default;
};

TEST_F(PlatformTest, piPlatformsGet) {
  pi_uint32 platformCount = 0;

  ASSERT_EQ(PI_CALL_RESULT(RT::piPlatformsGet(0, 0, &platformCount)),
            PI_SUCCESS)
      << "piPlatformsGet failed";

  ASSERT_GT(platformCount, 0u) << "piPlatformsGet found 0 platforms.\n";

  std::vector<pi_platform> platforms(platformCount);

  ASSERT_EQ(PI_CALL_RESULT(
                RT::piPlatformsGet(platformCount, platforms.data(), nullptr)),
            PI_SUCCESS)
      << "piPlatformsGet failed with nullptr for return size.\n";
}

TEST_F(PlatformTest, piPlatformGetInfo) {
  auto get_info_test = [](char *out_string, pi_platform platform,
                          _pi_platform_info info) {

    auto info_name = detail::pi::platformInfoToString(info);

    size_t reported_string_length = 0;
    memset(out_string, 0, out_string_size);

    ASSERT_EQ(PI_CALL_RESULT(RT::piPlatformGetInfo(platform, info,
                                                   out_string_size, out_string,
                                                   &reported_string_length)),
              PI_SUCCESS)
        << "piPlatformGetInfo for " << info_name << " failed.\n";

    auto returned_string_length = strlen(out_string) + 1;

    EXPECT_EQ(returned_string_length, reported_string_length)
        << "Returned string length " << returned_string_length
        << " does not equal reported string length " << reported_string_length
        << ".\n";
  };

  pi_uint32 platformCount = 0;
  PI_CALL(RT::piPlatformsGet(0, 0, &platformCount));
  std::vector<pi_platform> platforms(platformCount);
  PI_CALL(RT::piPlatformsGet(platformCount, platforms.data(), nullptr));

  auto out_string_buffer = std::unique_ptr<char[]>(new char[out_string_size]);
  auto out_string = out_string_buffer.get();

  for (auto i = 0u; i < platformCount; ++i) {
    const auto &platform = platforms[i];
    get_info_test(out_string, platform, PI_PLATFORM_INFO_NAME);
    get_info_test(out_string, platform, PI_PLATFORM_INFO_VENDOR);
    get_info_test(out_string, platform, PI_PLATFORM_INFO_PROFILE);
    get_info_test(out_string, platform, PI_PLATFORM_INFO_VERSION);
    get_info_test(out_string, platform, PI_PLATFORM_INFO_EXTENSIONS);
  }
}
} // Namespace
