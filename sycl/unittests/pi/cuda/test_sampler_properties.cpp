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

class SamplerPropertiesTest
    : public ::testing::TestWithParam<std::tuple<
          pi_bool, pi_sampler_filter_mode, pi_sampler_addressing_mode>> {
protected:
  detail::plugin *plugin = pi::initializeAndGet(backend::cuda);

  pi_platform platform_;
  pi_device device_;
  pi_context context_;
  pi_sampler sampler_;

  pi_bool normalizedCoords_;
  pi_sampler_filter_mode filterMode_;
  pi_sampler_addressing_mode addressMode_;

  SamplerPropertiesTest() = default;

  ~SamplerPropertiesTest() override = default;

  void SetUp() override {
    // skip the tests if the CUDA backend is not available
    if (plugin == nullptr) {
      GTEST_SKIP();
    }

    std::tie(normalizedCoords_, filterMode_, addressMode_) = GetParam();

    pi_uint32 numPlatforms = 0;
    ASSERT_EQ(plugin->getBackend(), backend::cuda);

    ASSERT_EQ((plugin->call_nocheck<detail::PiApiKind::piPlatformsGet>(
                  0, nullptr, &numPlatforms)),
              PI_SUCCESS)
        << "piPlatformsGet failed.\n";

    ASSERT_EQ((plugin->call_nocheck<detail::PiApiKind::piPlatformsGet>(
                  numPlatforms, &platform_, nullptr)),
              PI_SUCCESS)
        << "piPlatformsGet failed.\n";

    ASSERT_EQ((plugin->call_nocheck<detail::PiApiKind::piDevicesGet>(
                  platform_, PI_DEVICE_TYPE_GPU, 1, &device_, nullptr)),
              PI_SUCCESS);
    ASSERT_EQ((plugin->call_nocheck<detail::PiApiKind::piContextCreate>(
                  nullptr, 1, &device_, nullptr, nullptr, &context_)),
              PI_SUCCESS);
    EXPECT_NE(context_, nullptr);

    pi_sampler_properties sampler_properties[] = {
        PI_SAMPLER_PROPERTIES_NORMALIZED_COORDS,
        normalizedCoords_,
        PI_SAMPLER_PROPERTIES_ADDRESSING_MODE,
        addressMode_,
        PI_SAMPLER_PROPERTIES_FILTER_MODE,
        filterMode_,
        0};

    ASSERT_EQ((plugin->call_nocheck<detail::PiApiKind::piSamplerCreate>(
                  context_, sampler_properties, &sampler_)),
              PI_SUCCESS);
  }

  void TearDown() override {
    if (plugin) {
      plugin->call<detail::PiApiKind::piSamplerRelease>(sampler_);
      plugin->call<detail::PiApiKind::piDeviceRelease>(device_);
      plugin->call<detail::PiApiKind::piContextRelease>(context_);
    }
  }
};

TEST_P(SamplerPropertiesTest, piCheckNormalizedCoords) {
  pi_bool actualNormalizedCoords = !normalizedCoords_;

  plugin->call<detail::PiApiKind::piSamplerGetInfo>(
      sampler_, PI_SAMPLER_INFO_NORMALIZED_COORDS, sizeof(pi_bool),
      &actualNormalizedCoords, nullptr);

  ASSERT_EQ(actualNormalizedCoords, normalizedCoords_);
}

TEST_P(SamplerPropertiesTest, piCheckFilterMode) {
  pi_sampler_filter_mode actualFilterMode;

  plugin->call<detail::PiApiKind::piSamplerGetInfo>(
      sampler_, PI_SAMPLER_INFO_FILTER_MODE, sizeof(pi_sampler_filter_mode),
      &actualFilterMode, nullptr);

  ASSERT_EQ(actualFilterMode, filterMode_);
}

TEST_P(SamplerPropertiesTest, piCheckAddressingMode) {
  pi_sampler_addressing_mode actualAddressMode;

  plugin->call<detail::PiApiKind::piSamplerGetInfo>(
      sampler_, PI_SAMPLER_INFO_ADDRESSING_MODE,
      sizeof(pi_sampler_addressing_mode), &actualAddressMode, nullptr);

  ASSERT_EQ(actualAddressMode, addressMode_);
}

INSTANTIATE_TEST_CASE_P(
    SamplerPropertiesTesttImpl, SamplerPropertiesTest,
    ::testing::Combine(
        ::testing::Values(PI_TRUE, PI_FALSE),
        ::testing::Values(PI_SAMPLER_FILTER_MODE_LINEAR,
                          PI_SAMPLER_FILTER_MODE_NEAREST),
        ::testing::Values(PI_SAMPLER_ADDRESSING_MODE_CLAMP,
                          PI_SAMPLER_ADDRESSING_MODE_CLAMP_TO_EDGE,
                          PI_SAMPLER_ADDRESSING_MODE_NONE,
                          PI_SAMPLER_ADDRESSING_MODE_MIRRORED_REPEAT,
                          PI_SAMPLER_ADDRESSING_MODE_REPEAT)));
} // namespace
