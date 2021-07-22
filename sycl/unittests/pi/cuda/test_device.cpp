//==---- test_device.cpp --- PI unit tests ---------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include <cuda.h>

#include "TestGetPlugin.hpp"
#include <CL/sycl.hpp>
#include <CL/sycl/detail/pi.hpp>
#include <detail/plugin.hpp>
#include <pi_cuda.hpp>

using namespace cl::sycl;

struct CudaDeviceTests : public ::testing::Test {

protected:
  detail::plugin *plugin = pi::initializeAndGet(backend::cuda);

  pi_platform platform_;
  pi_device device_;
  pi_context context_;

  void SetUp() override {
    // skip the tests if the CUDA backend is not available
    if (!plugin) {
      GTEST_SKIP();
    }

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
  }

  void TearDown() override {
    if (plugin) {
      plugin->call<detail::PiApiKind::piDeviceRelease>(device_);
      plugin->call<detail::PiApiKind::piContextRelease>(context_);
    }
  }

  CudaDeviceTests() = default;
  ~CudaDeviceTests() = default;
};

TEST_F(CudaDeviceTests, PIDeviceGetInfoSimple) {

  size_t return_size = 0;
  pi_device_type device_type;
  ASSERT_EQ((plugin->call_nocheck<detail::PiApiKind::piDeviceGetInfo>(
                device_, PI_DEVICE_INFO_TYPE, sizeof(pi_device_type),
                &device_type, &return_size)),
            PI_SUCCESS);
  EXPECT_EQ(return_size, sizeof(pi_device_type));
  EXPECT_EQ(
      device_type,
      PI_DEVICE_TYPE_GPU); // backend pre-defined value, device must be a GPU

  pi_device parent_device = nullptr;
  ASSERT_EQ((plugin->call_nocheck<detail::PiApiKind::piDeviceGetInfo>(
                device_, PI_DEVICE_INFO_PARENT_DEVICE, sizeof(pi_device),
                &parent_device, &return_size)),
            PI_SUCCESS);
  EXPECT_EQ(return_size, sizeof(pi_device));
  EXPECT_EQ(parent_device,
            nullptr); // backend pre-set value, device cannot have a parent

  pi_platform platform = nullptr;
  ASSERT_EQ((plugin->call_nocheck<detail::PiApiKind::piDeviceGetInfo>(
                device_, PI_DEVICE_INFO_PLATFORM, sizeof(pi_platform),
                &platform, &return_size)),
            PI_SUCCESS);
  EXPECT_EQ(return_size, sizeof(pi_platform));
  EXPECT_EQ(platform, platform_); // test fixture device was created from the
                                  // test fixture platform

  cl_device_partition_property device_partition_property = -1;
  ASSERT_EQ((plugin->call_nocheck<detail::PiApiKind::piDeviceGetInfo>(
                device_, PI_DEVICE_INFO_PARTITION_TYPE,
                sizeof(cl_device_partition_property),
                &device_partition_property, &return_size)),
            PI_SUCCESS);
  EXPECT_EQ(device_partition_property,
            0); // PI CUDA backend will not support device partitioning, this
                // function should just return 0.
  EXPECT_EQ(return_size, sizeof(cl_device_partition_property));
}
