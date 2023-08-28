//==---- test_base_objects.cpp --- PI unit tests ---------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include <hip/hip_runtime.h>

#include "TestGetPlugin.hpp"
#include <detail/plugin.hpp>
#include <pi_hip.hpp>
#include <sycl/detail/pi.hpp>
#include <sycl/sycl.hpp>

#include <thread>

// https://sep5.readthedocs.io/en/latest/ROCm_API_References/
// HIP_API/Context-Management.html#_CPPv419hipCtxGetApiVersion8hipCtx_tPi
const int HIP_DRIVER_API_VERSION = 4;

using namespace sycl;

class HipBaseObjectsTest : public ::testing::Test {
protected:
  std::optional<detail::PluginPtr> &plugin =
      pi::initializeAndGet(backend::ext_oneapi_hip);

  void SetUp() override {
    // skip the tests if the HIP backend is not available
    if (!plugin.has_value()) {
      GTEST_SKIP();
    }
  }

  HipBaseObjectsTest() = default;

  ~HipBaseObjectsTest() = default;
};

TEST_F(HipBaseObjectsTest, piContextCreate) {
  pi_uint32 numPlatforms = 0;
  pi_platform platform = nullptr;
  pi_device device;
  ASSERT_EQ(plugin->hasBackend(backend::ext_oneapi_hip), PI_SUCCESS);

  ASSERT_EQ((plugin->call_nocheck<detail::PiApiKind::piPlatformsGet>(
                0, nullptr, &numPlatforms)),
            PI_SUCCESS)
      << "piPlatformsGet failed.\n";

  ASSERT_EQ((plugin->call_nocheck<detail::PiApiKind::piPlatformsGet>(
                numPlatforms, &platform, nullptr)),
            PI_SUCCESS)
      << "piPlatformsGet failed.\n";

  ASSERT_GE(numPlatforms, 1u);
  ASSERT_NE(platform, nullptr);

  ASSERT_EQ((plugin->call_nocheck<detail::PiApiKind::piDevicesGet>(
                platform, PI_DEVICE_TYPE_GPU, 1, &device, nullptr)),
            PI_SUCCESS)
      << "piDevicesGet failed.\n";

  pi_context ctxt = nullptr;
  ASSERT_EQ((plugin->call_nocheck<detail::PiApiKind::piContextCreate>(
                nullptr, 1, &device, nullptr, nullptr, &ctxt)),
            PI_SUCCESS)
      << "piContextCreate failed.\n";

  EXPECT_NE(ctxt, nullptr);
  EXPECT_EQ(ctxt->get_device(), device);

  // Retrieve the hipCtxt to check information is correct
  hipCtx_t hipContext = ctxt->get();
  int version = 0;
  auto hipErr = hipCtxGetApiVersion(hipContext, &version);
  EXPECT_EQ(hipErr, PI_SUCCESS);
  EXPECT_EQ(version, HIP_DRIVER_API_VERSION);

  ASSERT_EQ((plugin->call_nocheck<detail::PiApiKind::piContextRelease>(ctxt)),
            PI_SUCCESS);
}

TEST_F(HipBaseObjectsTest, piContextCreateChildThread) {
  pi_uint32 numPlatforms = 0;
  pi_platform platform;
  pi_device device;

  ASSERT_EQ((plugin->call_nocheck<detail::PiApiKind::piPlatformsGet>(
                0, nullptr, &numPlatforms)),
            PI_SUCCESS)
      << "piPlatformsGet failed.\n";

  ASSERT_EQ((plugin->call_nocheck<detail::PiApiKind::piPlatformsGet>(
                numPlatforms, &platform, nullptr)),
            PI_SUCCESS)
      << "piPlatformsGet failed.\n";

  ASSERT_EQ((plugin->call_nocheck<detail::PiApiKind::piDevicesGet>(
                platform, PI_DEVICE_TYPE_GPU, 1, &device, nullptr)),
            PI_SUCCESS);

  pi_context ctxt;
  ASSERT_EQ((plugin->call_nocheck<detail::PiApiKind::piContextCreate>(
                nullptr, 1, &device, nullptr, nullptr, &ctxt)),
            PI_SUCCESS);
  EXPECT_NE(ctxt, nullptr);

  // Retrieve the cuCtxt to check information is correct
  auto checkValue = [=]() {
    hipCtx_t hipContext = ctxt->get();
    int version = 0;
    auto hipErr = hipCtxGetApiVersion(hipContext, &version);
    EXPECT_EQ(hipErr, PI_SUCCESS);
    EXPECT_EQ(version, HIP_DRIVER_API_VERSION);

    // The current context is different from the current thread
    hipCtx_t current;
    hipErr = hipCtxGetCurrent(&current);
    EXPECT_EQ(hipErr, PI_SUCCESS);
    EXPECT_NE(hipContext, current);

    // Set the context from PI API as the current one
    hipErr = hipCtxPushCurrent(hipContext);
    EXPECT_EQ(hipErr, PI_SUCCESS);

    hipErr = hipCtxGetCurrent(&current);
    EXPECT_EQ(hipErr, PI_SUCCESS);
    EXPECT_EQ(hipContext, current);
  };
  auto callContextFromOtherThread = std::thread(checkValue);

  callContextFromOtherThread.join();

  ASSERT_EQ((plugin->call_nocheck<detail::PiApiKind::piContextRelease>(ctxt)),
            PI_SUCCESS);
}
