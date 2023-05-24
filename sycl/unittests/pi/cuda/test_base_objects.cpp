//==---- test_base_objects.cpp --- PI unit tests ---------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include <cuda.h>

#include "TestGetPlugin.hpp"
#include <detail/plugin.hpp>
#include <pi_cuda.hpp>
#include <sycl/detail/cuda_definitions.hpp>
#include <sycl/detail/pi.hpp>
#include <sycl/sycl.hpp>

#include <thread>

const unsigned int LATEST_KNOWN_CUDA_DRIVER_API_VERSION = 3020u;

using namespace sycl;

class CudaBaseObjectsTest : public ::testing::Test {
protected:
  std::optional<detail::PluginPtr> &plugin =
      pi::initializeAndGet(backend::ext_oneapi_cuda);

  void SetUp() override {
    // skip the tests if the CUDA backend is not available
    if (!plugin.has_value()) {
      GTEST_SKIP();
    }
  }

  CudaBaseObjectsTest() = default;

  ~CudaBaseObjectsTest() = default;
};

TEST_F(CudaBaseObjectsTest, piContextCreate) {
  pi_uint32 numPlatforms = 0;
  pi_platform platform = nullptr;
  pi_device device;
  ASSERT_EQ(plugin->hasBackend(backend::ext_oneapi_cuda), PI_SUCCESS);

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

  // Retrieve the cuCtxt to check information is correct
  CUcontext cudaContext = ctxt->get();
  unsigned int version = 0;
  cuCtxGetApiVersion(cudaContext, &version);
  EXPECT_EQ(version, LATEST_KNOWN_CUDA_DRIVER_API_VERSION);

  ASSERT_EQ((plugin->call_nocheck<detail::PiApiKind::piContextRelease>(ctxt)),
            PI_SUCCESS);
}

TEST_F(CudaBaseObjectsTest, piContextCreateChildThread) {
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
    CUcontext cudaContext = ctxt->get();
    unsigned int version = 0;
    auto cuErr = cuCtxGetApiVersion(cudaContext, &version);
    EXPECT_EQ(cuErr, CUDA_SUCCESS);
    EXPECT_EQ(version, LATEST_KNOWN_CUDA_DRIVER_API_VERSION);

    // The current context is different from the current thread
    CUcontext current;
    cuErr = cuCtxGetCurrent(&current);
    EXPECT_EQ(cuErr, CUDA_SUCCESS);
    EXPECT_NE(cudaContext, current);

    // Set the context from PI API as the current one
    cuErr = cuCtxPushCurrent(cudaContext);
    EXPECT_EQ(cuErr, CUDA_SUCCESS);

    cuErr = cuCtxGetCurrent(&current);
    EXPECT_EQ(cuErr, CUDA_SUCCESS);
    EXPECT_EQ(cudaContext, current);
  };
  auto callContextFromOtherThread = std::thread(checkValue);

  callContextFromOtherThread.join();

  ASSERT_EQ((plugin->call_nocheck<detail::PiApiKind::piContextRelease>(ctxt)),
            PI_SUCCESS);
}
