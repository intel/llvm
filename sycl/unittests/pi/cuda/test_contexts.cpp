//==---- test_contexts.cpp --- PI unit tests -------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include <condition_variable>
#include <thread>
#include <mutex>

#include <cuda.h>

#include "CudaUtils.hpp"
#include "TestGetPlugin.hpp"
#include <detail/plugin.hpp>
#include <pi_cuda.hpp>
#include <sycl/detail/pi.hpp>
#include <sycl/sycl.hpp>

using namespace sycl;

struct CudaContextsTest : public ::testing::Test {

protected:
  std::optional<detail::plugin> plugin =
      pi::initializeAndGet(backend::ext_oneapi_cuda);

  pi_platform platform_;
  pi_device device_;

  void SetUp() override {
    // skip the tests if the CUDA backend is not available
    if (!plugin.has_value()) {
      GTEST_SKIP();
    }

    pi_uint32 numPlatforms = 0;
    ASSERT_EQ(plugin->getBackend(), backend::ext_oneapi_cuda);

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
  }

  void TearDown() override {}

  CudaContextsTest() = default;

  ~CudaContextsTest() = default;
};

TEST_F(CudaContextsTest, ContextLifetime) {
  // start with no active context
  pi::clearCudaContext();

  // create a context
  pi_context context;
  ASSERT_EQ((plugin->call_nocheck<detail::PiApiKind::piContextCreate>(
                nullptr, 1, &device_, nullptr, nullptr, &context)),
            PI_SUCCESS);
  ASSERT_NE(context, nullptr);

  // create a queue from the context, this should use the ScopedContext
  pi_queue queue;
  ASSERT_EQ((plugin->call_nocheck<detail::PiApiKind::piQueueCreate>(
                context, device_, 0, &queue)),
            PI_SUCCESS);
  ASSERT_NE(queue, nullptr);

  // ensure the queue has the correct context
  ASSERT_EQ(context, queue->get_context());

  // check that the context is now the active CUDA context
  CUcontext cudaCtxt = nullptr;
  cuCtxGetCurrent(&cudaCtxt);
  ASSERT_EQ(cudaCtxt, queue->get_native_context());

  plugin->call<detail::PiApiKind::piQueueRelease>(queue);
  plugin->call<detail::PiApiKind::piContextRelease>(context);
}

TEST_F(CudaContextsTest, ContextLifetimeExisting) {
  // start by setting up a CUDA context on the thread
  CUcontext original;
  cuCtxCreate(&original, CU_CTX_MAP_HOST, device_->get());

  // ensure the CUDA context is active
  CUcontext current = nullptr;
  cuCtxGetCurrent(&current);
  ASSERT_EQ(original, current);

  // create a PI context
  pi_context context;
  ASSERT_EQ((plugin->call_nocheck<detail::PiApiKind::piContextCreate>(
                nullptr, 1, &device_, nullptr, nullptr, &context)),
            PI_SUCCESS);
  ASSERT_NE(context, nullptr);

  // create a queue from the context, this should use the ScopedContext
  pi_queue queue;
  ASSERT_EQ((plugin->call_nocheck<detail::PiApiKind::piQueueCreate>(
                context, device_, 0, &queue)),
            PI_SUCCESS);
  ASSERT_NE(queue, nullptr);

  // ensure the queue has the correct context
  ASSERT_EQ(context, queue->get_context());

  // check that the context is now the active CUDA context
  cuCtxGetCurrent(&current);
  ASSERT_EQ(current, queue->get_native_context());

  plugin->call<detail::PiApiKind::piQueueRelease>(queue);
  plugin->call<detail::PiApiKind::piContextRelease>(context);

  // release original context
  cuCtxDestroy(original);
}
