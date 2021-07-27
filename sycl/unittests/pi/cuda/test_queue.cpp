//==---- test_queue.cpp --- PI unit tests ----------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include <cuda.h>

#include "TestGetPlatforms.hpp"
#include "TestGetPlugin.hpp"
#include <CL/sycl.hpp>
#include <CL/sycl/backend/cuda.hpp>
#include <CL/sycl/detail/cuda_definitions.hpp>
#include <CL/sycl/detail/pi.hpp>
#include <detail/plugin.hpp>
#include <pi_cuda.hpp>

using namespace sycl;

struct CudaTestQueue : public ::testing::TestWithParam<platform> {

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

  CudaTestQueue() = default;

  ~CudaTestQueue() = default;
};

TEST_F(CudaTestQueue, PICreateQueueSimple) {
  pi_queue queue;
  ASSERT_EQ((plugin->call_nocheck<detail::PiApiKind::piQueueCreate>(
                context_, device_, 0, &queue)),
            PI_SUCCESS);
  ASSERT_NE(queue, nullptr);
  EXPECT_EQ(queue->get_context(), context_);

  unsigned int flags = 0;
  CUstream stream = queue->get();
  cuStreamGetFlags(stream, &flags);
  ASSERT_EQ(flags, CU_STREAM_NON_BLOCKING);

  ASSERT_EQ((plugin->call_nocheck<detail::PiApiKind::piQueueRelease>(queue)),
            PI_SUCCESS);
}

TEST_F(CudaTestQueue, PIQueueFinishSimple) {
  pi_queue queue;
  ASSERT_EQ((plugin->call_nocheck<detail::PiApiKind::piQueueCreate>(
                context_, device_, 0, &queue)),
            PI_SUCCESS);
  ASSERT_NE(queue, nullptr);

  // todo: post work on queue, ensure the results are valid and the work is
  // complete after piQueueFinish?

  ASSERT_EQ((plugin->call_nocheck<detail::PiApiKind::piQueueFinish>(queue)),
            PI_SUCCESS);

  ASSERT_EQ(cuStreamQuery(queue->get()), CUDA_SUCCESS);

  ASSERT_EQ((plugin->call_nocheck<detail::PiApiKind::piQueueRelease>(queue)),
            PI_SUCCESS);
}

TEST_F(CudaTestQueue, PICreateQueueSimpleDefault) {
  pi_queue queue;
  ASSERT_EQ((plugin->call_nocheck<detail::PiApiKind::piQueueCreate>(
                context_, device_, __SYCL_PI_CUDA_USE_DEFAULT_STREAM, &queue)),
            PI_SUCCESS);
  ASSERT_NE(queue, nullptr);
  EXPECT_EQ(queue->get_context(), context_);

  unsigned int flags = 0;
  CUstream stream = queue->get();
  cuStreamGetFlags(stream, &flags);
  ASSERT_EQ(flags, CU_STREAM_DEFAULT);

  ASSERT_EQ((plugin->call_nocheck<detail::PiApiKind::piQueueRelease>(queue)),
            PI_SUCCESS);
}

TEST_F(CudaTestQueue, PICreateQueueSyncWithDefault) {
  pi_queue queue;
  ASSERT_EQ((plugin->call_nocheck<detail::PiApiKind::piQueueCreate>(
                context_, device_, __SYCL_PI_CUDA_SYNC_WITH_DEFAULT, &queue)),
            PI_SUCCESS);
  ASSERT_NE(queue, nullptr);
  EXPECT_EQ(queue->get_context(), context_);

  unsigned int flags = 0;
  CUstream stream = queue->get();
  cuStreamGetFlags(stream, &flags);
  ASSERT_NE(flags, CU_STREAM_NON_BLOCKING);

  ASSERT_EQ((plugin->call_nocheck<detail::PiApiKind::piQueueRelease>(queue)),
            PI_SUCCESS);
}

TEST_F(CudaTestQueue, PICreateQueueInterop) {
  pi_queue queue;
  ASSERT_EQ((plugin->call_nocheck<detail::PiApiKind::piQueueCreate>(
                context_, device_, 0, &queue)),
            PI_SUCCESS);
  ASSERT_NE(queue, nullptr);
  EXPECT_EQ(queue->get_context(), context_);

  CUstream cuStream = queue->get();

  CUcontext cuCtx;
  CUresult res = cuStreamGetCtx(cuStream, &cuCtx);
  ASSERT_EQ(res, CUDA_SUCCESS);
  EXPECT_EQ(cuCtx, context_->get());

  ASSERT_EQ((plugin->call_nocheck<detail::PiApiKind::piQueueRelease>(queue)),
            PI_SUCCESS);
}

TEST_P(CudaTestQueue, SYCLQueueDefaultStream) {
  std::vector<device> CudaDevices = GetParam().get_devices();
  auto deviceA_ = CudaDevices[0];
  queue Queue(deviceA_, async_handler{},
              {property::queue::cuda::use_default_stream{}});

  CUstream CudaStream = get_native<backend::cuda>(Queue);
  unsigned int flags;
  cuStreamGetFlags(CudaStream, &flags);
  ASSERT_EQ(flags, CU_STREAM_DEFAULT);
}
