//==---- test_queue.cpp --- PI unit tests ----------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include <cuda.h>

#include <CL/sycl.hpp>
#include <CL/sycl/detail/cuda_definitions.hpp>
#include <CL/sycl/detail/pi.hpp>
#include <detail/plugin.hpp>
#include <pi_cuda.hpp>

using namespace cl::sycl;

struct DISABLED_CudaTestQueue : public ::testing::Test {

protected:
  std::vector<detail::plugin> Plugins;

  pi_platform platform_;
  pi_device device_;
  pi_context context_;

  void SetUp() override {
    pi_uint32 numPlatforms = 0;
    ASSERT_FALSE(Plugins.empty());

    ASSERT_EQ((Plugins[0].call_nocheck<detail::PiApiKind::piPlatformsGet>(
                  0, nullptr, &numPlatforms)),
              PI_SUCCESS)
        << "piPlatformsGet failed.\n";

    ASSERT_EQ((Plugins[0].call_nocheck<detail::PiApiKind::piPlatformsGet>(
                  numPlatforms, &platform_, nullptr)),
              PI_SUCCESS)
        << "piPlatformsGet failed.\n";

    ASSERT_EQ((Plugins[0].call_nocheck<detail::PiApiKind::piDevicesGet>(
                  platform_, PI_DEVICE_TYPE_GPU, 1, &device_, nullptr)),
              PI_SUCCESS);
    ASSERT_EQ((Plugins[0].call_nocheck<detail::PiApiKind::piContextCreate>(
                  nullptr, 1, &device_, nullptr, nullptr, &context_)),
              PI_SUCCESS);
    EXPECT_NE(context_, nullptr);
  }

  void TearDown() override {
    Plugins[0].call<detail::PiApiKind::piDeviceRelease>(device_);
    Plugins[0].call<detail::PiApiKind::piContextRelease>(context_);
  }

  DISABLED_CudaTestQueue() { detail::pi::initialize(); }

  ~DISABLED_CudaTestQueue() = default;
};

TEST_F(DISABLED_CudaTestQueue, PICreateQueueSimple) {
  pi_queue queue;
  ASSERT_EQ((Plugins[0].call_nocheck<detail::PiApiKind::piQueueCreate>(
                context_, device_, 0, &queue)),
            PI_SUCCESS);
  ASSERT_NE(queue, nullptr);
  EXPECT_EQ(queue->get_context(), context_);

  unsigned int flags = 0;
  CUstream stream = queue->get();
  cuStreamGetFlags(stream, &flags);
  ASSERT_EQ(flags, CU_STREAM_NON_BLOCKING);

  ASSERT_EQ((Plugins[0].call_nocheck<detail::PiApiKind::piQueueRelease>(queue)),
            PI_SUCCESS);
}

TEST_F(DISABLED_CudaTestQueue, PIQueueFinishSimple) {
  pi_queue queue;
  ASSERT_EQ((Plugins[0].call_nocheck<detail::PiApiKind::piQueueCreate>(
                context_, device_, 0, &queue)),
            PI_SUCCESS);
  ASSERT_NE(queue, nullptr);

  // todo: post work on queue, ensure the results are valid and the work is
  // complete after piQueueFinish?

  ASSERT_EQ((Plugins[0].call_nocheck<detail::PiApiKind::piQueueFinish>(queue)),
            PI_SUCCESS);

  ASSERT_EQ(cuStreamQuery(queue->get()), CUDA_SUCCESS);

  ASSERT_EQ((Plugins[0].call_nocheck<detail::PiApiKind::piQueueRelease>(queue)),
            PI_SUCCESS);
}

TEST_F(DISABLED_CudaTestQueue, PICreateQueueSimpleDefault) {
  pi_queue queue;
  ASSERT_EQ((Plugins[0].call_nocheck<detail::PiApiKind::piQueueCreate>(
                context_, device_, PI_CUDA_USE_DEFAULT_STREAM, &queue)),
            PI_SUCCESS);
  ASSERT_NE(queue, nullptr);
  EXPECT_EQ(queue->get_context(), context_);

  unsigned int flags = 0;
  CUstream stream = queue->get();
  cuStreamGetFlags(stream, &flags);
  ASSERT_EQ(flags, CU_STREAM_DEFAULT);

  ASSERT_EQ((Plugins[0].call_nocheck<detail::PiApiKind::piQueueRelease>(queue)),
            PI_SUCCESS);
}

TEST_F(DISABLED_CudaTestQueue, PICreateQueueSyncWithDefault) {
  pi_queue queue;
  ASSERT_EQ((Plugins[0].call_nocheck<detail::PiApiKind::piQueueCreate>(
                context_, device_, PI_CUDA_SYNC_WITH_DEFAULT, &queue)),
            PI_SUCCESS);
  ASSERT_NE(queue, nullptr);
  EXPECT_EQ(queue->get_context(), context_);

  unsigned int flags = 0;
  CUstream stream = queue->get();
  cuStreamGetFlags(stream, &flags);
  ASSERT_NE(flags, CU_STREAM_NON_BLOCKING);

  ASSERT_EQ((Plugins[0].call_nocheck<detail::PiApiKind::piQueueRelease>(queue)),
            PI_SUCCESS);
}

TEST_F(DISABLED_CudaTestQueue, PICreateQueueInterop) {
  pi_queue queue;
  ASSERT_EQ((Plugins[0].call_nocheck<detail::PiApiKind::piQueueCreate>(
                context_, device_, 0, &queue)),
            PI_SUCCESS);
  ASSERT_NE(queue, nullptr);
  EXPECT_EQ(queue->get_context(), context_);

  CUstream cuStream = queue->get();

  CUcontext cuCtx;
  CUresult res = cuStreamGetCtx(cuStream, &cuCtx);
  ASSERT_EQ(res, CUDA_SUCCESS);
  EXPECT_EQ(cuCtx, context_->get());

  ASSERT_EQ((Plugins[0].call_nocheck<detail::PiApiKind::piQueueRelease>(queue)),
            PI_SUCCESS);
}
