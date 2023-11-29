//==---- test_mem_obj.cpp --- PI unit tests --------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include <hip/hip_runtime.h>

#include "HipUtils.hpp"
#include "TestGetPlugin.hpp"
#include <detail/plugin.hpp>
#include <pi_hip.hpp>
#include <sycl/detail/pi.hpp>
#include <sycl/sycl.hpp>

using namespace sycl;

struct HipTestMemObj : public ::testing::Test {

protected:
  std::optional<detail::PluginPtr> &plugin =
      pi::initializeAndGet(backend::ext_oneapi_hip);

  pi_platform platform_;
  pi_device device_;
  pi_context context_;

  void SetUp() override {
    // skip the tests if the HIP backend is not available
    if (!plugin.has_value()) {
      GTEST_SKIP();
    }

    pi::clearHipContext();
    pi_uint32 numPlatforms = 0;
    ASSERT_EQ(plugin->hasBackend(backend::ext_oneapi_hip), PI_SUCCESS);

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
    if (plugin.has_value()) {
      plugin->call<detail::PiApiKind::piDeviceRelease>(device_);
      plugin->call<detail::PiApiKind::piContextRelease>(context_);
    }
  }

  HipTestMemObj() = default;

  ~HipTestMemObj() = default;
};

TEST_F(HipTestMemObj, piMemBufferCreateSimple) {
  const size_t memSize = 1024u;
  pi_mem memObj;
  ASSERT_EQ((plugin->call_nocheck<detail::PiApiKind::piMemBufferCreate>(
                context_, PI_MEM_FLAGS_ACCESS_RW, memSize, nullptr, &memObj,
                nullptr)),
            PI_SUCCESS);

  ASSERT_EQ((plugin->call_nocheck<detail::PiApiKind::piMemRelease>(memObj)),
            PI_SUCCESS);
}

TEST_F(HipTestMemObj, piMemBufferAllocHost) {
  const size_t memSize = 1024u;
  pi_mem memObj;
  ASSERT_EQ((plugin->call_nocheck<detail::PiApiKind::piMemBufferCreate>(
                context_, PI_MEM_FLAGS_ACCESS_RW | PI_MEM_FLAGS_HOST_PTR_ALLOC,
                memSize, nullptr, &memObj, nullptr)),
            PI_SUCCESS);

  ASSERT_EQ((plugin->call_nocheck<detail::PiApiKind::piMemRelease>(memObj)),
            PI_SUCCESS);
}

TEST_F(HipTestMemObj, piMemBufferCreateNoActiveContext) {
  const size_t memSize = 1024u;
  // Context has been destroyed

  hipCtx_t current = nullptr;

  // pop HIP contexts until there is not a HIP context bound to the thread
  do {
    hipCtx_t oldContext = nullptr;
    auto hipErr = hipCtxPopCurrent(&oldContext);
    EXPECT_EQ(hipErr, PI_SUCCESS);

    // There should not be any active HIP context
    hipErr = hipCtxGetCurrent(&current);
    ASSERT_EQ(hipErr, PI_SUCCESS);
  } while (current != nullptr);

  // The context object is passed, even if its not active it should be used
  // to allocate the memory object
  pi_mem memObj;
  ASSERT_EQ((plugin->call_nocheck<detail::PiApiKind::piMemBufferCreate>(
                context_, PI_MEM_FLAGS_ACCESS_RW, memSize, nullptr, &memObj,
                nullptr)),
            PI_SUCCESS);
  ASSERT_NE(memObj, nullptr);

  ASSERT_EQ((plugin->call_nocheck<detail::PiApiKind::piMemRelease>(memObj)),
            PI_SUCCESS);
}

TEST_F(HipTestMemObj, piMemBufferPinnedMappedRead) {
  const size_t memSize = sizeof(int);
  const int value = 20;

  pi_queue queue;
  ASSERT_EQ((plugin->call_nocheck<detail::PiApiKind::piQueueCreate>(
                context_, device_, 0, &queue)),
            PI_SUCCESS);
  ASSERT_NE(queue, nullptr);
  ASSERT_EQ(queue->get_context(), context_);

  pi_mem memObj;
  ASSERT_EQ((plugin->call_nocheck<detail::PiApiKind::piMemBufferCreate>(
                context_, PI_MEM_FLAGS_ACCESS_RW | PI_MEM_FLAGS_HOST_PTR_ALLOC,
                memSize, nullptr, &memObj, nullptr)),
            PI_SUCCESS);

  ASSERT_EQ(
      (plugin->call_nocheck<detail::PiApiKind::piEnqueueMemBufferWrite>(
          queue, memObj, true, 0, sizeof(int), &value, 0, nullptr, nullptr)),
      PI_SUCCESS);

  int *host_ptr = nullptr;
  ASSERT_EQ((plugin->call_nocheck<detail::PiApiKind::piEnqueueMemBufferMap>(
                queue, memObj, true, PI_MAP_READ, 0, sizeof(int), 0, nullptr,
                nullptr, (void **)&host_ptr)),
            PI_SUCCESS);

  ASSERT_EQ(*host_ptr, value);

  ASSERT_EQ((plugin->call_nocheck<detail::PiApiKind::piEnqueueMemUnmap>(
                queue, memObj, host_ptr, 0, nullptr, nullptr)),
            PI_SUCCESS);

  ASSERT_EQ((plugin->call_nocheck<detail::PiApiKind::piMemRelease>(memObj)),
            PI_SUCCESS);
  plugin->call<detail::PiApiKind::piQueueRelease>(queue);
}

TEST_F(HipTestMemObj, piMemBufferPinnedMappedWrite) {
  const size_t memSize = sizeof(int);
  const int value = 30;

  pi_queue queue;
  ASSERT_EQ((plugin->call_nocheck<detail::PiApiKind::piQueueCreate>(
                context_, device_, 0, &queue)),
            PI_SUCCESS);
  ASSERT_NE(queue, nullptr);
  ASSERT_EQ(queue->get_context(), context_);

  pi_mem memObj;
  ASSERT_EQ((plugin->call_nocheck<detail::PiApiKind::piMemBufferCreate>(
                context_, PI_MEM_FLAGS_ACCESS_RW | PI_MEM_FLAGS_HOST_PTR_ALLOC,
                memSize, nullptr, &memObj, nullptr)),
            PI_SUCCESS);

  int *host_ptr = nullptr;
  ASSERT_EQ((plugin->call_nocheck<detail::PiApiKind::piEnqueueMemBufferMap>(
                queue, memObj, true, PI_MAP_WRITE, 0, sizeof(int), 0, nullptr,
                nullptr, (void **)&host_ptr)),
            PI_SUCCESS);

  *host_ptr = value;

  ASSERT_EQ((plugin->call_nocheck<detail::PiApiKind::piEnqueueMemUnmap>(
                queue, memObj, host_ptr, 0, nullptr, nullptr)),
            PI_SUCCESS);

  int read_value = 0;
  ASSERT_EQ((plugin->call_nocheck<detail::PiApiKind::piEnqueueMemBufferRead>(
                queue, memObj, true, 0, sizeof(int), &read_value, 0, nullptr,
                nullptr)),
            PI_SUCCESS);

  ASSERT_EQ(read_value, value);

  ASSERT_EQ((plugin->call_nocheck<detail::PiApiKind::piMemRelease>(memObj)),
            PI_SUCCESS);
  plugin->call<detail::PiApiKind::piQueueRelease>(queue);
}
