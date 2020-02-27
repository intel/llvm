//==---- test_events.cpp --- PI unit tests ---------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include <cuda.h>

#include <CL/sycl.hpp>
#include <CL/sycl/detail/pi.hpp>
#include <detail/plugin.hpp>
#include <pi_cuda.hpp>
#include <thread>

using namespace cl::sycl;

namespace pi {
class DISABLED_CudaEventTests : public ::testing::Test {
protected:
  std::vector<detail::plugin> Plugins;

  pi_platform _platform;
  pi_context _context;
  pi_queue _queue;
  pi_device _device;

  DISABLED_CudaEventTests()
      : _context{nullptr}, _queue{nullptr}, _device{nullptr} {
    Plugins = detail::pi::initialize();
  }

  ~DISABLED_CudaEventTests() override = default;

  void SetUp() override {
    pi_uint32 numPlatforms = 0;
    ASSERT_FALSE(Plugins.empty());

    ASSERT_EQ((Plugins[0].call_nocheck<detail::PiApiKind::piPlatformsGet>(
                  0, nullptr, &numPlatforms)),
              PI_SUCCESS)
        << "piPlatformsGet failed.\n";

    ASSERT_EQ((Plugins[0].call_nocheck<detail::PiApiKind::piPlatformsGet>(
                  numPlatforms, &_platform, nullptr)),
              PI_SUCCESS)
        << "piPlatformsGet failed.\n";

    ASSERT_EQ((Plugins[0].call_nocheck<detail::PiApiKind::piDevicesGet>(
                  _platform, PI_DEVICE_TYPE_GPU, 1, &_device, nullptr)),
              PI_SUCCESS);

    ASSERT_EQ((Plugins[0].call_nocheck<detail::PiApiKind::piContextCreate>(
                  nullptr, 1, &_device, nullptr, nullptr, &_context)),
              PI_SUCCESS);

    ASSERT_EQ((Plugins[0].call_nocheck<detail::PiApiKind::piQueueCreate>(
                  _context, _device, 0, &_queue)),
              PI_SUCCESS);
  }

  void TearDown() override {
    Plugins[0].call<detail::PiApiKind::piQueueRelease>(_queue);
    Plugins[0].call<detail::PiApiKind::piContextRelease>(_context);
  }
};

TEST_F(DISABLED_CudaEventTests, PICreateEvent) {

  pi_event foo;
  ASSERT_EQ((Plugins[0].call_nocheck<detail::PiApiKind::piEventCreate>(_context,
                                                                       &foo)),
            PI_SUCCESS);
  ASSERT_NE(foo, nullptr);
  // There is no CUDA interop event for user events
  EXPECT_EQ(foo->get(), nullptr);
  ASSERT_EQ((Plugins[0].call_nocheck<detail::PiApiKind::piEventRelease>(foo)),
            PI_SUCCESS);
}

TEST_F(DISABLED_CudaEventTests, piGetInfoNativeEvent) {

  auto foo = _pi_event::make_native(PI_COMMAND_KERNEL_LAUNCH, _queue);
  ASSERT_NE(foo, nullptr);

  pi_event_status paramValue = {};
  size_t retSize = 0u;
  ASSERT_EQ((Plugins[0].call_nocheck<detail::PiApiKind::piEventGetInfo>(
                foo, PI_EVENT_INFO_COMMAND_EXECUTION_STATUS, sizeof(paramValue),
                &paramValue, &retSize)),
            PI_SUCCESS);
  EXPECT_EQ(retSize, sizeof(pi_int32));
  EXPECT_EQ(paramValue, PI_EVENT_SUBMITTED);

  auto cuEvent = foo->get();
  ASSERT_NE(cuEvent, nullptr);

  auto errCode = cuEventQuery(cuEvent);
  ASSERT_EQ(errCode, CUDA_SUCCESS);

  ASSERT_EQ((Plugins[0].call_nocheck<detail::PiApiKind::piEventRelease>(foo)),
            PI_SUCCESS);
}
} // namespace pi
