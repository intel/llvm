//==------------------------- DiscardEvent.cpp -----------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "sycl/platform.hpp"
#include <helpers/TestKernel.hpp>
#include <helpers/UrMock.hpp>

#include <gtest/gtest.h>

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>
#include <sycl/properties/all_properties.hpp>

using namespace sycl;

namespace oneapiext = ext::oneapi::experimental;

namespace {

thread_local size_t counter_urEnqueueKernelLaunch = 0;
inline ur_result_t redefined_urEnqueueKernelLaunch(void *pParams) {
  ++counter_urEnqueueKernelLaunch;
  auto params = *static_cast<ur_enqueue_kernel_launch_params_t *>(pParams);
  EXPECT_EQ(*params.pphEvent, nullptr);
  return UR_RESULT_SUCCESS;
}

thread_local size_t counter_urEnqueueEventsWaitWithBarrier = 0;
thread_local std::chrono::time_point<std::chrono::steady_clock>
    timestamp_urEnqueueEventsWaitWithBarrier;
inline ur_result_t after_urEnqueueEventsWaitWithBarrier(void *) {
  ++counter_urEnqueueEventsWaitWithBarrier;
  timestamp_urEnqueueEventsWaitWithBarrier = std::chrono::steady_clock::now();
  return UR_RESULT_SUCCESS;
}

class DiscardEventTests : public ::testing::Test {
public:
  DiscardEventTests()
      : Mock{}, Q{context(sycl::platform()), default_selector_v,
                  property::queue::in_order{}} {}

protected:
  void SetUp() override {
    counter_urEnqueueKernelLaunch = 0;
    counter_urEnqueueEventsWaitWithBarrier = 0;
  }

  unittest::UrMock<> Mock;
  queue Q;
};

TEST_F(DiscardEventTests, BarrierBeforeHostTask) {
  // Special test for case where host_task need an event after, so a barrier is
  // enqueued to create a usable event.
  mock::getCallbacks().set_replace_callback("urEnqueueKernelLaunch",
                                            &redefined_urEnqueueKernelLaunch);
  mock::getCallbacks().set_after_callback(
      "urEnqueueEventsWaitWithBarrier", &after_urEnqueueEventsWaitWithBarrier);

  oneapiext::single_task<TestKernel<>>(Q, []() {});

  std::chrono::time_point<std::chrono::steady_clock> HostTaskTimestamp;
  Q.submit([&](handler &CGH) {
     CGH.host_task(
         [&]() { HostTaskTimestamp = std::chrono::steady_clock::now(); });
   }).wait();

  ASSERT_EQ(counter_urEnqueueKernelLaunch, size_t{1});
  ASSERT_EQ(counter_urEnqueueEventsWaitWithBarrier, size_t{1});
  ASSERT_TRUE(HostTaskTimestamp > timestamp_urEnqueueEventsWaitWithBarrier);
}

} // namespace
