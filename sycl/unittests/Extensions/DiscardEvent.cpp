//==------------------------- DiscardEvent.cpp -----------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <helpers/PiMock.hpp>
#include <helpers/TestKernel.hpp>

#include <gtest/gtest.h>

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>
#include <sycl/properties/all_properties.hpp>

using namespace sycl;

namespace oneapiext = ext::oneapi::experimental;

namespace {

thread_local size_t counter_piEnqueueKernelLaunch = 0;
inline pi_result redefined_piEnqueueKernelLaunch(pi_queue, pi_kernel, pi_uint32,
                                                 const size_t *, const size_t *,
                                                 const size_t *, pi_uint32,
                                                 const pi_event *,
                                                 pi_event *event) {
  ++counter_piEnqueueKernelLaunch;
  EXPECT_EQ(event, nullptr);
  return PI_SUCCESS;
}

thread_local size_t counter_piEnqueueEventsWaitWithBarrier = 0;
thread_local std::chrono::time_point<std::chrono::steady_clock>
    timestamp_piEnqueueEventsWaitWithBarrier;
inline pi_result after_piEnqueueEventsWaitWithBarrier(pi_queue, pi_uint32,
                                                      const pi_event *,
                                                      pi_event *) {
  ++counter_piEnqueueEventsWaitWithBarrier;
  timestamp_piEnqueueEventsWaitWithBarrier = std::chrono::steady_clock::now();
  return PI_SUCCESS;
}

class DiscardEventTests : public ::testing::Test {
public:
  DiscardEventTests()
      : Mock{}, Q{context(Mock.getPlatform()), default_selector_v,
                  property::queue::in_order{}} {}

protected:
  void SetUp() override {
    counter_piEnqueueKernelLaunch = 0;
    counter_piEnqueueEventsWaitWithBarrier = 0;
  }

  unittest::PiMock Mock;
  queue Q;
};

TEST_F(DiscardEventTests, BarrierBeforeHostTask) {
  // Special test for case where host_task need an event after, so a barrier is
  // enqueued to create a usable event.
  Mock.redefine<detail::PiApiKind::piEnqueueKernelLaunch>(
      redefined_piEnqueueKernelLaunch);
  Mock.redefineAfter<detail::PiApiKind::piEnqueueEventsWaitWithBarrier>(
      after_piEnqueueEventsWaitWithBarrier);

  oneapiext::single_task<TestKernel<>>(Q, []() {});

  std::chrono::time_point<std::chrono::steady_clock> HostTaskTimestamp;
  Q.submit([&](handler &CGH) {
     CGH.host_task(
         [&]() { HostTaskTimestamp = std::chrono::steady_clock::now(); });
   }).wait();

  ASSERT_EQ(counter_piEnqueueKernelLaunch, size_t{1});
  ASSERT_EQ(counter_piEnqueueEventsWaitWithBarrier, size_t{1});
  ASSERT_TRUE(HostTaskTimestamp > timestamp_piEnqueueEventsWaitWithBarrier);
}

} // namespace
