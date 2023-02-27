//==-------- InOrderQueueHostTaskDeps.cpp --- Scheduler unit tests ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SchedulerTest.hpp"
#include "SchedulerTestUtils.hpp"

#include <helpers/PiImage.hpp>
#include <helpers/PiMock.hpp>
#include <helpers/ScopedEnvVar.hpp>

#include <detail/event_impl.hpp>
#include <detail/queue_impl.hpp>

#include <gtest/gtest.h>

#include <helpers/TestKernel.hpp>

using namespace sycl;

size_t GEventsWaitCounter = 0;

inline pi_result redefinedEventsWait(pi_uint32 num_events,
                                     const pi_event *event_list) {
  if (num_events > 0) {
    GEventsWaitCounter++;
  }
  return PI_SUCCESS;
}

TEST_F(SchedulerTest, InOrderQueueHostTaskDeps) {
  sycl::unittest::PiMock Mock;
  sycl::platform Plt = Mock.getPlatform();
  Mock.redefineBefore<detail::PiApiKind::piEventsWait>(redefinedEventsWait);

  context Ctx{Plt};
  queue InOrderQueue{Ctx, default_selector_v, property::queue::in_order()};

  kernel_bundle KernelBundle =
      sycl::get_kernel_bundle<sycl::bundle_state::input>(Ctx);
  auto ExecBundle = sycl::build(KernelBundle);

  event Evt = InOrderQueue.submit([&](sycl::handler &CGH) {
    CGH.use_kernel_bundle(ExecBundle);
    CGH.single_task<TestKernel<>>([] {});
  });
  InOrderQueue.submit([&](sycl::handler &CGH) { CGH.host_task([=] {}); })
      .wait();

  EXPECT_TRUE(GEventsWaitCounter == 1);
}

constexpr auto DisablePostEnqueueCleanupName =
    "SYCL_DISABLE_POST_ENQUEUE_CLEANUP";

TEST_F(SchedulerTest, InOrderQueueSubmissionOrder) {
  unittest::ScopedEnvVar DisabledCleanup{
      DisablePostEnqueueCleanupName, "1",
      detail::SYCLConfig<detail::SYCL_DISABLE_POST_ENQUEUE_CLEANUP>::reset};

  sycl::unittest::PiMock Mock;
  sycl::platform Plt = Mock.getPlatform();

  context Ctx{Plt};
  queue InOrderQueue{Ctx, default_selector_v, property::queue::in_order()};

  auto HostTaskEvent =
      InOrderQueue.submit([&](sycl::handler &CGH) { CGH.host_task([] {}); });
  auto Kernel1Event = InOrderQueue.single_task<TestKernel<>>([] {});
  detail::EventImplPtr Kernel1EventImpl =
      sycl::detail::getSyclObjImpl(Kernel1Event);
  detail::Command *Kernel1Command =
      static_cast<detail::Command *>(Kernel1EventImpl->getCommand());
  ASSERT_NE(Kernel1Command, nullptr);

  // Kernel waits for host task in submit call so always enqueued
  EXPECT_TRUE(Kernel1Command->isSuccessfullyEnqueued());

  auto Kernel2Event = InOrderQueue.single_task<TestKernel<>>([] {});
  detail::EventImplPtr Kernel2EventImpl =
      sycl::detail::getSyclObjImpl(Kernel2Event);
  detail::Command *Kernel2Command =
      static_cast<detail::Command *>(Kernel2EventImpl->getCommand());
  // Unfortunately now zero deps kernel command can be deleted on enqueue. Keep
  // it for future usage.
  if (Kernel2Command) {
    EXPECT_TRUE(Kernel2Command->isSuccessfullyEnqueued());
  }

  InOrderQueue.wait();
}
