//==-------- InOrderQueueHostTaskDeps.cpp --- Scheduler unit tests ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SchedulerTest.hpp"
#include "SchedulerTestUtils.hpp"

#include <helpers/CommonRedefinitions.hpp>
#include <helpers/PiImage.hpp>
#include <helpers/PiMock.hpp>

#include <detail/config.hpp>
#include <detail/event_impl.hpp>
#include <detail/queue_impl.hpp>

#include <gtest/gtest.h>

#include <helpers/ScopedEnvVar.hpp>
#include <helpers/TestKernel.hpp>

#include <iostream>
#include <map>

using namespace sycl;

inline constexpr auto DisablePostEnqueueCleanupName =
    "SYCL_DISABLE_POST_ENQUEUE_CLEANUP";

size_t GEventsWaitCounter = 0;
std::map<pi_event, int> UniqueEvents;

inline pi_result redefinedEventsWait(pi_uint32 num_events,
                                     const pi_event *event_list) {
  if (num_events > 0) {
    GEventsWaitCounter++;
    while (num_events--)
      UniqueEvents[event_list[num_events]]++;
  }

  return PI_SUCCESS;
}

TEST_F(SchedulerTest, InOrderQueueHostTaskDeps) {
  default_selector Selector;
  platform Plt{default_selector()};
  if (Plt.is_host()) {
    std::cout << "Not run due to host-only environment\n";
    return;
  }
  // This test only contains device image for SPIR-V capable devices.
  if (Plt.get_backend() != sycl::backend::opencl &&
      Plt.get_backend() != sycl::backend::ext_oneapi_level_zero) {
    std::cout << "Only OpenCL and Level Zero are supported for this test\n";
    return;
  }

  unittest::PiMock Mock{Plt};
  setupDefaultMockAPIs(Mock);
  Mock.redefine<detail::PiApiKind::piEventsWait>(redefinedEventsWait);

  context Ctx{Plt};
  queue InOrderQueue{Ctx, Selector, property::queue::in_order()};

  kernel_bundle KernelBundle =
      sycl::get_kernel_bundle<sycl::bundle_state::input>(Ctx);
  auto ExecBundle = sycl::build(KernelBundle);

  // kernel, host_task
  event Evt = InOrderQueue.submit([&](sycl::handler &CGH) {
    CGH.use_kernel_bundle(ExecBundle);
    CGH.single_task<TestKernel>([] {});
  });
  InOrderQueue
      .submit([&](sycl::handler &CGH) {
        CGH.use_kernel_bundle(ExecBundle);
        CGH.host_task([=] {});
      })
      .wait();

  EXPECT_TRUE(GEventsWaitCounter == 1);
}

inline pi_result redefinedEnqueueKernelLaunch(pi_queue, pi_kernel, pi_uint32,
                                              const size_t *, const size_t *,
                                              const size_t *,
                                              pi_uint32 pi_event_list_size,
                                              const pi_event *,
                                              pi_event *event) {
  EXPECT_TRUE(pi_event_list_size == 0);
  *event = reinterpret_cast<pi_event>(new int{});
  return PI_SUCCESS;
}

// Test for in-order queue event dependencies.
// Host vs host are synced explicitly by RT using events.
// e.g. Kernel vs kernel (mem read/write and other) are synced implicitly by
// backend. Host vs kernel should be also synced by events.
TEST_F(SchedulerTest, InOrderQueueHostTaskDepsExt) {
  // Prevent post enqueue cleanup from deleting commands.
  unittest::ScopedEnvVar DisabledCleanup{
      DisablePostEnqueueCleanupName, "1",
      detail::SYCLConfig<detail::SYCL_DISABLE_POST_ENQUEUE_CLEANUP>::reset};

  default_selector Selector;
  platform Plt{default_selector()};
  if (Plt.is_host()) {
    std::cout << "Not run due to host-only environment\n";
    return;
  }
  // This test only contains device image for SPIR-V capable devices.
  if (Plt.get_backend() != sycl::backend::opencl &&
      Plt.get_backend() != sycl::backend::ext_oneapi_level_zero) {
    std::cout << "Only OpenCL and Level Zero are supported for this test\n";
    return;
  }

  unittest::PiMock Mock{Plt};
  setupDefaultMockAPIs(Mock);
  Mock.redefine<detail::PiApiKind::piEventsWait>(redefinedEventsWait);
  Mock.redefine<detail::PiApiKind::piEnqueueKernelLaunch>(
      redefinedEnqueueKernelLaunch);

  context Ctx{Plt};
  queue InOrderQueue{Ctx, Selector, property::queue::in_order()};

  kernel_bundle KernelBundle =
      sycl::get_kernel_bundle<sycl::bundle_state::input>(Ctx);
  auto ExecBundle = sycl::build(KernelBundle);

  GEventsWaitCounter = 0;
  // host task, 1st command

  event HostTaskEvent1 = InOrderQueue.submit([&](sycl::handler &CGH) {
    CGH.use_kernel_bundle(ExecBundle);
    CGH.host_task([=] {});
  });

  std::shared_ptr<detail::event_impl> HostTaskEventImpl1 =
      detail::getSyclObjImpl(HostTaskEvent1);
  ASSERT_NE(HostTaskEventImpl1, nullptr);
  auto *HostCmd1 =
      static_cast<detail::Command *>(HostTaskEventImpl1->getCommand());
  ASSERT_NE(HostCmd1, nullptr);
  // MLastEvent -> +1 host event
  EXPECT_EQ(HostCmd1->getPreparedHostDepsEvents().size(), size_t(1));

  // kernel, 2nd command

  // Needs mem requirement to avoid cleanup after addCG
  buffer<int, 1> buf2{range<1>(1)};
  event SingleTaskEvent2 = InOrderQueue.submit([&](sycl::handler &CGH) {
    auto acc = buf2.template get_access<access::mode::read>(CGH);
    CGH.use_kernel_bundle(ExecBundle);
    CGH.single_task<TestKernel>([=] { (void)acc; });
  });

  std::shared_ptr<detail::event_impl> SingleTaskEventImpl2 =
      detail::getSyclObjImpl(SingleTaskEvent2);
  ASSERT_NE(SingleTaskEventImpl2, nullptr);
  auto *SingleCmd2 =
      static_cast<detail::Command *>(SingleTaskEventImpl2->getCommand());
  // 1st event is added in scheduler because of requirement presence
  // 2nd event is a dependency on host task
  ASSERT_NE(SingleCmd2, nullptr);
  EXPECT_EQ(SingleCmd2->getPreparedHostDepsEvents().size(), size_t(2));

  // kernel, 3rd command

  buffer<int, 1> buf3{range<1>(1)};
  event SingleTaskEventSecond3 = InOrderQueue.submit([&](sycl::handler &CGH) {
    auto acc = buf3.template get_access<access::mode::read>(CGH);
    CGH.use_kernel_bundle(ExecBundle);
    CGH.single_task<TestKernel>([=] { (void)acc; });
  });
  // pi events absence between kernels is checked by redefined pi calls
  std::shared_ptr<detail::event_impl> SingleTaskEventSecondImpl3 =
      detail::getSyclObjImpl(SingleTaskEventSecond3);
  ASSERT_NE(SingleTaskEventSecondImpl3, nullptr);
  auto *SingleCmdSecond3 =
      static_cast<detail::Command *>(SingleTaskEventSecondImpl3->getCommand());
  // 1st event is added in scheduler because of requirement presence
  ASSERT_NE(SingleCmdSecond3, nullptr);
  EXPECT_EQ(SingleCmdSecond3->getPreparedHostDepsEvents().size(), size_t(1));

  // host, 4th command
  event HostTaskEvent4 = InOrderQueue.submit([&](sycl::handler &CGH) {
    CGH.use_kernel_bundle(ExecBundle);
    CGH.host_task([=] {});
  });

  std::shared_ptr<detail::event_impl> HostTaskEventImpl4 =
      detail::getSyclObjImpl(HostTaskEvent4);
  ASSERT_NE(HostTaskEventImpl4, nullptr);
  auto *HostCmd4 =
      static_cast<detail::Command *>(HostTaskEventImpl4->getCommand());
  ASSERT_NE(HostCmd4, nullptr);

  // host should have dependency on pi events - to be checked by UniqueEvents
  // counter
  EXPECT_EQ(HostCmd4->getPreparedHostDepsEvents().size(), size_t(0));

  HostTaskEvent4.wait();

  EXPECT_TRUE(GEventsWaitCounter == 1);
  EXPECT_EQ(UniqueEvents.size(), size_t(2));
}
