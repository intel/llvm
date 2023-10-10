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
#include <helpers/TestKernel.hpp>

#include <detail/event_impl.hpp>
#include <detail/queue_impl.hpp>

#include <gtest/gtest.h>

#include <sycl/sycl.hpp>

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

  auto buf = sycl::malloc_shared<int>(1, InOrderQueue);
  event Evt = InOrderQueue.submit(
      [&](sycl::handler &CGH) { CGH.memset(buf, 0, sizeof(buf[0])); });
  InOrderQueue.submit([&](sycl::handler &CGH) { CGH.host_task([=] {}); })
      .wait();

  EXPECT_TRUE(GEventsWaitCounter == 1);
}

enum CommandType { KERNEL = 1, MEMSET = 2 };
std::vector<CommandType> ExecutedCommands;

inline pi_result customEnqueueKernelLaunch(pi_queue, pi_kernel, pi_uint32,
                                           const size_t *, const size_t *,
                                           const size_t *, pi_uint32,
                                           const pi_event *, pi_event *) {
  ExecutedCommands.push_back(CommandType::KERNEL);
  return PI_SUCCESS;
}
inline pi_result customextUSMEnqueueMemset(pi_queue, void *, pi_int32, size_t,
                                           pi_uint32, const pi_event *,
                                           pi_event *) {
  ExecutedCommands.push_back(CommandType::MEMSET);
  return PI_SUCCESS;
}

TEST_F(SchedulerTest, InOrderQueueCrossDeps) {
  sycl::unittest::PiMock Mock;
  Mock.redefineBefore<detail::PiApiKind::piEnqueueKernelLaunch>(
      customEnqueueKernelLaunch);
  Mock.redefineBefore<detail::PiApiKind::piextUSMEnqueueMemset>(
      customextUSMEnqueueMemset);

  sycl::platform Plt = Mock.getPlatform();
  if (Plt.is_host()) {
    std::cout << "Not run due to host-only environment\n";
    GTEST_SKIP();
  }
  MockScheduler *MockSchedulerPtr = new MockScheduler();
  sycl::detail::GlobalHandler::instance().attachScheduler(
      dynamic_cast<sycl::detail::Scheduler *>(MockSchedulerPtr));

  context Ctx{Plt};
  queue InOrderQueue{Ctx, default_selector_v, property::queue::in_order()};

  kernel_bundle KernelBundle =
      sycl::get_kernel_bundle<sycl::bundle_state::input>(Ctx);
  auto ExecBundle = sycl::build(KernelBundle);

  std::mutex CvMutex;
  std::condition_variable Cv;

  InOrderQueue.submit([&](sycl::handler &CGH) {
    CGH.host_task([&] {
      std::unique_lock lk(CvMutex);
      Cv.wait(lk);
    });
  });

  auto buf = sycl::malloc_shared<int>(1, InOrderQueue);

  event Ev1 = InOrderQueue.submit(
      [&](sycl::handler &CGH) { CGH.memset(buf, 0, sizeof(buf[0])); });

  event Ev2 = InOrderQueue.submit([&](sycl::handler &CGH) {
    CGH.use_kernel_bundle(ExecBundle);
    CGH.single_task<TestKernel<>>([] {});
  });

  Cv.notify_one();

  InOrderQueue.wait();

  sycl::detail::GlobalHandler::instance().attachScheduler(NULL);

  ASSERT_EQ(ExecutedCommands.size(), 2u);
  EXPECT_EQ(ExecutedCommands[0], MEMSET);
  EXPECT_EQ(ExecutedCommands[1], KERNEL);
}
