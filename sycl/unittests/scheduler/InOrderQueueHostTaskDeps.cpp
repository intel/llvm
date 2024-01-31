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
  GEventsWaitCounter = 0;
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

  EXPECT_EQ(GEventsWaitCounter, 1u);
}

enum class CommandType { KERNEL = 1, MEMSET = 2 };
std::vector<std::pair<CommandType, size_t>> ExecutedCommands;

inline pi_result customEnqueueKernelLaunch(pi_queue, pi_kernel, pi_uint32,
                                           const size_t *, const size_t *,
                                           const size_t *,
                                           pi_uint32 EventsCount,
                                           const pi_event *, pi_event *) {
  ExecutedCommands.push_back({CommandType::KERNEL, EventsCount});
  return PI_SUCCESS;
}
inline pi_result customextUSMEnqueueMemset(pi_queue, void *, pi_int32, size_t,
                                           pi_uint32 EventsCount,
                                           const pi_event *, pi_event *) {
  ExecutedCommands.push_back({CommandType::MEMSET, EventsCount});
  return PI_SUCCESS;
}

TEST_F(SchedulerTest, InOrderQueueCrossDeps) {
  ExecutedCommands.clear();
  sycl::unittest::PiMock Mock;
  Mock.redefineBefore<detail::PiApiKind::piEnqueueKernelLaunch>(
      customEnqueueKernelLaunch);
  Mock.redefineBefore<detail::PiApiKind::piextUSMEnqueueMemset>(
      customextUSMEnqueueMemset);

  sycl::platform Plt = Mock.getPlatform();

  context Ctx{Plt};
  queue InOrderQueue{Ctx, default_selector_v, property::queue::in_order()};

  kernel_bundle KernelBundle =
      sycl::get_kernel_bundle<sycl::bundle_state::input>(Ctx);
  auto ExecBundle = sycl::build(KernelBundle);

  std::mutex CvMutex;
  std::condition_variable Cv;
  bool ready = false;

  InOrderQueue.submit([&](sycl::handler &CGH) {
    CGH.host_task([&] {
      std::unique_lock<std::mutex> lk(CvMutex);
      Cv.wait(lk, [&ready] { return ready; });
    });
  });
  if (InOrderQueue.get_device().has(aspect::usm_shared_allocations)) {
    auto buf = sycl::malloc_shared<int>(1, InOrderQueue);

    event Ev1 = InOrderQueue.submit(
        [&](sycl::handler &CGH) { CGH.memset(buf, 0, sizeof(buf[0])); });

    event Ev2 = InOrderQueue.submit([&](sycl::handler &CGH) {
      CGH.use_kernel_bundle(ExecBundle);
      CGH.single_task<TestKernel<>>([] {});
    });

  {
    std::unique_lock<std::mutex> lk(CvMutex);
    ready = true;
  }
  Cv.notify_one();

    InOrderQueue.wait();

    ASSERT_EQ(ExecutedCommands.size(), 2u);
    EXPECT_EQ(ExecutedCommands[0].first /*CommandType*/, CommandType::MEMSET);
    EXPECT_EQ(ExecutedCommands[0].second /*EventsCount*/, 0u);
    EXPECT_EQ(ExecutedCommands[1].first /*CommandType*/, CommandType::KERNEL);
    EXPECT_EQ(ExecutedCommands[1].second /*EventsCount*/, 0u);
  }
}

TEST_F(SchedulerTest, InOrderQueueCrossDepsShortcutFuncs) {
  ExecutedCommands.clear();
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

  context Ctx{Plt};
  queue InOrderQueue{Ctx, default_selector_v, property::queue::in_order()};

  std::mutex CvMutex;
  std::condition_variable Cv;
  bool ready = false;

  InOrderQueue.submit([&](sycl::handler &CGH) {
    CGH.host_task([&] {
      std::unique_lock<std::mutex> lk(CvMutex);
      Cv.wait(lk, [&ready] { return ready; });
    });
  });
  if (InOrderQueue.get_device().has(aspect::usm_shared_allocations)) {
    auto buf = sycl::malloc_shared<int>(1, InOrderQueue);

    event Ev1 = InOrderQueue.memset(buf, 0, sizeof(buf[0]));

    event Ev2 = InOrderQueue.single_task<TestKernel<>>([] {});

  {
    std::unique_lock<std::mutex> lk(CvMutex);
    ready = true;
  }
  Cv.notify_one();

    InOrderQueue.wait();

  ASSERT_EQ(ExecutedCommands.size(), 2u);
  EXPECT_EQ(ExecutedCommands[0].first /*CommandType*/, CommandType::MEMSET);
  EXPECT_EQ(ExecutedCommands[0].second /*EventsCount*/, 0u);
  EXPECT_EQ(ExecutedCommands[1].first /*CommandType*/, CommandType::KERNEL);
  EXPECT_EQ(ExecutedCommands[1].second /*EventsCount*/, 0u);
}
