//==-------- InOrderQueueHostTaskDeps.cpp --- Scheduler unit tests ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SchedulerTest.hpp"
#include "SchedulerTestUtils.hpp"

#include <helpers/TestKernel.hpp>
#include <helpers/UrMock.hpp>

#include <detail/event_impl.hpp>
#include <detail/queue_impl.hpp>

#include <gtest/gtest.h>

#include <sycl/sycl.hpp>

using namespace sycl;

namespace oneapiext = ext::oneapi::experimental;

size_t GEventsWaitCounter = 0;

inline ur_result_t redefinedEventsWaitWithBarrier(void *pParams) {
  GEventsWaitCounter++;
  return UR_RESULT_SUCCESS;
}

TEST_F(SchedulerTest, InOrderQueueHostTaskDeps) {
  GEventsWaitCounter = 0;
  sycl::unittest::UrMock<> Mock;
  sycl::platform Plt = sycl::platform();
  mock::getCallbacks().set_before_callback("urEnqueueEventsWaitWithBarrier",
                                           &redefinedEventsWaitWithBarrier);

  context Ctx{Plt};
  queue InOrderQueue{Ctx, default_selector_v, property::queue::in_order()};

  auto buf = sycl::malloc_device<int>(1, InOrderQueue);
  event Evt = InOrderQueue.submit(
      [&](sycl::handler &CGH) { CGH.memset(buf, 0, sizeof(buf[0])); });
  InOrderQueue.submit([&](sycl::handler &CGH) { CGH.host_task([=] {}); })
      .wait();

  size_t expectedCount = 1u;
  EXPECT_EQ(GEventsWaitCounter, expectedCount);
}

enum class CommandType { KERNEL = 1, MEMSET = 2, HOST_TASK = 3 };
std::vector<std::tuple<CommandType, size_t, size_t>> ExecutedCommands;

inline ur_result_t customEnqueueKernelLaunchWithArgsExp(void *pParams) {
  auto params =
      *static_cast<ur_enqueue_kernel_launch_with_args_exp_params_t *>(pParams);
  ExecutedCommands.push_back({CommandType::KERNEL, *params.pnumEventsInWaitList,
                              **params.ppGlobalWorkSize});
  return UR_RESULT_SUCCESS;
}

inline ur_result_t customEnqueueUSMFill(void *pParams) {
  auto params = *static_cast<ur_enqueue_usm_fill_params_t *>(pParams);
  ExecutedCommands.push_back(
      {CommandType::MEMSET, *params.pnumEventsInWaitList, 0});
  return UR_RESULT_SUCCESS;
}

TEST_F(SchedulerTest, InOrderQueueCrossDeps) {
  ExecutedCommands.clear();
  sycl::unittest::UrMock<> Mock;
  mock::getCallbacks().set_before_callback(
      "urEnqueueKernelLaunchWithArgsExp",
      &customEnqueueKernelLaunchWithArgsExp);
  mock::getCallbacks().set_before_callback("urEnqueueUSMFill",
                                           &customEnqueueUSMFill);

  sycl::platform Plt = sycl::platform();

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
      ExecutedCommands.push_back({CommandType::HOST_TASK, 0, 0});
    });
  });

  auto buf = sycl::malloc_device<int>(1, InOrderQueue);

  event Ev1 = InOrderQueue.submit(
      [&](sycl::handler &CGH) { CGH.memset(buf, 0, sizeof(buf[0])); });

  event Ev2 = InOrderQueue.submit([&](sycl::handler &CGH) {
    CGH.use_kernel_bundle(ExecBundle);
    CGH.single_task<TestKernel>([] {});
  });

  {
    std::unique_lock<std::mutex> lk(CvMutex);
    ready = true;
  }
  Cv.notify_one();

  InOrderQueue.wait();

  ASSERT_EQ(ExecutedCommands.size(), 3u);
  EXPECT_EQ(std::get<0>(ExecutedCommands[0]) /*CommandType*/,
            CommandType::HOST_TASK);
  EXPECT_EQ(std::get<0>(ExecutedCommands[1]) /*CommandType*/,
            CommandType::MEMSET);
  EXPECT_EQ(std::get<1>(ExecutedCommands[1]) /*EventsCount*/, 0u);
  EXPECT_EQ(std::get<0>(ExecutedCommands[2]) /*CommandType*/,
            CommandType::KERNEL);
  EXPECT_EQ(std::get<1>(ExecutedCommands[2]) /*EventsCount*/, 0u);
}

TEST_F(SchedulerTest, InOrderQueueCrossDepsShortcutFuncs) {
  ExecutedCommands.clear();
  sycl::unittest::UrMock<> Mock;
  mock::getCallbacks().set_before_callback(
      "urEnqueueKernelLaunchWithArgsExp",
      &customEnqueueKernelLaunchWithArgsExp);
  mock::getCallbacks().set_before_callback("urEnqueueUSMFill",
                                           &customEnqueueUSMFill);

  sycl::platform Plt = sycl::platform();

  context Ctx{Plt};
  queue InOrderQueue{Ctx, default_selector_v, property::queue::in_order()};

  std::mutex CvMutex;
  std::condition_variable Cv;
  bool ready = false;

  InOrderQueue.submit([&](sycl::handler &CGH) {
    CGH.host_task([&] {
      std::unique_lock<std::mutex> lk(CvMutex);
      Cv.wait(lk, [&ready] { return ready; });
      ExecutedCommands.push_back({CommandType::HOST_TASK, 0, 0});
    });
  });

  auto buf = sycl::malloc_device<int>(1, InOrderQueue);

  event Ev1 = InOrderQueue.memset(buf, 0, sizeof(buf[0]));

  event Ev2 = InOrderQueue.single_task<TestKernel>([] {});

  {
    std::unique_lock<std::mutex> lk(CvMutex);
    ready = true;
  }
  Cv.notify_one();

  InOrderQueue.wait();

  ASSERT_EQ(ExecutedCommands.size(), 3u);
  EXPECT_EQ(std::get<0>(ExecutedCommands[0]) /*CommandType*/,
            CommandType::HOST_TASK);
  EXPECT_EQ(std::get<0>(ExecutedCommands[1]) /*CommandType*/,
            CommandType::MEMSET);
  EXPECT_EQ(std::get<1>(ExecutedCommands[1]) /*EventsCount*/, 0u);
  EXPECT_EQ(std::get<0>(ExecutedCommands[2]) /*CommandType*/,
            CommandType::KERNEL);
  EXPECT_EQ(std::get<1>(ExecutedCommands[2]) /*EventsCount*/, 0u);
}

TEST_F(SchedulerTest, InOrderQueueCrossDepsShortcutFuncsParallelFor) {
  ExecutedCommands.clear();
  sycl::unittest::UrMock<> Mock;
  mock::getCallbacks().set_before_callback(
      "urEnqueueKernelLaunchWithArgsExp",
      &customEnqueueKernelLaunchWithArgsExp);

  sycl::platform Plt = sycl::platform();

  context Ctx{Plt};
  queue InOrderQueue{Ctx, default_selector_v, property::queue::in_order()};

  std::mutex CvMutex;
  std::condition_variable Cv;
  bool ready = false;

  InOrderQueue.submit([&](sycl::handler &CGH) {
    CGH.host_task([&] {
      std::unique_lock<std::mutex> lk(CvMutex);
      Cv.wait(lk, [&ready] { return ready; });
      ExecutedCommands.push_back({CommandType::HOST_TASK, 0, 0});
    });
  });

  event Ev2 = InOrderQueue.parallel_for<TestKernel>(
      nd_range<1>{range{32}, range{32}}, [](nd_item<1>) {});

  {
    std::unique_lock<std::mutex> lk(CvMutex);
    ready = true;
  }
  Cv.notify_one();

  InOrderQueue.wait();

  ASSERT_EQ(ExecutedCommands.size(), 2u);
  EXPECT_EQ(std::get<0>(ExecutedCommands[0]) /*CommandType*/,
            CommandType::HOST_TASK);
  EXPECT_EQ(std::get<1>(ExecutedCommands[0]) /*EventsCount*/, 0u);
  EXPECT_EQ(std::get<0>(ExecutedCommands[1]) /*CommandType*/,
            CommandType::KERNEL);
  EXPECT_EQ(std::get<1>(ExecutedCommands[1]) /*EventsCount*/, 0u);
}

TEST_F(SchedulerTest, InOrderQueueCrossDepsEnqueueFunctions) {
  ExecutedCommands.clear();
  sycl::unittest::UrMock<> Mock;
  mock::getCallbacks().set_before_callback(
      "urEnqueueKernelLaunchWithArgsExp",
      &customEnqueueKernelLaunchWithArgsExp);

  sycl::platform Plt = sycl::platform();

  context Ctx{Plt};
  queue InOrderQueue{Ctx, default_selector_v, property::queue::in_order()};

  std::mutex CvMutex;
  std::condition_variable Cv;
  bool ready = false;

  InOrderQueue.submit([&](sycl::handler &CGH) {
    CGH.host_task([&] {
      std::unique_lock<std::mutex> lk(CvMutex);
      Cv.wait(lk, [&ready] { return ready; });
      ExecutedCommands.push_back({CommandType::HOST_TASK, 0, 0});
    });
  });

  oneapiext::nd_launch<TestKernel>(
      InOrderQueue, nd_range<1>{range<1>{32}, range<1>{32}}, [](nd_item<1>) {});

  oneapiext::nd_launch<TestKernel>(
      InOrderQueue, nd_range<1>{range<1>{64}, range<1>{32}}, [](nd_item<1>) {});

  {
    std::unique_lock<std::mutex> lk(CvMutex);
    ready = true;
  }
  Cv.notify_one();

  InOrderQueue.wait();

  ASSERT_EQ(ExecutedCommands.size(), 3u);
  EXPECT_EQ(std::get<0>(ExecutedCommands[0]) /*CommandType*/,
            CommandType::HOST_TASK);
  EXPECT_EQ(std::get<0>(ExecutedCommands[1]) /*CommandType*/,
            CommandType::KERNEL);
  EXPECT_EQ(std::get<1>(ExecutedCommands[1]) /*EventsCount*/, 0u);
  EXPECT_EQ(std::get<2>(ExecutedCommands[1]) /*GlobalWorkSize*/, 32u);
  EXPECT_EQ(std::get<0>(ExecutedCommands[2]) /*CommandType*/,
            CommandType::KERNEL);
  EXPECT_EQ(std::get<1>(ExecutedCommands[2]) /*EventsCount*/, 0u);
  EXPECT_EQ(std::get<2>(ExecutedCommands[2]) /*GlobalWorkSize*/, 64u);
}
