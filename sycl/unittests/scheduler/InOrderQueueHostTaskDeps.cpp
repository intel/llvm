//==-------- InOrderQueueHostTaskDeps.cpp --- Scheduler unit tests ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SchedulerTest.hpp"
#include "SchedulerTestUtils.hpp"

#include <helpers/ScopedEnvVar.hpp>
#include <helpers/TestKernel.hpp>
#include <helpers/UrMock.hpp>

#include <detail/event_impl.hpp>
#include <detail/queue_impl.hpp>

#include <gtest/gtest.h>

#include <sycl/sycl.hpp>

using namespace sycl;

size_t GEventsWaitCounter = 0;

auto DummyHostTaskEvent = mock::createDummyHandle<ur_event_handle_t>();
bool HostTaskReady = false;
std::mutex HostTaskMutex;
std::condition_variable HostTaskCV;

size_t EventSignaled = 0;
size_t EventCreated = 0;

inline ur_result_t redefinedEventsWait(void *pParams) {
  auto params = *static_cast<ur_event_wait_params_t *>(pParams);
  if (*params.pnumEvents > 0) {
    GEventsWaitCounter++;

    if (**params.pphEventWaitList == DummyHostTaskEvent) {
      std::unique_lock lk(HostTaskMutex);
      HostTaskCV.wait(lk, [] { return HostTaskReady; });
    }
  }

  return UR_RESULT_SUCCESS;
}

inline ur_result_t redefinedEventHostSignal(void *pParams) {
  EventSignaled = true;
  {
    std::lock_guard lk(HostTaskMutex);
    HostTaskReady = true;
  }
  HostTaskCV.notify_one();
  return UR_RESULT_SUCCESS;
}

inline ur_result_t redefinedEventCreateWithNativeHandle(void *pParams) {
  EventCreated = true;
  auto params =
      *static_cast<ur_event_create_with_native_handle_params_t *>(pParams);
  **params.pphEvent = DummyHostTaskEvent;
  return UR_RESULT_SUCCESS;
}

template <sycl::backend Backend> void InOrderQueueHostTaskDepsTestBody() {
  GEventsWaitCounter = 0;
  sycl::unittest::UrMock<Backend> Mock;
  sycl::platform Plt = sycl::platform();
  mock::getCallbacks().set_before_callback("urEventWait", &redefinedEventsWait);
  mock::getCallbacks().set_replace_callback("urEventHostSignal",
                                            &redefinedEventHostSignal);
  mock::getCallbacks().set_replace_callback(
      "urEventCreateWithNativeHandle", &redefinedEventCreateWithNativeHandle);

  context Ctx{Plt};
  queue InOrderQueue{Ctx, default_selector_v, property::queue::in_order()};

  auto buf = sycl::malloc_device<int>(1, InOrderQueue);
  event Evt = InOrderQueue.submit(
      [&](sycl::handler &CGH) { CGH.memset(buf, 0, sizeof(buf[0])); });
  InOrderQueue.submit([&](sycl::handler &CGH) { CGH.host_task([=] {}); })
      .wait();
  EXPECT_EQ(EventCreated, Backend == sycl::backend::ext_oneapi_level_zero);
  EXPECT_EQ(GEventsWaitCounter,
            1u + size_t(Backend == sycl::backend::ext_oneapi_level_zero));
  EXPECT_EQ(EventSignaled, Backend == sycl::backend::ext_oneapi_level_zero);
}

TEST_F(SchedulerTest, InOrderQueueHostTaskDepsOCL) {
  InOrderQueueHostTaskDepsTestBody<sycl::backend::opencl>();
}

TEST_F(SchedulerTest, InOrderQueueHostTaskDepsL0) {
  std::function<void()> DoNothing = [] {};
  unittest::ScopedEnvVar HostTaskViaNativeEvent{"SYCL_ENABLE_USER_EVENTS_PATH",
                                                "1", DoNothing};
  InOrderQueueHostTaskDepsTestBody<sycl::backend::ext_oneapi_level_zero>();
}

enum class CommandType { KERNEL = 1, MEMSET = 2 };
std::vector<std::pair<CommandType, size_t>> ExecutedCommands;

inline ur_result_t customEnqueueKernelLaunch(void *pParams) {
  auto params = *static_cast<ur_enqueue_kernel_launch_params_t *>(pParams);
  ExecutedCommands.push_back(
      {CommandType::KERNEL, *params.pnumEventsInWaitList});
  return UR_RESULT_SUCCESS;
}

inline ur_result_t customEnqueueUSMFill(void *pParams) {
  auto params = *static_cast<ur_enqueue_usm_fill_params_t *>(pParams);
  ExecutedCommands.push_back(
      {CommandType::MEMSET, *params.pnumEventsInWaitList});
  return UR_RESULT_SUCCESS;
}

TEST_F(SchedulerTest, InOrderQueueCrossDeps) {
  ExecutedCommands.clear();
  sycl::unittest::UrMock<> Mock;
  mock::getCallbacks().set_before_callback("urEnqueueKernelLaunch",
                                           &customEnqueueKernelLaunch);
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
    });
  });

  auto buf = sycl::malloc_device<int>(1, InOrderQueue);

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

TEST_F(SchedulerTest, InOrderQueueCrossDepsShortcutFuncs) {
  ExecutedCommands.clear();
  sycl::unittest::UrMock<> Mock;
  mock::getCallbacks().set_before_callback("urEnqueueKernelLaunch",
                                           &customEnqueueKernelLaunch);
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
    });
  });

  auto buf = sycl::malloc_device<int>(1, InOrderQueue);

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
