//===-- ReusableEvents.cpp --- Unit tests for reusable events extension --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <gtest/gtest.h>
#include <helpers/MockDeviceImage.hpp>
#include <helpers/MockKernelInfo.hpp>
#include <helpers/UrMock.hpp>
#include <sycl/sycl.hpp>

#include <sycl/ext/oneapi/experimental/reusable_events.hpp>

namespace syclex = sycl::ext::oneapi::experimental;

class TestKernel;
MOCK_INTEGRATION_HEADER(TestKernel)

namespace {

const auto DummyEventHandle = reinterpret_cast<ur_event_handle_t>(1u);
const auto DummyContextHandle = reinterpret_cast<ur_context_handle_t>(2u);
const auto DummyDeviceHandle = reinterpret_cast<ur_device_handle_t>(3u);

int UrEventCreateExp_counter = 0;
int UrEventRelease_counter = 0;
int RedefinedUrEnqueueEventsWaitWithBarrierExt_wait_counter = 0;
int RedefinedUrEnqueueEventsWaitWithBarrierExt_signal_counter = 0;

ur_result_t redefinedUrEventCreateExp(void *pParams) {
  auto params = *static_cast<ur_event_create_exp_params_t *>(pParams);
  **params.pphEvent = DummyEventHandle;
  EXPECT_EQ(*params.phContext, DummyContextHandle);

  UrEventCreateExp_counter++;

  return UR_RESULT_SUCCESS;
}

bool CheckUrEventReleaseHandle = true;

ur_result_t redefinedUrEventRelease(void *pParams) {
  auto params = *static_cast<ur_event_release_params_t *>(pParams);
  if (CheckUrEventReleaseHandle) {
    EXPECT_EQ(*params.phEvent, DummyEventHandle);
  }

  UrEventRelease_counter++;

  return UR_RESULT_SUCCESS;
}

ur_result_t redefinedUrEventGetInfo(void *pParams) {
  auto params = *static_cast<ur_event_get_info_params_t *>(pParams);
  if (*params.ppropName == UR_EVENT_INFO_COMMAND_EXECUTION_STATUS) {
    auto *result = reinterpret_cast<ur_event_status_t *>(*params.ppPropValue);
    *result = UR_EVENT_STATUS_COMPLETE;
  } else if (*params.ppropName == UR_EVENT_INFO_CONTEXT) {
    auto *result = reinterpret_cast<ur_context_handle_t *>(*params.ppPropValue);
    *result = DummyContextHandle;
  }
  return UR_RESULT_SUCCESS;
}

uint32_t ExpectedNumEventsInWaitList = 0;

ur_result_t redefinedUrEnqueueEventsWaitWithBarrierExt_wait(void *pParams) {
  auto params =
      *static_cast<ur_enqueue_events_wait_with_barrier_ext_params_t *>(pParams);
  EXPECT_EQ(*params.pnumEventsInWaitList, ExpectedNumEventsInWaitList);
  for (uint32_t i = 0; i != *params.pnumEventsInWaitList; ++i) {
    EXPECT_EQ((*params.pphEventWaitList)[i], DummyEventHandle);
  }

  RedefinedUrEnqueueEventsWaitWithBarrierExt_wait_counter++;

  return UR_RESULT_SUCCESS;
}

ur_result_t redefinedUrEnqueueEventsWaitWithBarrierExt_signal(void *pParams) {
  auto params =
      *static_cast<ur_enqueue_events_wait_with_barrier_ext_params_t *>(pParams);
  EXPECT_EQ(DummyEventHandle, **params.pphEvent);
  EXPECT_EQ(*params.pnumEventsInWaitList, 0u);

  RedefinedUrEnqueueEventsWaitWithBarrierExt_signal_counter++;

  return UR_RESULT_SUCCESS;
}

ur_result_t
redefinedUrEnqueueEventsWaitWithBarrierExt_signal_create_event(void *pParams) {
  auto params =
      *static_cast<ur_enqueue_events_wait_with_barrier_ext_params_t *>(pParams);
  **params.pphEvent = DummyEventHandle;
  EXPECT_EQ(*params.pnumEventsInWaitList, 0u);

  RedefinedUrEnqueueEventsWaitWithBarrierExt_signal_counter++;

  return UR_RESULT_SUCCESS;
}

ur_result_t redefinedUrEventGetProfilingInfo(void *pParams) {
  auto params = *static_cast<ur_event_get_profiling_info_params_t *>(pParams);
  if (*params.ppPropValue) {
    auto *result = reinterpret_cast<uint64_t *>(*params.ppPropValue);
    *result = 1000000; // 1ms in nanoseconds
  }
  return UR_RESULT_SUCCESS;
}

bool ReusableEventsSupported = true;

ur_result_t after_urDeviceGetInfo(void *pParams) {
  auto params = *static_cast<ur_device_get_info_params_t *>(pParams);
  switch (*params.ppropName) {
  case UR_DEVICE_INFO_REUSABLE_EVENTS_SUPPORT_EXP:
    if (*params.ppPropSizeRet)
      **params.ppPropSizeRet = sizeof(ur_bool_t);
    if (*params.ppPropValue)
      *static_cast<ur_bool_t *>(*params.ppPropValue) =
          ur_bool_t{ReusableEventsSupported};
    return UR_RESULT_SUCCESS;
  default:
    return UR_RESULT_SUCCESS;
  }
}

ur_result_t redefinedUrContextCreate(void *pParams) {
  auto params = *static_cast<ur_context_create_params_t *>(pParams);
  **params.pphContext = DummyContextHandle;
  return UR_RESULT_SUCCESS;
}

ur_result_t redefinedUrContextRelease(void *pParams) {
  auto params = *static_cast<ur_context_release_params_t *>(pParams);
  EXPECT_EQ(*params.phContext, DummyContextHandle);
  return UR_RESULT_SUCCESS;
}

uint32_t ExpectedNumEventsInWaitListKernelLaunch = 0;

ur_result_t redefinedUrEnqueueKernelLaunchWithArgsExp(void *pParams) {
  auto params =
      *static_cast<ur_enqueue_kernel_launch_with_args_exp_params_t *>(pParams);
  EXPECT_EQ(*params.pnumEventsInWaitList,
            ExpectedNumEventsInWaitListKernelLaunch);
  for (uint32_t i = 0; i != *params.pnumEventsInWaitList; ++i) {
    EXPECT_EQ((*params.pphEventWaitList)[i], DummyEventHandle);
  }
  **params.pphEvent = DummyEventHandle;
  return UR_RESULT_SUCCESS;
}

ur_result_t redefinedUrDeviceGet(void *pParams) {
  auto params = *static_cast<ur_device_get_params_t *>(pParams);
  if (*params.ppNumDevices)
    **params.ppNumDevices = 2;
  if (*params.pphDevices && *params.pNumEntries > 0) {
    (*params.pphDevices)[0] = DummyDeviceHandle;
    (*params.pphDevices)[1] = DummyDeviceHandle;
  }
  return UR_RESULT_SUCCESS;
}

ur_result_t redefinedUrDeviceRelease(void *pParams) {
  auto params = *static_cast<ur_device_release_params_t *>(pParams);
  EXPECT_EQ(*params.phDevice, DummyDeviceHandle);
  return UR_RESULT_SUCCESS;
}

} // namespace

class ReusableEventsTest : public ::testing::Test {
protected:
  void SetUp() override {

    UrEventCreateExp_counter = 0;
    UrEventRelease_counter = 0;
    RedefinedUrEnqueueEventsWaitWithBarrierExt_wait_counter = 0;
    RedefinedUrEnqueueEventsWaitWithBarrierExt_signal_counter = 0;
    ExpectedNumEventsInWaitList = 0;
    ExpectedNumEventsInWaitListKernelLaunch = 0;
    CheckUrEventReleaseHandle = true;
    ReusableEventsSupported = true;

    mock::getCallbacks().set_replace_callback("urEventCreateExp",
                                              &redefinedUrEventCreateExp);
    mock::getCallbacks().set_replace_callback("urEventRelease",
                                              &redefinedUrEventRelease);
    mock::getCallbacks().set_replace_callback("urEventGetInfo",
                                              &redefinedUrEventGetInfo);
    mock::getCallbacks().set_replace_callback(
        "urEventGetProfilingInfo", &redefinedUrEventGetProfilingInfo);
    mock::getCallbacks().set_after_callback("urDeviceGetInfo",
                                            &after_urDeviceGetInfo);
    mock::getCallbacks().set_replace_callback("urContextCreate",
                                              &redefinedUrContextCreate);
    mock::getCallbacks().set_replace_callback("urContextRelease",
                                              &redefinedUrContextRelease);
    mock::getCallbacks().set_replace_callback(
        "urEnqueueEventsWaitWithBarrierExt",
        &redefinedUrEnqueueEventsWaitWithBarrierExt_wait);
    mock::getCallbacks().set_replace_callback(
        "urEnqueueKernelLaunchWithArgsExp",
        &redefinedUrEnqueueKernelLaunchWithArgsExp);
  }

  sycl::unittest::UrMock<> Mock;
};

// make_event with no parameters
TEST_F(ReusableEventsTest, MakeEventDefault) {
  {
    auto event = syclex::make_event();

    auto status = event.get_info<sycl::info::event::command_execution_status>();
    EXPECT_EQ(status, sycl::info::event_command_status::complete);
  }

  // make_event should not call UR.
  EXPECT_EQ(UrEventCreateExp_counter, 0);
  EXPECT_EQ(UrEventRelease_counter, 0);
}

// enqueue_wait_event
TEST_F(ReusableEventsTest, EnqueueWaitEvent) {
  sycl::platform Plt = sycl::platform();
  const sycl::device Dev = Plt.get_devices()[0];
  sycl::context Ctx{Dev};
  sycl::queue Queue{Ctx, Dev};

  {
    auto event = syclex::make_event(Ctx);

    EXPECT_NO_THROW({ syclex::enqueue_wait_event(Queue, event); });
  }

  // Waiting for an event which was not enqueued for signaling should complete
  // immediately.
  EXPECT_EQ(RedefinedUrEnqueueEventsWaitWithBarrierExt_wait_counter, 0);
  EXPECT_EQ(UrEventCreateExp_counter, 0);
  EXPECT_EQ(UrEventRelease_counter, 0);

  Queue.wait();
}

// enqueue_wait_events with vector
TEST_F(ReusableEventsTest, EnqueueWaitEvents) {
  sycl::platform Plt = sycl::platform();
  const sycl::device Dev = Plt.get_devices()[0];
  sycl::context Ctx{Dev};
  sycl::queue Queue{Ctx, Dev};

  {
    auto event1 = syclex::make_event(Ctx);
    auto event2 = syclex::make_event(Ctx);
    std::vector<sycl::event> events{event1, event2};

    EXPECT_NO_THROW({ syclex::enqueue_wait_events(Queue, events); });
  }

  EXPECT_EQ(RedefinedUrEnqueueEventsWaitWithBarrierExt_wait_counter, 0);
  EXPECT_EQ(UrEventCreateExp_counter, 0);
  EXPECT_EQ(UrEventRelease_counter, 0);

  Queue.wait();
}

// enqueue_signal_event
TEST_F(ReusableEventsTest, EnqueueSignalEvent) {
  mock::getCallbacks().set_replace_callback(
      "urEnqueueEventsWaitWithBarrierExt",
      &redefinedUrEnqueueEventsWaitWithBarrierExt_signal);
  sycl::platform Plt = sycl::platform();
  const sycl::device Dev = Plt.get_devices()[0];
  sycl::context Ctx{Dev};
  sycl::queue Queue{Ctx, Dev, sycl::property::queue::in_order{}};

  {
    auto event = syclex::make_event(Ctx);

    EXPECT_NO_THROW({ syclex::enqueue_signal_event(Queue, event); });

    EXPECT_EQ(RedefinedUrEnqueueEventsWaitWithBarrierExt_signal_counter, 1);
    EXPECT_EQ(UrEventCreateExp_counter, 1);
    EXPECT_EQ(UrEventRelease_counter, 0);

    EXPECT_NO_THROW({ syclex::enqueue_signal_event(Queue, event); });

    EXPECT_EQ(RedefinedUrEnqueueEventsWaitWithBarrierExt_signal_counter, 2);
    EXPECT_EQ(UrEventCreateExp_counter, 1);
    EXPECT_EQ(UrEventRelease_counter, 0);
  }

  EXPECT_EQ(UrEventCreateExp_counter, 1);
  EXPECT_EQ(UrEventRelease_counter, 1);

  Queue.wait();
}

// enqueue_signal_event
TEST_F(ReusableEventsTest, EnqueueSignalEventReusableEventsNotSupported) {
  mock::getCallbacks().set_replace_callback(
      "urEnqueueEventsWaitWithBarrierExt",
      &redefinedUrEnqueueEventsWaitWithBarrierExt_signal_create_event);
  sycl::platform Plt = sycl::platform();
  const sycl::device Dev = Plt.get_devices()[0];
  sycl::context Ctx{Dev};
  sycl::queue Queue{Ctx, Dev, sycl::property::queue::in_order{}};

  ReusableEventsSupported = false;

  {
    auto event = syclex::make_event(Ctx);

    EXPECT_NO_THROW({ syclex::enqueue_signal_event(Queue, event); });

    EXPECT_EQ(RedefinedUrEnqueueEventsWaitWithBarrierExt_signal_counter, 1);
    EXPECT_EQ(UrEventRelease_counter, 0);

    EXPECT_NO_THROW({ syclex::enqueue_signal_event(Queue, event); });

    EXPECT_EQ(RedefinedUrEnqueueEventsWaitWithBarrierExt_signal_counter, 2);
    EXPECT_EQ(UrEventRelease_counter, 1);
  }

  EXPECT_EQ(UrEventCreateExp_counter, 0);
  EXPECT_EQ(UrEventRelease_counter, 2);

  Queue.wait();
}

// Default event constructor
TEST_F(ReusableEventsTest, DefaultConstructor) {
  {
    // Default constructor is equivalent to make_event() per spec
    sycl::event event;

    auto status = event.get_info<sycl::info::event::command_execution_status>();
    EXPECT_EQ(status, sycl::info::event_command_status::complete);
  }

  EXPECT_EQ(UrEventCreateExp_counter, 0);
  EXPECT_EQ(UrEventRelease_counter, 0);
}

// Queue-returned event can be used with extension functions
TEST_F(ReusableEventsTest, QueueReturnedEvent) {
  sycl::platform Plt = sycl::platform();
  const sycl::device Dev = Plt.get_devices()[0];
  sycl::context Ctx{Dev};
  sycl::queue Queue{Ctx, Dev};

  static sycl::unittest::MockDeviceImage DevImage =
      sycl::unittest::generateDefaultImage({"TestKernel"});
  static sycl::unittest::MockDeviceImageArray<1> DevImageArray = {&DevImage};

  auto event = Queue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel>([]() {}); });

  ExpectedNumEventsInWaitList = 1;

  // Queue-returned events are associated with the queue's context per spec;
  // verify the event is usable with extension wait functions
  EXPECT_NO_THROW({ syclex::enqueue_wait_event(Queue, event); });

  EXPECT_EQ(RedefinedUrEnqueueEventsWaitWithBarrierExt_wait_counter, 1);

  Queue.wait();
}

// TODO reusable events
/*
// Event with profiling can query profiling info
TEST_F(ReusableEventsTest, ProfilingInfoQuery) {
  sycl::device Dev;
  sycl::platform Plt = Dev.get_platform();

  bool supports_profiling =
Plt.get_info<syclex::info::platform::event_profiling>(); if
(!supports_profiling) { GTEST_SKIP() << "Platform does not support event
profiling";
  }

  sycl::context Ctx{Dev};
  auto event = syclex::make_event(Ctx, syclex::enable_profiling{true});
  sycl::queue Queue{sycl::property::queue::in_order{}};

  syclex::enqueue_signal_event(Queue, event);
  event.wait();

  // Should be able to query profiling info
  EXPECT_NO_THROW({
    auto submit_time =
event.get_profiling_info<sycl::info::event_profiling::command_submit>(); auto
start_time =
event.get_profiling_info<sycl::info::event_profiling::command_start>(); auto
end_time = event.get_profiling_info<sycl::info::event_profiling::command_end>();
    (void)submit_time;
    (void)start_time;
    (void)end_time;
  });
}
*/

// Event can be used in depends_on
TEST_F(ReusableEventsTest, EventInDependsOn) {
  mock::getCallbacks().set_replace_callback(
      "urEnqueueEventsWaitWithBarrierExt",
      &redefinedUrEnqueueEventsWaitWithBarrierExt_signal);
  sycl::platform Plt = sycl::platform();
  const sycl::device Dev = Plt.get_devices()[0];
  sycl::context Ctx{Dev};
  sycl::queue Queue{Ctx, Dev};

  static sycl::unittest::MockDeviceImage DevImage =
      sycl::unittest::generateDefaultImage({"TestKernel"});
  static sycl::unittest::MockDeviceImageArray<1> DevImageArray = {&DevImage};

  auto event = syclex::make_event(Ctx);
  syclex::enqueue_signal_event(Queue, event);

  EXPECT_EQ(RedefinedUrEnqueueEventsWaitWithBarrierExt_signal_counter, 1);

  ExpectedNumEventsInWaitListKernelLaunch = 1;

  EXPECT_NO_THROW({
    Queue.submit([&](sycl::handler &cgh) {
      cgh.depends_on(event);
      cgh.single_task<TestKernel>([]() {});
    });
  });

  Queue.wait();
}

// Cross-context events with wait
TEST_F(ReusableEventsTest, CrossContextEventWait) {
  mock::getCallbacks().set_replace_callback("urDeviceGet",
                                            &redefinedUrDeviceGet);
  mock::getCallbacks().set_replace_callback("urDeviceRelease",
                                            &redefinedUrDeviceRelease);

  sycl::platform Plt = sycl::platform();
  auto Devices = Plt.get_devices();

  if (Devices.size() < 2) {
    GTEST_SKIP() << "Need at least 2 devices for this test";
  }

  const sycl::device Dev1 = Devices[0];
  const sycl::device Dev2 = Devices[1];
  sycl::context Ctx1{Dev1};
  sycl::context Ctx2{Dev2};

  sycl::queue Queue1{Ctx1, Dev1};
  sycl::queue Queue2{Ctx2, Dev2};

  auto event = syclex::make_event(Ctx1);

  mock::getCallbacks().set_replace_callback(
      "urEnqueueEventsWaitWithBarrierExt",
      &redefinedUrEnqueueEventsWaitWithBarrierExt_signal);

  syclex::enqueue_signal_event(Queue1, event);

  EXPECT_EQ(RedefinedUrEnqueueEventsWaitWithBarrierExt_signal_counter, 1);

  ExpectedNumEventsInWaitList = 1;
  mock::getCallbacks().set_replace_callback(
      "urEnqueueEventsWaitWithBarrierExt",
      &redefinedUrEnqueueEventsWaitWithBarrierExt_wait);

  // Event from different context should still work with enqueue_wait_event
  EXPECT_NO_THROW({ syclex::enqueue_wait_event(Queue2, event); });

  EXPECT_EQ(RedefinedUrEnqueueEventsWaitWithBarrierExt_wait_counter, 1);

  Queue1.wait();
  Queue2.wait();
}

// Cross-context make_event and signal event - not allowed
TEST_F(ReusableEventsTest, CrossContextMakeEventSignalEvent) {
  mock::getCallbacks().set_replace_callback("urDeviceGet",
                                            &redefinedUrDeviceGet);
  mock::getCallbacks().set_replace_callback("urDeviceRelease",
                                            &redefinedUrDeviceRelease);

  sycl::platform Plt = sycl::platform();
  auto Devices = Plt.get_devices();

  if (Devices.size() < 2) {
    GTEST_SKIP() << "Need at least 2 devices for this test";
  }

  const sycl::device Dev1 = Devices[0];
  const sycl::device Dev2 = Devices[1];
  sycl::context Ctx1{Dev1};
  sycl::context Ctx2{Dev2};

  sycl::queue Queue1{Ctx1, Dev1};

  auto event = syclex::make_event(Ctx2);

  bool exception = false;

  try {
    syclex::enqueue_signal_event(Queue1, event);
  } catch (sycl::exception const &e) {
    exception = true;
    EXPECT_EQ(e.code(), sycl::errc::invalid);
    EXPECT_STREQ(e.what(), "Event context must match the queue context.");
  }

  EXPECT_TRUE(exception);

  Queue1.wait();
}

// Empty event vector wait
TEST_F(ReusableEventsTest, EmptyEventVectorWait) {
  sycl::platform Plt = sycl::platform();
  const sycl::device Dev = Plt.get_devices()[0];
  sycl::context Ctx{Dev};
  sycl::queue Queue{Ctx, Dev};

  std::vector<sycl::event> empty_events;

  EXPECT_NO_THROW({ syclex::enqueue_wait_events(Queue, empty_events); });

  EXPECT_EQ(RedefinedUrEnqueueEventsWaitWithBarrierExt_wait_counter, 0);

  Queue.wait();
}

TEST_F(ReusableEventsTest, SignalEventHostTask) {
  sycl::platform Plt = sycl::platform();
  const sycl::device Dev = Plt.get_devices()[0];
  sycl::context Ctx{Dev};
  sycl::queue Queue{Ctx, Dev, sycl::property::queue::in_order{}};

  mock::getCallbacks().set_replace_callback(
      "urEnqueueEventsWaitWithBarrierExt",
      &redefinedUrEnqueueEventsWaitWithBarrierExt_signal);

  // Host task submission generates additional event
  // for a helper barrier, so skip the handle checks related
  // to event release.
  CheckUrEventReleaseHandle = false;

  {
    auto event = syclex::make_event(Ctx);

    bool exception = false;
    std::mutex CvMutex;
    std::condition_variable Cv;
    bool ready = false;

    Queue.submit([&](sycl::handler &CGH) {
      CGH.host_task([&] {
        std::unique_lock<std::mutex> lk(CvMutex);
        Cv.wait(lk, [&ready] { return ready; });
      });
    });

    try {
      syclex::enqueue_signal_event(Queue, event);
    } catch (sycl::exception const &e) {
      exception = true;
      EXPECT_EQ(e.code(), sycl::errc::invalid);
      EXPECT_STREQ(e.what(), "An event cannot be enqueued for signaling behind "
                             "a command which is not enqueued in the backend.");
    }

    {
      std::unique_lock<std::mutex> lk(CvMutex);
      ready = true;
    }
    Cv.notify_one();

    EXPECT_TRUE(exception);
  }

  Queue.wait();
}

// TODO reusable events - support for queue with profiling enabled
// Queue with enable_profiling property
/*
TEST_F(ReusableEventsTest, QueueWithProfilingProperty) {
  sycl::platform Plt = sycl::platform();
  const sycl::device Dev = Plt.get_devices()[0];
  sycl::context Ctx{Dev};
  sycl::queue Queue{Ctx, Dev,
                    sycl::property_list{sycl::property::queue::enable_profiling{}}};

  auto event = syclex::make_event(Ctx, syclex::enable_profiling{false});

  syclex::enqueue_signal_event(Queue, event);
  event.wait();

  // Event should capture profiling info because queue has profiling enabled
  EXPECT_NO_THROW({
    auto end_time =
event.get_profiling_info<sycl::info::event_profiling::command_end>();
    (void)end_time;
  });
}
*/
