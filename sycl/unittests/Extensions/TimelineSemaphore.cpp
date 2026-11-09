//===-- TimelineSemaphore.cpp --- Unit tests for timeline semaphores ------===//
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

#include <sycl/ext/oneapi/experimental/graph.hpp>
#include <sycl/ext/oneapi/experimental/timeline_semaphore.hpp>

#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace syclex = sycl::ext::oneapi::experimental;

class TimelineSemaphoreKernel;
MOCK_INTEGRATION_HEADER(TimelineSemaphoreKernel)

namespace {

sycl::unittest::MockDeviceImage DevImage =
    sycl::unittest::generateDefaultImage({"TimelineSemaphoreKernel"});
sycl::unittest::MockDeviceImageArray<1> DevImageArray = {&DevImage};

bool TimelineSemaphoreSupported = true;
uint64_t MaxValue = UINT64_MAX;
int MaxValueQueries = 0;

ur_result_t after_urDeviceGetInfo(void *pParams) {
  auto params = *static_cast<ur_device_get_info_params_t *>(pParams);
  switch (*params.ppropName) {
  case UR_DEVICE_INFO_TIMELINE_SEMAPHORE_SUPPORT_EXP:
    if (*params.ppPropSizeRet)
      **params.ppPropSizeRet = sizeof(ur_bool_t);
    if (*params.ppPropValue)
      *static_cast<ur_bool_t *>(*params.ppPropValue) =
          ur_bool_t{TimelineSemaphoreSupported};
    return UR_RESULT_SUCCESS;
  case UR_DEVICE_INFO_TIMELINE_SEMAPHORE_MAX_VALUE_EXP:
    ++MaxValueQueries;
    if (*params.ppPropSizeRet)
      **params.ppPropSizeRet = sizeof(uint64_t);
    if (*params.ppPropValue)
      *static_cast<uint64_t *>(*params.ppPropValue) = MaxValue;
    return UR_RESULT_SUCCESS;
  default:
    return UR_RESULT_SUCCESS;
  }
}

// The order of the UR calls of interest, e.g. "barrier", "signal", "wait".
std::vector<std::string> CallLog;

ur_exp_timeline_semaphore_handle_t CreatedSemaphore = nullptr;
ur_context_handle_t CreateContext = nullptr;
uint64_t CreateInitialValue = 0;
int ReleaseCounter = 0;
ur_exp_timeline_semaphore_handle_t ReleasedSemaphore = nullptr;

ur_result_t after_urTimelineSemaphoreCreateExp(void *pParams) {
  auto params =
      *static_cast<ur_timeline_semaphore_create_exp_params_t *>(pParams);
  CreatedSemaphore = **params.pphSemaphore;
  CreateContext = *params.phContext;
  CreateInitialValue = *params.pinitialValue;
  CallLog.push_back("create");
  return UR_RESULT_SUCCESS;
}

ur_result_t before_urTimelineSemaphoreReleaseExp(void *pParams) {
  auto params =
      *static_cast<ur_timeline_semaphore_release_exp_params_t *>(pParams);
  ReleasedSemaphore = *params.phSemaphore;
  ++ReleaseCounter;
  return UR_RESULT_SUCCESS;
}

struct EnqueuedOp {
  ur_exp_timeline_semaphore_handle_t Semaphore;
  uint64_t Value;
  uint32_t NumWaitEvents;
  bool EventRequested;
  ur_event_handle_t Event;
};
std::vector<EnqueuedOp> Signals;
std::vector<EnqueuedOp> Waits;

template <typename ParamsT> EnqueuedOp recordOp(const ParamsT &params) {
  return {*params.phSemaphore, *params.pvalue, *params.pnumEventsInWaitList,
          *params.pphEvent != nullptr,
          *params.pphEvent ? **params.pphEvent : nullptr};
}

ur_result_t after_urEnqueueTimelineSemaphoreSignalExp(void *pParams) {
  auto params =
      *static_cast<ur_enqueue_timeline_semaphore_signal_exp_params_t *>(
          pParams);
  Signals.push_back(recordOp(params));
  CallLog.push_back("signal");
  return UR_RESULT_SUCCESS;
}

ur_result_t after_urEnqueueTimelineSemaphoreWaitExp(void *pParams) {
  auto params =
      *static_cast<ur_enqueue_timeline_semaphore_wait_exp_params_t *>(pParams);
  Waits.push_back(recordOp(params));
  CallLog.push_back("wait");
  return UR_RESULT_SUCCESS;
}

struct BarrierCall {
  std::vector<ur_event_handle_t> WaitList;
};
std::vector<BarrierCall> Barriers;

ur_result_t after_urEnqueueEventsWaitWithBarrierExt(void *pParams) {
  auto params =
      *static_cast<ur_enqueue_events_wait_with_barrier_ext_params_t *>(pParams);
  BarrierCall Call;
  for (uint32_t I = 0; I != *params.pnumEventsInWaitList; ++I)
    Call.WaitList.push_back((*params.pphEventWaitList)[I]);
  Barriers.push_back(std::move(Call));
  CallLog.push_back("barrier");
  return UR_RESULT_SUCCESS;
}

uint32_t KernelNumWaitEvents = 0;

ur_result_t after_urEnqueueKernelLaunchWithArgsExp(void *pParams) {
  auto params =
      *static_cast<ur_enqueue_kernel_launch_with_args_exp_params_t *>(pParams);
  KernelNumWaitEvents = *params.pnumEventsInWaitList;
  CallLog.push_back("kernel");
  return UR_RESULT_SUCCESS;
}

std::vector<uint64_t> HostWaitValues;
std::vector<uint64_t> HostSignalValues;

ur_result_t after_urTimelineSemaphoreHostWaitExp(void *pParams) {
  auto params =
      *static_cast<ur_timeline_semaphore_host_wait_exp_params_t *>(pParams);
  EXPECT_EQ(*params.phSemaphore, CreatedSemaphore);
  HostWaitValues.push_back(*params.pvalue);
  return UR_RESULT_SUCCESS;
}

ur_result_t after_urTimelineSemaphoreHostSignalExp(void *pParams) {
  auto params =
      *static_cast<ur_timeline_semaphore_host_signal_exp_params_t *>(pParams);
  EXPECT_EQ(*params.phSemaphore, CreatedSemaphore);
  HostSignalValues.push_back(*params.pvalue);
  return UR_RESULT_SUCCESS;
}

ur_result_t replace_urEnqueueTimelineSemaphoreWaitExp_error(void *) {
  return UR_RESULT_ERROR_INVALID_VALUE;
}

} // namespace

class TimelineSemaphoreTest : public ::testing::Test {
protected:
  void SetUp() override {
    TimelineSemaphoreSupported = true;
    MaxValue = UINT64_MAX;
    MaxValueQueries = 0;
    CallLog.clear();
    CreatedSemaphore = nullptr;
    CreateContext = nullptr;
    CreateInitialValue = 0;
    ReleaseCounter = 0;
    ReleasedSemaphore = nullptr;
    Signals.clear();
    Waits.clear();
    Barriers.clear();
    KernelNumWaitEvents = 0;
    HostWaitValues.clear();
    HostSignalValues.clear();

    mock::getCallbacks().set_after_callback("urDeviceGetInfo",
                                            &after_urDeviceGetInfo);
    mock::getCallbacks().set_after_callback(
        "urTimelineSemaphoreCreateExp", &after_urTimelineSemaphoreCreateExp);
    mock::getCallbacks().set_before_callback(
        "urTimelineSemaphoreReleaseExp", &before_urTimelineSemaphoreReleaseExp);
    mock::getCallbacks().set_after_callback(
        "urEnqueueTimelineSemaphoreSignalExp",
        &after_urEnqueueTimelineSemaphoreSignalExp);
    mock::getCallbacks().set_after_callback(
        "urEnqueueTimelineSemaphoreWaitExp",
        &after_urEnqueueTimelineSemaphoreWaitExp);
    mock::getCallbacks().set_after_callback(
        "urEnqueueEventsWaitWithBarrierExt",
        &after_urEnqueueEventsWaitWithBarrierExt);
    mock::getCallbacks().set_after_callback(
        "urEnqueueKernelLaunchWithArgsExp",
        &after_urEnqueueKernelLaunchWithArgsExp);
    mock::getCallbacks().set_after_callback(
        "urTimelineSemaphoreHostWaitExp",
        &after_urTimelineSemaphoreHostWaitExp);
    mock::getCallbacks().set_after_callback(
        "urTimelineSemaphoreHostSignalExp",
        &after_urTimelineSemaphoreHostSignalExp);
  }

  sycl::unittest::UrMock<> Mock;
};

TEST_F(TimelineSemaphoreTest, ConstructorThrowsWithoutAspect) {
  TimelineSemaphoreSupported = false;
  sycl::queue Queue;
  EXPECT_FALSE(
      Queue.get_device().has(sycl::aspect::ext_oneapi_timeline_semaphore));

  try {
    syclex::timeline_semaphore Sem{Queue.get_context()};
    FAIL() << "Expected an exception";
  } catch (const sycl::exception &E) {
    EXPECT_EQ(E.code(), sycl::errc::feature_not_supported);
  }
  EXPECT_EQ(CreatedSemaphore, nullptr);
}

TEST_F(TimelineSemaphoreTest, CreateAndReleaseOnce) {
  sycl::queue Queue;
  EXPECT_TRUE(
      Queue.get_device().has(sycl::aspect::ext_oneapi_timeline_semaphore));
  {
    syclex::timeline_semaphore Sem{Queue.get_context(), 5};
    syclex::timeline_semaphore Copy = Sem;
    EXPECT_EQ(Copy, Sem);
    EXPECT_EQ(Sem.get_context(), Queue.get_context());

    EXPECT_NE(CreatedSemaphore, nullptr);
    EXPECT_NE(CreateContext, nullptr);
    EXPECT_EQ(CreateInitialValue, 5u);
    EXPECT_EQ(ReleaseCounter, 0);
  }
  EXPECT_EQ(ReleaseCounter, 1);
  EXPECT_EQ(ReleasedSemaphore, CreatedSemaphore);
}

TEST_F(TimelineSemaphoreTest, InOrderQueueOperations) {
  sycl::queue Queue{sycl::property::queue::in_order{}};
  syclex::timeline_semaphore Sem{Queue.get_context()};

  syclex::enqueue_signal(Queue, Sem, 3);
  syclex::enqueue_wait(Queue, Sem, 2);
  Queue.wait();

  ASSERT_EQ(Signals.size(), 1u);
  EXPECT_EQ(Signals[0].Semaphore, CreatedSemaphore);
  EXPECT_EQ(Signals[0].Value, 3u);
  EXPECT_FALSE(Signals[0].EventRequested);

  ASSERT_EQ(Waits.size(), 1u);
  EXPECT_EQ(Waits[0].Semaphore, CreatedSemaphore);
  EXPECT_EQ(Waits[0].Value, 2u);
  EXPECT_FALSE(Waits[0].EventRequested);

  EXPECT_TRUE(Barriers.empty());
}

TEST_F(TimelineSemaphoreTest, OutOfOrderQueueActsAsBarrier) {
  sycl::queue Queue;
  syclex::timeline_semaphore Sem{Queue.get_context()};

  syclex::enqueue_signal(Queue, Sem, 1);
  syclex::enqueue_wait(Queue, Sem, 1);
  Queue.single_task<TimelineSemaphoreKernel>([]() {});
  Queue.wait();

  const std::vector<std::string> Expected = {"create",  "barrier", "signal",
                                             "barrier", "barrier", "wait",
                                             "barrier", "kernel"};
  EXPECT_EQ(CallLog, Expected);

  ASSERT_EQ(Signals.size(), 1u);
  ASSERT_EQ(Waits.size(), 1u);
  ASSERT_EQ(Barriers.size(), 4u);
  // The opening barriers wait for all previous commands, the closing barriers
  // wait for the semaphore operation.
  EXPECT_TRUE(Barriers[0].WaitList.empty());
  EXPECT_TRUE(Signals[0].EventRequested);
  EXPECT_EQ(Barriers[1].WaitList,
            std::vector<ur_event_handle_t>{Signals[0].Event});
  EXPECT_TRUE(Barriers[2].WaitList.empty());
  EXPECT_TRUE(Waits[0].EventRequested);
  EXPECT_EQ(Barriers[3].WaitList,
            std::vector<ur_event_handle_t>{Waits[0].Event});
}

TEST_F(TimelineSemaphoreTest, InOrderSignalAfterKernelHasNoExtraWaits) {
  sycl::queue Queue{sycl::property::queue::in_order{}};
  syclex::timeline_semaphore Sem{Queue.get_context()};

  Queue.single_task<TimelineSemaphoreKernel>([]() {});
  syclex::enqueue_signal(Queue, Sem, 1);
  Queue.wait();

  const std::vector<std::string> Expected = {"create", "kernel", "signal"};
  EXPECT_EQ(CallLog, Expected);
  ASSERT_EQ(Signals.size(), 1u);
  EXPECT_EQ(Signals[0].NumWaitEvents, 0u);
}

TEST_F(TimelineSemaphoreTest, HostTaskDependencyUsesScheduler) {
  sycl::queue Queue{sycl::property::queue::in_order{}};
  syclex::timeline_semaphore Sem{Queue.get_context()};

  std::mutex Mtx;
  std::unique_lock<std::mutex> Lock{Mtx};
  Queue.submit([&](sycl::handler &CGH) {
    CGH.host_task([&]() { std::lock_guard<std::mutex> Guard{Mtx}; });
  });

  syclex::enqueue_signal(Queue, Sem, 1);
  syclex::enqueue_wait(Queue, Sem, 1);
  Queue.single_task<TimelineSemaphoreKernel>([]() {});
  EXPECT_TRUE(Signals.empty());
  EXPECT_TRUE(Waits.empty());
  const std::vector<std::string> ExpectedBefore = {"create"};
  EXPECT_EQ(CallLog, ExpectedBefore);

  Lock.unlock();
  Queue.wait();

  ASSERT_EQ(Signals.size(), 1u);
  EXPECT_EQ(Signals[0].Value, 1u);
  ASSERT_EQ(Waits.size(), 1u);
  EXPECT_EQ(Waits[0].Value, 1u);
  const std::vector<std::string> ExpectedAfter = {"create", "signal", "wait",
                                                  "kernel"};
  EXPECT_EQ(CallLog, ExpectedAfter);
}

// The operations and the kernel after them are held by the scheduler behind
// the host task. The kernel depends on the operations because they are the last
// barrier of the out-of-order queue.
TEST_F(TimelineSemaphoreTest, OutOfOrderHostTaskDependency) {
  sycl::queue Queue;
  syclex::timeline_semaphore Sem{Queue.get_context()};

  std::mutex Mtx;
  std::unique_lock<std::mutex> Lock{Mtx};
  Queue.submit([&](sycl::handler &CGH) {
    CGH.host_task([&]() { std::lock_guard<std::mutex> Guard{Mtx}; });
  });

  syclex::enqueue_signal(Queue, Sem, 1);
  syclex::enqueue_wait(Queue, Sem, 1);
  Queue.single_task<TimelineSemaphoreKernel>([]() {});
  const std::vector<std::string> ExpectedBefore = {"create"};
  EXPECT_EQ(CallLog, ExpectedBefore);

  Lock.unlock();
  Queue.wait();

  const std::vector<std::string> ExpectedAfter = {
      "create",  "barrier", "signal",  "barrier",
      "barrier", "wait",    "barrier", "kernel"};
  EXPECT_EQ(CallLog, ExpectedAfter);
  ASSERT_EQ(Signals.size(), 1u);
  ASSERT_EQ(Waits.size(), 1u);
  ASSERT_EQ(Barriers.size(), 4u);
  EXPECT_EQ(Barriers[1].WaitList,
            std::vector<ur_event_handle_t>{Signals[0].Event});
  EXPECT_EQ(Barriers[3].WaitList,
            std::vector<ur_event_handle_t>{Waits[0].Event});
}

// The signal and the wait are each held behind a host task on their own queue.
// Releasing the consumer first submits the wait to the backend while the signal
// that satisfies it is still held by the scheduler.
TEST_F(TimelineSemaphoreTest, CrossQueueOperationsBehindHostTasks) {
  sycl::queue Producer{sycl::property::queue::in_order{}};
  sycl::queue Consumer{Producer.get_context(), Producer.get_device(),
                       sycl::property::queue::in_order{}};
  syclex::timeline_semaphore Sem{Producer.get_context()};

  std::mutex ProducerMtx;
  std::unique_lock<std::mutex> ProducerLock{ProducerMtx};
  Producer.submit([&](sycl::handler &CGH) {
    CGH.host_task([&]() { std::lock_guard<std::mutex> Guard{ProducerMtx}; });
  });
  syclex::enqueue_signal(Producer, Sem, 1);

  std::mutex ConsumerMtx;
  std::unique_lock<std::mutex> ConsumerLock{ConsumerMtx};
  Consumer.submit([&](sycl::handler &CGH) {
    CGH.host_task([&]() { std::lock_guard<std::mutex> Guard{ConsumerMtx}; });
  });
  syclex::enqueue_wait(Consumer, Sem, 1);
  Consumer.single_task<TimelineSemaphoreKernel>([]() {});

  const std::vector<std::string> ExpectedBefore = {"create"};
  EXPECT_EQ(CallLog, ExpectedBefore);

  ConsumerLock.unlock();
  Consumer.wait();

  const std::vector<std::string> ExpectedAfterConsumer = {"create", "wait",
                                                          "kernel"};
  EXPECT_EQ(CallLog, ExpectedAfterConsumer);
  EXPECT_TRUE(Signals.empty());

  ProducerLock.unlock();
  Producer.wait();

  const std::vector<std::string> ExpectedAfterProducer = {"create", "wait",
                                                          "kernel", "signal"};
  EXPECT_EQ(CallLog, ExpectedAfterProducer);
  ASSERT_EQ(Signals.size(), 1u);
  EXPECT_EQ(Signals[0].Value, 1u);
}

// A host signal is performed immediately, even while an enqueued wait for the
// same value is still held by the scheduler.
TEST_F(TimelineSemaphoreTest, HostSignalWhileWaitPendingInScheduler) {
  sycl::queue Queue{sycl::property::queue::in_order{}};
  syclex::timeline_semaphore Sem{Queue.get_context()};

  std::mutex Mtx;
  std::unique_lock<std::mutex> Lock{Mtx};
  Queue.submit([&](sycl::handler &CGH) {
    CGH.host_task([&]() { std::lock_guard<std::mutex> Guard{Mtx}; });
  });
  syclex::enqueue_wait(Queue, Sem, 1);
  Queue.single_task<TimelineSemaphoreKernel>([]() {});

  Sem.signal(1);
  EXPECT_EQ(HostSignalValues, std::vector<uint64_t>{1});
  EXPECT_TRUE(Waits.empty());

  Lock.unlock();
  Queue.wait();

  const std::vector<std::string> Expected = {"create", "wait", "kernel"};
  EXPECT_EQ(CallLog, Expected);
  ASSERT_EQ(Waits.size(), 1u);
  EXPECT_EQ(Waits[0].Value, 1u);
}

TEST_F(TimelineSemaphoreTest, SemaphoreOutlivesPendingOperation) {
  sycl::queue Queue{sycl::property::queue::in_order{}};

  std::mutex Mtx;
  std::unique_lock<std::mutex> Lock{Mtx};
  Queue.submit([&](sycl::handler &CGH) {
    CGH.host_task([&]() { std::lock_guard<std::mutex> Guard{Mtx}; });
  });

  {
    syclex::timeline_semaphore Sem{Queue.get_context()};
    syclex::enqueue_signal(Queue, Sem, 1);
  }
  // The pending command keeps the semaphore alive.
  EXPECT_EQ(ReleaseCounter, 0);

  Lock.unlock();
  Queue.wait();

  ASSERT_EQ(Signals.size(), 1u);
  EXPECT_EQ(Signals[0].Semaphore, CreatedSemaphore);
}

TEST_F(TimelineSemaphoreTest, ContextMismatchThrows) {
  sycl::queue Queue;
  sycl::context OtherContext{Queue.get_device()};
  syclex::timeline_semaphore Sem{OtherContext};

  try {
    syclex::enqueue_signal(Queue, Sem, 1);
    FAIL() << "Expected an exception";
  } catch (const sycl::exception &E) {
    EXPECT_EQ(E.code(), sycl::errc::invalid);
  }
  try {
    syclex::enqueue_wait(Queue, Sem, 1);
    FAIL() << "Expected an exception";
  } catch (const sycl::exception &E) {
    EXPECT_EQ(E.code(), sycl::errc::invalid);
  }
  EXPECT_TRUE(Signals.empty());
  EXPECT_TRUE(Waits.empty());
}

TEST_F(TimelineSemaphoreTest, GraphRecordingThrows) {
  sycl::queue Queue;
  syclex::timeline_semaphore Sem{Queue.get_context()};
  syclex::command_graph Graph{Queue.get_context(), Queue.get_device()};
  Graph.begin_recording(Queue);

  try {
    syclex::enqueue_signal(Queue, Sem, 1);
    FAIL() << "Expected an exception";
  } catch (const sycl::exception &E) {
    EXPECT_EQ(E.code(), sycl::errc::invalid);
  }
  try {
    syclex::enqueue_wait(Queue, Sem, 1);
    FAIL() << "Expected an exception";
  } catch (const sycl::exception &E) {
    EXPECT_EQ(E.code(), sycl::errc::invalid);
  }

  Graph.end_recording(Queue);
  EXPECT_TRUE(Signals.empty());
  EXPECT_TRUE(Waits.empty());
}

TEST_F(TimelineSemaphoreTest, HostWaitAndSignal) {
  sycl::queue Queue;
  syclex::timeline_semaphore Sem{Queue.get_context()};

  Sem.signal(7);
  Sem.wait(4);

  EXPECT_EQ(HostSignalValues, std::vector<uint64_t>{7});
  EXPECT_EQ(HostWaitValues, std::vector<uint64_t>{4});
}

static void expectInvalid(const std::function<void()> &F) {
  try {
    F();
    FAIL() << "Expected an exception";
  } catch (const sycl::exception &E) {
    EXPECT_EQ(E.code(), sycl::errc::invalid);
  }
}

TEST_F(TimelineSemaphoreTest, ValuesAboveMaximumThrow) {
  MaxValue = 10;
  sycl::queue Queue;

  expectInvalid(
      [&] { syclex::timeline_semaphore Sem(Queue.get_context(), 11); });
  EXPECT_EQ(CreatedSemaphore, nullptr);

  syclex::timeline_semaphore Sem{Queue.get_context(), 10};
  EXPECT_NE(CreatedSemaphore, nullptr);

  syclex::enqueue_wait(Queue, Sem, 10);
  expectInvalid([&] { syclex::enqueue_wait(Queue, Sem, 11); });
  expectInvalid([&] { Sem.wait(11); });
  expectInvalid([&] { syclex::enqueue_signal(Queue, Sem, 11); });
  expectInvalid([&] { Sem.signal(11); });
  Queue.wait();

  ASSERT_EQ(Waits.size(), 1u);
  EXPECT_EQ(Waits[0].Value, 10u);
  EXPECT_TRUE(Signals.empty());
  EXPECT_TRUE(HostWaitValues.empty());
  EXPECT_TRUE(HostSignalValues.empty());
}

TEST_F(TimelineSemaphoreTest, MaximumIsQueriedOncePerContext) {
  sycl::queue Queue;
  const size_t NumDevices = Queue.get_context().get_devices().size();

  syclex::timeline_semaphore First{Queue.get_context()};
  syclex::timeline_semaphore Second{Queue.get_context()};
  syclex::enqueue_wait(Queue, Second, 0);
  Queue.wait();

  EXPECT_EQ(MaxValueQueries, static_cast<int>(NumDevices));
}

TEST_F(TimelineSemaphoreTest, BackendErrorIsReported) {
  mock::getCallbacks().set_replace_callback(
      "urEnqueueTimelineSemaphoreWaitExp",
      &replace_urEnqueueTimelineSemaphoreWaitExp_error);
  sycl::queue Queue{sycl::property::queue::in_order{}};
  syclex::timeline_semaphore Sem{Queue.get_context()};

  EXPECT_THROW(syclex::enqueue_wait(Queue, Sem, 1), sycl::exception);
}
