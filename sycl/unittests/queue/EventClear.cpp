//==------------------ EventClear.cpp --- queue unit tests -----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/context_impl.hpp>
#include <gtest/gtest.h>
#include <helpers/UrMock.hpp>
#include <sycl/sycl.hpp>

using namespace sycl;

struct TestCtx {
  TestCtx(context &Ctx) : Ctx{Ctx} {};

  context &Ctx;
  int NEventsWaitedFor = 0;
  int EventReferenceCount = 0;
};

std::unique_ptr<TestCtx> TestContext;

const int ExpectedEventThreshold = 128;

ur_result_t redefinedQueueCreate(void *pParams) {
  auto params = *static_cast<ur_queue_create_params_t *>(pParams);
  assert(*params.ppProperties);
  // Use in-order queues to force storing events for calling wait on them,
  // rather than calling urQueueFinish.
  if ((*params.ppProperties)->flags &
      UR_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE) {
    return UR_RESULT_ERROR_INVALID_QUEUE_PROPERTIES;
  }
  return UR_RESULT_SUCCESS;
}

ur_result_t redefinedEventsWait(void *) {
  ++TestContext->NEventsWaitedFor;
  return UR_RESULT_SUCCESS;
}

ur_result_t redefinedEventGetInfoAfter(void *pParams) {
  auto params = *static_cast<ur_event_get_info_params_t *>(pParams);
  EXPECT_EQ(*params.ppropName, UR_EVENT_INFO_COMMAND_EXECUTION_STATUS)
      << "Unexpected event info requested";
  // Report first half of events as complete.
  // Report second half of events as running.
  // This is important, because removal algorithm assumes that
  // events are likely to be removed oldest first, and stops removing
  // at the first non-completed event.
  static int Counter = 0;
  auto *Result = reinterpret_cast<ur_event_status_t *>(*params.ppPropValue);
  *Result = (Counter < (ExpectedEventThreshold / 2)) ? UR_EVENT_STATUS_COMPLETE
                                                     : UR_EVENT_STATUS_RUNNING;
  Counter++;
  return UR_RESULT_SUCCESS;
}

ur_result_t redefinedEventRetain(void *) {
  ++TestContext->EventReferenceCount;
  return UR_RESULT_SUCCESS;
}

ur_result_t redefinedEventRelease(void *) {
  --TestContext->EventReferenceCount;
  return UR_RESULT_SUCCESS;
}

void prepareUrMock(unittest::UrMock<> &Mock) {
  mock::getCallbacks().set_before_callback("urQueueCreate",
                                           &redefinedQueueCreate);
  mock::getCallbacks().set_before_callback("urEventWait", &redefinedEventsWait);
  mock::getCallbacks().set_after_callback("urEventGetInfo",
                                          &redefinedEventGetInfoAfter);
  mock::getCallbacks().set_before_callback("urEventRetain",
                                           &redefinedEventRetain);
  mock::getCallbacks().set_before_callback("urEventRelease",
                                           &redefinedEventRelease);
}

// Check that the USM events are cleared from the queue upon call to wait(),
// so that they are not waited for multiple times.
TEST(QueueEventClear, ClearOnQueueWait) {
  sycl::unittest::UrMock<> Mock;
  sycl::platform Plt = sycl::platform();
  prepareUrMock(Mock);

  context Ctx{Plt.get_devices()[0]};
  TestContext.reset(new TestCtx(Ctx));
  queue Q{Ctx, default_selector()};

  unsigned char *HostAlloc = (unsigned char *)malloc_host(1, Ctx);
  TestContext->EventReferenceCount = 1;
  Q.memset(HostAlloc, 42, 1);
  Q.wait();
  ASSERT_EQ(TestContext->NEventsWaitedFor, 1);
  ASSERT_EQ(TestContext->EventReferenceCount, 0);
  Q.wait();
  ASSERT_EQ(TestContext->NEventsWaitedFor, 1);
}

// Check that shared events are cleaned up from the queue once their number
// exceeds a threshold.
TEST(QueueEventClear, CleanupOnThreshold) {
  sycl::unittest::UrMock<> Mock;
  sycl::platform Plt = sycl::platform();
  prepareUrMock(Mock);

  context Ctx{Plt.get_devices()[0]};
  TestContext.reset(new TestCtx(Ctx));
  queue Q{Ctx, default_selector()};

  unsigned char *HostAlloc = (unsigned char *)malloc_host(1, Ctx);
  TestContext->EventReferenceCount = ExpectedEventThreshold;
  for (size_t I = 0; I < ExpectedEventThreshold; ++I) {
    Q.memset(HostAlloc, 42, 1).wait();
  }
  // Half of the events (those reported as completed) should be released.
  Q.memset(HostAlloc, 42, 1);
  ASSERT_EQ(TestContext->EventReferenceCount, ExpectedEventThreshold / 2);
}
