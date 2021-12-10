//==------------------ EventClear.cpp --- queue unit tests -----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>
#include <detail/context_impl.hpp>
#include <gtest/gtest.h>
#include <helpers/PiMock.hpp>

using namespace cl::sycl;

struct TestCtx {
  TestCtx(context &Ctx) : Ctx{Ctx} {};

  context &Ctx;
  int NEventsWaitedFor = 0;
  int EventReferenceCount = 0;
};

std::unique_ptr<TestCtx> TestContext;

const int ExpectedEventThreshold = 128;

pi_result redefinedQueueCreate(pi_context context, pi_device device,
                               pi_queue_properties properties,
                               pi_queue *queue) {
  // Use in-order queues to force storing events for calling wait on them,
  // rather than calling piQueueFinish.
  if (properties & PI_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) {
    return PI_INVALID_QUEUE_PROPERTIES;
  }
  return PI_SUCCESS;
}

pi_result redefinedQueueRelease(pi_queue Queue) { return PI_SUCCESS; }

pi_result redefinedUSMEnqueueMemset(pi_queue queue, void *ptr, pi_int32 value,
                                    size_t count,
                                    pi_uint32 num_events_in_waitlist,
                                    const pi_event *events_waitlist,
                                    pi_event *event) {
  // Provide a dummy non-nullptr value
  *event = reinterpret_cast<pi_event>(1);
  return PI_SUCCESS;
}

pi_result redefinedEventsWait(pi_uint32 num_events,
                              const pi_event *event_list) {
  ++TestContext->NEventsWaitedFor;
  return PI_SUCCESS;
}

pi_result redefinedEventGetInfo(pi_event event, pi_event_info param_name,
                                size_t param_value_size, void *param_value,
                                size_t *param_value_size_ret) {
  EXPECT_EQ(param_name, PI_EVENT_INFO_COMMAND_EXECUTION_STATUS)
      << "Unexpected event info requested";
  // Report first half of events as complete.
  // Report second half of events as running.
  // This is important, because removal algorithm assumes that
  // events are likely to be removed oldest first, and stops removing
  // at the first non-completed event.
  static int Counter = 0;
  auto *Result = reinterpret_cast<pi_event_status *>(param_value);
  *Result = (Counter < (ExpectedEventThreshold / 2)) ? PI_EVENT_COMPLETE
                                                     : PI_EVENT_RUNNING;
  Counter++;
  return PI_SUCCESS;
}

pi_result redefinedEventRetain(pi_event event) {
  ++TestContext->EventReferenceCount;
  return PI_SUCCESS;
}

pi_result redefinedEventRelease(pi_event event) {
  --TestContext->EventReferenceCount;
  return PI_SUCCESS;
}

bool preparePiMock(platform &Plt) {
  if (Plt.is_host()) {
    std::cout << "Not run on host - no PI events created in that case"
              << std::endl;
    return false;
  }

  unittest::PiMock Mock{Plt};
  Mock.redefine<detail::PiApiKind::piQueueCreate>(redefinedQueueCreate);
  Mock.redefine<detail::PiApiKind::piQueueRelease>(redefinedQueueRelease);
  Mock.redefine<detail::PiApiKind::piextUSMEnqueueMemset>(
      redefinedUSMEnqueueMemset);
  Mock.redefine<detail::PiApiKind::piEventsWait>(redefinedEventsWait);
  Mock.redefine<detail::PiApiKind::piEventGetInfo>(redefinedEventGetInfo);
  Mock.redefine<detail::PiApiKind::piEventRetain>(redefinedEventRetain);
  Mock.redefine<detail::PiApiKind::piEventRelease>(redefinedEventRelease);
  return true;
}

// Check that the USM events are cleared from the queue upon call to wait(),
// so that they are not waited for multiple times.
TEST(QueueEventClear, ClearOnQueueWait) {
  platform Plt{default_selector()};
  if (!preparePiMock(Plt))
    return;

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
  platform Plt{default_selector()};
  if (!preparePiMock(Plt))
    return;

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
