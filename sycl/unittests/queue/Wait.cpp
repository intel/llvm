//==--------------------- Wait.cpp --- queue unit tests --------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/event_impl.hpp>
#include <detail/platform_impl.hpp>
#include <detail/scheduler/commands.hpp>
#include <gtest/gtest.h>
#include <helpers/PiMock.hpp>
#include <sycl/sycl.hpp>

#include <memory>

namespace {
using namespace sycl;

struct TestCtx {
  bool SupportOOO = true;
  bool PiQueueFinishCalled = false;
  int NEventsWaitedFor = 0;
  int EventReferenceCount = 0;
};
static TestCtx TestContext;

pi_result redefinedQueueCreateEx(pi_context context, pi_device device,
                                 pi_queue_properties *properties,
                                 pi_queue *queue) {
  assert(properties && properties[0] == PI_QUEUE_FLAGS);
  if (!TestContext.SupportOOO &&
      properties[1] & PI_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE) {
    return PI_ERROR_INVALID_QUEUE_PROPERTIES;
  }
  return PI_SUCCESS;
}

pi_result redefinedUSMEnqueueMemset(pi_queue Queue, void *Ptr,
                                    const void *Pattern, size_t PatternSize,
                                    size_t Count,
                                    pi_uint32 Num_events_in_waitlist,
                                    const pi_event *Events_waitlist,
                                    pi_event *Event) {
  TestContext.EventReferenceCount = 1;
  return PI_SUCCESS;
}
pi_result redefinedEnqueueMemBufferFill(pi_queue Queue, pi_mem Buffer,
                                        const void *Pattern, size_t PatternSize,
                                        size_t Offset, size_t Size,
                                        pi_uint32 NumEventsInWaitList,
                                        const pi_event *EventWaitList,
                                        pi_event *Event) {
  TestContext.EventReferenceCount = 1;
  return PI_SUCCESS;
}

pi_result redefinedQueueFinish(pi_queue Queue) {
  TestContext.PiQueueFinishCalled = true;
  return PI_SUCCESS;
}
pi_result redefinedEventsWait(pi_uint32 num_events,
                              const pi_event *event_list) {
  ++TestContext.NEventsWaitedFor;
  return PI_SUCCESS;
}

pi_result redefinedEventRetain(pi_event event) {
  ++TestContext.EventReferenceCount;
  return PI_SUCCESS;
}

pi_result redefinedEventRelease(pi_event event) {
  --TestContext.EventReferenceCount;
  return PI_SUCCESS;
}

event submitTask(queue &Q, buffer<int, 1> &Buf) {
  return Q.submit([&](handler &Cgh) {
    auto Acc = Buf.template get_access<access::mode::read_write>(Cgh);
    Cgh.fill(Acc, 42);
  });
}

TEST(QueueWait, QueueWaitTest) {
  sycl::unittest::PiMock Mock;
  sycl::platform Plt = Mock.getPlatform();
  Mock.redefineBefore<detail::PiApiKind::piextQueueCreate>(
      redefinedQueueCreateEx);
  Mock.redefineBefore<detail::PiApiKind::piQueueFinish>(redefinedQueueFinish);
  Mock.redefineBefore<detail::PiApiKind::piextUSMEnqueueFill>(
      redefinedUSMEnqueueMemset);
  Mock.redefineBefore<detail::PiApiKind::piEventsWait>(redefinedEventsWait);
  Mock.redefineBefore<detail::PiApiKind::piEnqueueMemBufferFill>(
      redefinedEnqueueMemBufferFill);
  Mock.redefineBefore<detail::PiApiKind::piEventRetain>(redefinedEventRetain);
  Mock.redefineBefore<detail::PiApiKind::piEventRelease>(redefinedEventRelease);
  context Ctx{Plt.get_devices()[0]};
  queue Q{Ctx, default_selector()};

  unsigned char *HostAlloc = (unsigned char *)malloc_host(1, Ctx);

  // USM API event
  TestContext = {};
  Q.memset(HostAlloc, 42, 1);
  // No need to keep the event since we'll use piQueueFinish.
  ASSERT_EQ(TestContext.EventReferenceCount, 0);
  Q.wait();
  ASSERT_EQ(TestContext.NEventsWaitedFor, 0);
  ASSERT_TRUE(TestContext.PiQueueFinishCalled);

  // Events with temporary ownership
  {
    TestContext = {};
    buffer<int, 1> Buf{range<1>(1)};
    submitTask(Q, Buf);
    Q.wait();
    // Still owned by the execution graph
    ASSERT_EQ(TestContext.EventReferenceCount, 1);
    ASSERT_EQ(TestContext.NEventsWaitedFor, 0);
    ASSERT_TRUE(TestContext.PiQueueFinishCalled);
  }

  // Blocked commands
  {
    TestContext = {};
    buffer<int, 1> Buf{range<1>(1)};

    event DepEvent = submitTask(Q, Buf);

    // Manually block the next commands.
    std::shared_ptr<detail::event_impl> DepEventImpl =
        detail::getSyclObjImpl(DepEvent);
    auto *Cmd = static_cast<detail::Command *>(DepEventImpl->getCommand());
    Cmd->MIsBlockable = true;
    Cmd->MEnqueueStatus = detail::EnqueueResultT::SyclEnqueueBlocked;

    submitTask(Q, Buf);
    submitTask(Q, Buf);

    Cmd->MEnqueueStatus = detail::EnqueueResultT::SyclEnqueueSuccess;
    Q.wait();
    // Only a single event (the last one) should be waited for here.
    ASSERT_EQ(TestContext.NEventsWaitedFor, 1);
    ASSERT_TRUE(TestContext.PiQueueFinishCalled);
  }

  // Test behaviour for emulating an OOO queue with multiple in-order ones.
  TestContext = {};
  TestContext.SupportOOO = false;
  Q = queue{Ctx, default_selector()};
  Q.memset(HostAlloc, 42, 1);
  // The event is kept alive in this case to call wait.
  ASSERT_EQ(TestContext.EventReferenceCount, 1);
  Q.wait();
  ASSERT_EQ(TestContext.EventReferenceCount, 0);
  ASSERT_EQ(TestContext.NEventsWaitedFor, 1);
  ASSERT_FALSE(TestContext.PiQueueFinishCalled);
}

} // namespace
