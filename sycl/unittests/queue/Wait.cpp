//==--------------------- Wait.cpp --- queue unit tests --------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>
#include <detail/event_impl.hpp>
#include <detail/platform_impl.hpp>
#include <detail/scheduler/commands.hpp>
#include <gtest/gtest.h>
#include <helpers/PiMock.hpp>

#include <memory>

namespace {
using namespace cl::sycl;

struct TestCtx {
  bool SupportOOO = true;
  bool PiQueueFinishCalled = false;
  int NEventsWaitedFor = 0;
  int EventReferenceCount = 0;
};
static TestCtx TestContext;

pi_result redefinedQueueCreate(pi_context context, pi_device device,
                               pi_queue_properties properties,
                               pi_queue *queue) {
  if (!TestContext.SupportOOO &&
      properties & PI_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) {
    return PI_INVALID_QUEUE_PROPERTIES;
  }
  return PI_SUCCESS;
}

pi_result redefinedQueueRelease(pi_queue Queue) { return PI_SUCCESS; }

pi_result redefinedUSMEnqueueMemset(pi_queue Queue, void *Ptr, pi_int32 Value,
                                    size_t Count,
                                    pi_uint32 Num_events_in_waitlist,
                                    const pi_event *Events_waitlist,
                                    pi_event *Event) {
  // Provide a dummy non-nullptr value
  TestContext.EventReferenceCount = 1;
  *Event = reinterpret_cast<pi_event>(1);
  return PI_SUCCESS;
}
pi_result redefinedEnqueueMemBufferFill(pi_queue Queue, pi_mem Buffer,
                                        const void *Pattern, size_t PatternSize,
                                        size_t Offset, size_t Size,
                                        pi_uint32 NumEventsInWaitList,
                                        const pi_event *EventWaitList,
                                        pi_event *Event) {
  // Provide a dummy non-nullptr value
  TestContext.EventReferenceCount = 1;
  *Event = reinterpret_cast<pi_event>(1);
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

pi_result redefinedEventGetInfo(pi_event event, pi_event_info param_name,
                                size_t param_value_size, void *param_value,
                                size_t *param_value_size_ret) {
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

bool preparePiMock(platform &Plt) {
  if (Plt.is_host()) {
    std::cout << "Not run on host - no PI events created in that case"
              << std::endl;
    return false;
  }

  unittest::PiMock Mock{Plt};
  Mock.redefine<detail::PiApiKind::piQueueCreate>(redefinedQueueCreate);
  Mock.redefine<detail::PiApiKind::piQueueRelease>(redefinedQueueRelease);
  Mock.redefine<detail::PiApiKind::piQueueFinish>(redefinedQueueFinish);
  Mock.redefine<detail::PiApiKind::piextUSMEnqueueMemset>(
      redefinedUSMEnqueueMemset);
  Mock.redefine<detail::PiApiKind::piEventsWait>(redefinedEventsWait);
  Mock.redefine<detail::PiApiKind::piEnqueueMemBufferFill>(
      redefinedEnqueueMemBufferFill);
  Mock.redefine<detail::PiApiKind::piEventGetInfo>(redefinedEventGetInfo);
  Mock.redefine<detail::PiApiKind::piEventRetain>(redefinedEventRetain);
  Mock.redefine<detail::PiApiKind::piEventRelease>(redefinedEventRelease);
  return true;
}

TEST(QueueWait, QueueWaitTest) {
  platform Plt{default_selector()};
  if (!preparePiMock(Plt))
    return;
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
    buffer<int, 1> buf{range<1>(1)};
    Q.submit([&](handler &Cgh) {
      auto acc = buf.template get_access<access::mode::read_write>(Cgh);
      Cgh.fill(acc, 42);
    });
    Q.wait();
    // Still owned by the execution graph
    ASSERT_EQ(TestContext.EventReferenceCount, 1);
    ASSERT_EQ(TestContext.NEventsWaitedFor, 0);
    ASSERT_TRUE(TestContext.PiQueueFinishCalled);
  }

  // Blocked commands
  {
    TestContext = {};
    buffer<int, 1> buf{range<1>(1)};
    event HostTaskEvent = Q.submit([&](handler &Cgh) {
      auto acc = buf.template get_access<access::mode::read>(Cgh);
      Cgh.host_task([=]() { (void)acc; });
    });
    std::shared_ptr<detail::event_impl> HostTaskEventImpl =
        detail::getSyclObjImpl(HostTaskEvent);
    auto *Cmd = static_cast<detail::Command *>(HostTaskEventImpl->getCommand());
    detail::Command *EmptyTask = *Cmd->MUsers.begin();
    ASSERT_EQ(EmptyTask->getType(), detail::Command::EMPTY_TASK);
    HostTaskEvent.wait();
    // Use the empty task produced by the host task to block the next commands
    while (EmptyTask->MEnqueueStatus !=
           detail::EnqueueResultT::SyclEnqueueSuccess)
      continue;
    EmptyTask->MEnqueueStatus = detail::EnqueueResultT::SyclEnqueueBlocked;
    Q.submit([&](handler &Cgh) {
      auto acc = buf.template get_access<access::mode::discard_write>(Cgh);
      Cgh.fill(acc, 42);
    });
    Q.submit([&](handler &Cgh) {
      auto acc = buf.template get_access<access::mode::discard_write>(Cgh);
      Cgh.fill(acc, 42);
    });
    // Unblock the empty task to allow the submitted events to complete once
    // enqueued.
    EmptyTask->MEnqueueStatus = detail::EnqueueResultT::SyclEnqueueSuccess;
    Q.wait();
    // Only a single event (the last one) should be waited for here.
    ASSERT_EQ(TestContext.NEventsWaitedFor, 1);
    ASSERT_TRUE(TestContext.PiQueueFinishCalled);
  }

  // Test for event::get_wait_list
  {
    sycl::event eA =
        Q.submit([&](sycl::handler &cgh) { cgh.host_task([]() {}); });
    sycl::event eB = Q.submit([&](sycl::handler &cgh) {
      cgh.depends_on(eA);
      cgh.host_task([]() {});
    });

    auto res = eB.get_wait_list();
    assert(res.size() == 1);
    ASSERT_EQ(res[0], eA);

    sycl::event eC = Q.submit([&](sycl::handler &cgh) {
      cgh.depends_on({eA, eB});
      cgh.host_task([]() {});
    });

    res = eC.get_wait_list();
    assert(res.size() == 2);
    ASSERT_EQ(res[0], eA);
    ASSERT_EQ(res[1], eB);

    eC.wait();
  }
  // Test behaviour for emulating an OOO queue with multiple in-order ones.
  TestContext = {};
  TestContext.SupportOOO = false;
  Q = {Ctx, default_selector()};
  Q.memset(HostAlloc, 42, 1);
  // The event is kept alive in this case to call wait.
  ASSERT_EQ(TestContext.EventReferenceCount, 1);
  Q.wait();
  ASSERT_EQ(TestContext.EventReferenceCount, 0);
  ASSERT_EQ(TestContext.NEventsWaitedFor, 1);
  ASSERT_FALSE(TestContext.PiQueueFinishCalled);
}

} // namespace
