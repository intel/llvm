//==-------- CommandsWaitForEvents.cpp --- Scheduler unit tests ------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SchedulerTest.hpp"
#include "SchedulerTestUtils.hpp"
#include <helpers/PiMock.hpp>

using namespace cl::sycl;

struct TestCtx {
  queue &Q1;
  queue &Q2;

  std::shared_ptr<detail::context_impl> Ctx1;
  std::shared_ptr<detail::context_impl> Ctx2;

  pi_event EventCtx1 = reinterpret_cast<pi_event>(0x01);
  pi_event EventCtx2 = reinterpret_cast<pi_event>(0x02);

  bool EventCtx1WasWaited = false;
  bool EventCtx2WasWaited = false;

  TestCtx(queue &Queue1, queue &Queue2)
      : Q1(Queue1), Q2(Queue2), Ctx1{detail::getSyclObjImpl(Q1.get_context())},
        Ctx2{detail::getSyclObjImpl(Q2.get_context())} {}
};

std::unique_ptr<TestCtx> TestContext;

pi_result waitFunc(pi_uint32 N, const pi_event *List) {
  EXPECT_EQ(N, 1u) << "piEventsWait called for different contexts\n";

  EXPECT_TRUE((TestContext->EventCtx1 == *List) ||
              (TestContext->EventCtx2 == *List))
      << "piEventsWait called for unknown event";

  if (TestContext->EventCtx1 == *List)
    TestContext->EventCtx1WasWaited = true;

  if (TestContext->EventCtx2 == *List)
    TestContext->EventCtx2WasWaited = true;

  return PI_SUCCESS;
}

pi_result retainReleaseFunc(pi_event) { return PI_SUCCESS; }

pi_result getEventInfoFunc(pi_event Event, pi_event_info PName, size_t PVSize,
                           void *PV, size_t *PVSizeRet) {
  EXPECT_EQ(PName, PI_EVENT_INFO_CONTEXT) << "Unknown param name";

  if (Event == TestContext->EventCtx1)
    *reinterpret_cast<pi_context *>(PV) =
        reinterpret_cast<pi_context>(TestContext->Ctx1->getHandleRef());
  else if (Event == TestContext->EventCtx2)
    *reinterpret_cast<pi_context *>(PV) =
        reinterpret_cast<pi_context>(TestContext->Ctx2->getHandleRef());

  return PI_SUCCESS;
}

TEST_F(SchedulerTest, CommandsWaitForEvents) {
  default_selector Selector{};
  if (Selector.select_device().is_host()) {
    std::cerr << "Not run due to host-only environment\n";
    return;
  }

  platform Plt{Selector};
  unittest::PiMock Mock{Plt};

  Mock.redefine<detail::PiApiKind::piEventsWait>(waitFunc);
  Mock.redefine<detail::PiApiKind::piEventRetain>(retainReleaseFunc);
  Mock.redefine<detail::PiApiKind::piEventRelease>(retainReleaseFunc);
  Mock.redefine<detail::PiApiKind::piEventGetInfo>(getEventInfoFunc);

  context Ctx1{Plt.get_devices()[0]};
  queue Q1{Ctx1, Selector};
  context Ctx2{Plt.get_devices()[0]};
  queue Q2{Ctx2, Selector};

  TestContext.reset(new TestCtx(Q1, Q2));

  std::shared_ptr<detail::event_impl> E1(
      new detail::event_impl(TestContext->EventCtx1, Q1.get_context()));
  std::shared_ptr<detail::event_impl> E2(
      new detail::event_impl(TestContext->EventCtx2, Q2.get_context()));

  sycl::device HostDevice;
  std::shared_ptr<detail::queue_impl> DefaultHostQueue(new detail::queue_impl(
      detail::getSyclObjImpl(HostDevice), /*AsyncHandler=*/{},
      /*PropList=*/{}));

  MockCommand Cmd(DefaultHostQueue);

  std::vector<std::shared_ptr<detail::event_impl>> Events;
  Events.push_back(E1);
  Events.push_back(E2);

  pi_event EventResult = nullptr;

  Cmd.waitForEventsCall(DefaultHostQueue, Events, EventResult);

  ASSERT_TRUE(TestContext->EventCtx1WasWaited &&
              TestContext->EventCtx2WasWaited)
      << "Not all events were waited for";
  delete TestContext.release(); // explicitly delete here is important for CUDA
                                // BE to ensure that cuda driver is still in
                                // memory while cuda objects are being freed.
}
