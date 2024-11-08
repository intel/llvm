//==-------- CommandsWaitForEvents.cpp --- Scheduler unit tests ------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SchedulerTest.hpp"
#include "SchedulerTestUtils.hpp"
#include "ur_mock_helpers.hpp"
#include <helpers/MockDeviceImage.hpp>
#include <helpers/MockKernelInfo.hpp>
#include <helpers/UrMock.hpp>

#include <cassert>

using namespace sycl;

struct TestCtx {
  queue &Q1;
  queue &Q2;

  // These used to be shared_ptr but that was causing problems due to Mock
  // teardown clearing default overrides between tests.
  const detail::context_impl &Ctx1;
  const detail::context_impl &Ctx2;

  ur_event_handle_t EventCtx1 = nullptr;

  ur_event_handle_t EventCtx2 = nullptr;

  bool EventCtx1WasWaited = false;
  bool EventCtx2WasWaited = false;

  TestCtx(queue &Queue1, queue &Queue2)
      : Q1(Queue1), Q2(Queue2),
        Ctx1(*detail::getSyclObjImpl(Q1.get_context()).get()),
        Ctx2(*detail::getSyclObjImpl(Q2.get_context()).get()) {

    EventCtx1 = mock::createDummyHandle<ur_event_handle_t>();
    EventCtx2 = mock::createDummyHandle<ur_event_handle_t>();
  }
};

std::unique_ptr<TestCtx> TestContext;

ur_result_t urEventWaitRedefineCheckEvents(void *pParams) {
  auto params = *static_cast<ur_event_wait_params_t *>(pParams);
  EXPECT_EQ(*params.pnumEvents, 1u)
      << "urEnqueueEventsWait called for different contexts\n";

  EXPECT_TRUE((TestContext->EventCtx1 == **params.pphEventWaitList) ||
              (TestContext->EventCtx2 == **params.pphEventWaitList))
      << "urEventsWait called for unknown event";

  if (TestContext->EventCtx1 == **params.pphEventWaitList)
    TestContext->EventCtx1WasWaited = true;

  if (TestContext->EventCtx2 == **params.pphEventWaitList)
    TestContext->EventCtx2WasWaited = true;

  return UR_RESULT_SUCCESS;
}

ur_result_t getEventInfoFunc(void *pParams) {
  auto params = *static_cast<ur_event_get_info_params_t *>(pParams);
  EXPECT_EQ(*params.ppropName, UR_EVENT_INFO_CONTEXT) << "Unknown param name";

  if (*params.phEvent == TestContext->EventCtx1)
    *reinterpret_cast<ur_context_handle_t *>(*params.ppPropValue) =
        reinterpret_cast<ur_context_handle_t>(TestContext->Ctx1.getHandleRef());
  else if (*params.phEvent == TestContext->EventCtx2)
    *reinterpret_cast<ur_context_handle_t *>(*params.ppPropValue) =
        reinterpret_cast<ur_context_handle_t>(TestContext->Ctx2.getHandleRef());

  return UR_RESULT_SUCCESS;
}

static bool GpiEventsWaitRedefineCalled = false;
ur_result_t urEventsWaitRedefineCheckCalled(void *) {
  GpiEventsWaitRedefineCalled = true;
  return UR_RESULT_SUCCESS;
}

class StreamAUXCmdsWait_TestKernel;

namespace sycl {
inline namespace _V1 {
namespace detail {
template <>
struct KernelInfo<StreamAUXCmdsWait_TestKernel>
    : public unittest::MockKernelInfoBase {
  static constexpr const char *getName() {
    return "StreamAUXCmdsWait_TestKernel";
  }
  static constexpr bool isESIMD() { return true; }
  static constexpr int64_t getKernelSize() { return sizeof(sycl::stream); }
};

} // namespace detail
} // namespace _V1
} // namespace sycl

static sycl::unittest::MockDeviceImage generateDefaultImage() {
  using namespace sycl::unittest;

  MockPropertySet PropSet;
  addESIMDFlag(PropSet);

  std::vector<MockOffloadEntry> Entries =
      makeEmptyKernels({"StreamAUXCmdsWait_TestKernel"});

  MockDeviceImage Img(std::move(Entries), std::move(PropSet));

  return Img;
}

sycl::unittest::MockDeviceImage Img = generateDefaultImage();
sycl::unittest::MockDeviceImageArray<1> ImgArray{&Img};

class EventImplProxyT : public sycl::detail::event_impl {
public:
  using sycl::detail::event_impl::MPostCompleteEvents;
  using sycl::detail::event_impl::MState;
  using sycl::detail::event_impl::MWeakPostCompleteEvents;
};

class QueueImplProxyT : public sycl::detail::queue_impl {
public:
  using sycl::detail::queue_impl::MStreamsServiceEvents;
};

TEST_F(SchedulerTest, StreamAUXCmdsWait) {

  {
    sycl::unittest::UrMock<> Mock;
    sycl::platform Plt = sycl::platform();
    sycl::queue Q(Plt.get_devices()[0]);
    std::shared_ptr<sycl::detail::queue_impl> QueueImpl =
        detail::getSyclObjImpl(Q);

    auto QueueImplProxy = std::static_pointer_cast<QueueImplProxyT>(QueueImpl);

    ASSERT_TRUE(QueueImplProxy->MStreamsServiceEvents.empty())
        << "No stream service events are expected at the beggining";

    event Event = Q.submit([&](handler &CGH) {
      stream Out(1024, 80, CGH);
      CGH.single_task<StreamAUXCmdsWait_TestKernel>(
          [=]() { Out << "Hello, World!" << endl; });
    });

    ASSERT_TRUE(QueueImplProxy->MStreamsServiceEvents.size() == 1)
        << "Expected 1 service stream event";

    std::shared_ptr<sycl::detail::event_impl> EventImpl =
        detail::getSyclObjImpl(Event);

    auto EventImplProxy = std::static_pointer_cast<EventImplProxyT>(EventImpl);

    ASSERT_EQ(EventImplProxy->MWeakPostCompleteEvents.size(), 1u)
        << "Expected 1 post complete event";

    Q.wait();

    ASSERT_TRUE(QueueImplProxy->MStreamsServiceEvents.empty())
        << "No stream service events are expected to left after wait";
  }

  {
    sycl::unittest::UrMock<> Mock;
    sycl::platform Plt = sycl::platform();
    sycl::queue Q(Plt.get_devices()[0]);
    std::shared_ptr<sycl::detail::queue_impl> QueueImpl =
        detail::getSyclObjImpl(Q);

    mock::getCallbacks().set_before_callback("urEventWait",
                                             &urEventsWaitRedefineCheckCalled);

    auto QueueImplProxy = std::static_pointer_cast<QueueImplProxyT>(QueueImpl);

    ur_event_handle_t UREvent = mock::createDummyHandle<ur_event_handle_t>();

    auto EventImpl = std::make_shared<sycl::detail::event_impl>(QueueImpl);
    EventImpl->setHandle(UREvent);

    QueueImplProxy->registerStreamServiceEvent(EventImpl);

    QueueImplProxy->wait();

    ASSERT_TRUE(GpiEventsWaitRedefineCalled)
        << "No stream service events are expected to left after wait";
  }
}

TEST_F(SchedulerTest, CommandsWaitForEvents) {
  sycl::unittest::UrMock<> Mock;
  sycl::platform Plt = sycl::platform();

  mock::getCallbacks().set_before_callback("urEventWait",
                                           &urEventWaitRedefineCheckEvents);
  mock::getCallbacks().set_before_callback("urEventGetInfo", &getEventInfoFunc);

  context Ctx1{Plt.get_devices()[0]};
  queue Q1{Ctx1, default_selector_v};
  context Ctx2{Plt.get_devices()[0]};
  queue Q2{Ctx2, default_selector_v};

  TestContext.reset(new TestCtx(Q1, Q2));

  std::shared_ptr<detail::event_impl> E1(
      new detail::event_impl(TestContext->EventCtx1, Q1.get_context()));
  std::shared_ptr<detail::event_impl> E2(
      new detail::event_impl(TestContext->EventCtx2, Q2.get_context()));

  MockCommand Cmd(nullptr);

  std::vector<std::shared_ptr<detail::event_impl>> Events;
  Events.push_back(E1);
  Events.push_back(E2);

  ur_event_handle_t EventResult = nullptr;

  Cmd.waitForEventsCall(nullptr, Events, EventResult);

  ASSERT_TRUE(TestContext->EventCtx1WasWaited &&
              TestContext->EventCtx2WasWaited)
      << "Not all events were waited for";
  delete TestContext.release(); // explicitly delete here is important for CUDA
                                // BE to ensure that cuda driver is still in
                                // memory while cuda objects are being freed.
}
