//==-------- CommandsWaitForEvents.cpp --- Scheduler unit tests ------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SchedulerTest.hpp"
#include "SchedulerTestUtils.hpp"
#include <helpers/MockKernelInfo.hpp>
#include <helpers/PiImage.hpp>
#include <helpers/PiMock.hpp>

#include <cassert>

using namespace sycl;

struct TestCtx {
  queue &Q1;
  queue &Q2;

  std::shared_ptr<detail::context_impl> Ctx1;
  std::shared_ptr<detail::context_impl> Ctx2;

  pi_event EventCtx1 = nullptr;

  pi_event EventCtx2 = nullptr;

  bool EventCtx1WasWaited = false;
  bool EventCtx2WasWaited = false;

  TestCtx(queue &Queue1, queue &Queue2)
      : Q1(Queue1), Q2(Queue2), Ctx1{detail::getSyclObjImpl(Q1.get_context())},
        Ctx2{detail::getSyclObjImpl(Q2.get_context())} {

    pi_result Res = mock_piEventCreate((pi_context)0x0, &EventCtx1);
    EXPECT_TRUE(PI_SUCCESS == Res);

    Res = mock_piEventCreate((pi_context)0x0, &EventCtx2);
    EXPECT_TRUE(PI_SUCCESS == Res);
  }
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

static bool GpiEventsWaitRedefineCalled = false;
pi_result piEventsWaitRedefine(pi_uint32 num_events,
                               const pi_event *event_list) {
  GpiEventsWaitRedefineCalled = true;
  return PI_SUCCESS;
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

static sycl::unittest::PiImage generateDefaultImage() {
  using namespace sycl::unittest;

  PiPropertySet PropSet;
  addESIMDFlag(PropSet);
  std::vector<unsigned char> Bin{0, 1, 2, 3, 4, 5}; // Random data

  PiArray<PiOffloadEntry> Entries =
      makeEmptyKernels({"StreamAUXCmdsWait_TestKernel"});

  PiImage Img{PI_DEVICE_BINARY_TYPE_SPIRV,            // Format
              __SYCL_PI_DEVICE_BINARY_TARGET_SPIRV64, // DeviceTargetSpec
              "",                                     // Compile options
              "",                                     // Link options
              std::move(Bin),
              std::move(Entries),
              std::move(PropSet)};

  return Img;
}

sycl::unittest::PiImage Img = generateDefaultImage();
sycl::unittest::PiImageArray<1> ImgArray{&Img};

class EventImplProxyT : public sycl::detail::event_impl {
public:
  using sycl::detail::event_impl::MPostCompleteEvents;
  using sycl::detail::event_impl::MState;
};

class QueueImplProxyT : public sycl::detail::queue_impl {
public:
  using sycl::detail::queue_impl::MStreamsServiceEvents;
};

TEST_F(SchedulerTest, StreamAUXCmdsWait) {

  {
    sycl::unittest::PiMock Mock;
    sycl::platform Plt = Mock.getPlatform();
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

    ASSERT_EQ(EventImplProxy->MPostCompleteEvents.size(), 1u)
        << "Expected 1 post complete event";

    Q.wait();

    ASSERT_TRUE(QueueImplProxy->MStreamsServiceEvents.empty())
        << "No stream service events are expected to left after wait";
  }

  {
    sycl::unittest::PiMock Mock;
    sycl::platform Plt = Mock.getPlatform();
    sycl::queue Q(Plt.get_devices()[0]);
    std::shared_ptr<sycl::detail::queue_impl> QueueImpl =
        detail::getSyclObjImpl(Q);

    Mock.redefineBefore<detail::PiApiKind::piEventsWait>(piEventsWaitRedefine);

    auto QueueImplProxy = std::static_pointer_cast<QueueImplProxyT>(QueueImpl);

    pi_event PIEvent = nullptr;
    pi_result Res =
        mock_piEventCreate(/*context = */ (pi_context)0x1, &PIEvent);
    ASSERT_TRUE(PI_SUCCESS == Res);

    auto EventImpl = std::make_shared<sycl::detail::event_impl>(QueueImpl);
    EventImpl->getHandleRef() = PIEvent;

    QueueImplProxy->registerStreamServiceEvent(EventImpl);

    QueueImplProxy->wait();

    ASSERT_TRUE(GpiEventsWaitRedefineCalled)
        << "No stream service events are expected to left after wait";
  }
}

TEST_F(SchedulerTest, CommandsWaitForEvents) {
  sycl::unittest::PiMock Mock;
  sycl::platform Plt = Mock.getPlatform();

  Mock.redefineBefore<detail::PiApiKind::piEventsWait>(waitFunc);
  Mock.redefineBefore<detail::PiApiKind::piEventGetInfo>(getEventInfoFunc);

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

  pi_event EventResult = nullptr;

  Cmd.waitForEventsCall(nullptr, Events, EventResult);

  ASSERT_TRUE(TestContext->EventCtx1WasWaited &&
              TestContext->EventCtx2WasWaited)
      << "Not all events were waited for";
  delete TestContext.release(); // explicitly delete here is important for CUDA
                                // BE to ensure that cuda driver is still in
                                // memory while cuda objects are being freed.
}
