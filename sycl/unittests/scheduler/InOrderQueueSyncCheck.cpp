//==---------- InOrderQueueSyncCheck.cpp --- Scheduler unit tests ----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SchedulerTest.hpp"
#include "SchedulerTestUtils.hpp"
#include <detail/queue_impl.hpp>
#include <detail/scheduler/commands.hpp>
#include <helpers/PiMock.hpp>
#include <sycl/sycl.hpp>

#include <gtest/gtest.h>

using namespace sycl;

using ::testing::An;

class MockQueueImpl : public sycl::detail::queue_impl {
public:
  MockQueueImpl(const sycl::detail::DeviceImplPtr &Device,
                const sycl::async_handler &AsyncHandler,
                const sycl::property_list &PropList)
      : sycl::detail::queue_impl(Device, AsyncHandler, PropList) {}
  using sycl::detail::queue_impl::finalizeHandler;
};

// Define type with the only methods called by finalizeHandler
class LimitedHandler {
public:
  LimitedHandler(sycl::detail::CG::CGTYPE CGType,
                 std::shared_ptr<MockQueueImpl> Queue)
      : MCGType(CGType), MQueue(Queue) {}

  virtual ~LimitedHandler() {}
  virtual void depends_on(const sycl::detail::EventImplPtr &) {}
  virtual void depends_on(const std::vector<detail::EventImplPtr> &Events) {}
  virtual void depends_on(event Event) {};

  virtual event finalize() {
    sycl::detail::EventImplPtr NewEvent =
        std::make_shared<detail::event_impl>();
    return sycl::detail::createSyclObjFromImpl<sycl::event>(NewEvent);
  }

  sycl::detail::CG::CGTYPE getType() { return MCGType; }

  sycl::detail::CG::CGTYPE MCGType;
  std::shared_ptr<MockQueueImpl> MQueue;
};

// Needed to use EXPECT_CALL to verify depends_on that originally appends lst
// event as dependency to the new CG
class LimitedHandlerSimulation : public LimitedHandler {
public:
  LimitedHandlerSimulation(sycl::detail::CG::CGTYPE CGType,
                           std::shared_ptr<MockQueueImpl> Queue)
      : LimitedHandler(CGType, Queue) {}

  MOCK_METHOD1(depends_on, void(const sycl::detail::EventImplPtr &));
  MOCK_METHOD1(depends_on, void(event Event));
  MOCK_METHOD1(depends_on,
               void(const std::vector<detail::EventImplPtr> &Events));
};

// Only check events dependency in queue_impl::finalizeHandler
TEST_F(SchedulerTest, InOrderQueueSyncCheck) {
  sycl::unittest::PiMock Mock;
  platform Plt = Mock.getPlatform();

  const sycl::device Dev = Plt.get_devices()[0];
  auto Queue = std::make_shared<MockQueueImpl>(
      sycl::detail::getSyclObjImpl(Dev), sycl::async_handler{},
      sycl::property::queue::in_order());

  // Check that tasks submitted to an in-order queue implicitly depend_on the
  // previous task, this is needed to properly sync blocking & blocked tasks.
  sycl::event Event;
  {
    LimitedHandlerSimulation MockCGH{detail::CG::CGTYPE::CodeplayHostTask,
                                     Queue};
    EXPECT_CALL(MockCGH, depends_on(An<const sycl::detail::EventImplPtr &>()))
        .Times(0);
    Queue->finalizeHandler<LimitedHandlerSimulation>(MockCGH, Event);
  }
  {
    LimitedHandlerSimulation MockCGH{detail::CG::CGTYPE::CodeplayHostTask,
                                     Queue};
    EXPECT_CALL(MockCGH, depends_on(An<const sycl::detail::EventImplPtr &>()))
        .Times(1);
    Queue->finalizeHandler<LimitedHandlerSimulation>(MockCGH, Event);
  }
}
