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

// Define type with the only methods called by finalizeHandler
class LimitedHandler {
public:
  virtual ~LimitedHandler() {}
  virtual void depends_on(sycl::event) {}

  virtual event finalize() {
    sycl::detail::EventImplPtr NewEvent =
        std::make_shared<detail::event_impl>();
    return sycl::detail::createSyclObjFromImpl<sycl::event>(NewEvent);
  }
};

// Needed to use EXPECT_CALL to verify depends_on that originally appends lst
// event as dependency to the new CG
class LimitedHandlerSimulation : public LimitedHandler {
public:
  MOCK_METHOD1(depends_on, void(sycl::event));
};

class MockQueueImpl : public sycl::detail::queue_impl {
public:
  MockQueueImpl(const sycl::detail::DeviceImplPtr &Device,
                const sycl::async_handler &AsyncHandler,
                const sycl::property_list &PropList)
      : sycl::detail::queue_impl(Device, AsyncHandler, PropList) {}
  using sycl::detail::queue_impl::finalizeHandler;
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
    LimitedHandlerSimulation MockCGH;
    EXPECT_CALL(MockCGH, depends_on).Times(0);
    Queue->finalizeHandler<LimitedHandlerSimulation>(MockCGH, Event);
  }
  {
    LimitedHandlerSimulation MockCGH;
    EXPECT_CALL(MockCGH, depends_on).Times(1);
    Queue->finalizeHandler<LimitedHandlerSimulation>(MockCGH, Event);
  }
}
