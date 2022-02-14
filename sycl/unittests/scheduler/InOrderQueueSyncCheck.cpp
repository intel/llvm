//==---------- InOrderQueueSyncCheck.cpp --- Scheduler unit tests ----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SchedulerTest.hpp"
#include "SchedulerTestUtils.hpp"
#include <CL/sycl.hpp>
#include <detail/queue_impl.hpp>
#include <detail/scheduler/commands.hpp>

#include <gtest/gtest.h>

using namespace sycl;

// Define type with the only methods called by finalizeHandler
class LimitedHandler {
public:
  virtual void depends_on(sycl::event){};

  virtual event finalize() {
    cl::sycl::detail::EventImplPtr NewEvent =
        std::make_shared<detail::event_impl>();
    return sycl::detail::createSyclObjFromImpl<sycl::event>(NewEvent);
  };
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
  sycl::platform Plt{sycl::default_selector()};
  if (Plt.is_host() || Plt.get_backend() == sycl::backend::ext_oneapi_cuda ||
      Plt.get_backend() == sycl::backend::ext_oneapi_hip) {
    std::cerr << "Test is not supported on "
              << Plt.get_info<sycl::info::platform::name>() << ", skipping\n";
    GTEST_SKIP(); // test is not supported on selected platform.
  }

  const sycl::device Dev = Plt.get_devices()[0];
  auto Queue = std::make_shared<MockQueueImpl>(
      sycl::detail::getSyclObjImpl(Dev), sycl::async_handler{},
      sycl::property::queue::in_order());

  // What we are testing here:
  // Task type  | Must depend on
  //  host      | yes - always, separate sync management
  //  host      | yes - always, separate sync management
  //  kernel    | yes - change of sync approach
  //  kernel    | no  - sync between pi calls must be done by backend
  //  host      | yes - always, separate sync management

  sycl::event Event;
  // host task
  {
    LimitedHandlerSimulation MockCGH;
    EXPECT_CALL(MockCGH, depends_on).Times(1);
    Queue->finalizeHandler<LimitedHandlerSimulation>(
        MockCGH, detail::CG::CGTYPE::CodeplayHostTask, Event);
  }
  // host task
  {
    LimitedHandlerSimulation MockCGH;
    EXPECT_CALL(MockCGH, depends_on).Times(1);
    Queue->finalizeHandler<LimitedHandlerSimulation>(
        MockCGH, detail::CG::CGTYPE::CodeplayHostTask, Event);
  }
  // kernel task
  {
    LimitedHandlerSimulation MockCGH;
    EXPECT_CALL(MockCGH, depends_on).Times(1);
    Queue->finalizeHandler<LimitedHandlerSimulation>(
        MockCGH, detail::CG::CGTYPE::Kernel, Event);
  }
  // kernel task
  {
    LimitedHandlerSimulation MockCGH;
    EXPECT_CALL(MockCGH, depends_on).Times(0);
    Queue->finalizeHandler<LimitedHandlerSimulation>(
        MockCGH, detail::CG::CGTYPE::Kernel, Event);
  }
  // host task
  {
    LimitedHandlerSimulation MockCGH;
    EXPECT_CALL(MockCGH, depends_on).Times(1);
    Queue->finalizeHandler<LimitedHandlerSimulation>(
        MockCGH, detail::CG::CGTYPE::CodeplayHostTask, Event);
  }
}