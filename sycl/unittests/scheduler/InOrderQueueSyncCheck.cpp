//==---------- InOrderQueueSyncCheck.cpp --- Scheduler unit tests ----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SchedulerTest.hpp"
#include "SchedulerTestUtils.hpp"
#include <detail/handler_impl.hpp>
#include <detail/queue_impl.hpp>
#include <detail/scheduler/commands.hpp>
#include <helpers/UrMock.hpp>
#include <sycl/sycl.hpp>

#include <gtest/gtest.h>

using namespace sycl;

using ::testing::An;

class MockQueueImpl : public sycl::detail::queue_impl {
public:
  MockQueueImpl(sycl::detail::device_impl &Device,
                const sycl::async_handler &AsyncHandler,
                const sycl::property_list &PropList)
      : sycl::detail::queue_impl(Device, AsyncHandler, PropList,
                                 sycl::detail::queue_impl::private_tag{}) {}
  using sycl::detail::queue_impl::finalizeHandlerInOrderHostTaskUnlocked;
};

// Define type with the only methods called by finalizeHandler
class LimitedHandler {
public:
  LimitedHandler(sycl::detail::CGType CGType,
                 std::shared_ptr<MockQueueImpl> Queue)
      : MCGType(CGType), impl(std::make_shared<handler_impl>(Queue)) {}

  virtual ~LimitedHandler() {}
  virtual void depends_on(const sycl::detail::EventImplPtr &) {}
  virtual void depends_on(const std::vector<detail::EventImplPtr> &Events) {}
  virtual void depends_on(event Event) {};

#ifdef __INTEL_PREVIEW_BREAKING_CHANGES
  virtual sycl::detail::EventImplPtr finalize() {
    return std::make_shared<detail::event_impl>();
  }
#else
  virtual event finalize() {
    sycl::detail::EventImplPtr NewEvent =
        detail::event_impl::create_completed_host_event();
    return sycl::detail::createSyclObjFromImpl<sycl::event>(NewEvent);
  }
#endif

  sycl::detail::CGType getType() { return MCGType; }

  sycl::detail::CGType MCGType;
  struct handler_impl {
    handler_impl(std::shared_ptr<MockQueueImpl> Queue) : MQueue(Queue) {}
    std::shared_ptr<MockQueueImpl> MQueue;
    MockQueueImpl &get_queue() { return *MQueue; }
    std::shared_ptr<ext::oneapi::experimental::detail::exec_graph_impl>
        MExecGraph;
  };
  std::shared_ptr<handler_impl> impl;
  std::shared_ptr<detail::kernel_impl> MKernel;
  detail::string MKernelName;
};

// Needed to use EXPECT_CALL to verify depends_on that originally appends lst
// event as dependency to the new CG
class LimitedHandlerSimulation : public LimitedHandler {
public:
  LimitedHandlerSimulation(sycl::detail::CGType CGType,
                           std::shared_ptr<MockQueueImpl> Queue)
      : LimitedHandler(CGType, Queue) {}

  MOCK_METHOD1(depends_on, void(const sycl::detail::EventImplPtr &));
  MOCK_METHOD1(depends_on, void(event Event));
  MOCK_METHOD1(depends_on,
               void(const std::vector<detail::EventImplPtr> &Events));
};

// Only check events dependency in queue_impl::finalizeHandler
TEST_F(SchedulerTest, InOrderQueueSyncCheck) {
  sycl::unittest::UrMock<> Mock;
  platform Plt = sycl::platform();

  const sycl::device Dev = Plt.get_devices()[0];
  auto Queue = std::make_shared<MockQueueImpl>(
      *sycl::detail::getSyclObjImpl(Dev), sycl::async_handler{},
      sycl::property::queue::in_order());

  // Check that tasks submitted to an in-order queue implicitly depend_on the
  // previous task, this is needed to properly sync blocking & blocked tasks.
  {
    LimitedHandlerSimulation MockCGH{detail::CGType::CodeplayHostTask, Queue};
    EXPECT_CALL(MockCGH, depends_on(An<const sycl::detail::EventImplPtr &>()))
        .Times(1);
    Queue->finalizeHandlerInOrderHostTaskUnlocked<LimitedHandlerSimulation>(
        MockCGH);
  }
  {
    LimitedHandlerSimulation MockCGH{detail::CGType::CodeplayHostTask, Queue};
    EXPECT_CALL(MockCGH, depends_on(An<const sycl::detail::EventImplPtr &>()))
        .Times(1);
    Queue->finalizeHandlerInOrderHostTaskUnlocked<LimitedHandlerSimulation>(
        MockCGH);
  }
}
