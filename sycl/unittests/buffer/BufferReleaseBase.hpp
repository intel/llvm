//==- BufferReleaseBase.hpp --- check delayed destruction of buffer --------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <helpers/PiMock.hpp>
#include <helpers/TestKernel.hpp>

#include <sycl/accessor.hpp>
#include <sycl/sycl.hpp>
#include <sycl/usm/usm_allocator.hpp>

#include <gtest/gtest.h>

#include <detail/buffer_impl.hpp>
#include <detail/global_handler.hpp>
#include <detail/scheduler/scheduler.hpp>
#include <gmock/gmock.h>

#include "../scheduler/SchedulerTestUtils.hpp"

class MockCmdWithReleaseTracking : public MockCommand {
public:
  MockCmdWithReleaseTracking(
      sycl::detail::QueueImplPtr Queue, sycl::detail::Requirement Req,
      sycl::detail::Command::CommandType Type = sycl::detail::Command::RUN_CG)
      : MockCommand(Queue, Req, Type){};
  MockCmdWithReleaseTracking(
      sycl::detail::QueueImplPtr Queue,
      sycl::detail::Command::CommandType Type = sycl::detail::Command::RUN_CG)
      : MockCommand(Queue, Type){};
  ~MockCmdWithReleaseTracking() { Release(); }
  MOCK_METHOD0(Release, void());
};

template <sycl::backend Backend>
class BufferDestructionCheckCommon : public ::testing::Test {
public:
  BufferDestructionCheckCommon() : Mock(Backend), Plt(Mock.getPlatform()) {}

protected:
  void SetUp() override {
    MockSchedulerPtr = new MockScheduler();
    sycl::detail::GlobalHandler::instance().attachScheduler(
        dynamic_cast<sycl::detail::Scheduler *>(MockSchedulerPtr));
  }
  void TearDown() override {
    sycl::detail::GlobalHandler::instance().attachScheduler(NULL);
  }

  template <typename Buffer>
  MockCmdWithReleaseTracking *addCommandToBuffer(Buffer &Buf, sycl::queue &Q) {
    sycl::detail::Requirement MockReq = getMockRequirement(Buf);
    sycl::detail::MemObjRecord *Rec = MockSchedulerPtr->getOrInsertMemObjRecord(
        sycl::detail::getSyclObjImpl(Q), &MockReq);
    MockCmdWithReleaseTracking *MockCmd = new MockCmdWithReleaseTracking(
        sycl::detail::getSyclObjImpl(Q), MockReq);
    std::vector<sycl::detail::Command *> ToEnqueue;
    MockSchedulerPtr->addNodeToLeaves(Rec, MockCmd, sycl::access::mode::write,
                                      ToEnqueue);
    // we do not want to enqueue commands, just keep not enqueued and not
    // completed, otherwise check is not possible
    return MockCmd;
  }

protected:
  sycl::unittest::PiMock Mock;
  sycl::platform Plt;
  MockScheduler *MockSchedulerPtr;
};
