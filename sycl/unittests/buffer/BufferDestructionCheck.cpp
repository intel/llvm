//==- BufferDestructionCheck.cpp --- check delayed destruction of buffer --==//
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

class BufferDestructionCheck : public ::testing::Test {
public:
  BufferDestructionCheck() : Mock{}, Plt{Mock.getPlatform()} {}

protected:
  void SetUp() override {
    if (Plt.is_host()) {
      std::cout << "Not run due to host-only environment\n";
      GTEST_SKIP();
    }
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
    std::vector<sycl::detail::Command *> AuxCmds;
    sycl::detail::MemObjRecord *Rec = MockSchedulerPtr->getOrInsertMemObjRecord(
        sycl::detail::getSyclObjImpl(Q), &MockReq, AuxCmds);
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

TEST_F(BufferDestructionCheck, BufferWithSizeOnlyDefault) {
  sycl::context Context{Plt};
  sycl::queue Q = sycl::queue{Context, sycl::default_selector{}};

  MockCmdWithReleaseTracking *MockCmd = NULL;
  sycl::detail::buffer_impl *RawBufferImplPtr = NULL;
  {
    sycl::buffer<int, 1> Buf(1);
    std::shared_ptr<sycl::detail::buffer_impl> BufImpl =
        sycl::detail::getSyclObjImpl(Buf);
    RawBufferImplPtr = BufImpl.get();
    MockCmd = addCommandToBuffer(Buf, Q);
  }
  ASSERT_EQ(MockSchedulerPtr->MDeferredMemObjRelease.size(), 1u);
  EXPECT_EQ(MockSchedulerPtr->MDeferredMemObjRelease[0].get(),
            RawBufferImplPtr);
  EXPECT_CALL(*MockCmd, Release).Times(1);
}

TEST_F(BufferDestructionCheck, BufferWithSizeOnlyDefaultSetFinalData) {
  sycl::context Context{Plt};
  sycl::queue Q = sycl::queue{Context, sycl::default_selector{}};

  MockCmdWithReleaseTracking *MockCmd = NULL;
  {
    int FinalData = 0;
    sycl::buffer<int, 1> Buf(1);
    Buf.set_final_data(&FinalData);
    MockCmd = addCommandToBuffer(Buf, Q);
    EXPECT_CALL(*MockCmd, Release).Times(1);
  }
  ASSERT_EQ(MockSchedulerPtr->MDeferredMemObjRelease.size(), 0u);
}

TEST_F(BufferDestructionCheck, BufferWithSizeOnlyNonDefaultAllocator) {
  sycl::context Context{Plt};
  sycl::queue Q = sycl::queue{Context, sycl::default_selector{}};

  MockCmdWithReleaseTracking *MockCmd = NULL;
  sycl::detail::buffer_impl *RawBufferImplPtr = NULL;
  {
    using AllocatorTypeTest =
        sycl::usm_allocator<int, sycl::usm::alloc::shared>;
    AllocatorTypeTest allocator(Q);
    sycl::buffer<int, 1, AllocatorTypeTest> Buf(1, allocator);
    std::shared_ptr<sycl::detail::buffer_impl> BufImpl =
        sycl::detail::getSyclObjImpl(Buf);
    RawBufferImplPtr = BufImpl.get();
    MockCmd = addCommandToBuffer(Buf, Q);
    EXPECT_CALL(*MockCmd, Release).Times(1);
  }
  ASSERT_EQ(MockSchedulerPtr->MDeferredMemObjRelease.size(), 1u);
  EXPECT_EQ(MockSchedulerPtr->MDeferredMemObjRelease[0].get(),
            RawBufferImplPtr);
}

TEST_F(BufferDestructionCheck, BufferWithSizeOnlyDefaultAllocator) {
  sycl::context Context{Plt};
  sycl::queue Q = sycl::queue{Context, sycl::default_selector{}};

  MockCmdWithReleaseTracking *MockCmd = NULL;
  sycl::detail::buffer_impl *RawBufferImplPtr = NULL;
  {
    using AllocatorTypeTest = sycl::buffer_allocator<int>;
    AllocatorTypeTest allocator;
    sycl::buffer<int, 1, AllocatorTypeTest> Buf(1, allocator);
    std::shared_ptr<sycl::detail::buffer_impl> BufImpl =
        sycl::detail::getSyclObjImpl(Buf);
    RawBufferImplPtr = BufImpl.get();
    MockCmd = addCommandToBuffer(Buf, Q);
    EXPECT_CALL(*MockCmd, Release).Times(1);
  }
  ASSERT_EQ(MockSchedulerPtr->MDeferredMemObjRelease.size(), 1u);
  EXPECT_EQ(MockSchedulerPtr->MDeferredMemObjRelease[0].get(),
            RawBufferImplPtr);
}

TEST_F(BufferDestructionCheck, BufferWithRawHostPtr) {
  sycl::context Context{Plt};
  sycl::queue Q = sycl::queue{Context, sycl::default_selector{}};

  MockCmdWithReleaseTracking *MockCmd = NULL;
  {
    int InitialVal = 8;
    sycl::buffer<int, 1> Buf(&InitialVal, 1);
    MockCmd = addCommandToBuffer(Buf, Q);
    EXPECT_CALL(*MockCmd, Release).Times(1);
  }
  ASSERT_EQ(MockSchedulerPtr->MDeferredMemObjRelease.size(), 0u);
}

TEST_F(BufferDestructionCheck, BufferWithRawHostPtrWithNonDefaultAllocator) {
  sycl::context Context{Plt};
  sycl::queue Q = sycl::queue{Context, sycl::default_selector{}};

  MockCmdWithReleaseTracking *MockCmd = NULL;
  {
    int InitialVal = 8;
    using AllocatorTypeTest =
        sycl::usm_allocator<int, sycl::usm::alloc::shared>;
    AllocatorTypeTest allocator(Q);
    sycl::buffer<int, 1, AllocatorTypeTest> Buf(&InitialVal, 1, allocator);
    MockCmd = addCommandToBuffer(Buf, Q);
    EXPECT_CALL(*MockCmd, Release).Times(1);
  }
  ASSERT_EQ(MockSchedulerPtr->MDeferredMemObjRelease.size(), 0u);
}

TEST_F(BufferDestructionCheck, BufferWithConstRawHostPtr) {
  sycl::context Context{Plt};
  sycl::queue Q = sycl::queue{Context, sycl::default_selector{}};

  MockCmdWithReleaseTracking *MockCmd = NULL;
  {
    const int InitialVal = 8;
    sycl::buffer<int, 1> Buf(&InitialVal, 1);
    MockCmd = addCommandToBuffer(Buf, Q);
    EXPECT_CALL(*MockCmd, Release).Times(1);
  }
  ASSERT_EQ(MockSchedulerPtr->MDeferredMemObjRelease.size(), 0u);
}

TEST_F(BufferDestructionCheck, BufferWithContainer) {
  sycl::context Context{Plt};
  sycl::queue Q = sycl::queue{Context, sycl::default_selector{}};

  MockCmdWithReleaseTracking *MockCmd = NULL;
  {
    std::vector<int> data{3, 4};
    sycl::buffer<int, 1> Buf(data);
    MockCmd = addCommandToBuffer(Buf, Q);
    EXPECT_CALL(*MockCmd, Release).Times(1);
  }
  ASSERT_EQ(MockSchedulerPtr->MDeferredMemObjRelease.size(), 0u);
}

TEST_F(BufferDestructionCheck, BufferWithSharedPtr) {
  sycl::context Context{Plt};
  sycl::queue Q = sycl::queue{Context, sycl::default_selector{}};

  MockCmdWithReleaseTracking *MockCmd = NULL;
  {
    std::shared_ptr<int> InitialVal(new int(5));
    sycl::buffer<int, 1> Buf(InitialVal, 1);
    MockCmd = addCommandToBuffer(Buf, Q);
    EXPECT_CALL(*MockCmd, Release).Times(1);
  }
  ASSERT_EQ(MockSchedulerPtr->MDeferredMemObjRelease.size(), 0u);
}

TEST_F(BufferDestructionCheck, BufferWithSharedPtrArray) {
  sycl::context Context{Plt};
  sycl::queue Q = sycl::queue{Context, sycl::default_selector{}};

  MockCmdWithReleaseTracking *MockCmd = NULL;
  {
    std::shared_ptr<int[]> InitialVal(new int[2]);
    sycl::buffer<int, 1> Buf(InitialVal, 1);
    MockCmd = addCommandToBuffer(Buf, Q);
    EXPECT_CALL(*MockCmd, Release).Times(1);
  }
  ASSERT_EQ(MockSchedulerPtr->MDeferredMemObjRelease.size(), 0u);
}

TEST_F(BufferDestructionCheck, BufferWithIterators) {
  sycl::context Context{Plt};
  sycl::queue Q = sycl::queue{Context, sycl::default_selector{}};

  MockCmdWithReleaseTracking *MockCmd = NULL;
  sycl::detail::buffer_impl *RawBufferImplPtr = NULL;
  {
    std::vector<int> data{3, 4};
    sycl::buffer<int, 1> Buf(data.begin(), data.end());
    std::shared_ptr<sycl::detail::buffer_impl> BufImpl =
        sycl::detail::getSyclObjImpl(Buf);
    RawBufferImplPtr = BufImpl.get();
    MockCmd = addCommandToBuffer(Buf, Q);
    EXPECT_CALL(*MockCmd, Release).Times(1);
  }
  ASSERT_EQ(MockSchedulerPtr->MDeferredMemObjRelease.size(), 1u);
  EXPECT_EQ(MockSchedulerPtr->MDeferredMemObjRelease[0].get(),
            RawBufferImplPtr);
}

std::map<pi_event, pi_int32> ExpectedEventStatus;
pi_result getEventInfoFunc(pi_event Event, pi_event_info PName, size_t PVSize,
                           void *PV, size_t *PVSizeRet) {
  EXPECT_EQ(PName, PI_EVENT_INFO_COMMAND_EXECUTION_STATUS)
      << "Unknown param name";
  // could not use assert here
  EXPECT_EQ(PVSize, 4u);
  auto it = ExpectedEventStatus.find(Event);
  if (it != ExpectedEventStatus.end()) {
    *(static_cast<pi_int32 *>(PV)) = it->second;
    return PI_SUCCESS;
  } else
    return PI_ERROR_INVALID_OPERATION;
}

TEST_F(BufferDestructionCheck, ReadyToReleaseLogic) {
  sycl::context Context{Plt};
  sycl::queue Q = sycl::queue{Context, sycl::default_selector{}};

  sycl::buffer<int, 1> Buf(1);
  sycl::detail::Requirement MockReq = getMockRequirement(Buf);
  std::vector<sycl::detail::Command *> AuxCmds;
  sycl::detail::MemObjRecord *Rec = MockSchedulerPtr->getOrInsertMemObjRecord(
      sycl::detail::getSyclObjImpl(Q), &MockReq, AuxCmds);

  std::shared_ptr<sycl::detail::context_impl> CtxImpl =
      sycl::detail::getSyclObjImpl(Context);
  MockCmdWithReleaseTracking *ReadCmd = nullptr;
  MockCmdWithReleaseTracking *WriteCmd = nullptr;
  ReadCmd =
      new MockCmdWithReleaseTracking(sycl::detail::getSyclObjImpl(Q), MockReq);
  ReadCmd->getEvent()->getHandleRef() =
      createDummyHandle<pi_event>(); // just assign to be able to use mock
  WriteCmd =
      new MockCmdWithReleaseTracking(sycl::detail::getSyclObjImpl(Q), MockReq);
  WriteCmd->getEvent()->getHandleRef() =
      createDummyHandle<pi_event>(); // just assign to be able to use mock
  ReadCmd->MEnqueueStatus = sycl::detail::EnqueueResultT::SyclEnqueueSuccess;
  WriteCmd->MEnqueueStatus = sycl::detail::EnqueueResultT::SyclEnqueueSuccess;

  std::vector<sycl::detail::Command *> ToCleanUp;
  std::vector<sycl::detail::Command *> ToEnqueue;
  MockSchedulerPtr->addNodeToLeaves(Rec, ReadCmd, sycl::access::mode::read,
                                    ToEnqueue);
  MockSchedulerPtr->addNodeToLeaves(Rec, WriteCmd, sycl::access::mode::write,
                                    ToEnqueue);

  Mock.redefine<sycl::detail::PiApiKind::piEventGetInfo>(getEventInfoFunc);
  testing::InSequence S;

  ExpectedEventStatus[ReadCmd->getEvent()->getHandleRef()] = PI_EVENT_SUBMITTED;
  ExpectedEventStatus[WriteCmd->getEvent()->getHandleRef()] =
      PI_EVENT_SUBMITTED;

  EXPECT_FALSE(MockSchedulerPtr->checkLeavesCompletion(Rec));

  ExpectedEventStatus[ReadCmd->getEvent()->getHandleRef()] = PI_EVENT_COMPLETE;
  ExpectedEventStatus[WriteCmd->getEvent()->getHandleRef()] =
      PI_EVENT_SUBMITTED;

  EXPECT_FALSE(MockSchedulerPtr->checkLeavesCompletion(Rec));

  ExpectedEventStatus[ReadCmd->getEvent()->getHandleRef()] = PI_EVENT_COMPLETE;
  ExpectedEventStatus[WriteCmd->getEvent()->getHandleRef()] = PI_EVENT_COMPLETE;
  EXPECT_TRUE(MockSchedulerPtr->checkLeavesCompletion(Rec));
  // previous expect_call is still valid and will generate failure if we recieve
  // call here, no need for extra limitation
  EXPECT_CALL(*ReadCmd, Release).Times(1);
  EXPECT_CALL(*WriteCmd, Release).Times(1);
}