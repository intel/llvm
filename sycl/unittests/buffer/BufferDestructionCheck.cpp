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

class FairMockScheduler : public sycl::detail::Scheduler {
public:
  using sycl::detail::Scheduler::MDeferredMemObjRelease;
  using sycl::detail::Scheduler::MGraphBuilder;
  using sycl::detail::Scheduler::MGraphLock;
  using sycl::detail::Scheduler::waitForRecordToFinish;
  MOCK_METHOD1(deferMemObjRelease,
               void(const std::shared_ptr<sycl::detail::SYCLMemObjI> &));
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
    MockSchedulerPtr = new testing::NiceMock<FairMockScheduler>();
    sycl::detail::GlobalHandler::instance().attachScheduler(
        dynamic_cast<sycl::detail::Scheduler *>(MockSchedulerPtr));
  }
  void TearDown() override {
    sycl::detail::GlobalHandler::instance().attachScheduler(NULL);
  }

  inline void
  CheckBufferDestruction(std::shared_ptr<sycl::detail::buffer_impl> BufImpl,
                         bool ShouldBeDeferred) {
    ASSERT_NE(BufImpl, nullptr);
    const std::function<bool(
        const std::shared_ptr<sycl::detail::SYCLMemObjI> &)>
        checkerNotEqual =
            [&BufImpl](
                const std::shared_ptr<sycl::detail::SYCLMemObjI> &memObj) {
              return BufImpl.get() != memObj.get();
            };
    const std::function<bool(
        const std::shared_ptr<sycl::detail::SYCLMemObjI> &)>
        checkerEqual =
            [&BufImpl](
                const std::shared_ptr<sycl::detail::SYCLMemObjI> &memObj) {
              return BufImpl.get() == memObj.get();
            };
    if (ShouldBeDeferred) {
      testing::Sequence S;
      // first is check that explicitly created buffer is deferred
      EXPECT_CALL(*MockSchedulerPtr,
                  deferMemObjRelease(testing::Truly(checkerEqual)))
          .Times(1)
          .InSequence(S)
          .RetiresOnSaturation();
      // we have two queues - non host and host queue. Currently queue contains
      // its own buffer as class member, buffer as used for assert handling.
      // those buffers also created with size only so it also to be deferred on
      // deletion.
      EXPECT_CALL(*MockSchedulerPtr, deferMemObjRelease(testing::_))
          .Times(/*testing::AnyNumber()*/ 2)
          .InSequence(S);
    } else {
      // buffer created above should not be deferred on deletion because has non
      // default allocator
      EXPECT_CALL(*MockSchedulerPtr,
                  deferMemObjRelease(testing::Truly(checkerNotEqual)))
          .Times(testing::AnyNumber());
    }
  }

protected:
  sycl::unittest::PiMock Mock;
  sycl::platform Plt;
  testing::NiceMock<FairMockScheduler> *MockSchedulerPtr;
};

TEST_F(BufferDestructionCheck, BufferWithSizeOnlyDefault) {
  sycl::context Context{Plt};
  sycl::queue Q = sycl::queue{Context, sycl::default_selector{}};
  {
    sycl::buffer<int, 1> Buf(1);
    std::shared_ptr<sycl::detail::buffer_impl> BufImpl =
        sycl::detail::getSyclObjImpl(Buf);
    CheckBufferDestruction(BufImpl, true);
  }
}

TEST_F(BufferDestructionCheck, BufferWithSizeOnlyNonDefaultAllocator) {
  sycl::context Context{Plt};
  sycl::queue Q = sycl::queue{Context, sycl::default_selector{}};
  {
    using AllocatorTypeTest =
        sycl::usm_allocator<int, sycl::usm::alloc::shared>;
    AllocatorTypeTest allocator(Q);
    sycl::buffer<int, 1, AllocatorTypeTest> Buf(1, allocator);
    std::shared_ptr<sycl::detail::buffer_impl> BufImpl =
        sycl::detail::getSyclObjImpl(Buf);
    CheckBufferDestruction(BufImpl, false);
  }
}

TEST_F(BufferDestructionCheck, BufferWithSizeOnlyDefaultAllocator) {
  sycl::context Context{Plt};
  sycl::queue Q = sycl::queue{Context, sycl::default_selector{}};
  {
    using AllocatorTypeTest = sycl::buffer_allocator<int>;
    AllocatorTypeTest allocator;
    sycl::buffer<int, 1, AllocatorTypeTest> Buf(1, allocator);
    std::shared_ptr<sycl::detail::buffer_impl> BufImpl =
        sycl::detail::getSyclObjImpl(Buf);
    CheckBufferDestruction(BufImpl, true);
  }
}

TEST_F(BufferDestructionCheck, BufferWithRawHostPtr) {
  sycl::context Context{Plt};
  sycl::queue Q = sycl::queue{Context, sycl::default_selector{}};
  {
    int InitialVal = 8;
    sycl::buffer<int, 1> Buf(&InitialVal, 1);
    std::shared_ptr<sycl::detail::buffer_impl> BufImpl =
        sycl::detail::getSyclObjImpl(Buf);
    CheckBufferDestruction(BufImpl, false);
  }
}

TEST_F(BufferDestructionCheck, BufferWithRawHostPtrWithNonDefaultAllocator) {
  sycl::context Context{Plt};
  sycl::queue Q = sycl::queue{Context, sycl::default_selector{}};
  {
    int InitialVal = 8;
    using AllocatorTypeTest =
        sycl::usm_allocator<int, sycl::usm::alloc::shared>;
    AllocatorTypeTest allocator(Q);
    sycl::buffer<int, 1, AllocatorTypeTest> Buf(&InitialVal, 1, allocator);
    std::shared_ptr<sycl::detail::buffer_impl> BufImpl =
        sycl::detail::getSyclObjImpl(Buf);
    CheckBufferDestruction(BufImpl, false);
  }
}

TEST_F(BufferDestructionCheck, BufferWithConstRawHostPtr) {
  sycl::context Context{Plt};
  sycl::queue Q = sycl::queue{Context, sycl::default_selector{}};
  {
    const int InitialVal = 8;
    sycl::buffer<int, 1> Buf(&InitialVal, 1);
    std::shared_ptr<sycl::detail::buffer_impl> BufImpl =
        sycl::detail::getSyclObjImpl(Buf);
    CheckBufferDestruction(BufImpl, false);
  }
}

TEST_F(BufferDestructionCheck,
       BufferWithConstRawHostPtrWithNonDefaultAllocator) {
  sycl::context Context{Plt};
  sycl::queue Q = sycl::queue{Context, sycl::default_selector{}};
  {
    const int InitialVal = 8;
    using AllocatorTypeTest =
        sycl::usm_allocator<int, sycl::usm::alloc::shared>;
    AllocatorTypeTest allocator(Q);
    sycl::buffer<int, 1, AllocatorTypeTest> Buf(&InitialVal, 1, allocator);
    std::shared_ptr<sycl::detail::buffer_impl> BufImpl =
        sycl::detail::getSyclObjImpl(Buf);
    CheckBufferDestruction(BufImpl, false);
  }
}

TEST_F(BufferDestructionCheck, BufferWithContainer) {
  sycl::context Context{Plt};
  sycl::queue Q = sycl::queue{Context, sycl::default_selector{}};
  {
    std::vector<int> data{3, 4};
    sycl::buffer<int, 1> Buf(data);
    std::shared_ptr<sycl::detail::buffer_impl> BufImpl =
        sycl::detail::getSyclObjImpl(Buf);
    CheckBufferDestruction(BufImpl, false);
  }
}

TEST_F(BufferDestructionCheck, BufferWithContainerWithNonDefaultAllocator) {
  sycl::context Context{Plt};
  sycl::queue Q = sycl::queue{Context, sycl::default_selector{}};
  {
    std::vector<int> data{3, 4};
    using AllocatorTypeTest =
        sycl::usm_allocator<int, sycl::usm::alloc::shared>;
    AllocatorTypeTest allocator(Q);
    sycl::buffer<int, 1, AllocatorTypeTest> Buf(data, allocator);
    std::shared_ptr<sycl::detail::buffer_impl> BufImpl =
        sycl::detail::getSyclObjImpl(Buf);
    CheckBufferDestruction(BufImpl, false);
  }
}

TEST_F(BufferDestructionCheck, BufferWithSharedPtr) {
  sycl::context Context{Plt};
  sycl::queue Q = sycl::queue{Context, sycl::default_selector{}};
  {
    std::shared_ptr<int> InitialVal(new int(5));
    sycl::buffer<int, 1> Buf(InitialVal, 1);
    std::shared_ptr<sycl::detail::buffer_impl> BufImpl =
        sycl::detail::getSyclObjImpl(Buf);
    CheckBufferDestruction(BufImpl, false);
  }
}

TEST_F(BufferDestructionCheck, BufferWithSharedPtrWithNonDefaultAllocator) {
  sycl::context Context{Plt};
  sycl::queue Q = sycl::queue{Context, sycl::default_selector{}};
  {
    std::shared_ptr<int> InitialVal(new int(5));
    using AllocatorTypeTest =
        sycl::usm_allocator<int, sycl::usm::alloc::shared>;
    AllocatorTypeTest allocator(Q);
    sycl::buffer<int, 1, AllocatorTypeTest> Buf(InitialVal, 1, allocator);
    std::shared_ptr<sycl::detail::buffer_impl> BufImpl =
        sycl::detail::getSyclObjImpl(Buf);
    CheckBufferDestruction(BufImpl, false);
  }
}

TEST_F(BufferDestructionCheck, BufferWithSharedPtrArray) {
  sycl::context Context{Plt};
  sycl::queue Q = sycl::queue{Context, sycl::default_selector{}};
  {
    std::shared_ptr<int[]> InitialVal(new int[2]);
    sycl::buffer<int, 1> Buf(InitialVal, 1);
    std::shared_ptr<sycl::detail::buffer_impl> BufImpl =
        sycl::detail::getSyclObjImpl(Buf);
    CheckBufferDestruction(BufImpl, false);
  }
}

TEST_F(BufferDestructionCheck,
       BufferWithSharedPtrArrayWithNonDefaultAllocator) {
  sycl::context Context{Plt};
  sycl::queue Q = sycl::queue{Context, sycl::default_selector{}};
  {
    std::shared_ptr<int[]> InitialVal(new int[2]);
    using AllocatorTypeTest =
        sycl::usm_allocator<int, sycl::usm::alloc::shared>;
    AllocatorTypeTest allocator(Q);
    sycl::buffer<int, 1, AllocatorTypeTest> Buf(InitialVal, 2, allocator);
    std::shared_ptr<sycl::detail::buffer_impl> BufImpl =
        sycl::detail::getSyclObjImpl(Buf);
    CheckBufferDestruction(BufImpl, false);
  }
}

TEST_F(BufferDestructionCheck, BufferWithIterators) {
  sycl::context Context{Plt};
  sycl::queue Q = sycl::queue{Context, sycl::default_selector{}};
  {
    std::vector<int> data{3, 4};
    sycl::buffer<int, 1> Buf(data.begin(), data.end());
    std::shared_ptr<sycl::detail::buffer_impl> BufImpl =
        sycl::detail::getSyclObjImpl(Buf);
    CheckBufferDestruction(BufImpl, true);
  }
}

// TEST_F(BufferDestructionCheck, BufferWithIteratorsWithNonDefaultAllocator) {
//   sycl::context Context{Plt};
//   sycl::queue Q = sycl::queue{Context, sycl::default_selector{}};
//   {
//     std::vector<int> data{3, 4};
//     using AllocatorTypeTest = sycl::usm_allocator<int,
//     sycl::usm::alloc::shared>; AllocatorTypeTest allocator(Q);
//     sycl::buffer<int, 1, AllocatorTypeTest> Buf(data.begin(), data.end(),
//     allocator); std::shared_ptr<sycl::detail::buffer_impl> BufImpl =
//         sycl::detail::getSyclObjImpl(Buf);
//     CheckBufferDestruction(BufImpl, false);
//   }
// }

TEST_F(BufferDestructionCheck, BufferDeferringCheckWriteLock) {
  sycl::context Context{Plt};
  sycl::queue Q = sycl::queue{Context, sycl::default_selector{}};
  {
    testing::Sequence S;
    sycl::detail::buffer_impl *unsafePtr = nullptr;
    EXPECT_EQ(MockSchedulerPtr->MDeferredMemObjRelease.size(), 0u);
    std::unique_lock<std::shared_timed_mutex> Lock(MockSchedulerPtr->MGraphLock,
                                                   std::defer_lock);
    {
      sycl::buffer<int, 1> Buf(1);
      {
        std::shared_ptr<sycl::detail::buffer_impl> BufImpl =
            sycl::detail::getSyclObjImpl(Buf);
        unsafePtr = BufImpl.get();
      }
      Lock.lock();
      // gmock warning will be generated - simply tell gtest that now we do not
      // want to mock the function
      ON_CALL(*MockSchedulerPtr, deferMemObjRelease)
          .WillByDefault(
              [this](const std::shared_ptr<sycl::detail::SYCLMemObjI> &MemObj) {
                return MockSchedulerPtr
                    ->sycl::detail::Scheduler::deferMemObjRelease(MemObj);
              });
    }
    // Record is empty but lock should prevent from being deleted
    ASSERT_EQ(MockSchedulerPtr->MDeferredMemObjRelease.size(), 1u);
    EXPECT_EQ(MockSchedulerPtr->MDeferredMemObjRelease.front().get(),
              unsafePtr);
    Lock.unlock();
    MockSchedulerPtr->releaseResources();

    ASSERT_EQ(MockSchedulerPtr->MDeferredMemObjRelease.size(), 0u);
  }
}

TEST_F(BufferDestructionCheck, BufferDeferringCheckReadLock) {
  sycl::context Context{Plt};
  sycl::queue Q = sycl::queue{Context, sycl::default_selector{}};
  {
    testing::Sequence S;
    EXPECT_EQ(MockSchedulerPtr->MDeferredMemObjRelease.size(), 0u);
    std::shared_lock<std::shared_timed_mutex> Lock(MockSchedulerPtr->MGraphLock,
                                                   std::defer_lock);
    {
      sycl::buffer<int, 1> Buf(1);
      Lock.lock();
      // gmock warning will be generated - simply tell gtest that now we do not
      // want to mock the function
      ON_CALL(*MockSchedulerPtr, deferMemObjRelease)
          .WillByDefault(
              [this](const std::shared_ptr<sycl::detail::SYCLMemObjI> &MemObj) {
                return MockSchedulerPtr
                    ->sycl::detail::Scheduler::deferMemObjRelease(MemObj);
              });
    }
    // Record is empty and read lock do not prevent from being deleted
    ASSERT_EQ(MockSchedulerPtr->MDeferredMemObjRelease.size(), 0u);
    Lock.unlock();
  }
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
  sycl::detail::MemObjRecord *Rec =
      MockSchedulerPtr->MGraphBuilder.getOrInsertMemObjRecord(
          sycl::detail::getSyclObjImpl(Q), &MockReq, AuxCmds);
  MockCommand *ReadCmd = nullptr;
  MockCommand *WriteCmd = nullptr;
  ReadCmd = new MockCommand(sycl::detail::getSyclObjImpl(Q), MockReq);
  ReadCmd->getEvent()->getHandleRef() = reinterpret_cast<sycl::RT::PiEvent>(
      0x01); // just assign to be able to use mock
  WriteCmd = new MockCommand(sycl::detail::getSyclObjImpl(Q), MockReq);
  WriteCmd->getEvent()->getHandleRef() =
      reinterpret_cast<sycl::RT::PiEvent>(0x02);

  std::vector<sycl::detail::Command *> ToCleanUp;
  std::vector<sycl::detail::Command *> ToEnqueue;
  MockSchedulerPtr->MGraphBuilder.addNodeToLeaves(
      Rec, ReadCmd, sycl::access::mode::read, ToEnqueue);
  MockSchedulerPtr->MGraphBuilder.addNodeToLeaves(
      Rec, WriteCmd, sycl::access::mode::write, ToEnqueue);

  Mock.redefine<sycl::detail::PiApiKind::piEventGetInfo>(getEventInfoFunc);
  std::shared_lock<std::shared_timed_mutex> Lock(MockSchedulerPtr->MGraphLock);
  testing::InSequence S;

  ExpectedEventStatus[ReadCmd->getEvent()->getHandleRef()] = PI_EVENT_SUBMITTED;
  ExpectedEventStatus[WriteCmd->getEvent()->getHandleRef()] =
      PI_EVENT_SUBMITTED;

  EXPECT_CALL(*ReadCmd, enqueue).Times(1).RetiresOnSaturation();
  EXPECT_FALSE(MockSchedulerPtr->waitForRecordToFinish(Rec, Lock, false));
  EXPECT_CALL(*ReadCmd, enqueue).Times(0);

  ExpectedEventStatus[ReadCmd->getEvent()->getHandleRef()] = PI_EVENT_COMPLETE;
  ExpectedEventStatus[WriteCmd->getEvent()->getHandleRef()] =
      PI_EVENT_SUBMITTED;

  EXPECT_CALL(*WriteCmd, enqueue).Times(1).RetiresOnSaturation();
  EXPECT_FALSE(MockSchedulerPtr->waitForRecordToFinish(Rec, Lock, false));
  EXPECT_CALL(*WriteCmd, enqueue).Times(0);

  ExpectedEventStatus[ReadCmd->getEvent()->getHandleRef()] = PI_EVENT_COMPLETE;
  ExpectedEventStatus[WriteCmd->getEvent()->getHandleRef()] = PI_EVENT_COMPLETE;
  EXPECT_TRUE(MockSchedulerPtr->waitForRecordToFinish(Rec, Lock, true));
  // previous expect_call is still valid and will generate failure if we recieve
  // call here, no need for extra limitation
}