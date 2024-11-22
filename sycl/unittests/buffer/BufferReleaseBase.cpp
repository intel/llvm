//==- BufferReleaseBase.cpp --- check delayed destruction of buffer --------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "BufferReleaseBase.hpp"
#include "gmock/gmock.h"

class BufferDestructionCheck
    : public BufferDestructionCheckCommon<sycl::backend::opencl> {};

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

std::map<ur_event_handle_t, ur_event_status_t> ExpectedEventStatus;
ur_result_t replaceEventGetInfo(void *pParams) {
  auto params = *reinterpret_cast<ur_event_get_info_params_t *>(pParams);
  EXPECT_EQ(*params.ppropName, UR_EVENT_INFO_COMMAND_EXECUTION_STATUS)
      << "Unknown param name";
  // could not use assert here
  EXPECT_EQ(*params.ppropSize, 4u);
  auto it = ExpectedEventStatus.find(*params.phEvent);
  if (it != ExpectedEventStatus.end()) {
    *(static_cast<int32_t *>(*params.ppPropValue)) = it->second;
    return UR_RESULT_SUCCESS;
  } else
    return UR_RESULT_ERROR_INVALID_OPERATION;
}

TEST_F(BufferDestructionCheck, ReadyToReleaseLogic) {
  sycl::context Context{Plt};
  sycl::queue Q = sycl::queue{Context, sycl::default_selector{}};

  sycl::buffer<int, 1> Buf(1);
  sycl::detail::Requirement MockReq = getMockRequirement(Buf);
  sycl::detail::MemObjRecord *Rec = MockSchedulerPtr->getOrInsertMemObjRecord(
      sycl::detail::getSyclObjImpl(Q), &MockReq);

  std::shared_ptr<sycl::detail::context_impl> CtxImpl =
      sycl::detail::getSyclObjImpl(Context);
  MockCmdWithReleaseTracking *ReadCmd = nullptr;
  MockCmdWithReleaseTracking *WriteCmd = nullptr;
  ReadCmd =
      new MockCmdWithReleaseTracking(sycl::detail::getSyclObjImpl(Q), MockReq);
  // These dummy handles are automatically cleaned up by the runtime
  ReadCmd->getEvent()->setHandle(reinterpret_cast<ur_event_handle_t>(
      mock::createDummyHandle<ur_event_handle_t>()));
  WriteCmd =
      new MockCmdWithReleaseTracking(sycl::detail::getSyclObjImpl(Q), MockReq);
  WriteCmd->getEvent()->setHandle(reinterpret_cast<ur_event_handle_t>(
      mock::createDummyHandle<ur_event_handle_t>()));
  ReadCmd->MEnqueueStatus = sycl::detail::EnqueueResultT::SyclEnqueueSuccess;
  WriteCmd->MEnqueueStatus = sycl::detail::EnqueueResultT::SyclEnqueueSuccess;

  std::vector<sycl::detail::Command *> ToCleanUp;
  std::vector<sycl::detail::Command *> ToEnqueue;
  MockSchedulerPtr->addNodeToLeaves(Rec, ReadCmd, sycl::access::mode::read,
                                    ToEnqueue);
  MockSchedulerPtr->addNodeToLeaves(Rec, WriteCmd, sycl::access::mode::write,
                                    ToEnqueue);

  mock::getCallbacks().set_replace_callback("urEventGetInfo",
                                            &replaceEventGetInfo);
  testing::InSequence S;

  ExpectedEventStatus[ReadCmd->getEvent()->getHandle()] =
      UR_EVENT_STATUS_SUBMITTED;
  ExpectedEventStatus[WriteCmd->getEvent()->getHandle()] =
      UR_EVENT_STATUS_SUBMITTED;

  EXPECT_FALSE(MockSchedulerPtr->checkLeavesCompletion(Rec));

  ExpectedEventStatus[ReadCmd->getEvent()->getHandle()] =
      UR_EVENT_STATUS_COMPLETE;
  ExpectedEventStatus[WriteCmd->getEvent()->getHandle()] =
      UR_EVENT_STATUS_SUBMITTED;

  EXPECT_FALSE(MockSchedulerPtr->checkLeavesCompletion(Rec));

  ExpectedEventStatus[ReadCmd->getEvent()->getHandle()] =
      UR_EVENT_STATUS_COMPLETE;
  ExpectedEventStatus[WriteCmd->getEvent()->getHandle()] =
      UR_EVENT_STATUS_COMPLETE;
  EXPECT_TRUE(MockSchedulerPtr->checkLeavesCompletion(Rec));
  // previous expect_call is still valid and will generate failure if we recieve
  // call here, no need for extra limitation
  EXPECT_CALL(*ReadCmd, Release).Times(1);
  EXPECT_CALL(*WriteCmd, Release).Times(1);
}
