//==- BufferReleaseBase.cpp --- check delayed destruction of buffer --------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "BufferReleaseBase.hpp"

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

pi_device GlobalDeviceHandle(createDummyHandle<pi_device>());

inline pi_result customMockDevicesGet(pi_platform platform,
                                      pi_device_type device_type,
                                      pi_uint32 num_entries, pi_device *devices,
                                      pi_uint32 *num_devices) {
  if (num_devices)
    *num_devices = 1;

  if (devices && num_entries > 0)
    devices[0] = GlobalDeviceHandle;

  return PI_SUCCESS;
}

inline pi_result customMockContextGetInfo(pi_context context,
                                          pi_context_info param_name,
                                          size_t param_value_size,
                                          void *param_value,
                                          size_t *param_value_size_ret) {
  switch (param_name) {
  case PI_CONTEXT_INFO_NUM_DEVICES: {
    if (param_value)
      *static_cast<pi_uint32 *>(param_value) = 1;
    if (param_value_size_ret)
      *param_value_size_ret = sizeof(pi_uint32);
    return PI_SUCCESS;
  }
  case PI_CONTEXT_INFO_DEVICES: {
    if (param_value)
      *static_cast<pi_device *>(param_value) = GlobalDeviceHandle;
    if (param_value_size_ret)
      *param_value_size_ret = sizeof(GlobalDeviceHandle);
    break;
  }
  default:;
  }
  return PI_SUCCESS;
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
  sycl::detail::MemObjRecord *Rec = MockSchedulerPtr->getOrInsertMemObjRecord(
      sycl::detail::getSyclObjImpl(Q), &MockReq);

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
