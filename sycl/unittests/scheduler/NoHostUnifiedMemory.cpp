//==----------- NoHostUnifiedMemory.cpp --- Scheduler unit tests -----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SchedulerTest.hpp"
#include "SchedulerTestUtils.hpp"

#include <helpers/PiMock.hpp>

#include <iostream>
#include <memory>

using namespace cl::sycl;

static pi_result redefinedDeviceGetInfo(pi_device Device,
                                        pi_device_info ParamName,
                                        size_t ParamValueSize, void *ParamValue,
                                        size_t *ParamValueSizeRet) {
  if (ParamName == PI_DEVICE_INFO_HOST_UNIFIED_MEMORY) {
    auto *Result = reinterpret_cast<pi_bool *>(ParamValue);
    *Result = false;
  } else if (ParamName == PI_DEVICE_INFO_TYPE) {
    auto *Result = reinterpret_cast<_pi_device_type *>(ParamValue);
    *Result = PI_DEVICE_TYPE_CPU;
  }
  return PI_SUCCESS;
}

static RT::PiMemFlags ExpectedMemObjFlags;

static pi_result
redefinedMemBufferCreate(pi_context context, pi_mem_flags flags, size_t size,
                         void *host_ptr, pi_mem *ret_mem,
                         const pi_mem_properties *properties = nullptr) {
  EXPECT_EQ(flags, ExpectedMemObjFlags);
  return PI_SUCCESS;
}

static pi_result redefinedEnqueueMemBufferReadRect(
    pi_queue command_queue, pi_mem buffer, pi_bool blocking_read,
    pi_buff_rect_offset buffer_offset, pi_buff_rect_offset host_offset,
    pi_buff_rect_region region, size_t buffer_row_pitch,
    size_t buffer_slice_pitch, size_t host_row_pitch, size_t host_slice_pitch,
    void *ptr, pi_uint32 num_events_in_wait_list,
    const pi_event *event_wait_list, pi_event *event) {
  return PI_SUCCESS;
}

static pi_result redefinedEnqueueMemBufferWriteRect(
    pi_queue command_queue, pi_mem buffer, pi_bool blocking_write,
    pi_buff_rect_offset buffer_offset, pi_buff_rect_offset host_offset,
    pi_buff_rect_region region, size_t buffer_row_pitch,
    size_t buffer_slice_pitch, size_t host_row_pitch, size_t host_slice_pitch,
    const void *ptr, pi_uint32 num_events_in_wait_list,
    const pi_event *event_wait_list, pi_event *event) {
  return PI_SUCCESS;
}

static pi_result redefinedMemRelease(pi_mem mem) { return PI_SUCCESS; }

TEST_F(SchedulerTest, NoHostUnifiedMemory) {
  platform Plt{default_selector()};
  if (Plt.is_host()) {
    std::cout << "Not run due to host-only environment\n";
    return;
  }

  queue Q;
  unittest::PiMock Mock{Q};
  Mock.redefine<detail::PiApiKind::piDeviceGetInfo>(redefinedDeviceGetInfo);
  Mock.redefine<detail::PiApiKind::piMemBufferCreate>(redefinedMemBufferCreate);
  Mock.redefine<detail::PiApiKind::piEnqueueMemBufferReadRect>(
      redefinedEnqueueMemBufferReadRect);
  Mock.redefine<detail::PiApiKind::piEnqueueMemBufferWriteRect>(
      redefinedEnqueueMemBufferWriteRect);
  Mock.redefine<detail::PiApiKind::piMemRelease>(redefinedMemRelease);
  cl::sycl::detail::QueueImplPtr QImpl = detail::getSyclObjImpl(Q);

  device HostDevice;
  std::shared_ptr<detail::queue_impl> DefaultHostQueue{
      new detail::queue_impl(detail::getSyclObjImpl(HostDevice), {}, {})};

  MockScheduler MS;
  // Check non-host -> host alloca with non-discard access mode
  {
    int val;
    buffer<int, 1> Buf(&val, range<1>(1));
    detail::Requirement Req = getMockRequirement(Buf);

    // The host pointer should be copied during the non-host allocation in this
    // case.
    ExpectedMemObjFlags = PI_MEM_FLAGS_ACCESS_RW | PI_MEM_FLAGS_HOST_PTR_COPY;

    detail::MemObjRecord *Record = MS.getOrInsertMemObjRecord(QImpl, &Req);
    detail::AllocaCommandBase *NonHostAllocaCmd =
        MS.getOrCreateAllocaForReq(Record, &Req, QImpl);

    detail::AllocaCommandBase *HostAllocaCmd =
        MS.getOrCreateAllocaForReq(Record, &Req, DefaultHostQueue);
    EXPECT_TRUE(!HostAllocaCmd->MLinkedAllocaCmd);
    EXPECT_TRUE(!NonHostAllocaCmd->MLinkedAllocaCmd);

    detail::Command *MemoryMove =
        MS.insertMemoryMove(Record, &Req, DefaultHostQueue);
    EXPECT_EQ(MemoryMove->getType(), detail::Command::COPY_MEMORY);
  }
  // Check non-host -> host alloca with discard access modes
  {
    int val;
    buffer<int, 1> Buf(&val, range<1>(1));
    detail::Requirement Req = getMockRequirement(Buf);
    // The host pointer should be ignored due to the discard access mode.
    ExpectedMemObjFlags = PI_MEM_FLAGS_ACCESS_RW;

    detail::Requirement DiscardReq = getMockRequirement(Buf);
    DiscardReq.MAccessMode = access::mode::discard_read_write;

    detail::MemObjRecord *Record = MS.getOrInsertMemObjRecord(QImpl, &Req);
    MS.getOrCreateAllocaForReq(Record, &DiscardReq, QImpl);
  }
  // Check host -> non-host alloca
  {
    int val;
    buffer<int, 1> Buf(&val, range<1>(1));
    detail::Requirement Req = getMockRequirement(Buf);

    // No copy expected during the second allocation, it is performed as a
    // separate command.
    ExpectedMemObjFlags = PI_MEM_FLAGS_ACCESS_RW;

    detail::MemObjRecord *Record =
        MS.getOrInsertMemObjRecord(DefaultHostQueue, &Req);
    detail::AllocaCommandBase *HostAllocaCmd =
        MS.getOrCreateAllocaForReq(Record, &Req, DefaultHostQueue);
    detail::AllocaCommandBase *NonHostAllocaCmd =
        MS.getOrCreateAllocaForReq(Record, &Req, QImpl);
    EXPECT_TRUE(!HostAllocaCmd->MLinkedAllocaCmd);
    EXPECT_TRUE(!NonHostAllocaCmd->MLinkedAllocaCmd);

    detail::Command *MemoryMove = MS.insertMemoryMove(Record, &Req, QImpl);
    EXPECT_EQ(MemoryMove->getType(), detail::Command::COPY_MEMORY);
  }
  // Check that memory movement operations work correctly with/after discard
  // access modes.
  {
    int val;
    buffer<int, 1> Buf(&val, range<1>(1));
    detail::Requirement Req = getMockRequirement(Buf);
    ExpectedMemObjFlags = PI_MEM_FLAGS_ACCESS_RW | PI_MEM_FLAGS_HOST_PTR_COPY;

    detail::Requirement DiscardReq = getMockRequirement(Buf);
    DiscardReq.MAccessMode = access::mode::discard_read_write;

    detail::MemObjRecord *Record = MS.getOrInsertMemObjRecord(QImpl, &Req);
    MS.getOrCreateAllocaForReq(Record, &Req, QImpl);
    MS.getOrCreateAllocaForReq(Record, &Req, DefaultHostQueue);

    // Memory movement operations should be omitted for discard access modes.
    detail::Command *MemoryMove =
        MS.insertMemoryMove(Record, &DiscardReq, DefaultHostQueue);
    EXPECT_EQ(MemoryMove, nullptr);
    // The current context for the record should still be modified.
    EXPECT_EQ(Record->MCurContext, DefaultHostQueue->getContextImplPtr());
  }
}
