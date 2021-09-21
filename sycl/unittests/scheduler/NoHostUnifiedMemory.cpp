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

static pi_result
redefinedMemBufferCreate(pi_context context, pi_mem_flags flags, size_t size,
                         void *host_ptr, pi_mem *ret_mem,
                         const pi_mem_properties *properties = nullptr) {
  EXPECT_EQ(flags, PI_MEM_FLAGS_ACCESS_RW);
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

static pi_result redefinedMemRetain(pi_mem mem) { return PI_SUCCESS; }
static pi_result redefinedMemRelease(pi_mem mem) { return PI_SUCCESS; }

static pi_context InteropPiContext = nullptr;
static pi_result redefinedMemGetInfo(pi_mem mem, cl_mem_info param_name,
                                     size_t param_value_size, void *param_value,
                                     size_t *param_value_size_ret) {
  EXPECT_EQ(param_name, static_cast<cl_mem_info>(CL_MEM_CONTEXT));
  auto *Result = reinterpret_cast<pi_context *>(param_value);
  *Result = InteropPiContext;
  return PI_SUCCESS;
}

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
  Mock.redefine<detail::PiApiKind::piMemRetain>(redefinedMemRetain);
  Mock.redefine<detail::PiApiKind::piMemRelease>(redefinedMemRelease);
  Mock.redefine<detail::PiApiKind::piMemGetInfo>(redefinedMemGetInfo);
  cl::sycl::detail::QueueImplPtr QImpl = detail::getSyclObjImpl(Q);

  device HostDevice;
  std::shared_ptr<detail::queue_impl> DefaultHostQueue{
      new detail::queue_impl(detail::getSyclObjImpl(HostDevice), {}, {})};

  MockScheduler MS;
  // Check non-host alloca with non-discard access mode
  {
    int val;
    buffer<int, 1> Buf(&val, range<1>(1));
    detail::Requirement Req = getMockRequirement(Buf);

    std::vector<detail::Command *> AuxCmds;
    detail::MemObjRecord *Record =
        MS.getOrInsertMemObjRecord(QImpl, &Req, AuxCmds);
    detail::AllocaCommandBase *NonHostAllocaCmd =
        MS.getOrCreateAllocaForReq(Record, &Req, QImpl, AuxCmds);

    // Both non-host and host allocations should be created in this case in
    // order to perform a memory move.
    EXPECT_EQ(Record->MAllocaCommands.size(), 2U);
    detail::AllocaCommandBase *HostAllocaCmd = Record->MAllocaCommands[0];
    EXPECT_TRUE(HostAllocaCmd->getQueue()->is_host());
    EXPECT_TRUE(!HostAllocaCmd->MLinkedAllocaCmd);
    EXPECT_TRUE(!NonHostAllocaCmd->MLinkedAllocaCmd);
    EXPECT_TRUE(Record->MCurContext->is_host());

    detail::Command *MemoryMove =
        MS.insertMemoryMove(Record, &Req, QImpl, AuxCmds);
    EXPECT_EQ(MemoryMove->getType(), detail::Command::COPY_MEMORY);
  }
  // Check non-host alloca with discard access modes
  {
    int val;
    buffer<int, 1> Buf(&val, range<1>(1));
    detail::Requirement Req = getMockRequirement(Buf);

    detail::Requirement DiscardReq = getMockRequirement(Buf);
    DiscardReq.MAccessMode = access::mode::discard_read_write;

    // No need to create a host allocation in this case since the data can be
    // discarded.
    std::vector<detail::Command *> AuxCmds;
    detail::MemObjRecord *Record =
        MS.getOrInsertMemObjRecord(QImpl, &Req, AuxCmds);
    MS.getOrCreateAllocaForReq(Record, &DiscardReq, QImpl, AuxCmds);
    EXPECT_EQ(Record->MAllocaCommands.size(), 1U);
  }
  // Check non-host alloca without user pointer
  {
    buffer<int, 1> Buf(range<1>(1));
    detail::Requirement Req = getMockRequirement(Buf);

    // No need to create a host allocation in this case since there's no data to
    // initialize the buffer with.
    std::vector<detail::Command *> AuxCmds;
    detail::MemObjRecord *Record =
        MS.getOrInsertMemObjRecord(QImpl, &Req, AuxCmds);
    MS.getOrCreateAllocaForReq(Record, &Req, QImpl, AuxCmds);
    EXPECT_EQ(Record->MAllocaCommands.size(), 1U);
  }
  // Check host -> non-host alloca
  {
    int val;
    buffer<int, 1> Buf(&val, range<1>(1));
    detail::Requirement Req = getMockRequirement(Buf);

    // No special handling required: alloca commands are created one after
    // another and the transfer is done via a write operation.
    std::vector<detail::Command *> AuxCmds;
    detail::MemObjRecord *Record =
        MS.getOrInsertMemObjRecord(DefaultHostQueue, &Req, AuxCmds);
    detail::AllocaCommandBase *HostAllocaCmd =
        MS.getOrCreateAllocaForReq(Record, &Req, DefaultHostQueue, AuxCmds);
    EXPECT_EQ(Record->MAllocaCommands.size(), 1U);
    detail::AllocaCommandBase *NonHostAllocaCmd =
        MS.getOrCreateAllocaForReq(Record, &Req, QImpl, AuxCmds);
    EXPECT_EQ(Record->MAllocaCommands.size(), 2U);
    EXPECT_TRUE(!HostAllocaCmd->MLinkedAllocaCmd);
    EXPECT_TRUE(!NonHostAllocaCmd->MLinkedAllocaCmd);

    detail::Command *MemoryMove =
        MS.insertMemoryMove(Record, &Req, QImpl, AuxCmds);
    EXPECT_EQ(MemoryMove->getType(), detail::Command::COPY_MEMORY);
  }
  // Check that memory movement operations work correctly with/after discard
  // access modes.
  {
    int val;
    buffer<int, 1> Buf(&val, range<1>(1));
    detail::Requirement Req = getMockRequirement(Buf);

    detail::Requirement DiscardReq = getMockRequirement(Buf);
    DiscardReq.MAccessMode = access::mode::discard_read_write;

    std::vector<detail::Command *> AuxCmds;
    detail::MemObjRecord *Record =
        MS.getOrInsertMemObjRecord(QImpl, &Req, AuxCmds);
    MS.getOrCreateAllocaForReq(Record, &Req, QImpl, AuxCmds);
    MS.getOrCreateAllocaForReq(Record, &Req, DefaultHostQueue, AuxCmds);

    // Memory movement operations should be omitted for discard access modes.
    detail::Command *MemoryMove =
        MS.insertMemoryMove(Record, &DiscardReq, DefaultHostQueue, AuxCmds);
    EXPECT_TRUE(MemoryMove == nullptr);
    // The current context for the record should still be modified.
    EXPECT_EQ(Record->MCurContext, DefaultHostQueue->getContextImplPtr());
  }
  // Check that interoperability memory objects are initialized.
  {
    cl_mem MockInteropBuffer = reinterpret_cast<cl_mem>(1);
    context InteropContext = Q.get_context();
    InteropPiContext = detail::getSyclObjImpl(InteropContext)->getHandleRef();
    std::shared_ptr<detail::buffer_impl> BufI = std::make_shared<
        detail::buffer_impl>(
        MockInteropBuffer, Q.get_context(), /*BufSize*/ 8,
        make_unique_ptr<detail::SYCLMemObjAllocatorHolder<buffer_allocator>>(),
        event());

    detail::Requirement Req = getMockRequirement();
    Req.MSYCLMemObj = BufI.get();
    std::vector<detail::Command *> AuxCmds;
    detail::MemObjRecord *Record =
        MS.getOrInsertMemObjRecord(QImpl, &Req, AuxCmds);
    detail::AllocaCommandBase *InteropAlloca =
        MS.getOrCreateAllocaForReq(Record, &Req, QImpl, AuxCmds);
    detail::EnqueueResultT Res;
    MockScheduler::enqueueCommand(InteropAlloca, Res, detail::BLOCKING);

    EXPECT_EQ(Record->MAllocaCommands.size(), 1U);
    EXPECT_EQ(InteropAlloca->MMemAllocation, MockInteropBuffer);
  }
}
