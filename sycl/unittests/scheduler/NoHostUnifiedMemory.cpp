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

#include <detail/buffer_impl.hpp>

#include <iostream>
#include <memory>

using namespace sycl;

static pi_result redefinedDeviceGetInfoAfter(pi_device Device,
                                             pi_device_info ParamName,
                                             size_t ParamValueSize,
                                             void *ParamValue,
                                             size_t *ParamValueSizeRet) {
  if (ParamName == PI_DEVICE_INFO_HOST_UNIFIED_MEMORY) {
    auto *Result = reinterpret_cast<pi_bool *>(ParamValue);
    *Result = false;
  } else if (ParamName == PI_DEVICE_INFO_TYPE) {
    auto *Result = reinterpret_cast<_pi_device_type *>(ParamValue);
    *Result = PI_DEVICE_TYPE_CPU;
  }

  // This mock device has no sub-devices
  if (ParamName == PI_DEVICE_INFO_PARTITION_PROPERTIES) {
    if (ParamValueSizeRet) {
      *ParamValueSizeRet = 0;
    }
  }
  if (ParamName == PI_DEVICE_INFO_PARTITION_AFFINITY_DOMAIN) {
    assert(ParamValueSize == sizeof(pi_device_affinity_domain));
    if (ParamValue) {
      *static_cast<pi_device_affinity_domain *>(ParamValue) = 0;
    }
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

static pi_context InteropPiContext = nullptr;
static pi_result redefinedMemGetInfoAfter(pi_mem mem, pi_mem_info param_name,
                                          size_t param_value_size,
                                          void *param_value,
                                          size_t *param_value_size_ret) {
  auto *Result = reinterpret_cast<pi_context *>(param_value);
  *Result = InteropPiContext;
  return PI_SUCCESS;

  if (param_name == PI_MEM_CONTEXT) {
    auto *Result = reinterpret_cast<pi_context *>(param_value);
    *Result = InteropPiContext;
  } else if (param_name == PI_MEM_SIZE) {
    auto *Result = reinterpret_cast<size_t *>(param_value);
    *Result = 8;
  }
}
static pi_result
redefinedMemCreateWithNativeHandle(pi_native_handle native_handle,
                                   pi_context context, bool own_native_handle,
                                   pi_mem *mem) {
  *mem = detail::pi::cast<pi_mem>(native_handle);
  return PI_SUCCESS;
}

TEST_F(SchedulerTest, NoHostUnifiedMemory) {
  unittest::PiMock Mock;
  queue Q{Mock.getPlatform().get_devices()[0]};
  Mock.redefineAfter<detail::PiApiKind::piDeviceGetInfo>(
      redefinedDeviceGetInfoAfter);
  Mock.redefineBefore<detail::PiApiKind::piMemBufferCreate>(
      redefinedMemBufferCreate);
  Mock.redefineAfter<detail::PiApiKind::piMemGetInfo>(redefinedMemGetInfoAfter);
  Mock.redefineBefore<detail::PiApiKind::piextMemCreateWithNativeHandle>(
      redefinedMemCreateWithNativeHandle);
  sycl::detail::QueueImplPtr QImpl = detail::getSyclObjImpl(Q);

  MockScheduler MS;
  // Check non-host alloca with non-discard access mode
  {
    int val;
    buffer<int, 1> Buf(&val, range<1>(1));
    detail::Requirement Req = getMockRequirement(Buf);

    detail::MemObjRecord *Record = MS.getOrInsertMemObjRecord(QImpl, &Req);
    std::vector<detail::Command *> AuxCmds;
    detail::AllocaCommandBase *NonHostAllocaCmd =
        MS.getOrCreateAllocaForReq(Record, &Req, QImpl, AuxCmds);

    // Both non-host and host allocations should be created in this case in
    // order to perform a memory move.
    EXPECT_EQ(Record->MAllocaCommands.size(), 2U);
    detail::AllocaCommandBase *HostAllocaCmd = Record->MAllocaCommands[0];
    EXPECT_TRUE(HostAllocaCmd->getQueue() == nullptr);
    EXPECT_TRUE(!HostAllocaCmd->MLinkedAllocaCmd);
    EXPECT_TRUE(!NonHostAllocaCmd->MLinkedAllocaCmd);
    EXPECT_TRUE(Record->MCurContext == nullptr);

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
    detail::MemObjRecord *Record = MS.getOrInsertMemObjRecord(QImpl, &Req);
    std::vector<detail::Command *> AuxCmds;
    MS.getOrCreateAllocaForReq(Record, &DiscardReq, QImpl, AuxCmds);
    EXPECT_EQ(Record->MAllocaCommands.size(), 1U);
  }
  // Check non-host alloca without user pointer
  {
    buffer<int, 1> Buf(range<1>(1));
    detail::Requirement Req = getMockRequirement(Buf);

    // No need to create a host allocation in this case since there's no data to
    // initialize the buffer with.
    detail::MemObjRecord *Record = MS.getOrInsertMemObjRecord(QImpl, &Req);
    std::vector<detail::Command *> AuxCmds;
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
    detail::MemObjRecord *Record = MS.getOrInsertMemObjRecord(nullptr, &Req);
    std::vector<detail::Command *> AuxCmds;
    detail::AllocaCommandBase *HostAllocaCmd =
        MS.getOrCreateAllocaForReq(Record, &Req, nullptr, AuxCmds);
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

    detail::MemObjRecord *Record = MS.getOrInsertMemObjRecord(QImpl, &Req);
    std::vector<detail::Command *> AuxCmds;
    MS.getOrCreateAllocaForReq(Record, &Req, QImpl, AuxCmds);
    MS.getOrCreateAllocaForReq(Record, &Req, nullptr, AuxCmds);

    // Memory movement operations should be omitted for discard access modes.
    detail::Command *MemoryMove =
        MS.insertMemoryMove(Record, &DiscardReq, nullptr, AuxCmds);
    EXPECT_TRUE(MemoryMove == nullptr);
    // The current context for the record should still be modified.
    EXPECT_EQ(Record->MCurContext, nullptr);
  }
  // Check that interoperability memory objects are initialized.
  {
    pi_mem MockInteropBuffer = nullptr;
    pi_result PIRes = mock_piMemBufferCreate(
        /*pi_context=*/0x0, /*pi_mem_flags=*/PI_MEM_FLAGS_ACCESS_RW, /*size=*/1,
        /*host_ptr=*/nullptr, &MockInteropBuffer);
    EXPECT_TRUE(PI_SUCCESS == PIRes);

    context InteropContext = Q.get_context();
    InteropPiContext = detail::getSyclObjImpl(InteropContext)->getHandleRef();
    auto BufI = std::make_shared<detail::buffer_impl>(
        detail::pi::cast<pi_native_handle>(MockInteropBuffer), Q.get_context(),
        std::make_unique<
            detail::SYCLMemObjAllocatorHolder<buffer_allocator<char>, char>>(),
        /* OwnNativeHandle */ true, event());

    detail::Requirement Req = getMockRequirement();
    Req.MSYCLMemObj = BufI.get();

    detail::MemObjRecord *Record = MS.getOrInsertMemObjRecord(QImpl, &Req);
    std::vector<detail::Command *> AuxCmds;
    detail::AllocaCommandBase *InteropAlloca =
        MS.getOrCreateAllocaForReq(Record, &Req, QImpl, AuxCmds);
    detail::EnqueueResultT Res;
    MockScheduler::enqueueCommand(InteropAlloca, Res, detail::BLOCKING);

    EXPECT_EQ(Record->MAllocaCommands.size(), 1U);
    EXPECT_EQ(InteropAlloca->MMemAllocation, MockInteropBuffer);
  }
}
