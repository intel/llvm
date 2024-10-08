//==----------- NoHostUnifiedMemory.cpp --- Scheduler unit tests -----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SchedulerTest.hpp"
#include "SchedulerTestUtils.hpp"
#include "ur_mock_helpers.hpp"

#include <helpers/UrMock.hpp>

#include <detail/buffer_impl.hpp>

#include <iostream>
#include <memory>

using namespace sycl;

static ur_result_t redefinedDeviceGetInfoAfter(void *pParams) {
  auto params = *static_cast<ur_device_get_info_params_t *>(pParams);
  if (*params.ppropName == UR_DEVICE_INFO_HOST_UNIFIED_MEMORY) {
    auto *Result = reinterpret_cast<ur_bool_t *>(*params.ppPropValue);
    *Result = false;
  } else if (*params.ppropName == UR_DEVICE_INFO_TYPE) {
    auto *Result = reinterpret_cast<ur_device_type_t *>(*params.ppPropValue);
    *Result = UR_DEVICE_TYPE_CPU;
  }

  // This mock device has no sub-devices
  if (*params.ppropName == UR_DEVICE_INFO_SUPPORTED_PARTITIONS) {
    if (*params.ppPropSizeRet) {
      **params.ppPropSizeRet = 0;
    }
  }
  if (*params.ppropName == UR_DEVICE_INFO_PARTITION_AFFINITY_DOMAIN) {
    assert(*params.ppropSize == sizeof(ur_device_affinity_domain_flags_t));
    if (*params.ppPropValue) {
      *static_cast<ur_device_affinity_domain_flags_t *>(*params.ppPropValue) =
          0;
    }
  }
  return UR_RESULT_SUCCESS;
}

static ur_result_t redefinedMemBufferCreate(void *pParams) {
  auto params = *static_cast<ur_mem_buffer_create_params_t *>(pParams);
  EXPECT_EQ(*params.pflags, UR_MEM_FLAG_READ_WRITE);
  return UR_RESULT_SUCCESS;
}

static ur_context_handle_t InteropUrContext = nullptr;

static ur_result_t redefinedMemGetInfoAfter(void *pParams) {
  auto params = *static_cast<ur_mem_get_info_params_t *>(pParams);
  auto *Result = reinterpret_cast<ur_context_handle_t *>(*params.ppPropValue);
  *Result = InteropUrContext;
  return UR_RESULT_SUCCESS;

  if (*params.ppropName == UR_MEM_INFO_CONTEXT) {
    auto *Result = reinterpret_cast<ur_context_handle_t *>(*params.ppPropValue);
    *Result = InteropUrContext;
  } else if (*params.ppropName == UR_MEM_INFO_SIZE) {
    auto *Result = reinterpret_cast<size_t *>(*params.ppPropValue);
    *Result = 8;
  }
}

static ur_result_t redefinedMemCreateWithNativeHandle(void *pParams) {
  auto params =
      *static_cast<ur_mem_buffer_create_with_native_handle_params_t *>(pParams);
  **params.pphMem = detail::ur::cast<ur_mem_handle_t>(*params.phNativeMem);
  return UR_RESULT_SUCCESS;
}

TEST_F(SchedulerTest, NoHostUnifiedMemory) {
  unittest::UrMock<> Mock;
  queue Q{sycl::platform().get_devices()[0]};
  mock::getCallbacks().set_after_callback("urDeviceGetInfo",
                                          &redefinedDeviceGetInfoAfter);
  mock::getCallbacks().set_before_callback("urMemBufferCreate",
                                           &redefinedMemBufferCreate);
  mock::getCallbacks().set_after_callback("urMemGetInfo",
                                          &redefinedMemGetInfoAfter);
  mock::getCallbacks().set_before_callback("urMemBufferCreateWithNativeHandle",
                                           &redefinedMemCreateWithNativeHandle);
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
    ur_mem_handle_t MockInteropBuffer =
        mock::createDummyHandle<ur_mem_handle_t>();

    context InteropContext = Q.get_context();
    InteropUrContext = detail::getSyclObjImpl(InteropContext)->getHandleRef();
    auto BufI = std::make_shared<detail::buffer_impl>(
        detail::ur::cast<ur_native_handle_t>(MockInteropBuffer),
        Q.get_context(),
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
