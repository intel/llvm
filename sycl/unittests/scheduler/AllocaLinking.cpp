//==-------------- AllocaLinking.cpp --- Scheduler unit tests --------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SchedulerTest.hpp"
#include "SchedulerTestUtils.hpp"

#include <helpers/UrMock.hpp>

#include <iostream>

using namespace sycl;

static bool HostUnifiedMemory = false;

static ur_result_t redefinedDeviceGetInfoAfter(void *pParams) {
  auto params = *static_cast<ur_device_get_info_params_t *>(pParams);
  if (*params.ppropName == UR_DEVICE_INFO_HOST_UNIFIED_MEMORY) {
    auto *Result = reinterpret_cast<ur_bool_t *>(*params.ppPropValue);
    *Result = HostUnifiedMemory;
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
    assert(**params.ppPropSizeRet == sizeof(ur_device_affinity_domain_flags_t));
    if (*params.ppPropValue) {
      *static_cast<ur_device_affinity_domain_flags_t *>(*params.ppPropValue) =
          0;
    }
  }
  return UR_RESULT_SUCCESS;
}

TEST_F(SchedulerTest, AllocaLinking) {
  HostUnifiedMemory = false;

  sycl::unittest::UrMock<> Mock;
  sycl::queue Q{sycl::platform().get_devices()[0]};
  mock::getCallbacks().set_after_callback("urDeviceGetInfo",
                                          &redefinedDeviceGetInfoAfter);
  sycl::detail::QueueImplPtr QImpl = detail::getSyclObjImpl(Q);

  MockScheduler MS;
  // Should not be linked w/o host unified memory or pinned host memory
  {
    buffer<int, 1> Buf(range<1>(1));
    detail::Requirement Req = getMockRequirement(Buf);

    detail::MemObjRecord *Record = MS.getOrInsertMemObjRecord(QImpl, &Req);
    std::vector<detail::Command *> AuxCmds;
    detail::AllocaCommandBase *NonHostAllocaCmd =
        MS.getOrCreateAllocaForReq(Record, &Req, QImpl, AuxCmds);
    detail::AllocaCommandBase *HostAllocaCmd =
        MS.getOrCreateAllocaForReq(Record, &Req, nullptr, AuxCmds);

    EXPECT_FALSE(HostAllocaCmd->MLinkedAllocaCmd);
    EXPECT_FALSE(NonHostAllocaCmd->MLinkedAllocaCmd);
  }
  // Should be linked in case of pinned host memory
  {
    buffer<int, 1> Buf(
        range<1>(1), {ext::oneapi::property::buffer::use_pinned_host_memory()});
    detail::Requirement Req = getMockRequirement(Buf);

    detail::MemObjRecord *Record = MS.getOrInsertMemObjRecord(QImpl, &Req);
    std::vector<detail::Command *> AuxCmds;
    detail::AllocaCommandBase *NonHostAllocaCmd =
        MS.getOrCreateAllocaForReq(Record, &Req, QImpl, AuxCmds);
    detail::AllocaCommandBase *HostAllocaCmd =
        MS.getOrCreateAllocaForReq(Record, &Req, nullptr, AuxCmds);

    EXPECT_EQ(HostAllocaCmd->MLinkedAllocaCmd, NonHostAllocaCmd);
    EXPECT_EQ(NonHostAllocaCmd->MLinkedAllocaCmd, HostAllocaCmd);
  }
  // Should be linked in case of host unified memory
  {
    HostUnifiedMemory = true;
    buffer<int, 1> Buf(range<1>(1));
    detail::Requirement Req = getMockRequirement(Buf);

    detail::MemObjRecord *Record = MS.getOrInsertMemObjRecord(QImpl, &Req);
    std::vector<detail::Command *> AuxCmds;
    detail::AllocaCommandBase *NonHostAllocaCmd =
        MS.getOrCreateAllocaForReq(Record, &Req, QImpl, AuxCmds);
    detail::AllocaCommandBase *HostAllocaCmd =
        MS.getOrCreateAllocaForReq(Record, &Req, nullptr, AuxCmds);

    EXPECT_EQ(HostAllocaCmd->MLinkedAllocaCmd, NonHostAllocaCmd);
    EXPECT_EQ(NonHostAllocaCmd->MLinkedAllocaCmd, HostAllocaCmd);
  }
}
