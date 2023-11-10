//==-------------- AllocaLinking.cpp --- Scheduler unit tests --------------==//
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

using namespace sycl;

static bool HostUnifiedMemory = false;

static pi_result redefinedDeviceGetInfoAfter(pi_device Device,
                                             pi_device_info ParamName,
                                             size_t ParamValueSize,
                                             void *ParamValue,
                                             size_t *ParamValueSizeRet) {
  if (ParamName == PI_DEVICE_INFO_HOST_UNIFIED_MEMORY) {
    auto *Result = reinterpret_cast<pi_bool *>(ParamValue);
    *Result = HostUnifiedMemory;
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

TEST_F(SchedulerTest, AllocaLinking) {
  HostUnifiedMemory = false;
  // This host device constructor should be placed before Mock.redefine
  // because it overrides the real implementation of get_device_info
  // which is needed when creating a host device.
  device HostDevice = detail::createSyclObjFromImpl<device>(
      detail::device_impl::getHostDeviceImpl());
  std::shared_ptr<detail::queue_impl> DefaultHostQueue{
      new detail::queue_impl(detail::getSyclObjImpl(HostDevice), {}, {})};

  sycl::unittest::PiMock Mock;
  sycl::queue Q{Mock.getPlatform().get_devices()[0]};
  Mock.redefineAfter<detail::PiApiKind::piDeviceGetInfo>(
      redefinedDeviceGetInfoAfter);
  sycl::detail::QueueImplPtr QImpl = detail::getSyclObjImpl(Q);

  MockScheduler MS;
  // Should not be linked w/o host unified memory or pinned host memory
  {
    buffer<int, 1> Buf(range<1>(1));
    detail::Requirement Req = getMockRequirement(Buf);

    std::vector<detail::Command *> AuxCmds;
    detail::MemObjRecord *Record =
        MS.getOrInsertMemObjRecord(QImpl, &Req, AuxCmds);
    detail::AllocaCommandBase *NonHostAllocaCmd =
        MS.getOrCreateAllocaForReq(Record, &Req, QImpl, AuxCmds);
    detail::AllocaCommandBase *HostAllocaCmd =
        MS.getOrCreateAllocaForReq(Record, &Req, DefaultHostQueue, AuxCmds);

    EXPECT_FALSE(HostAllocaCmd->MLinkedAllocaCmd);
    EXPECT_FALSE(NonHostAllocaCmd->MLinkedAllocaCmd);
  }
  // Should be linked in case of pinned host memory
  {
    buffer<int, 1> Buf(
        range<1>(1), {ext::oneapi::property::buffer::use_pinned_host_memory()});
    detail::Requirement Req = getMockRequirement(Buf);

    std::vector<detail::Command *> AuxCmds;
    detail::MemObjRecord *Record =
        MS.getOrInsertMemObjRecord(QImpl, &Req, AuxCmds);
    detail::AllocaCommandBase *NonHostAllocaCmd =
        MS.getOrCreateAllocaForReq(Record, &Req, QImpl, AuxCmds);
    detail::AllocaCommandBase *HostAllocaCmd =
        MS.getOrCreateAllocaForReq(Record, &Req, DefaultHostQueue, AuxCmds);

    EXPECT_EQ(HostAllocaCmd->MLinkedAllocaCmd, NonHostAllocaCmd);
    EXPECT_EQ(NonHostAllocaCmd->MLinkedAllocaCmd, HostAllocaCmd);
  }
  // Should be linked in case of host unified memory
  {
    HostUnifiedMemory = true;
    buffer<int, 1> Buf(range<1>(1));
    detail::Requirement Req = getMockRequirement(Buf);

    std::vector<detail::Command *> AuxCmds;
    detail::MemObjRecord *Record =
        MS.getOrInsertMemObjRecord(QImpl, &Req, AuxCmds);
    detail::AllocaCommandBase *NonHostAllocaCmd =
        MS.getOrCreateAllocaForReq(Record, &Req, QImpl, AuxCmds);
    detail::AllocaCommandBase *HostAllocaCmd =
        MS.getOrCreateAllocaForReq(Record, &Req, DefaultHostQueue, AuxCmds);

    EXPECT_EQ(HostAllocaCmd->MLinkedAllocaCmd, NonHostAllocaCmd);
    EXPECT_EQ(NonHostAllocaCmd->MLinkedAllocaCmd, HostAllocaCmd);
  }
}
