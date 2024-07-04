//==------ LinkedAllocaDependencies.cpp --- Scheduler unit tests -----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SchedulerTest.hpp"
#include "SchedulerTestUtils.hpp"

#include <helpers/PiMock.hpp>

using namespace sycl;

class MemObjMock : public sycl::detail::SYCLMemObjI {
public:
  using ContextImplPtr = std::shared_ptr<sycl::detail::context_impl>;

  MemObjMock(const std::shared_ptr<sycl::detail::MemObjRecord> &Record)
      : SYCLMemObjI() {
    MRecord = Record;
  }

  ~MemObjMock() = default;

  MemObjType getType() const override { return MemObjType::Buffer; }

  void *allocateMem(ContextImplPtr, bool, void *, sycl::detail::pi::PiEvent &) {
    return nullptr;
  }

  void *allocateHostMem() { return nullptr; }
  void releaseMem(ContextImplPtr, void *) {}
  void releaseHostMem(void *) {}
  size_t getSizeInBytes() const noexcept override { return 10; }
  bool isInterop() const override { return false; }
  bool hasUserDataPtr() const override { return false; }
  bool isHostPointerReadOnly() const override { return false; }
  bool usesPinnedHostMemory() const override { return false; }

  detail::ContextImplPtr getInteropContext() const override { return nullptr; }
};

static sycl::device getDeviceWithHostUnifiedMemory(sycl::platform &Plt) {
  for (sycl::device &D : Plt.get_devices()) {
    if (D.get_info<sycl::info::device::host_unified_memory>())
      return D;
  }
  return {};
}

TEST_F(SchedulerTest, LinkedAllocaDependencies) {
  sycl::unittest::PiMock Mock;
  sycl::platform Plt = Mock.getPlatform();
  sycl::device Dev = getDeviceWithHostUnifiedMemory(Plt);

  // 1. create two commands: alloca + alloca and link them
  // 2. call Scheduler::GraphBuilder::getOrCreateAllocaForReq
  detail::Requirement Req = getMockRequirement();

  // Commands are linked only if the device supports host unified memory.

  sycl::queue Queue1{Dev};
  sycl::detail::QueueImplPtr Q1 = sycl::detail::getSyclObjImpl(Queue1);

  auto AllocaDep = [](sycl::detail::Command *, sycl::detail::Command *,
                      sycl::detail::MemObjRecord *,
                      std::vector<sycl::detail::Command *> &) {};

  std::shared_ptr<sycl::detail::MemObjRecord> Record{
      new sycl::detail::MemObjRecord(nullptr, 10, AllocaDep)};

  MemObjMock MemObj(Record);
  Req.MSYCLMemObj = &MemObj;

  sycl::detail::AllocaCommand AllocaCmd1(nullptr, Req, false);
  Record->MAllocaCommands.push_back(&AllocaCmd1);

  MockCommand DepCmd(nullptr, Req);
  MockCommand DepDepCmd(nullptr, Req);
  DepCmd.MDeps.push_back({&DepDepCmd, DepDepCmd.getRequirement(), &AllocaCmd1});
  DepDepCmd.MUsers.insert(&DepCmd);
  std::vector<sycl::detail::Command *> ToEnqueue;
  Record->MWriteLeaves.push_back(&DepCmd, ToEnqueue);

  MockScheduler MS;
  sycl::detail::Command *AllocaCmd2 =
      MS.getOrCreateAllocaForReq(Record.get(), &Req, Q1, ToEnqueue);

  ASSERT_TRUE(!!AllocaCmd1.MLinkedAllocaCmd)
      << "No link appeared in existing command";
  ASSERT_EQ(AllocaCmd1.MLinkedAllocaCmd, AllocaCmd2) << "Invalid link appeared";
  ASSERT_GT(AllocaCmd1.MUsers.count(AllocaCmd2), 0u)
      << "New alloca isn't in users of the old one";
  ASSERT_GT(AllocaCmd2->MDeps.size(), 1u)
      << "No deps appeared in the new alloca";
  ASSERT_GT(DepCmd.MUsers.count(AllocaCmd2), 0u)
      << "No deps appeared for leaves of record (i.e. deps of existing alloca)";
  ASSERT_TRUE(std::find_if(AllocaCmd2->MDeps.begin(), AllocaCmd2->MDeps.end(),
                           [&](const sycl::detail::DepDesc &Dep) -> bool {
                             return Dep.MDepCommand == &AllocaCmd1;
                           }) != AllocaCmd2->MDeps.end())
      << "No deps for existing alloca appeared in new alloca";
  ASSERT_TRUE(std::find_if(AllocaCmd2->MDeps.begin(), AllocaCmd2->MDeps.end(),
                           [&](const sycl::detail::DepDesc &Dep) -> bool {
                             return Dep.MDepCommand == &DepCmd;
                           }) != AllocaCmd2->MDeps.end())
      << "No deps for leaves (deps of existing alloca) appeared in new alloca";
}
