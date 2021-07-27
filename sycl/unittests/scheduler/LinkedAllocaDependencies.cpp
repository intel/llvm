//==------ LinkedAllocaDependencies.cpp --- Scheduler unit tests -----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SchedulerTest.hpp"
#include "SchedulerTestUtils.hpp"

using namespace cl::sycl;

class MemObjMock : public cl::sycl::detail::SYCLMemObjI {
public:
  using ContextImplPtr = std::shared_ptr<cl::sycl::detail::context_impl>;

  MemObjMock(const std::shared_ptr<cl::sycl::detail::MemObjRecord> &Record)
      : SYCLMemObjI() {
    MRecord = Record;
  }

  ~MemObjMock() = default;

  MemObjType getType() const override { return MemObjType::Buffer; }

  void *allocateMem(ContextImplPtr, bool, void *,
                    cl::sycl::detail::pi::PiEvent &) {
    return nullptr;
  }

  void *allocateHostMem() { return nullptr; }
  void releaseMem(ContextImplPtr, void *) {}
  void releaseHostMem(void *) {}
  size_t getSize() const override { return 10; }
  detail::ContextImplPtr getInteropContext() const override { return nullptr; }
};

static cl::sycl::device getDeviceWithHostUnifiedMemory() {
  for (cl::sycl::device &D : cl::sycl::device::get_devices()) {
    if (!D.is_host() &&
        D.get_info<cl::sycl::info::device::host_unified_memory>())
      return D;
  }
  return {};
}

TEST_F(SchedulerTest, LinkedAllocaDependencies) {
  cl::sycl::device Dev = getDeviceWithHostUnifiedMemory();
  if (Dev.is_host()) {
    std::cerr << "Not run: no non-host devices with host unified memory support"
              << std::endl;
    return;
  }

  // 1. create two commands: alloca + alloca and link them
  // 2. call Scheduler::GraphBuilder::getOrCreateAllocaForReq
  detail::Requirement Req = getMockRequirement();

  // Commands are linked only if the device supports host unified memory.

  cl::sycl::queue Queue1{Dev};
  cl::sycl::detail::QueueImplPtr Q1 = cl::sycl::detail::getSyclObjImpl(Queue1);

  sycl::device HostDevice;
  std::shared_ptr<detail::queue_impl> DefaultHostQueue(new detail::queue_impl(
      detail::getSyclObjImpl(HostDevice), /*AsyncHandler=*/{},
      /*PropList=*/{}));

  auto AllocaDep = [](cl::sycl::detail::Command *, cl::sycl::detail::Command *,
                      cl::sycl::detail::MemObjRecord *,
                      std::vector<cl::sycl::detail::Command *> &) {};

  std::shared_ptr<cl::sycl::detail::MemObjRecord> Record{
      new cl::sycl::detail::MemObjRecord(DefaultHostQueue->getContextImplPtr(),
                                         10, AllocaDep)};

  MemObjMock MemObj(Record);
  Req.MSYCLMemObj = &MemObj;

  cl::sycl::detail::AllocaCommand AllocaCmd1(DefaultHostQueue, Req, false);
  Record->MAllocaCommands.push_back(&AllocaCmd1);

  MockCommand DepCmd(DefaultHostQueue, Req);
  MockCommand DepDepCmd(DefaultHostQueue, Req);
  DepCmd.MDeps.push_back({&DepDepCmd, DepDepCmd.getRequirement(), &AllocaCmd1});
  DepDepCmd.MUsers.insert(&DepCmd);
  std::vector<cl::sycl::detail::Command *> ToEnqueue;
  Record->MWriteLeaves.push_back(&DepCmd, ToEnqueue);

  MockScheduler MS;
  cl::sycl::detail::Command *AllocaCmd2 =
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
                           [&](const cl::sycl::detail::DepDesc &Dep) -> bool {
                             return Dep.MDepCommand == &AllocaCmd1;
                           }) != AllocaCmd2->MDeps.end())
      << "No deps for existing alloca appeared in new alloca";
  ASSERT_TRUE(std::find_if(AllocaCmd2->MDeps.begin(), AllocaCmd2->MDeps.end(),
                           [&](const cl::sycl::detail::DepDesc &Dep) -> bool {
                             return Dep.MDepCommand == &DepCmd;
                           }) != AllocaCmd2->MDeps.end())
      << "No deps for leaves (deps of existing alloca) appeared in new alloca";
}
