//==------ LinkedAllocaDependencies.cpp --- Scheduler unit tests -----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SchedulerTest.hpp"
#include "SchedulerTestUtils.hpp"

#include <detail/scheduler/scheduler_helpers.hpp>

using namespace cl::sycl;

class MockHandler : public sycl::handler {
public:
  MockHandler(shared_ptr_class<detail::queue_impl> Queue, bool IsHost)
      : sycl::handler(Queue, IsHost) {}

  void setType(detail::CG::CGTYPE Type) {
    static_cast<sycl::handler *>(this)->MCGType = Type;
  }

  void addStream(const detail::StreamImplPtr &Stream) {
    sycl::handler::addStream(Stream);
  }

  unique_ptr_class<detail::CG> finalize() {
    auto CGH = static_cast<sycl::handler *>(this);
    unique_ptr_class<detail::CG> CommandGroup;
    switch (CGH->MCGType) {
    case detail::CG::KERNEL:
    case detail::CG::RUN_ON_HOST_INTEL: {
      CommandGroup.reset(new detail::CGExecKernel(
          std::move(CGH->MNDRDesc), std::move(CGH->MHostKernel),
          std::move(CGH->MKernel), std::move(CGH->MArgsStorage),
          std::move(CGH->MAccStorage), std::move(CGH->MSharedPtrStorage),
          std::move(CGH->MRequirements), std::move(CGH->MEvents),
          std::move(CGH->MArgs), std::move(CGH->MKernelName),
          std::move(CGH->MOSModuleHandle), std::move(CGH->MStreamStorage),
          CGH->MCGType, CGH->MCodeLoc));
      break;
    }
    default:
      throw runtime_error("Unhandled type of command group",
                          PI_INVALID_OPERATION);
    }

    return CommandGroup;
  }
};

using CmdTypeTy = cl::sycl::detail::Command::CommandType;

static bool ValidateDepCommandsTree(const detail::Command *Cmd,
                                    std::queue<CmdTypeTy> DepCmdsTypes,
                                    const detail::SYCLMemObjI *MemObj) {
  if (DepCmdsTypes.empty())
    return true;
  else if (!Cmd)
    return false;

  CmdTypeTy DepCmdType = DepCmdsTypes.front();
  DepCmdsTypes.pop();

  for (const detail::DepDesc &Dep : Cmd->MDeps) {
    if (Dep.MDepCommand && (Dep.MDepCommand->getType() == DepCmdType) &&
        Dep.MDepRequirement && (Dep.MDepRequirement->MSYCLMemObj == MemObj) &&
        ValidateDepCommandsTree(Dep.MDepCommand, DepCmdsTypes, MemObj))
      return true;
  }

  return false;
}

TEST_F(SchedulerTest, StreamInitDependencyOnHost) {
  cl::sycl::queue HQueue(host_selector{});
  detail::QueueImplPtr HQueueImpl = detail::getSyclObjImpl(HQueue);

  // Emulating processing of command group function
  MockHandler MockCGH(HQueueImpl, true);
  MockCGH.setType(detail::CG::KERNEL);

  // Emulating construction of stream object inside command group
  detail::StreamImplPtr StreamImpl =
      std::make_shared<detail::stream_impl>(1024, 200, MockCGH);
  detail::GlobalBufAccessorT FlushBufAcc =
      StreamImpl->accessGlobalFlushBuf(MockCGH);
  MockCGH.addStream(StreamImpl);

  detail::SYCLMemObjI *FlushBufMemObjPtr =
      detail::getSyclObjImpl(FlushBufAcc)->MSYCLMemObj;
  ASSERT_TRUE(!!FlushBufMemObjPtr)
      << "Memory object for stream flush buffer not initialized";

  unique_ptr_class<detail::CG> MainCG = MockCGH.finalize();

  // Emulate call of Scheduler::addCG
  vector_class<detail::StreamImplPtr> Streams =
      static_cast<detail::CGExecKernel *>(MainCG.get())->getStreams();
  ASSERT_EQ(Streams.size(), 1u) << "Invalid number of stream objects";

  initStream(Streams[0], HQueueImpl);

  MockScheduler MS;
  detail::Command *NewCmd = MS.addCG(std::move(MainCG), HQueueImpl);
  ASSERT_TRUE(!!NewCmd) << "Failed to add command group into scheduler";
  ASSERT_GT(NewCmd->MDeps.size(), 0u)
      << "No deps appeared in the new exec kernel command";

  // Searching in dependencies for CG execution command that initializes flush
  // buffer of a stream that is supposed to be used inside NewCmd's CG.
  // Tree of dependencies should look like:
  // [MAIN_CG] -> [EMPTY_NODE {FlushBufMemObj}] -> [FILL_CG {FlushBufMemObj}] ->
  //     [[ALLOC_TASK {FlushBufMemObj}]
  std::queue<CmdTypeTy> DepCmdsTypes({CmdTypeTy::EMPTY_TASK,
                                      CmdTypeTy::RUN_CG, // FILL_CG
                                      CmdTypeTy::ALLOCA});
  ASSERT_TRUE(ValidateDepCommandsTree(NewCmd, DepCmdsTypes, FlushBufMemObjPtr))
      << "Dependency on stream flush buffer initialization not found";
}
