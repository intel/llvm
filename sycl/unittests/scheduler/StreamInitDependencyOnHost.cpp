//==------ LinkedAllocaDependencies.cpp --- Scheduler unit tests -----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SchedulerTest.hpp"
#include "SchedulerTestUtils.hpp"

#include <detail/config.hpp>
#include <detail/handler_impl.hpp>
#include <helpers/ScopedEnvVar.hpp>

using namespace sycl;

inline constexpr auto DisableCleanupName =
    "SYCL_DISABLE_EXECUTION_GRAPH_CLEANUP";

class MockHandlerStreamInit : public MockHandler {
public:
  MockHandlerStreamInit(std::shared_ptr<detail::queue_impl> Queue, bool IsHost)
      : MockHandler(Queue, IsHost) {}
  std::unique_ptr<detail::CG> finalize() {
    std::unique_ptr<detail::CG> CommandGroup;
    switch (getType()) {
    case detail::CG::Kernel: {
      CommandGroup.reset(new detail::CGExecKernel(
          getNDRDesc(), std::move(getHostKernel()), getKernel(),
          std::move(MImpl->MKernelBundle),
          detail::CG::StorageInitHelper(getArgsStorage(), getAccStorage(),
                                        getSharedPtrStorage(),
                                        getRequirements(), getEvents()),
          getArgs(), getKernelName(), getStreamStorage(),
          std::move(MImpl->MAuxiliaryResources), getCGType(), {},
          MImpl->MKernelIsCooperative, getCodeLoc()));
      break;
    }
    default:
      throw sycl::runtime_error("Unhandled type of command group",
                                PI_ERROR_INVALID_OPERATION);
    }

    return CommandGroup;
  }
};

using CmdTypeTy = sycl::detail::Command::CommandType;

// Function recursively checks that initial command has dependency on chain of
// other commands that should have type DepCmdsTypes[Depth] (Depth is a distance
// - 1 in a command dependencies tree from initial command to a currently
// checked one) and requirement on memory object of stream's flush buffer.
static bool ValidateDepCommandsTree(const detail::Command *Cmd,
                                    const std::vector<CmdTypeTy> &DepCmdsTypes,
                                    const detail::SYCLMemObjI *MemObj,
                                    size_t Depth = 0) {
  if (!Cmd || Depth >= DepCmdsTypes.size())
    throw sycl::runtime_error("Command parameters are invalid",
                              PI_ERROR_INVALID_VALUE);

  for (const detail::DepDesc &Dep : Cmd->MDeps) {
    if (Dep.MDepCommand &&
        (Dep.MDepCommand->getType() == DepCmdsTypes[Depth]) &&
        Dep.MDepRequirement && (Dep.MDepRequirement->MSYCLMemObj == MemObj) &&
        ((Depth == DepCmdsTypes.size() - 1) ||
         ValidateDepCommandsTree(Dep.MDepCommand, DepCmdsTypes, MemObj,
                                 Depth + 1))) {
      return true;
    }
  }

  return false;
}

TEST_F(SchedulerTest, StreamInitDependencyOnHost) {
  // Disable post enqueue cleanup so that it doesn't interfere with dependency
  // checks.
  unittest::ScopedEnvVar DisabledCleanup{
      DisableCleanupName, "1",
      detail::SYCLConfig<detail::SYCL_DISABLE_EXECUTION_GRAPH_CLEANUP>::reset};
  std::shared_ptr<detail::queue_impl> HQueueImpl(new detail::queue_impl(
      detail::device_impl::getHostDeviceImpl(), /*AsyncHandler=*/{},
      /*PropList=*/{}));

  // Emulating processing of command group function
  MockHandlerStreamInit MockCGH(HQueueImpl, true);
  MockCGH.setType(detail::CG::Kernel);

  auto EmptyKernel = [](sycl::nd_item<1>) {};
  MockCGH
      .setHostKernel<decltype(EmptyKernel), sycl::nd_item<1>, 1, class Empty>(
          EmptyKernel);
  MockCGH.setNDRangeDesc(
      sycl::nd_range<1>{sycl::range<1>{1}, sycl::range<1>{1}});

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

  std::unique_ptr<detail::CG> MainCG = MockCGH.finalize();

  // Emulate call of Scheduler::addCG
  std::vector<detail::StreamImplPtr> Streams =
      static_cast<detail::CGExecKernel *>(MainCG.get())->getStreams();
  ASSERT_EQ(Streams.size(), 1u) << "Invalid number of stream objects";

  Streams[0]->initStreamHost(HQueueImpl);

  MockScheduler MS;
  std::vector<detail::Command *> AuxCmds;
  detail::Command *NewCmd = MS.addCG(std::move(MainCG), HQueueImpl, AuxCmds);
  ASSERT_TRUE(!!NewCmd) << "Failed to add command group into scheduler";
  ASSERT_GT(NewCmd->MDeps.size(), 0u)
      << "No deps appeared in the new exec kernel command";

  // Searching in dependencies for CG execution command that initializes flush
  // buffer of a stream that is supposed to be used inside NewCmd's CG.
  // Tree of dependencies should look like:
  // [MAIN_CG] -> [EMPTY_NODE {FlushBufMemObj}] -> [FILL_CG {FlushBufMemObj}] ->
  //     [[ALLOC_TASK {FlushBufMemObj}]
  std::vector<CmdTypeTy> DepCmdsTypes({CmdTypeTy::RUN_CG, // FILL_CG
                                       CmdTypeTy::ALLOCA});
  ASSERT_TRUE(ValidateDepCommandsTree(NewCmd, DepCmdsTypes, FlushBufMemObjPtr))
      << "Dependency on stream flush buffer initialization not found";
}
