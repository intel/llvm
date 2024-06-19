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
#include <helpers/PiMock.hpp>
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
