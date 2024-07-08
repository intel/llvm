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
#include <helpers/UrMock.hpp>

using namespace sycl;

inline constexpr auto DisableCleanupName =
    "SYCL_DISABLE_EXECUTION_GRAPH_CLEANUP";

class MockHandlerStreamInit : public MockHandler {
public:
  MockHandlerStreamInit(std::shared_ptr<detail::queue_impl> Queue,
                        bool CallerNeedsEvent)
      : MockHandler(Queue, CallerNeedsEvent) {}
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
                                UR_RESULT_ERROR_INVALID_OPERATION);
    }

    return CommandGroup;
  }
};
