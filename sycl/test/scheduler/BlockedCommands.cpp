// RUN: %clangxx -fsycl %s -o %t.out
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out
//==------------------- BlockedCommands.cpp --------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <iostream>

#include <CL/cl.h>
#include <CL/sycl.hpp>

using namespace cl::sycl;

class FakeCommand : public detail::Command {
public:
  FakeCommand(detail::QueueImplPtr Queue)
      : Command(detail::Command::ALLOCA, Queue) {}
  void printDot(std::ostream &Stream) const override {}

  cl_int enqueueImp() override { return MRetVal; }

  cl_int MRetVal = CL_SUCCESS;
};

class TestScheduler : public detail::Scheduler {
public:
  static bool enqueueCommand(detail::Command *Cmd,
                             detail::EnqueueResultT &EnqueueResult,
                             detail::BlockingT Blocking) {
    return GraphProcessor::enqueueCommand(Cmd, EnqueueResult, Blocking);
  }
};

int main() {
  cl::sycl::queue Queue;
  FakeCommand FakeCmd(detail::getSyclObjImpl(Queue));

  FakeCmd.MIsBlockable = true;
  FakeCmd.MCanEnqueue = false;
  FakeCmd.MRetVal = CL_DEVICE_PARTITION_EQUALLY;

  {
    detail::EnqueueResultT Res;
    bool Enqueued =
        TestScheduler::enqueueCommand(&FakeCmd, Res, detail::NON_BLOCKING);

    if (Enqueued) {
      std::cerr << "Blocked command should not be enqueued" << std::endl;
      return 1;
    }

    if (detail::EnqueueResultT::BLOCKED != Res.MResult) {
      std::cerr << "Result of enqueueing blocked command should be BLOCKED"
                << std::endl;
      return 1;
    }
  }

  FakeCmd.MCanEnqueue = true;

  {
    detail::EnqueueResultT Res;
    bool Enqueued =
        TestScheduler::enqueueCommand(&FakeCmd, Res, detail::BLOCKING);

    if (Enqueued) {
      std::cerr << "The command is expected to fail to enqueue." << std::endl;
      return 1;
    }

    if (detail::EnqueueResultT::FAILED != Res.MResult) {
      std::cerr << "The command is expected to fail to enqueue." << std::endl;
      return 1;
    }

    if (CL_DEVICE_PARTITION_EQUALLY != Res.MErrCode) {
      std::cerr << "Expected different error code." << std::endl;
      return 1;
    }

    if (&FakeCmd != Res.MCmd) {
      std::cerr << "Expected different failed command." << std::endl;
      return 1;
    }
  }

  FakeCmd.MRetVal = CL_SUCCESS;

  {
    detail::EnqueueResultT Res;
    bool Enqueued =
        TestScheduler::enqueueCommand(&FakeCmd, Res, detail::BLOCKING);

    if (!Enqueued || detail::EnqueueResultT::SUCCESS != Res.MResult) {
      std::cerr << "The command is expected to be successfully enqueued."
                << std::endl;
      return 1;
    }
  }

  return 0;
}
