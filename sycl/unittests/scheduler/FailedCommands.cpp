//==----------- FailedCommands.cpp ---- Scheduler unit tests ---------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SchedulerTest.hpp"

#include <CL/cl.h>
#include <CL/sycl.hpp>
#include <detail/scheduler/scheduler.hpp>

#include <gtest/gtest.h>

using namespace cl::sycl;

class MockCommand : public detail::Command {
public:
  MockCommand(detail::QueueImplPtr Queue)
      : Command(detail::Command::ALLOCA, Queue) {}
  void printDot(std::ostream &Stream) const override {}
  void emitInstrumentationData() override {}
  cl_int enqueueImp() override { return CL_SUCCESS; }
};

class MockScheduler : public detail::Scheduler {
public:
  static bool enqueueCommand(detail::Command *Cmd,
                             detail::EnqueueResultT &EnqueueResult,
                             detail::BlockingT Blocking) {
    return GraphProcessor::enqueueCommand(Cmd, EnqueueResult, Blocking);
  }
};

TEST_F(SchedulerTest, FailedDependency) {
  detail::Requirement MockReq(/*Offset*/ {0, 0, 0}, /*AccessRange*/ {1, 1, 1},
                              /*MemoryRange*/ {1, 1, 1},
                              access::mode::read_write, /*SYCLMemObjT*/ nullptr,
                              /*Dims*/ 1, /*ElementSize*/ 1);
  MockCommand MDep(detail::getSyclObjImpl(MQueue));
  MockCommand MUser(detail::getSyclObjImpl(MQueue));
  MDep.addUser(&MUser);
  MUser.addDep(detail::DepDesc{&MDep, &MockReq, nullptr});
  MUser.MEnqueueStatus = detail::EnqueueResultT::SyclEnqueueReady;
  MDep.MEnqueueStatus = detail::EnqueueResultT::SyclEnqueueFailed;

  detail::EnqueueResultT Res;
  bool Enqueued =
      MockScheduler::enqueueCommand(&MUser, Res, detail::NON_BLOCKING);

  ASSERT_FALSE(Enqueued) << "Enqueue process must fail\n";
  ASSERT_EQ(Res.MCmd, &MDep) << "Wrong failed command\n";
  ASSERT_EQ(Res.MResult, detail::EnqueueResultT::SyclEnqueueFailed)
      << "Enqueue process must fail\n";
  ASSERT_EQ(MUser.MEnqueueStatus, detail::EnqueueResultT::SyclEnqueueReady)
      << "MUser shouldn't be marked as failed\n";
  ASSERT_EQ(MDep.MEnqueueStatus, detail::EnqueueResultT::SyclEnqueueFailed)
      << "MDep should be marked as failed\n";
}
