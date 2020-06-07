//==---------- SchedulerTestUtils.hpp --- Scheduler unit tests -------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once
#include <CL/sycl.hpp>
#include <CL/sycl/detail/cl.h>
#include <detail/queue_impl.hpp>
#include <detail/scheduler/scheduler.hpp>

#include <functional>
// This header contains a few common classes/methods used in
// execution graph testing.

cl::sycl::detail::Requirement getMockRequirement();

class MockCommand : public cl::sycl::detail::Command {
public:
  MockCommand(cl::sycl::detail::QueueImplPtr Queue,
              cl::sycl::detail::Requirement Req)
      : Command{cl::sycl::detail::Command::EMPTY_TASK, Queue},
        MRequirement{std::move(Req)} {}

  MockCommand(cl::sycl::detail::QueueImplPtr Queue)
      : Command{cl::sycl::detail::Command::EMPTY_TASK, Queue},
        MRequirement{std::move(getMockRequirement())} {}

  void printDot(std::ostream &) const override {}
  void emitInstrumentationData() override {}

  const cl::sycl::detail::Requirement *getRequirement() const final {
    return &MRequirement;
  };

  cl_int enqueueImp() override { return MRetVal; }

  cl_int MRetVal = CL_SUCCESS;

  void waitForEventsCall(
      std::shared_ptr<cl::sycl::detail::queue_impl> Queue,
      std::vector<std::shared_ptr<cl::sycl::detail::event_impl>> &RawEvents,
      pi_event &Event) {
    Command::waitForEvents(Queue, RawEvents, Event);
  }

protected:
  cl::sycl::detail::Requirement MRequirement;
};

class MockCommandWithCallback : public MockCommand {
public:
  MockCommandWithCallback(cl::sycl::detail::QueueImplPtr Queue,
                          cl::sycl::detail::Requirement Req,
                          std::function<void()> Callback)
      : MockCommand(Queue, Req), MCallback(std::move(Callback)) {}

  ~MockCommandWithCallback() override { MCallback(); }

protected:
  std::function<void()> MCallback;
};

class MockScheduler : public cl::sycl::detail::Scheduler {
public:
  cl::sycl::detail::MemObjRecord *
  getOrInsertMemObjRecord(const cl::sycl::detail::QueueImplPtr &Queue,
                          cl::sycl::detail::Requirement *Req) {
    return MGraphBuilder.getOrInsertMemObjRecord(Queue, Req);
  }

  void removeRecordForMemObj(cl::sycl::detail::SYCLMemObjI *MemObj) {
    MGraphBuilder.removeRecordForMemObj(MemObj);
  }

  void cleanupCommandsForRecord(cl::sycl::detail::MemObjRecord *Rec) {
    MGraphBuilder.cleanupCommandsForRecord(Rec);
  }

  void addNodeToLeaves(
      cl::sycl::detail::MemObjRecord *Rec, cl::sycl::detail::Command *Cmd,
      cl::sycl::access::mode Mode = cl::sycl::access::mode::read_write) {
    return MGraphBuilder.addNodeToLeaves(Rec, Cmd, Mode);
  }

  static bool enqueueCommand(cl::sycl::detail::Command *Cmd,
                             cl::sycl::detail::EnqueueResultT &EnqueueResult,
                             cl::sycl::detail::BlockingT Blocking) {
    return GraphProcessor::enqueueCommand(Cmd, EnqueueResult, Blocking);
  }
};

void addEdge(cl::sycl::detail::Command *User, cl::sycl::detail::Command *Dep,
             cl::sycl::detail::AllocaCommandBase *Alloca);

template <typename MemObjT>
cl::sycl::detail::Requirement getMockRequirement(const MemObjT &MemObj) {
  return {/*Offset*/ {0, 0, 0},
          /*AccessRange*/ {0, 0, 0},
          /*MemoryRange*/ {0, 0, 0},
          /*AccessMode*/ cl::sycl::access::mode::read_write,
          /*SYCLMemObj*/ cl::sycl::detail::getSyclObjImpl(MemObj).get(),
          /*Dims*/ 0,
          /*ElementSize*/ 0};
}
