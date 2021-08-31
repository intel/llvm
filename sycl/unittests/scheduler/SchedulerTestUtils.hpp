//==---------- SchedulerTestUtils.hpp --- Scheduler unit tests -------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once
#include <sycl/sycl.hpp>
#include <CL/sycl/detail/cl.h>
#include <detail/queue_impl.hpp>
#include <detail/scheduler/scheduler.hpp>
#include <detail/stream_impl.hpp>

#include <functional>
#include <gmock/gmock.h>
#include <vector>

// This header contains a few common classes/methods used in
// execution graph testing.

sycl::detail::Requirement getMockRequirement();

__SYCL_OPEN_NS() {
namespace detail {
class Command;
} // namespace detail
} // __SYCL_OPEN_NS()
__SYCL_CLOSE_NS()

class MockCommand : public sycl::detail::Command {
public:
  MockCommand(sycl::detail::QueueImplPtr Queue,
              sycl::detail::Requirement Req,
              sycl::detail::Command::CommandType Type =
                  sycl::detail::Command::RUN_CG)
      : Command{Type, Queue}, MRequirement{std::move(Req)} {
    using namespace testing;
    ON_CALL(*this, enqueue(_, _))
        .WillByDefault(Invoke(this, &MockCommand::enqueueOrigin));
    EXPECT_CALL(*this, enqueue(_, _)).Times(AnyNumber());
  }

  MockCommand(sycl::detail::QueueImplPtr Queue,
              sycl::detail::Command::CommandType Type =
                  sycl::detail::Command::RUN_CG)
      : Command{Type, Queue}, MRequirement{std::move(getMockRequirement())} {
    using namespace testing;
    ON_CALL(*this, enqueue(_, _))
        .WillByDefault(Invoke(this, &MockCommand::enqueueOrigin));
    EXPECT_CALL(*this, enqueue(_, _)).Times(AnyNumber());
  }

  void printDot(std::ostream &) const override {}
  void emitInstrumentationData() override {}

  const sycl::detail::Requirement *getRequirement() const final {
    return &MRequirement;
  };

  cl_int enqueueImp() override { return MRetVal; }

  MOCK_METHOD2(enqueue, bool(sycl::detail::EnqueueResultT &,
                             sycl::detail::BlockingT));
  bool enqueueOrigin(sycl::detail::EnqueueResultT &EnqueueResult,
                     sycl::detail::BlockingT Blocking) {
    return sycl::detail::Command::enqueue(EnqueueResult, Blocking);
  }

  cl_int MRetVal = CL_SUCCESS;

  void waitForEventsCall(
      std::shared_ptr<sycl::detail::queue_impl> Queue,
      std::vector<std::shared_ptr<sycl::detail::event_impl>> &RawEvents,
      pi_event &Event) {
    Command::waitForEvents(Queue, RawEvents, Event);
  }

protected:
  sycl::detail::Requirement MRequirement;
};

class MockCommandWithCallback : public MockCommand {
public:
  MockCommandWithCallback(sycl::detail::QueueImplPtr Queue,
                          sycl::detail::Requirement Req,
                          std::function<void()> Callback)
      : MockCommand(Queue, Req), MCallback(std::move(Callback)) {}

  ~MockCommandWithCallback() override { MCallback(); }

protected:
  std::function<void()> MCallback;
};

class MockScheduler : public sycl::detail::Scheduler {
public:
  sycl::detail::MemObjRecord *
  getOrInsertMemObjRecord(const sycl::detail::QueueImplPtr &Queue,
                          sycl::detail::Requirement *Req,
                          std::vector<sycl::detail::Command *> &ToEnqueue) {
    return MGraphBuilder.getOrInsertMemObjRecord(Queue, Req, ToEnqueue);
  }

  void removeRecordForMemObj(sycl::detail::SYCLMemObjI *MemObj) {
    MGraphBuilder.removeRecordForMemObj(MemObj);
  }

  void cleanupCommandsForRecord(sycl::detail::MemObjRecord *Rec) {
    std::vector<std::shared_ptr<sycl::detail::stream_impl>>
        StreamsToDeallocate;
    MGraphBuilder.cleanupCommandsForRecord(Rec, StreamsToDeallocate);
  }

  void addNodeToLeaves(sycl::detail::MemObjRecord *Rec,
                       sycl::detail::Command *Cmd,
                       sycl::access::mode Mode,
                       std::vector<sycl::detail::Command *> &ToEnqueue) {
    return MGraphBuilder.addNodeToLeaves(Rec, Cmd, Mode, ToEnqueue);
  }

  static bool enqueueCommand(sycl::detail::Command *Cmd,
                             sycl::detail::EnqueueResultT &EnqueueResult,
                             sycl::detail::BlockingT Blocking) {
    return GraphProcessor::enqueueCommand(Cmd, EnqueueResult, Blocking);
  }

  sycl::detail::AllocaCommandBase *
  getOrCreateAllocaForReq(sycl::detail::MemObjRecord *Record,
                          const sycl::detail::Requirement *Req,
                          sycl::detail::QueueImplPtr Queue,
                          std::vector<sycl::detail::Command *> &ToEnqueue) {
    return MGraphBuilder.getOrCreateAllocaForReq(Record, Req, Queue, ToEnqueue);
  }

  ReadLockT acquireGraphReadLock() { return ReadLockT{MGraphLock}; }

  sycl::detail::Command *
  insertMemoryMove(sycl::detail::MemObjRecord *Record,
                   sycl::detail::Requirement *Req,
                   const sycl::detail::QueueImplPtr &Queue,
                   std::vector<sycl::detail::Command *> &ToEnqueue) {
    return MGraphBuilder.insertMemoryMove(Record, Req, Queue, ToEnqueue);
  }

  sycl::detail::Command *
  addCG(std::unique_ptr<sycl::detail::CG> CommandGroup,
        sycl::detail::QueueImplPtr Queue,
        std::vector<sycl::detail::Command *> &ToEnqueue) {
    return MGraphBuilder.addCG(std::move(CommandGroup), Queue, ToEnqueue);
  }
};

void addEdge(sycl::detail::Command *User, sycl::detail::Command *Dep,
             sycl::detail::AllocaCommandBase *Alloca);

template <typename MemObjT>
sycl::detail::Requirement getMockRequirement(const MemObjT &MemObj) {
  return {/*Offset*/ {0, 0, 0},
          /*AccessRange*/ {0, 0, 0},
          /*MemoryRange*/ {0, 0, 0},
          /*AccessMode*/ sycl::access::mode::read_write,
          /*SYCLMemObj*/ sycl::detail::getSyclObjImpl(MemObj).get(),
          /*Dims*/ 0,
          /*ElementSize*/ 0};
}
