//==---------- SchedulerTestUtils.hpp --- Scheduler unit tests -------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once
#include <detail/queue_impl.hpp>
#include <detail/scheduler/commands.hpp>
#include <detail/scheduler/scheduler.hpp>
#include <detail/stream_impl.hpp>
#include <sycl/detail/cl.h>

#include <functional>
#include <gmock/gmock.h>
#include <vector>

// This header contains a few common classes/methods used in
// execution graph testing.

sycl::detail::Requirement getMockRequirement();

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace detail {
class Command;
} // namespace detail
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl

class MockCommand : public sycl::detail::Command {
public:
  MockCommand(
      sycl::detail::QueueImplPtr Queue, sycl::detail::Requirement Req,
      sycl::detail::Command::CommandType Type = sycl::detail::Command::RUN_CG)
      : Command{Type, Queue}, MRequirement{std::move(Req)} {
    using namespace testing;
    ON_CALL(*this, enqueue)
        .WillByDefault(Invoke(this, &MockCommand::enqueueOrigin));
    EXPECT_CALL(*this, enqueue).Times(AnyNumber());
  }

  MockCommand(
      sycl::detail::QueueImplPtr Queue,
      sycl::detail::Command::CommandType Type = sycl::detail::Command::RUN_CG)
      : Command{Type, Queue}, MRequirement{std::move(getMockRequirement())} {
    using namespace testing;
    ON_CALL(*this, enqueue)
        .WillByDefault(Invoke(this, &MockCommand::enqueueOrigin));
    EXPECT_CALL(*this, enqueue).Times(AnyNumber());
  }

  void printDot(std::ostream &) const override {}
  void emitInstrumentationData() override {}

  const sycl::detail::Requirement *getRequirement() const final {
    return &MRequirement;
  };

  cl_int enqueueImp() override { return MRetVal; }

  MOCK_METHOD3(enqueue,
               bool(sycl::detail::EnqueueResultT &, sycl::detail::BlockingT,
                    std::vector<sycl::detail::Command *> &));
  bool enqueueOrigin(sycl::detail::EnqueueResultT &EnqueueResult,
                     sycl::detail::BlockingT Blocking,
                     std::vector<sycl::detail::Command *> &ToCleanUp) {
    return sycl::detail::Command::enqueue(EnqueueResult, Blocking, ToCleanUp);
  }

  cl_int MRetVal = CL_SUCCESS;

  void waitForEventsCall(
      std::shared_ptr<sycl::detail::queue_impl> Queue,
      std::vector<std::shared_ptr<sycl::detail::event_impl>> &RawEvents,
      pi_event &Event) {
    Command::waitForEvents(Queue, RawEvents, Event);
  }

  std::shared_ptr<sycl::detail::event_impl> getEvent() { return MEvent; }

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
  using sycl::detail::Scheduler::addCG;
  using sycl::detail::Scheduler::addCopyBack;
  using sycl::detail::Scheduler::cleanupCommands;

  sycl::detail::MemObjRecord *
  getOrInsertMemObjRecord(const sycl::detail::QueueImplPtr &Queue,
                          sycl::detail::Requirement *Req,
                          std::vector<sycl::detail::Command *> &ToEnqueue) {
    return MGraphBuilder.getOrInsertMemObjRecord(Queue, Req, ToEnqueue);
  }

  void decrementLeafCountersForRecord(sycl::detail::MemObjRecord *Rec) {
    MGraphBuilder.decrementLeafCountersForRecord(Rec);
  }

  void removeRecordForMemObj(sycl::detail::SYCLMemObjI *MemObj) {
    MGraphBuilder.removeRecordForMemObj(MemObj);
  }

  void cleanupCommandsForRecord(sycl::detail::MemObjRecord *Rec) {
    std::vector<std::shared_ptr<sycl::detail::stream_impl>> StreamsToDeallocate;
    std::vector<std::shared_ptr<const void>> AuxiliaryResourcesToDeallocate;
    MGraphBuilder.cleanupCommandsForRecord(Rec, StreamsToDeallocate,
                                           AuxiliaryResourcesToDeallocate);
  }

  void addNodeToLeaves(sycl::detail::MemObjRecord *Rec,
                       sycl::detail::Command *Cmd, sycl::access::mode Mode,
                       std::vector<sycl::detail::Command *> &ToEnqueue) {
    return MGraphBuilder.addNodeToLeaves(Rec, Cmd, Mode, ToEnqueue);
  }

  void updateLeaves(const std::set<sycl::detail::Command *> &Cmds,
                    sycl::detail::MemObjRecord *Record,
                    sycl::access::mode AccessMode,
                    std::vector<sycl::detail::Command *> &ToCleanUp) {
    return MGraphBuilder.updateLeaves(Cmds, Record, AccessMode, ToCleanUp);
  }

  static bool enqueueCommand(sycl::detail::Command *Cmd,
                             sycl::detail::EnqueueResultT &EnqueueResult,
                             sycl::detail::BlockingT Blocking) {
    std::vector<sycl::detail::Command *> ToCleanUp;
    return GraphProcessor::enqueueCommand(Cmd, EnqueueResult, ToCleanUp,
                                          Blocking);
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
  addCopyBack(sycl::detail::Requirement *Req,
              std::vector<sycl::detail::Command *> &ToEnqueue) {
    return MGraphBuilder.addCopyBack(Req, ToEnqueue);
  }

  sycl::detail::UpdateHostRequirementCommand *
  insertUpdateHostReqCmd(sycl::detail::MemObjRecord *Record,
                         sycl::detail::Requirement *Req,
                         const sycl::detail::QueueImplPtr &Queue,
                         std::vector<sycl::detail::Command *> &ToEnqueue) {
    return MGraphBuilder.insertUpdateHostReqCmd(Record, Req, Queue, ToEnqueue);
  }

  sycl::detail::EmptyCommand *
  addEmptyCmd(sycl::detail::Command *Cmd,
              const std::vector<sycl::detail::Requirement *> &Reqs,
              const sycl::detail::QueueImplPtr &Queue,
              sycl::detail::Command::BlockReason Reason,
              std::vector<sycl::detail::Command *> &ToEnqueue) {
    return MGraphBuilder.addEmptyCmd(Cmd, Reqs, Queue, Reason, ToEnqueue);
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
