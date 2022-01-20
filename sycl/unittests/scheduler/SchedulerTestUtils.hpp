//==---------- SchedulerTestUtils.hpp --- Scheduler unit tests -------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once
#include <CL/sycl/detail/cl.h>
#include <detail/queue_impl.hpp>
#include <detail/scheduler/commands.hpp>
#include <detail/scheduler/scheduler.hpp>
#include <detail/stream_impl.hpp>

#include <functional>
#include <gmock/gmock.h>
#include <vector>

// This header contains a few common classes/methods used in
// execution graph testing.

cl::sycl::detail::Requirement getMockRequirement();

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {
class Command;
} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)

class MockCommand : public cl::sycl::detail::Command {
public:
  MockCommand(cl::sycl::detail::QueueImplPtr Queue,
              cl::sycl::detail::Requirement Req,
              cl::sycl::detail::Command::CommandType Type =
                  cl::sycl::detail::Command::RUN_CG)
      : Command{Type, Queue}, MRequirement{std::move(Req)} {
    using namespace testing;
    ON_CALL(*this, enqueue)
        .WillByDefault(Invoke(this, &MockCommand::enqueueOrigin));
    EXPECT_CALL(*this, enqueue).Times(AnyNumber());
  }

  MockCommand(cl::sycl::detail::QueueImplPtr Queue,
              cl::sycl::detail::Command::CommandType Type =
                  cl::sycl::detail::Command::RUN_CG)
      : Command{Type, Queue}, MRequirement{std::move(getMockRequirement())} {
    using namespace testing;
    ON_CALL(*this, enqueue)
        .WillByDefault(Invoke(this, &MockCommand::enqueueOrigin));
    EXPECT_CALL(*this, enqueue).Times(AnyNumber());
  }

  void printDot(std::ostream &) const override {}
  void emitInstrumentationData() override {}

  const cl::sycl::detail::Requirement *getRequirement() const final {
    return &MRequirement;
  };

  cl_int enqueueImp() override { return MRetVal; }

  MOCK_METHOD3(enqueue, bool(cl::sycl::detail::EnqueueResultT &,
                             cl::sycl::detail::BlockingT,
                             std::vector<cl::sycl::detail::Command *> &));
  bool enqueueOrigin(cl::sycl::detail::EnqueueResultT &EnqueueResult,
                     cl::sycl::detail::BlockingT Blocking,
                     std::vector<cl::sycl::detail::Command *> &ToCleanUp) {
    return cl::sycl::detail::Command::enqueue(EnqueueResult, Blocking,
                                              ToCleanUp);
  }

  cl_int MRetVal = CL_SUCCESS;

  void waitForEventsCall(
      std::shared_ptr<cl::sycl::detail::queue_impl> Queue,
      std::vector<std::shared_ptr<cl::sycl::detail::event_impl>> &RawEvents,
      pi_event &Event) {
    Command::waitForEvents(Queue, RawEvents, Event);
  }

  std::shared_ptr<cl::sycl::detail::event_impl> getEvent() { return MEvent; }

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
  using cl::sycl::detail::Scheduler::addCG;
  using cl::sycl::detail::Scheduler::addCopyBack;
  using cl::sycl::detail::Scheduler::cleanupCommands;

  cl::sycl::detail::MemObjRecord *
  getOrInsertMemObjRecord(const cl::sycl::detail::QueueImplPtr &Queue,
                          cl::sycl::detail::Requirement *Req,
                          std::vector<cl::sycl::detail::Command *> &ToEnqueue) {
    return MGraphBuilder.getOrInsertMemObjRecord(Queue, Req, ToEnqueue);
  }

  void decrementLeafCountersForRecord(cl::sycl::detail::MemObjRecord *Rec) {
    MGraphBuilder.decrementLeafCountersForRecord(Rec);
  }

  void removeRecordForMemObj(cl::sycl::detail::SYCLMemObjI *MemObj) {
    MGraphBuilder.removeRecordForMemObj(MemObj);
  }

  void cleanupCommandsForRecord(cl::sycl::detail::MemObjRecord *Rec) {
    std::vector<std::shared_ptr<cl::sycl::detail::stream_impl>>
        StreamsToDeallocate;
    MGraphBuilder.cleanupCommandsForRecord(Rec, StreamsToDeallocate);
  }

  void addNodeToLeaves(cl::sycl::detail::MemObjRecord *Rec,
                       cl::sycl::detail::Command *Cmd,
                       cl::sycl::access::mode Mode,
                       std::vector<cl::sycl::detail::Command *> &ToEnqueue) {
    return MGraphBuilder.addNodeToLeaves(Rec, Cmd, Mode, ToEnqueue);
  }

  void updateLeaves(const std::set<cl::sycl::detail::Command *> &Cmds,
                    cl::sycl::detail::MemObjRecord *Record,
                    cl::sycl::access::mode AccessMode,
                    std::vector<cl::sycl::detail::Command *> &ToCleanUp) {
    return MGraphBuilder.updateLeaves(Cmds, Record, AccessMode, ToCleanUp);
  }

  static bool enqueueCommand(cl::sycl::detail::Command *Cmd,
                             cl::sycl::detail::EnqueueResultT &EnqueueResult,
                             cl::sycl::detail::BlockingT Blocking) {
    std::vector<cl::sycl::detail::Command *> ToCleanUp;
    return GraphProcessor::enqueueCommand(Cmd, EnqueueResult, ToCleanUp,
                                          Blocking);
  }

  cl::sycl::detail::AllocaCommandBase *
  getOrCreateAllocaForReq(cl::sycl::detail::MemObjRecord *Record,
                          const cl::sycl::detail::Requirement *Req,
                          cl::sycl::detail::QueueImplPtr Queue,
                          std::vector<cl::sycl::detail::Command *> &ToEnqueue) {
    return MGraphBuilder.getOrCreateAllocaForReq(Record, Req, Queue, ToEnqueue);
  }

  ReadLockT acquireGraphReadLock() { return ReadLockT{MGraphLock}; }

  cl::sycl::detail::Command *
  insertMemoryMove(cl::sycl::detail::MemObjRecord *Record,
                   cl::sycl::detail::Requirement *Req,
                   const cl::sycl::detail::QueueImplPtr &Queue,
                   std::vector<cl::sycl::detail::Command *> &ToEnqueue) {
    return MGraphBuilder.insertMemoryMove(Record, Req, Queue, ToEnqueue);
  }

  cl::sycl::detail::Command *
  addCopyBack(cl::sycl::detail::Requirement *Req,
              std::vector<cl::sycl::detail::Command *> &ToEnqueue) {
    return MGraphBuilder.addCopyBack(Req, ToEnqueue);
  }

  cl::sycl::detail::UpdateHostRequirementCommand *
  insertUpdateHostReqCmd(cl::sycl::detail::MemObjRecord *Record,
                         cl::sycl::detail::Requirement *Req,
                         const cl::sycl::detail::QueueImplPtr &Queue,
                         std::vector<cl::sycl::detail::Command *> &ToEnqueue) {
    return MGraphBuilder.insertUpdateHostReqCmd(Record, Req, Queue, ToEnqueue);
  }

  cl::sycl::detail::EmptyCommand *
  addEmptyCmd(cl::sycl::detail::Command *Cmd,
              const std::vector<cl::sycl::detail::Requirement *> &Reqs,
              const cl::sycl::detail::QueueImplPtr &Queue,
              cl::sycl::detail::Command::BlockReason Reason,
              std::vector<cl::sycl::detail::Command *> &ToEnqueue) {
    return MGraphBuilder.addEmptyCmd(Cmd, Reqs, Queue, Reason, ToEnqueue);
  }

  cl::sycl::detail::Command *
  addCG(std::unique_ptr<cl::sycl::detail::CG> CommandGroup,
        cl::sycl::detail::QueueImplPtr Queue,
        std::vector<cl::sycl::detail::Command *> &ToEnqueue) {
    return MGraphBuilder.addCG(std::move(CommandGroup), Queue, ToEnqueue);
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
