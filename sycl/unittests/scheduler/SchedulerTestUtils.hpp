//==---------- SchedulerTestUtils.hpp --- Scheduler unit tests -------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once
#include <detail/handler_impl.hpp>
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
inline namespace _V1 {
namespace detail {
class Command;
} // namespace detail
} // namespace _V1
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
  using sycl::detail::Scheduler::checkLeavesCompletion;
  using sycl::detail::Scheduler::cleanupCommands;
  using sycl::detail::Scheduler::MDeferredMemObjRelease;

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
    MGraphBuilder.cleanupCommandsForRecord(Rec);
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
    RWLockT MockLock;
    ReadLockT MockReadLock(MockLock);
    std::vector<sycl::detail::Command *> ToCleanUp;
    return GraphProcessor::enqueueCommand(Cmd, MockReadLock, EnqueueResult,
                                          ToCleanUp, Cmd, Blocking);
  }

  sycl::detail::AllocaCommandBase *
  getOrCreateAllocaForReq(sycl::detail::MemObjRecord *Record,
                          const sycl::detail::Requirement *Req,
                          sycl::detail::QueueImplPtr Queue,
                          std::vector<sycl::detail::Command *> &ToEnqueue) {
    return MGraphBuilder.getOrCreateAllocaForReq(Record, Req, Queue, ToEnqueue);
  }

  ReadLockT acquireGraphReadLock() { return ReadLockT{MGraphLock}; }
  WriteLockT acquireOriginSchedGraphWriteLock() {
    Scheduler &Sched = Scheduler::getInstance();
    return WriteLockT(Sched.MGraphLock, std::defer_lock);
  }

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
    return MGraphBuilder.addCG(std::move(CommandGroup), Queue, ToEnqueue)
        .NewCmd;
  }

  void cancelFusion(sycl::detail::QueueImplPtr Queue,
                    std::vector<sycl::detail::Command *> &ToEnqueue) {
    MGraphBuilder.cancelFusion(Queue, ToEnqueue);
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
          /*ElementSize*/ 0,
          /*Offset*/ size_t(0)};
}

class MockHandler : public sycl::handler {
public:
  MockHandler(std::shared_ptr<sycl::detail::queue_impl> Queue, bool IsHost)
      : sycl::handler(Queue, IsHost) {}
  // Methods
  using sycl::handler::addReduction;
  using sycl::handler::getType;
  using sycl::handler::MImpl;

  sycl::detail::NDRDescT &getNDRDesc() { return MNDRDesc; }
  sycl::detail::code_location &getCodeLoc() { return MCodeLoc; }
  sycl::detail::CG::CGTYPE &getCGType() { return MCGType; }
  std::vector<std::shared_ptr<sycl::detail::stream_impl>> &getStreamStorage() {
    return MStreamStorage;
  }
  std::unique_ptr<sycl::detail::HostKernelBase> &getHostKernel() {
    return MHostKernel;
  }
  std::vector<std::vector<char>> &getArgsStorage() {
    return CGData.MArgsStorage;
  }
  std::vector<sycl::detail::AccessorImplPtr> &getAccStorage() {
    return CGData.MAccStorage;
  }
  std::vector<std::shared_ptr<const void>> &getSharedPtrStorage() {
    return CGData.MSharedPtrStorage;
  }
  std::vector<sycl::detail::Requirement *> &getRequirements() {
    return CGData.MRequirements;
  }
  std::vector<sycl::detail::EventImplPtr> &getEvents() {
    return CGData.MEvents;
  }
  std::vector<sycl::detail::ArgDesc> &getArgs() { return MArgs; }
#ifdef __INTEL_PREVIEW_BREAKING_CHANGES
  std::string getKernelName() { return MKernelName.c_str(); }
#else
  std::string &getKernelName() { return MKernelName; }
#endif
  std::shared_ptr<sycl::detail::kernel_impl> &getKernel() { return MKernel; }
  std::unique_ptr<sycl::detail::HostTask> &getHostTask() { return MHostTask; }
  std::shared_ptr<sycl::detail::queue_impl> &getQueue() { return MQueue; }

  void setType(sycl::detail::CG::CGTYPE Type) {
    static_cast<sycl::handler *>(this)->MCGType = Type;
  }

  template <typename KernelType, typename ArgType, int Dims,
            typename KernelName>
  void setHostKernel(KernelType Kernel) {
    static_cast<sycl::handler *>(this)->MHostKernel.reset(
        new sycl::detail::HostKernel<KernelType, ArgType, Dims>(Kernel));
  }

  template <int Dims> void setNDRangeDesc(sycl::nd_range<Dims> Range) {
    static_cast<sycl::handler *>(this)->MNDRDesc.set(std::move(Range));
  }

  void addStream(const sycl::detail::StreamImplPtr &Stream) {
    sycl::handler::addStream(Stream);
  }

  std::unique_ptr<sycl::detail::CG> finalize() {
    throw sycl::runtime_error("Unhandled type of command group",
                              PI_ERROR_INVALID_OPERATION);

    return nullptr;
  }
};

class MockHandlerCustomFinalize : public MockHandler {
public:
  MockHandlerCustomFinalize(std::shared_ptr<sycl::detail::queue_impl> Queue,
                            bool IsHost)
      : MockHandler(Queue, IsHost) {}

  std::unique_ptr<sycl::detail::CG> finalize() {
    std::unique_ptr<sycl::detail::CG> CommandGroup;
    sycl::detail::CG::StorageInitHelper CGData(
        getArgsStorage(), getAccStorage(), getSharedPtrStorage(),
        getRequirements(), getEvents());
    switch (getType()) {
    case sycl::detail::CG::Kernel: {
      CommandGroup.reset(new sycl::detail::CGExecKernel(
          getNDRDesc(), std::move(getHostKernel()), getKernel(),
          std::move(MImpl->MKernelBundle), std::move(CGData), getArgs(),
          getKernelName(), getStreamStorage(), MImpl->MAuxiliaryResources,
          getCGType(), {}, getCodeLoc()));
      break;
    }
    case sycl::detail::CG::CodeplayHostTask: {
      CommandGroup.reset(new sycl::detail::CGHostTask(
          std::move(getHostTask()), getQueue(), getQueue()->getContextImplPtr(),
          getArgs(), std::move(CGData), getCGType(), getCodeLoc()));
      break;
    }
    default:
      throw sycl::runtime_error("Unhandled type of command group",
                                PI_ERROR_INVALID_OPERATION);
    }

    return CommandGroup;
  }
};
