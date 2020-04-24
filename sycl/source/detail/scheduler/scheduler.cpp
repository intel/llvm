//===-- scheduler.cpp - SYCL Scheduler --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CL/sycl/detail/sycl_mem_obj_i.hpp"
#include <CL/sycl/device_selector.hpp>
#include <detail/queue_impl.hpp>
#include <detail/scheduler/scheduler.hpp>

#include <memory>
#include <mutex>
#include <set>
#include <vector>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

EventImplPtr addHostAccessorToSchedulerInstance(Requirement *Req,
                                                const bool destructor) {
  return cl::sycl::detail::Scheduler::getInstance().
                                              addHostAccessor(Req, destructor);
}

void Scheduler::waitForRecordToFinish(MemObjRecord *Record) {
#ifdef XPTI_ENABLE_INSTRUMENTATION
  // Will contain the list of dependencies for the Release Command
  std::set<Command *> DepCommands;
#endif
  for (Command *Cmd : Record->MReadLeaves) {
    EnqueueResultT Res;
    bool Enqueued = GraphProcessor::enqueueCommand(Cmd, Res);
    if (!Enqueued && EnqueueResultT::SyclEnqueueFailed == Res.MResult)
      throw runtime_error("Enqueue process failed.", PI_INVALID_OPERATION);
#ifdef XPTI_ENABLE_INSTRUMENTATION
    // Capture the dependencies
    DepCommands.insert(Cmd);
#endif
    GraphProcessor::waitForEvent(Cmd->getEvent());
  }
  for (Command *Cmd : Record->MWriteLeaves) {
    EnqueueResultT Res;
    bool Enqueued = GraphProcessor::enqueueCommand(Cmd, Res);
    if (!Enqueued && EnqueueResultT::SyclEnqueueFailed == Res.MResult)
      throw runtime_error("Enqueue process failed.", PI_INVALID_OPERATION);
#ifdef XPTI_ENABLE_INSTRUMENTATION
    DepCommands.insert(Cmd);
#endif
    GraphProcessor::waitForEvent(Cmd->getEvent());
  }
  for (AllocaCommandBase *AllocaCmd : Record->MAllocaCommands) {
    Command *ReleaseCmd = AllocaCmd->getReleaseCmd();
    EnqueueResultT Res;
    bool Enqueued = GraphProcessor::enqueueCommand(ReleaseCmd, Res);
    if (!Enqueued && EnqueueResultT::SyclEnqueueFailed == Res.MResult)
      throw runtime_error("Enqueue process failed.", PI_INVALID_OPERATION);
#ifdef XPTI_ENABLE_INSTRUMENTATION
    // Report these dependencies to the Command so these dependencies can be
    // reported as edges
    ReleaseCmd->resolveReleaseDependencies(DepCommands);
#endif
    GraphProcessor::waitForEvent(ReleaseCmd->getEvent());
  }
}

EventImplPtr Scheduler::addCG(std::unique_ptr<detail::CG> CommandGroup,
                              QueueImplPtr Queue) {
  Command *NewCmd = nullptr;
  const bool IsKernel = CommandGroup->getType() == CG::KERNEL;
  {
    std::lock_guard<std::mutex> Lock(MGraphLock);

    switch (CommandGroup->getType()) {
    case CG::UPDATE_HOST:
      NewCmd = MGraphBuilder.addCGUpdateHost(std::move(CommandGroup),
                                             DefaultHostQueue);
      break;
    case CG::CODEPLAY_HOST_TASK:
      NewCmd = MGraphBuilder.addCG(std::move(CommandGroup), DefaultHostQueue);
      break;
    default:
      NewCmd = MGraphBuilder.addCG(std::move(CommandGroup), std::move(Queue));
    }

    // TODO: Check if lazy mode.
    EnqueueResultT Res;
    bool Enqueued = GraphProcessor::enqueueCommand(NewCmd, Res);
    if (!Enqueued && EnqueueResultT::SyclEnqueueFailed == Res.MResult)
      throw runtime_error("Enqueue process failed.", PI_INVALID_OPERATION);
  }

  if (IsKernel)
    ((ExecCGCommand *)NewCmd)->flushStreams();

  return NewCmd->getEvent();
}

EventImplPtr Scheduler::addCopyBack(Requirement *Req) {
  std::lock_guard<std::mutex> lock(MGraphLock);
  Command *NewCmd = MGraphBuilder.addCopyBack(Req);
  // Command was not creted because there were no operations with
  // buffer.
  if (!NewCmd)
    return nullptr;

  try {
    EnqueueResultT Res;
    bool Enqueued = GraphProcessor::enqueueCommand(NewCmd, Res);
    if (!Enqueued && EnqueueResultT::SyclEnqueueFailed == Res.MResult)
      throw runtime_error("Enqueue process failed.", PI_INVALID_OPERATION);
  } catch (...) {
    NewCmd->getQueue()->reportAsyncException(std::current_exception());
  }
  return NewCmd->getEvent();
}

#ifdef __GNUC__
// The init_priority here causes the constructor for scheduler to run relatively
// early, and therefore the destructor to run relatively late (after anything
// else that has no priority set, or has a priority higher than 2000).
Scheduler Scheduler::instance __attribute__((init_priority(2000)));
#else
#pragma warning(disable:4073)
#pragma init_seg(lib)
Scheduler Scheduler::instance;
#endif

Scheduler &Scheduler::getInstance() {
  return instance;
}

std::vector<EventImplPtr> Scheduler::getWaitList(EventImplPtr Event) {
  std::lock_guard<std::mutex> lock(MGraphLock);
  return GraphProcessor::getWaitList(std::move(Event));
}

void Scheduler::waitForEvent(EventImplPtr Event) {
  GraphProcessor::waitForEvent(std::move(Event));
}

void Scheduler::cleanupFinishedCommands(EventImplPtr FinishedEvent) {
  std::lock_guard<std::mutex> lock(MGraphLock);
  Command *FinishedCmd = static_cast<Command *>(FinishedEvent->getCommand());
  // The command might have been cleaned up (and set to nullptr) by another
  // thread
  if (FinishedCmd)
    MGraphBuilder.cleanupFinishedCommands(FinishedCmd);
}

void Scheduler::removeMemoryObject(detail::SYCLMemObjI *MemObj) {
  std::lock_guard<std::mutex> lock(MGraphLock);

  MemObjRecord *Record = MGraphBuilder.getMemObjRecord(MemObj);
  if (!Record)
    // No operations were performed on the mem object
    return;

  waitForRecordToFinish(Record);
  MGraphBuilder.decrementLeafCountersForRecord(Record);
  MGraphBuilder.cleanupCommandsForRecord(Record);
  MGraphBuilder.removeRecordForMemObj(MemObj);
}

EventImplPtr Scheduler::addHostAccessor(Requirement *Req,
                                        const bool destructor) {
  std::lock_guard<std::mutex> lock(MGraphLock);

  Command *NewCmd = MGraphBuilder.addHostAccessor(Req, destructor);

  if (!NewCmd)
    return nullptr;
  EnqueueResultT Res;
  bool Enqueued = GraphProcessor::enqueueCommand(NewCmd, Res);
  if (!Enqueued && EnqueueResultT::SyclEnqueueFailed == Res.MResult)
    throw runtime_error("Enqueue process failed.", PI_INVALID_OPERATION);
  return NewCmd->getEvent();
}

void Scheduler::releaseHostAccessor(Requirement *Req) {
  Command *const BlockedCmd = Req->MBlockedCmd;

  assert(BlockedCmd && "Can't find appropriate command to unblock");

  if (EventImplPtr Event = BlockedCmd->getEvent())
    if (Event->is_host()) {
      Event->setComplete();
    }

  BlockedCmd->MEnqueueStatus = EnqueueResultT::SyclEnqueueReady;

  enqueueLeavesOfReq(Req);
}

// static
void Scheduler::enqueueLeavesOfReq(const Requirement *const Req) {
  MemObjRecord* Record = Req->MSYCLMemObj->MRecord.get();
  auto EnqueueLeaves = [](CircularBuffer<Command *> &Leaves) {
    for (Command *Cmd : Leaves) {
      EnqueueResultT Res;
      bool Enqueued = GraphProcessor::enqueueCommand(Cmd, Res);
      if (!Enqueued && EnqueueResultT::SyclEnqueueFailed == Res.MResult)
        throw runtime_error("Enqueue process failed.", PI_INVALID_OPERATION);
    }
  };
  EnqueueLeaves(Record->MReadLeaves);
  EnqueueLeaves(Record->MWriteLeaves);
}

Scheduler::Scheduler() {
  sycl::device HostDevice;
  DefaultHostQueue = QueueImplPtr(new queue_impl(
      detail::getSyclObjImpl(HostDevice), /*AsyncHandler=*/{},
          QueueOrder::Ordered, /*PropList=*/{}));
}

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
