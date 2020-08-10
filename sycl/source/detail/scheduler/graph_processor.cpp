//===-- graph_processor.cpp - SYCL Graph Processor --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/event_impl.hpp>
#include <detail/queue_impl.hpp>
#include <detail/scheduler/scheduler.hpp>

#include <memory>
#include <vector>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

static Command *getCommand(const EventImplPtr &Event) {
  return (Command *)Event->getCommand();
}

std::vector<EventImplPtr>
Scheduler::GraphProcessor::getWaitList(EventImplPtr Event) {
  Command *Cmd = getCommand(Event);
  // Command can be nullptr if user creates cl::sycl::event explicitly,
  // as such event is not mapped to any SYCL task.
  if (!Cmd)
    return {};
  std::vector<EventImplPtr> Result;
  for (const DepDesc &Dep : Cmd->MDeps) {
    if (Dep.MDepCommand)
      Result.push_back(Dep.MDepCommand->getEvent());
  }
  return Result;
}

void Scheduler::GraphProcessor::waitForEvent(EventImplPtr Event,
                                             ReadLockT &GraphReadLock) {
  Command *Cmd = getCommand(Event);
  // Command can be nullptr if user creates cl::sycl::event explicitly or the
  // event has been waited on by another thread
  if (!Cmd)
    return;

  EnqueueResultT Res;
  bool Enqueued = enqueueCommand(Cmd, Res, GraphReadLock, BLOCKING);
  if (!Enqueued && EnqueueResultT::SyclEnqueueFailed == Res.MResult)
    // TODO: Reschedule commands.
    throw runtime_error("Enqueue process failed.", PI_INVALID_OPERATION);

  GraphReadLock.unlock();

  Cmd->getEvent()->waitInternal();
}

bool Scheduler::GraphProcessor::enqueueCommand(Command *Cmd,
                                               EnqueueResultT &EnqueueResult,
                                               ReadLockT &GraphReadLock,
                                               BlockingT Blocking) {
  if (!Cmd || Cmd->isSuccessfullyEnqueued())
    return true;

  // Indicates whether dependency cannot be enqueued
  bool BlockedByDep = false;

  for (DepDesc &Dep : Cmd->MDeps) {
    const bool Enqueued =
        enqueueCommand(Dep.MDepCommand, EnqueueResult, GraphReadLock, Blocking);
    if (!Enqueued)
      switch (EnqueueResult.MResult) {
      case EnqueueResultT::SyclEnqueueFailed:
      default:
        // Exit immediately if a command fails to avoid enqueueing commands
        // result of which will be discarded.
        return false;
      case EnqueueResultT::SyclEnqueueBlocked:
        // If some dependency is blocked from enqueueing remember that, but
        // try to enqueue other dependencies(that can be ready for
        // enqueueing).
        BlockedByDep = true;
        break;
      }
  }

  // Exit if some command is blocked from enqueueing, the EnqueueResult is set
  // by the latest dependency which was blocked.
  if (BlockedByDep)
    return false;

  {
    GraphReadLock.unlock();
    bool Result = Cmd->enqueue(EnqueueResult, Blocking);
    GraphReadLock.lock();

    return Result;
  }
}

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
