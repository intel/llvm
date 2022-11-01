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

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace detail {

Command *Scheduler::GraphProcessor::getCommand(const EventImplPtr &Event) {
  return (Command *)Event->getCommand();
}

std::vector<EventImplPtr>
Scheduler::GraphProcessor::collectEventsForRecToFinish(MemObjRecord *Record) {
  std::vector<EventImplPtr> DepEvents;

#ifdef XPTI_ENABLE_INSTRUMENTATION
  // Will contain the list of dependencies for the Release Command
  std::set<Command *> DepCommands;
#endif
  for (Command *Cmd : Record->MReadLeaves) {
#ifdef XPTI_ENABLE_INSTRUMENTATION
    DepCommands.insert(Cmd);
#endif
    DepEvents.push_back(Cmd->getEvent());
  }
  for (Command *Cmd : Record->MWriteLeaves) {
#ifdef XPTI_ENABLE_INSTRUMENTATION
    DepCommands.insert(Cmd);
#endif
    DepEvents.push_back(Cmd->getEvent());
  }

  for (AllocaCommandBase *AllocaCmd : Record->MAllocaCommands) {
    Command *ReleaseCmd = AllocaCmd->getReleaseCmd();
#ifdef XPTI_ENABLE_INSTRUMENTATION
    // Report these dependencies to the Command so these dependencies can be
    // reported as edges
    ReleaseCmd->resolveReleaseDependencies(DepCommands);
#endif
    DepEvents.push_back(ReleaseCmd->getEvent());
  }
  return DepEvents;
/* void Scheduler::GraphProcessor::waitForEvent(const EventImplPtr &Event, */
/*                                              ReadLockT &GraphReadLock, */
/*                                              std::vector<Command *> &ToCleanUp, */
/*                                              bool LockTheLock) { */
/*   Command *Cmd = getCommand(Event); */
/*   // Command can be nullptr if user creates sycl::event explicitly or the */
/*   // event has been waited on by another thread */
/*   if (!Cmd) */
/*     return; */

/*   EnqueueResultT Res; */
/*   bool Enqueued = enqueueCommand(Cmd, Res, ToCleanUp, BLOCKING); */
/*   if (!Enqueued && EnqueueResultT::SyclEnqueueFailed == Res.MResult) */
/*     // TODO: Reschedule commands. */
/*     throw runtime_error("Enqueue process failed.", PI_ERROR_INVALID_OPERATION); */

/*   assert(Cmd->getEvent() == Event); */

/*   GraphReadLock.unlock(); */
/*   Event->waitInternal(); */

/*   if (LockTheLock) */
/*     GraphReadLock.lock(); */
}

/* bool Scheduler::GraphProcessor::enqueueCommand( */
/*     Command *Cmd, EnqueueResultT &EnqueueResult, */
/*     std::vector<Command *> &ToCleanUp, BlockingT Blocking) { */
/*   // Repeat enqueue process until we finally enqueue the target command */
/*   while (true) { */
/*     EnqueueResultT Res; */
/*     { */
/*       ReadLockT Lock{MGraphLock}; */
/*       if (GraphProcessor::enqueueCommandImpl(GraphProcessor::getCommand(Event), */
/*                                              Res, ToCleanUp, Blocking)) */
/*         break; */
/*     } */

/*     if (Blocking == NON_BLOCKING) */
/*       break; */

/*     if (EnqueueResultT::SyclEnqueueFailed == Res.MResult) */
/*       // TODO: Reschedule commands. */
/*       throw runtime_error("Enqueue process failed.", */
/*                           PI_ERROR_INVALID_OPERATION); */

/*     assert(EnqueueResultT::SyclEnqueueBlocked == Res.MResult); */
/*     assert(!Res.MBlockingEvents.empty()); */

/*     // Wait for state change of the commands blocking the target command from */
/*     // being enqueued. The state may change to completed or ready to enqueue. */
/*     // In both cases need to repeat enqueue. */
/*     for (EventImplPtr &Event : Res.MBlockingEvents) */
/*       Event->waitStateChange(); */
/*   } */
/* } */

bool Scheduler::GraphProcessor::enqueueCommand(
    Command *Cmd, EnqueueResultT &EnqueueResult,
    std::vector<Command *> &ToCleanUp, BlockingT Blocking) {
  if (!Cmd || Cmd->isSuccessfullyEnqueued())
    return true;

  // Exit early if the command is blocked and the enqueue type is non-blocking
  if (Cmd->isEnqueueBlocked() && !Blocking) {
    EnqueueResult = EnqueueResultT(EnqueueResultT::SyclEnqueueBlocked, Cmd);
    return false;
  }

  // Recursively enqueue all the implicit + explicit backend level dependencies
  // first and exit immediately if any of the commands cannot be enqueued.
  for (const EventImplPtr &Event : Cmd->getPreparedDepsEvents())
    if (Command *DepCmd = static_cast<Command *>(Event->getCommand()))
      if (!enqueueCommand(DepCmd, EnqueueResult, ToCleanUp, Blocking))
        return false;

  // Recursively enqueue all the implicit + explicit host dependencies and
  // exit immediately if any of the commands cannot be enqueued.
  // Host task execution is asynchronous. In current implementation enqueue for
  // this command will wait till host task completion by waitInternal call on
  // MHostDepsEvents. TO FIX: implement enqueue of blocked commands on host task
  // completion stage and eliminate this event waiting in enqueue.
  for (const EventImplPtr &Event : Cmd->getPreparedHostDepsEvents())
    if (Command *DepCmd = static_cast<Command *>(Event->getCommand()))
      if (!enqueueCommand(DepCmd, EnqueueResult, ToCleanUp, Blocking))
        return false;

  if (Cmd->isEnqueueBlocked()) {
    EnqueueResult = EnqueueResultT(EnqueueResultT::SyclEnqueueBlocked, Cmd);
    EnqueueResult.MBlockingEvents.push_back(Cmd->getEvent());
    return false;
  }

  // Only graph read lock is to be held here.
  // Enqueue process of a command may last quite a time. Having graph locked can
  // introduce some thread starving (i.e. when the other thread attempts to
  // acquire write lock and add a command to graph). Releasing read lock without
  // other safety measures isn't an option here as the other thread could go
  // into graph cleanup process (due to some event complete) and remove some
  // dependencies from dependencies of the user of this command.
  // An example: command A depends on commands B and C. This thread wants to
  // enqueue A. Hence, it needs to enqueue B and C. So this thread gets into
  // dependency list and starts enqueueing B right away. The other thread waits
  // on completion of C and starts cleanup process. This thread is still in the
  // middle of enqueue of B. The other thread modifies dependency list of A by
  // removing C out of it. Iterators become invalid.
  return Cmd->enqueue(EnqueueResult, ToCleanUp);
}

} // namespace detail
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
