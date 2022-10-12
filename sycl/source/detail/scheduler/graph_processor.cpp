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

/* void Scheduler::GraphProcessor::waitForEvent(EventImplPtr Event, */
/*                                              ReadLockT &GraphReadLock, */
/*                                              std::vector<Command *> &ToCleanUp, */
/*                                              bool LockTheLock) { */
/*   Command *Cmd = getCommand(Event); */
/*   // Command can be nullptr if user creates sycl::event explicitly or the */
/*   // event has been waited on by another thread */
/*   if (!Cmd) */
/*     return; */


/*   EnqueueResultT Res; */
/*   std::vector<EventImplPtr> WaitList; */

/*   bool Enqueued = false; */

/*   while(true) { */
/*     Enqueued = enqueueCommand(Cmd, Res, ToCleanUp, WaitList, BLOCKING); */

/*     if (Enqueued) */
/*       break; */

/*     if (!Enqueued && EnqueueResultT::SyclEnqueueFailed == Res.MResult) */
/*       // TODO: Reschedule commands. */
/*       throw runtime_error("Enqueue process failed.", */
/*                           PI_ERROR_INVALID_OPERATION); */

/*     GraphReadLock.unlock(); */

/*     // Wait for the blocking commands. */
/*     for (EventImplPtr &Event : WaitList) */
/*       // If the command enqueued we can wait for its event completion directly */
/*       if (Event->getEnqueueStatus() == EnqueueResultT::SyclEnqueueSuccess) */
/*         Event->waitInternal(); */
/*       // Otherwise wait until unblocked and move further(we will enqueue it */
/*       // during the next attempt) */
/*       else { */
/*         if (Event->getEnqueueStatus() == EnqueueResultT::SyclEnqueueBlocked) { */
/* #ifdef VLAD_PRINT */
/*             std::cout << "Waiting Event.get() " << Event.get() << std::endl; */
/*             std::cout << "Waiting Event.getCommand() " << Event->getCommand() << std::endl; */
/* #endif */
/*         } */

/*         static bool ThrowOnBlock = getenv("SYCL_THROW_ON_BLOCK") != nullptr; */
/*         if (Event->getEnqueueStatus() == EnqueueResultT::SyclEnqueueBlocked && ThrowOnBlock) */
/*             throw sycl::runtime_error( */
/*                 std::string("Waiting for blocked command. Block reason: "), */
/*                 PI_ERROR_INVALID_OPERATION); */
/*         while (Event->getEnqueueStatus() == EnqueueResultT::SyclEnqueueBlocked) */
/*             // TODO: Sleep */
/*           ; */
/* #ifdef VLAD_PRINT */
/*             std::cout << "Waiting for  Event.getCommand() done " << Event->getCommand() << std::endl; */
/* #endif */
/*       } */
/*     // TODO: Handle other statuses? */

/*     WaitList.clear(); */

/*     GraphReadLock.lock(); */
/*   } */

/*   assert(Cmd->getEvent() == Event); */

/*   GraphReadLock.unlock(); */
/*   Event->waitInternal(); */

/*   if (LockTheLock) */
/*     GraphReadLock.lock(); */
/* } */

std::vector<EventImplPtr>
Scheduler::GraphProcessor::collectEventsForRecToFinish(MemObjRecord *Record) {
#ifdef XPTI_ENABLE_INSTRUMENTATION
  // Will contain the list of dependencies for the Release Command
  std::set<Command *> DepCommands;
  std::vector<EventImplPtr> DepEvents;
#endif
  for (Command *Cmd : Record->MReadLeaves) {
#ifdef XPTI_ENABLE_INSTRUMENTATION
    // Capture the dependencies
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
}

bool Scheduler::GraphProcessor::enqueueCommand(
    Command *Cmd, EnqueueResultT &EnqueueResult,
    std::vector<Command *> &ToCleanUp, std::vector<EventImplPtr> &WaitList,
    BlockingT Blocking) {

    bool Blocked = false;
  if (!Cmd || Cmd->isSuccessfullyEnqueued()) {
#ifdef VLAD_PRINT
    std::cout <<  "GraphProcessor::enqueueCommand " << Cmd << " already enqueued. Exiting" << std::endl;
#endif
    return true;
  }

#ifdef VLAD_PRINT
    std::cout <<  "GraphProcessor::enqueueCommand " << Cmd << std::endl;
    if(Cmd->getType() == Command::CommandType::RUN_CG)
        std::cout <<  "This is a host task " << Cmd << std::endl;
#endif


  /* if (!WaitList.empty()) { */
  /*   EnqueueResult = EnqueueResultT(EnqueueResultT::SyclEnqueueBlocked, Cmd); */
  /*   return false; */
  /* } */


  // Exit early if the command is blocked and the enqueue type is non-blocking
  /* if (Cmd->isEnqueueBlocked() && !Blocking) { */
  /*   EnqueueResult = EnqueueResultT(EnqueueResultT::SyclEnqueueBlocked, Cmd); */
  /*   return false; */
  /* } */

  // Recursively enqueue all the implicit + explicit backend level dependencies
  // first and exit immediately if any of the commands cannot be enqueued.
  for (const EventImplPtr &Event : Cmd->getPreparedDepsEvents()) {
    if (Command *DepCmd = static_cast<Command *>(Event->getCommand())){
#ifdef VLAD_PRINT
        std::cout <<  "GraphProcessor::enqueueCommand " << Cmd << ". ENqueue Dep = " << DepCmd << std::endl;
#endif
      if (!enqueueCommand(DepCmd, EnqueueResult, ToCleanUp, WaitList, Blocking)) {

#ifdef VLAD_PRINT
        std::cout <<  "GraphProcessor::enqueueCommand " << Cmd << ". Normal Dep is failed to enqueue, Exiting" << std::endl;
#endif
        Blocked = true;
        return false;
      }

    }
  }

  // Recursively enqueue all the implicit + explicit host dependencies and
  // exit immediately if any of the commands cannot be enqueued.
  // Host task execution is asynchronous. In current implementation enqueue for
  // this command will wait till host task completion by waitInternal call on
  // MHostDepsEvents. TO FIX: implement enqueue of blocked commands on host task
  // completion stage and eliminate this event waiting in enqueue.
  for (const EventImplPtr &Event : Cmd->getPreparedHostDepsEvents())
    if (Command *DepCmd = static_cast<Command *>(Event->getCommand())) {
#ifdef VLAD_PRINT
        std::cout <<  "GraphProcessor::enqueueCommand " << Cmd << ". ENqueue hostj Dep = " << DepCmd << std::endl;
#endif
      if(!enqueueCommand(DepCmd, EnqueueResult, ToCleanUp, WaitList, NON_BLOCKING)) {
        Blocked = true;
      }
    }


  if (!Blocked && Cmd->isEnqueueBlocked()) {
#ifdef VLAD_PRINT
    std::cout <<  "GraphProcessor::enqueueCommand " << Cmd << " is blocked . Adding to waitlit and Exiting" << std::endl;
#endif
    WaitList.push_back(Cmd->getEvent());
    EnqueueResult = EnqueueResultT(EnqueueResultT::SyclEnqueueBlocked, Cmd);
    Blocked = true;
    return false;
  }


  if (Blocked) {
#ifdef VLAD_PRINT
        std::cout <<  "GraphProcessor::enqueueCommand " << Cmd << ". Blocked Exiting" << std::endl;
        std::cout <<  "GraphProcessor::enqueueCommand " << Cmd << ". Wait list is not empty Exiting" << std::endl;
#endif
    EnqueueResult = EnqueueResultT(EnqueueResultT::SyclEnqueueBlocked, Cmd);
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
#ifdef VLAD_PRINT
        std::cout <<  "GraphProcessor::enqueueCommand " << Cmd << ". Finally enqueuing " << std::endl;
#endif
  return Cmd->enqueue(EnqueueResult, Blocking, ToCleanUp);
}

} // namespace detail
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
