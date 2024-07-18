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
inline namespace _V1 {
namespace detail {

static Command *getCommand(const EventImplPtr &Event) {
  return (Command *)Event->getCommand();
}

void Scheduler::GraphProcessor::waitForEvent(const EventImplPtr &Event,
                                             ReadLockT &GraphReadLock,
                                             std::vector<Command *> &ToCleanUp,
                                             bool LockTheLock, bool *Success) {
  Command *Cmd = getCommand(Event);
  // Command can be nullptr if user creates sycl::event explicitly or the
  // event has been waited on by another thread
  if (!Cmd)
    return;

  EnqueueResultT Res;
  bool Enqueued =
      enqueueCommand(Cmd, GraphReadLock, Res, ToCleanUp, Cmd, BLOCKING);
  if (!Enqueued && EnqueueResultT::SyclEnqueueFailed == Res.MResult)
    // TODO: Reschedule commands.
    throw exception(make_error_code(errc::runtime), "Enqueue process failed.");

  assert(Cmd->getEvent() == Event);

  GraphReadLock.unlock();
  Event->waitInternal(Success);

  if (LockTheLock)
    GraphReadLock.lock();
}

bool Scheduler::GraphProcessor::handleBlockingCmd(Command *Cmd,
                                                  EnqueueResultT &EnqueueResult,
                                                  Command *RootCommand,
                                                  BlockingT Blocking) {
  if (Cmd == RootCommand || Blocking)
    return true;
  {
    std::lock_guard<std::mutex> Guard(Cmd->MBlockedUsersMutex);
    if (Cmd->isBlocking()) {
      const EventImplPtr &RootCmdEvent = RootCommand->getEvent();
      Cmd->addBlockedUserUnique(RootCmdEvent);
      EnqueueResult = EnqueueResultT(EnqueueResultT::SyclEnqueueBlocked, Cmd);

      // Blocked command will be enqueued asynchronously from submission so we
      // need to keep current root code location to report failure properly.
      RootCommand->copySubmissionCodeLocation();
      return false;
    }
  }
  return true;
}

bool Scheduler::GraphProcessor::enqueueCommand(
    Command *Cmd, ReadLockT &GraphReadLock, EnqueueResultT &EnqueueResult,
    std::vector<Command *> &ToCleanUp, Command *RootCommand,
    BlockingT Blocking) {
  if (!Cmd)
    return true;
  if (Cmd->isSuccessfullyEnqueued())
    return handleBlockingCmd(Cmd, EnqueueResult, RootCommand, Blocking);

  if (KernelFusionCommand *FusionCmd = isPartOfActiveFusion(Cmd)) {
    // The fusion is still in-flight, but some other event/command depending
    // on one of the kernels in the fusion list has triggered it to be
    // enqueued. To avoid circular dependencies and deadlocks, we will need to
    // cancel fusion here and enqueue the kernels in the fusion list right
    // away.
    printFusionWarning("Aborting fusion because synchronization with one of "
                       "the kernels in the fusion list was requested");
    // We need to unlock the read lock, as cancelFusion in the scheduler will
    // acquire a write lock to alter the graph.
    GraphReadLock.unlock();
    // Cancel fusion will take care of enqueueing all the kernels.
    Scheduler::getInstance().cancelFusion(FusionCmd->getQueue());
    // Lock the read lock again.
    GraphReadLock.lock();
    // The fusion (placeholder) command should have been enqueued by
    // cancelFusion.
    if (FusionCmd->isSuccessfullyEnqueued()) {
      return true;
    }
  }

  // Exit early if the command is blocked and the enqueue type is non-blocking
  if (Cmd->isEnqueueBlocked() && !Blocking) {
    EnqueueResult = EnqueueResultT(EnqueueResultT::SyclEnqueueBlocked, Cmd);
    return false;
  }

  // Recursively enqueue all the implicit + explicit backend level dependencies
  // first and exit immediately if any of the commands cannot be enqueued.
  for (const EventImplPtr &Event : Cmd->getPreparedDepsEvents()) {
    if (Command *DepCmd = static_cast<Command *>(Event->getCommand()))
      if (!enqueueCommand(DepCmd, GraphReadLock, EnqueueResult, ToCleanUp,
                          RootCommand, Blocking))
        return false;
  }

  // Recursively enqueue all the implicit + explicit host dependencies and
  // exit immediately if any of the commands cannot be enqueued.
  // Host task execution is asynchronous. In current implementation enqueue for
  // this command will wait till host task completion by waitInternal call on
  // MHostDepsEvents. TO FIX: implement enqueue of blocked commands on host task
  // completion stage and eliminate this event waiting in enqueue.
  for (const EventImplPtr &Event : Cmd->getPreparedHostDepsEvents()) {
    if (Command *DepCmd = static_cast<Command *>(Event->getCommand()))
      if (!enqueueCommand(DepCmd, GraphReadLock, EnqueueResult, ToCleanUp,
                          RootCommand, Blocking))
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
  bool Result = Cmd->enqueue(EnqueueResult, Blocking, ToCleanUp);
  if (Result)
    Result = handleBlockingCmd(Cmd, EnqueueResult, RootCommand, Blocking);
  return Result;
}

} // namespace detail
} // namespace _V1
} // namespace sycl
