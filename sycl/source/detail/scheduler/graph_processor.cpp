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

static Command *getCommand(const EventImplPtr &Event) {
  return (Command *)Event->getCommand();
}

void Scheduler::GraphProcessor::waitForEvent(const EventImplPtr &Event,
                                             ReadLockT &GraphReadLock,
                                             std::vector<Command *> &ToCleanUp,
                                             bool LockTheLock) {
  Command *Cmd = getCommand(Event);
  // Command can be nullptr if user creates sycl::event explicitly or the
  // event has been waited on by another thread
  if (!Cmd)
    return;

  EnqueueResultT Res;
  bool Enqueued = enqueueCommand(Cmd, Res, ToCleanUp, Cmd, BLOCKING);
  if (!Enqueued && EnqueueResultT::SyclEnqueueFailed == Res.MResult)
    // TODO: Reschedule commands.
    throw runtime_error("Enqueue process failed.", PI_ERROR_INVALID_OPERATION);

  assert(Cmd->getEvent() == Event);

  GraphReadLock.unlock();

  static bool ThrowOnBlock = getenv("SYCL_THROW_ON_BLOCK") != nullptr;
    if (ThrowOnBlock)
      throw sycl::runtime_error(
          std::string("Waiting for blocked command. Block reason: ") +
              std::string(CmdAfterWait->getBlockReason()),
          PI_ERROR_INVALID_OPERATION);
  Event->waitInternal();

  if (Command* CmdAfterWait = Event->getCommand() && CmdAfterWait->isBlocking())
  {
    static bool ThrowOnBlock = getenv("SYCL_THROW_ON_BLOCK") != nullptr;
    if (ThrowOnBlock)
      throw sycl::runtime_error(
          std::string("Waiting for blocked command. Block reason: ") +
              std::string(CmdAfterWait->getBlockReason()),
          PI_ERROR_INVALID_OPERATION);
  #ifdef XPTI_ENABLE_INSTRUMENTATION
    // Scoped trace event notifier that emits a barrier begin and barrier end
    // event, which models the barrier while enqueuing along with the blocked
    // reason, as determined by the scheduler
    std::string Info = "enqueue.barrier[";
    Info += std::string(Cmd->getBlockReason()) + "]";
    emitInstrumentation(xpti::trace_barrier_begin, Info.c_str());
#endif

    // Wait if blocking. isBlocked path for task completion is handled above with Event->waitInternal().
    while (CmdAfterWait->MIsManuallyBlocked == true)
      ;
#ifdef XPTI_ENABLE_INSTRUMENTATION
    emitInstrumentation(xpti::trace_barrier_end, Info.c_str());
#endif
  }

  if (LockTheLock)
    GraphReadLock.lock();
}

bool Scheduler::GraphProcessor::handleBlockingCmd(Command *Cmd,
                                                  EnqueueResultT &EnqueueResult,
                                                  Command *RootCommand,
                                                  BlockingT Blocking) {
 
  static bool ThrowOnBlock = getenv("SYCL_THROW_ON_BLOCK") != nullptr;
  // No error to be returned for root command.
  // Blocking && !ThrowOnBlock means that we will wait for task in parent command enqueue if it is blocking and do not report it to user.
  if ((Cmd == RootCommand) || (Blocking && !ThrowOnBlock))
    return true;

  {
    std::lock_guard<std::mutex> Guard(Cmd->MBlockedUsersMutex);
    if (Cmd->isBlocking()) {
      if (Blocking && ThrowOnBlock)
        // Means that we are going to wait on Blocking command
        throw sycl::runtime_error(
          std::string("Waiting for blocked command. Block reason: ") +
              std::string(Cmd->getBlockReason()),
          PI_ERROR_INVALID_OPERATION);
      const EventImplPtr &RootCmdEvent = RootCommand->getEvent();
      Cmd->addBlockedUserUnique(RootCmdEvent);
      EnqueueResult = EnqueueResultT(EnqueueResultT::SyclEnqueueBlocked, Cmd);
      return false;
    }
  }
  return true;
}

bool Scheduler::GraphProcessor::enqueueCommand(
    Command *Cmd, EnqueueResultT &EnqueueResult,
    std::vector<Command *> &ToCleanUp, Command *RootCommand,
    BlockingT Blocking) {
  if (!Cmd)
    return true;
  if (Cmd->isSuccessfullyEnqueued())
    return handleBlockingCmd(Cmd, EnqueueResult, RootCommand, Blocking);

  // Recursively enqueue all the implicit + explicit backend level dependencies
  // first and exit immediately if any of the commands cannot be enqueued.
  for (const EventImplPtr &Event : Cmd->getPreparedDepsEvents()) {
    if (Command *DepCmd = static_cast<Command *>(Event->getCommand()))
      if (!enqueueCommand(DepCmd, EnqueueResult, ToCleanUp, RootCommand,
                          Blocking))
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
      if (!enqueueCommand(DepCmd, EnqueueResult, ToCleanUp, RootCommand,
                          Blocking))
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
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
