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
#include <iostream>

namespace sycl {
inline namespace _V1 {
namespace detail {

// Using the names being generated and the string are subject to change to
// something more meaningful to end-users as this will be visible in analysis
// tools that subscribe to this data
static std::string commandToName(Command::CommandType Type) {
  switch (Type) {
  case Command::CommandType::RUN_CG            : return "Command Group Action";
  case Command::CommandType::COPY_MEMORY       : return "Memory Transfer (Copy)";
  case Command::CommandType::ALLOCA            : return "Memory Allocation";
  case Command::CommandType::ALLOCA_SUB_BUF    : return "Sub Buffer Creation";
  case Command::CommandType::RELEASE           : return "Memory Deallocation";
  case Command::CommandType::MAP_MEM_OBJ       : return "Memory Transfer (Map)";
  case Command::CommandType::UNMAP_MEM_OBJ     : return "Memory Transfer (Unmap)";
  case Command::CommandType::UPDATE_REQUIREMENT: return "Host Accessor Creation/Buffer Lock";
  case Command::CommandType::EMPTY_TASK        : return "Host Accessor Destruction/Buffer Lock Release";
  default: return "Unknown Action";
  }
}

void Scheduler::GraphProcessor::waitForEvent(event_impl &Event,
                                             ReadLockT &GraphReadLock,
                                             std::vector<Command *> &ToCleanUp,
                                             bool LockTheLock, bool *Success) {
  Command *Cmd = Event.getCommand();
  // Command can be nullptr if user creates sycl::event explicitly or the
  // event has been waited on by another thread
  if (!Cmd)
    return;

  EnqueueResultT Res;
  bool Enqueued =
      enqueueCommand(Cmd, GraphReadLock, Res, ToCleanUp, Cmd, BLOCKING);
  if (!Enqueued && EnqueueResultT::SyclEnqueueFailed == Res.MResult)
    // TODO: Reschedule commands.
    throw exception(make_error_code(errc::runtime), "Enqueue process failed(0).");

  assert(Cmd->getEvent().get() == &Event);

  GraphReadLock.unlock();
  Event.waitInternal(Success);

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
  std::cerr << "[SYCL] ENTER: Scheduler::GraphProcessor::enqueueCommand(Command *, ReadLockT &, ...)\n";
  if (!Cmd) {
    std::cerr << "[SYCL]    -enqueueCommand: Cmd is nullptr --> return true\n";
    return true;
  }
  std::cerr << "[SYCL]    -enqueueCommand: Cmd = " << commandToName(Cmd->getType()) << "\n";
  if (Cmd->isSuccessfullyEnqueued()) {
    auto ret = handleBlockingCmd(Cmd, EnqueueResult, RootCommand, Blocking);
    std::cerr << "[SYCL]    -enqueueCommand: Cmd is successfully enqueued --> \n";
    std::cerr << "[SYCL] LEAVE: Scheduler::GraphProcessor::enqueueCommand(Command *, ReadLockT &, ...) --> " << (ret ? "true" : "false") << "\n";
    return ret;
  }

  // Exit early if the command is blocked and the enqueue type is non-blocking
  if (Cmd->isEnqueueBlocked() && !Blocking) {
    EnqueueResult = EnqueueResultT(EnqueueResultT::SyclEnqueueBlocked, Cmd);
    std::cerr << "[SYCL]    -enqueueCommand: Cmd is enqueue blocked --> Result = {" << static_cast<int>(EnqueueResult.MResult) << ", " << static_cast<int>(EnqueueResult.MErrCode) << "}\n";
    std::cerr << "[SYCL] LEAVE: Scheduler::GraphProcessor::enqueueCommand(Command *, ReadLockT &, ...) --> false\n";
    return false;
  }

  // Recursively enqueue all the implicit + explicit backend level dependencies
  // first and exit immediately if any of the commands cannot be enqueued.
  for (const EventImplPtr &Event : Cmd->getPreparedDepsEvents()) {
    if (Command *DepCmd = Event->getCommand()) {
      std::cerr << "[SYCL]    -enqueueCommand: enqueueing backend dependency command --> " << (DepCmd ? commandToName(DepCmd->getType()) : "nullptr") << "\n";
      if (!enqueueCommand(DepCmd, GraphReadLock, EnqueueResult, ToCleanUp,
                          RootCommand, Blocking)) {
        std::cerr << "[SYCL]    -enqueueCommand: failed to enqueue command --> Result = {" << static_cast<int>(EnqueueResult.MResult) << ", " << static_cast<int>(EnqueueResult.MErrCode) << "}\n";
        std::cerr << "[SYCL] LEAVE: Scheduler::GraphProcessor::enqueueCommand(Command *, ReadLockT &, ...) --> false\n";
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
  for (const EventImplPtr &Event : Cmd->getPreparedHostDepsEvents()) {
    if (Command *DepCmd = Event->getCommand()) {
      std::cerr << "[SYCL]    -enqueueCommand: enqueueing host dependency command --> " << (DepCmd ? commandToName(DepCmd->getType()) : "nullptr") << "\n";
      if (!enqueueCommand(DepCmd, GraphReadLock, EnqueueResult, ToCleanUp,
                          RootCommand, Blocking)) {
        std::cerr << "[SYCL]    -enqueueCommand: failed to enqueue command --> Result = {" << static_cast<int>(EnqueueResult.MResult) << ", " << static_cast<int>(EnqueueResult.MErrCode) << "}\n";
        std::cerr << "[SYCL] LEAVE: Scheduler::GraphProcessor::enqueueCommand(Command *, ReadLockT &, ...) --> false\n";
        return false;
      }
    }
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
  std::cerr << "[SYCL]    -enqueueCommand: enqueueing source command --> " << (Cmd ? commandToName(Cmd->getType()) : "nullptr") << "\n";
  bool Result = Cmd->enqueue(EnqueueResult, Blocking, ToCleanUp);
  if (Result) {
    std::cerr << "[SYCL]    -enqueueCommand: check if command blocks dependent commands\n";
    Result = handleBlockingCmd(Cmd, EnqueueResult, RootCommand, Blocking);
    std::cerr << "[SYCL]    -enqueueCommand: has blocking command result --> " << (Result ? "true" : "false") << "\n";
  }
  std::cerr << "[SYCL]    -enqueueCommand: final enqueue result --> Result = {" << static_cast<int>(EnqueueResult.MResult) << ", " << static_cast<int>(EnqueueResult.MErrCode) << "}\n";
  std::cerr << "[SYCL] LEAVE: Scheduler::GraphProcessor::enqueueCommand(Command *, ReadLockT &, ...) --> " << (Result ? "true" : "false") << "\n";
  return Result;
}

} // namespace detail
} // namespace _V1
} // namespace sycl
