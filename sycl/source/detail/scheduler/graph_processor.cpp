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

void Scheduler::GraphProcessor::waitForEvent(EventImplPtr Event) {
  Command *Cmd = getCommand(Event);
  // Command can be nullptr if user creates cl::sycl::event explicitly or the
  // event has been waited on by another thread
  if (!Cmd)
    return;

  EnqueueResultT Res;
  bool Enqueued = enqueueCommand(Cmd, Res, BLOCKING);
  if (!Enqueued && EnqueueResultT::SyclEnqueueFailed == Res.MResult)
    // TODO: Reschedule commands.
    throw runtime_error("Enqueue process failed.", PI_INVALID_OPERATION);

  Cmd->getEvent()->waitInternal();
}

bool Scheduler::GraphProcessor::enqueueCommand(Command *Cmd,
                                               EnqueueResultT &EnqueueResult,
                                               BlockingT Blocking) {
  if (!Cmd || Cmd->isSuccessfullyEnqueued())
    return true;

  // Exit early if the command is blocked and the enqueue type is non-blocking
  if (Cmd->isEnqueueBlocked() && !Blocking) {
    EnqueueResult = EnqueueResultT(EnqueueResultT::SyclEnqueueBlocked, Cmd);
    return false;
  }

  // Recursively enqueue all the dependencies first and
  // exit immediately if any of the commands cannot be enqueued.
  for (DepDesc &Dep : Cmd->MDeps) {
    if (!enqueueCommand(Dep.MDepCommand, EnqueueResult, Blocking))
      return false;
  }

  // Asynchronous host operations (amongst dependencies of an arbitrary command)
  // are not supported (see Command::processDepEvent method). This impacts
  // operation of host-task feature a lot with hangs and long-runs. Hence we
  // have this workaround here.
  // This workaround is safe as long as the only asynchronous host operation we
  // have is a host task.
  // This may iterate over some of dependencies in Cmd->MDeps. Though, the
  // enqueue operation is idempotent and the second call will result in no-op.
  // TODO remove the workaround when proper fix for host-task dispatching is
  // implemented.
  for (const EventImplPtr &Event : Cmd->getPreparedHostDepsEvents()) {
    if (Command *DepCmd = static_cast<Command *>(Event->getCommand()))
      if (!enqueueCommand(DepCmd, EnqueueResult, Blocking))
        return false;
  }

  return Cmd->enqueue(EnqueueResult, Blocking);
}

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
