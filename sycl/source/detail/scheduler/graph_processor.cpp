//===-- graph_processor.cpp - SYCL Graph Processor --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/detail/event_impl.hpp>
#include <CL/sycl/detail/queue_impl.hpp>
#include <CL/sycl/detail/scheduler/scheduler.hpp>

#include <memory>
#include <vector>

namespace cl {
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
  assert(Cmd && "Event has no associated command?");
  EnqueueResultT Res;
  bool Enqueued = enqueueCommand(Cmd, Res, BLOCKING);
  if (!Enqueued && EnqueueResultT::FAILED == Res.MResult)
    // TODO: Reschedule commands.
    throw runtime_error("Enqueue process failed.");

  RT::PiEvent &CLEvent = Cmd->getEvent()->getHandleRef();
  if (CLEvent)
    PI_CALL(RT::piEventsWait, 1, &CLEvent);
}

bool Scheduler::GraphProcessor::enqueueCommand(Command *Cmd,
                                               EnqueueResultT &EnqueueResult,
                                               BlockingT Blocking) {
  if (!Cmd || Cmd->isEnqueued())
    return true;

  // Indicates whether dependency cannot be enqueued
  bool BlockedByDep = false;

  for (DepDesc &Dep : Cmd->MDeps) {
    const bool Enqueued =
        enqueueCommand(Dep.MDepCommand, EnqueueResult, Blocking);
    if (!Enqueued)
      switch (EnqueueResult.MResult) {
      case EnqueueResultT::FAILED:
      default:
        // Exit immediately if a command fails to avoid enqueueing commands
        // result of which will be discarded.
        return false;
      case EnqueueResultT::BLOCKED:
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

  return Cmd->enqueue(EnqueueResult, Blocking);
}

} // namespace detail
} // namespace sycl
} // namespace cl
