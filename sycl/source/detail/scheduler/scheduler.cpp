//===-- scheduler.cpp - SYCL Scheduler --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CL/sycl/detail/sycl_mem_obj_i.hpp"
#include <CL/sycl/detail/queue_impl.hpp>
#include <CL/sycl/detail/scheduler/scheduler.hpp>
#include <CL/sycl/device_selector.hpp>

#include <memory>
#include <mutex>
#include <set>
#include <vector>

namespace cl {
namespace sycl {
namespace detail {

void Scheduler::waitForRecordToFinish(MemObjRecord *Record) {
  for (Command *Cmd : Record->MReadLeafs) {
    EnqueueResultT Res;
    bool Enqueued = GraphProcessor::enqueueCommand(Cmd, Res);
    if (!Enqueued && EnqueueResultT::FAILED == Res.MResult)
      throw runtime_error("Enqueue process failed.");
    GraphProcessor::waitForEvent(Cmd->getEvent());
  }
  for (Command *Cmd : Record->MWriteLeafs) {
    EnqueueResultT Res;
    bool Enqueued = GraphProcessor::enqueueCommand(Cmd, Res);
    if (!Enqueued && EnqueueResultT::FAILED == Res.MResult)
      throw runtime_error("Enqueue process failed.");
    GraphProcessor::waitForEvent(Cmd->getEvent());
  }
  for (AllocaCommandBase *AllocaCmd : Record->MAllocaCommands) {
    Command *ReleaseCmd = AllocaCmd->getReleaseCmd();
    EnqueueResultT Res;
    bool Enqueued = GraphProcessor::enqueueCommand(ReleaseCmd, Res);
    if (!Enqueued && EnqueueResultT::FAILED == Res.MResult)
      throw runtime_error("Enqueue process failed.");
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
    default:
      NewCmd = MGraphBuilder.addCG(std::move(CommandGroup), std::move(Queue));
    }

    // TODO: Check if lazy mode.
    EnqueueResultT Res;
    bool Enqueued = GraphProcessor::enqueueCommand(NewCmd, Res);
    if (!Enqueued && EnqueueResultT::FAILED == Res.MResult)
      throw runtime_error("Enqueue process failed.");
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
  EnqueueResultT Res;
  bool Enqueued = GraphProcessor::enqueueCommand(NewCmd, Res);
  if (!Enqueued && EnqueueResultT::FAILED == Res.MResult)
    throw runtime_error("Enqueue process failed.");
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
  std::lock_guard<std::mutex> lock(MGraphLock);
  GraphProcessor::waitForEvent(std::move(Event));
}

void Scheduler::removeMemoryObject(detail::SYCLMemObjI *MemObj) {
  std::lock_guard<std::mutex> lock(MGraphLock);

  MemObjRecord *Record = MGraphBuilder.getMemObjRecord(MemObj);
  if (!Record)
    // No operations were performed on the mem object
    return;
  waitForRecordToFinish(Record);
  MGraphBuilder.cleanupCommandsForRecord(Record);
  MGraphBuilder.removeRecordForMemObj(MemObj);
}

EventImplPtr Scheduler::addHostAccessor(Requirement *Req) {
  std::lock_guard<std::mutex> lock(MGraphLock);

  EventImplPtr RetEvent;
  Command *NewCmd = MGraphBuilder.addHostAccessor(Req, RetEvent);

  if (!NewCmd)
    return nullptr;
  EnqueueResultT Res;
  bool Enqueued = GraphProcessor::enqueueCommand(NewCmd, Res);
  if (!Enqueued && EnqueueResultT::FAILED == Res.MResult)
    throw runtime_error("Enqueue process failed.");
  return RetEvent;
}

Scheduler::Scheduler() {
  sycl::device HostDevice;
  DefaultHostQueue = QueueImplPtr(new queue_impl(
      HostDevice, /*AsyncHandler=*/{}, QueueOrder::Ordered, /*PropList=*/{}));
}

} // namespace detail
} // namespace sycl
} // namespace cl
