//===-- scheduler.cpp - SYCL Scheduler --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "detail/sycl_mem_obj_i.hpp"
#include <detail/global_handler.hpp>
#include <detail/queue_impl.hpp>
#include <detail/scheduler/scheduler.hpp>
#include <detail/scheduler/scheduler_helpers.hpp>
#include <detail/stream_impl.hpp>
#include <sycl/device_selector.hpp>

#include <chrono>
#include <cstdio>
#include <memory>
#include <mutex>
#include <set>
#include <thread>
#include <vector>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace detail {

// PUBLIC
EventImplPtr Scheduler::addCG(std::unique_ptr<detail::CG> CommandGroup,
                              const QueueImplPtr &Queue) {
  EventImplPtr NewEvent = nullptr;
  const CG::CGTYPE Type = CommandGroup->getType();
  std::vector<Command *> AuxiliaryCmds;
  std::vector<StreamImplPtr> Streams;

  if (Type == CG::Kernel) {
    Streams = ((CGExecKernel *)CommandGroup.get())->getStreams();
    // Stream's flush buffer memory is mainly initialized in stream's __init
    // method. However, this method is not available on host device.
    // Initializing stream's flush buffer on the host side in a separate task.
    if (Queue->is_host()) {
      for (const StreamImplPtr &Stream : Streams) {
        initStream(Stream, Queue);
      }
    }
  }

  { // Start graph write lock scope
    WriteLockT Lock = acquireWriteLock();

    Command *NewCmd = nullptr;
    switch (Type) {
    case CG::UpdateHost:
      NewCmd = MGraphBuilder.addCGUpdateHost(std::move(CommandGroup),
                                             DefaultHostQueue, AuxiliaryCmds);
      break;
    case CG::CodeplayHostTask:
      NewCmd = MGraphBuilder.addCG(std::move(CommandGroup), DefaultHostQueue,
                                   AuxiliaryCmds);
      break;
    default:
      NewCmd = MGraphBuilder.addCG(std::move(CommandGroup), std::move(Queue),
                                   AuxiliaryCmds);
    }
    NewEvent = NewCmd->getEvent();
  } // End graph write lock scope

  std::vector<Command *> ToCleanUp;

  { // Start graph read lock scope
    ReadLockT Lock = acquireReadLock();

    Command *NewCmd = static_cast<Command *>(NewEvent->getCommand());

    EnqueueResultT Res;
    bool Enqueued = false;

    auto CleanUp = [&]() {
      if (NewCmd && (NewCmd->MDeps.size() == 0 && NewCmd->MUsers.size() == 0)) {
        NewEvent->setCommand(nullptr);
        delete NewCmd;
      }
    };

    for (Command *Cmd : AuxiliaryCmds) {
      std::vector<EventImplPtr> WaitList;
      Enqueued = GraphProcessor::enqueueCommand(Cmd, Res, ToCleanUp, WaitList);
      try {
        if (!Enqueued && EnqueueResultT::SyclEnqueueFailed == Res.MResult)
          throw runtime_error("Auxiliary enqueue process failed.",
                              PI_ERROR_INVALID_OPERATION);
      } catch (...) {
        // enqueueCommand() func and if statement above may throw an exception,
        // so destroy required resources to avoid memory leak
        CleanUp();
        std::rethrow_exception(std::current_exception());
      }
    }

    if (NewCmd) {
      // TODO: Check if lazy mode.
      EnqueueResultT Res;
      try {
        std::vector<EventImplPtr> WaitList;
        bool Enqueued =
            GraphProcessor::enqueueCommand(NewCmd, Res, ToCleanUp, WaitList);
        if (!Enqueued && EnqueueResultT::SyclEnqueueFailed == Res.MResult)
          throw runtime_error("Enqueue process failed.",
                              PI_ERROR_INVALID_OPERATION);
      } catch (...) {
        // enqueueCommand() func and if statement above may throw an exception,
        // so destroy required resources to avoid memory leak
        CleanUp();
        std::rethrow_exception(std::current_exception());
      }
    }
  } // End graph read lock scope

  {
    WriteLockT Lock = acquireWriteLock();
    cleanupCommands(ToCleanUp);
  }

  for (auto &StreamImplPtr : Streams)
    StreamImplPtr->flush(NewEvent);

  return NewEvent;
}

// PUBLIC
EventImplPtr Scheduler::addCopyBack(Requirement *Req) {
  std::vector<Command *> AuxiliaryCmds;
  Command *NewCmd = nullptr;
  EventImplPtr NewEvent;
  { // Start graph write lock
    WriteLockT Lock = acquireWriteLock();
    NewCmd = MGraphBuilder.addCopyBack(Req, AuxiliaryCmds);
    // Command was not creted because there were no operations with
    // buffer.
    if (!NewCmd)
      return nullptr;
    NewEvent = NewCmd->getEvent();
  } // End graph write lock

  std::vector<Command *> ToCleanUp;
  try { // Start graph read lock
    ReadLockT Lock = acquireReadLock();
    EnqueueResultT Res;
    bool Enqueued = false;

    for (Command *Cmd : AuxiliaryCmds) {
      std::vector<EventImplPtr> WaitList;
      Enqueued = GraphProcessor::enqueueCommand(Cmd, Res, ToCleanUp, WaitList);
      if (!Enqueued && EnqueueResultT::SyclEnqueueFailed == Res.MResult)
        throw runtime_error("Enqueue process failed.",
                            PI_ERROR_INVALID_OPERATION);
    }

    std::vector<EventImplPtr> WaitList;
    Enqueued = GraphProcessor::enqueueCommand(NewCmd, Res, ToCleanUp, WaitList);
    if (!Enqueued && EnqueueResultT::SyclEnqueueFailed == Res.MResult)
      throw runtime_error("Enqueue process failed.",
                          PI_ERROR_INVALID_OPERATION);
  } // End graph read lock
  catch (...) {
    NewCmd->getQueue()->reportAsyncException(std::current_exception());
  }

  {
    WriteLockT Lock = acquireWriteLock();
    cleanupCommands(ToCleanUp);
  }
  return NewEvent;
}

// PUBLIC
Scheduler &Scheduler::getInstance() {
  return GlobalHandler::instance().getScheduler();
}

// PUBLIC
void Scheduler::waitForEvent(const EventImplPtr &Event) {
  // It's fine to leave the lock unlocked upon return from waitForEvent as
  // there's no more actions to do here with graph
  std::vector<Command *> ToCleanUp;
  {
    EnqueueResultT Res;
    std::vector<EventImplPtr> WaitList;
    bool Enqueued = false;

    while (true) {
      {
        ReadLockT Lock = acquireReadLock();

        Command *Cmd = GraphProcessor::getCommand(Event);
        // Command can be nullptr if user creates sycl::event explicitly or the
        // event has been waited on by another thread
        if (!Cmd)
          break;

        if ((Enqueued = GraphProcessor::enqueueCommand(Cmd, Res, ToCleanUp,
                                                      WaitList, BLOCKING)))
          break;
      }

      if (EnqueueResultT::SyclEnqueueFailed == Res.MResult)
        // TODO: CleanUp commands?
        // TODO: Reschedule commands.
        throw runtime_error("Enqueue process failed.",
                            PI_ERROR_INVALID_OPERATION);

      // Wait for the blocking commands.
      for (EventImplPtr &Event : WaitList)
        Event->waitStateChange();
    }
  }

  Event->waitInternal();

  {
    WriteLockT Lock = acquireWriteLock();
    cleanupCommands(ToCleanUp);
  }
}

static void deallocateStreams(
    std::vector<std::shared_ptr<stream_impl>> &StreamsToDeallocate) {
  // Deallocate buffers for stream objects of the finished commands. Iterate in
  // reverse order because it is the order of commands execution.
  for (auto StreamImplPtr = StreamsToDeallocate.rbegin();
       StreamImplPtr != StreamsToDeallocate.rend(); ++StreamImplPtr)
    detail::Scheduler::getInstance().deallocateStreamBuffers(
        StreamImplPtr->get());
}

// PUBLIC
void Scheduler::cleanupFinishedCommands(EventImplPtr FinishedEvent) {
  // We are going to traverse a graph of finished commands. Gather stream
  // objects from these commands if any and deallocate buffers for these stream
  // objects, this is needed to guarantee that streamed data is printed and
  // resources are released.
  std::vector<std::shared_ptr<stream_impl>> StreamsToDeallocate;
  // Similar to streams, we also collect the auxiliary resources used by the
  // commands. Cleanup will make sure the commands do not own the resources
  // anymore, so we just need them to survive the graph lock then they can die
  // as they go out of scope.
  std::vector<std::shared_ptr<const void>> AuxResourcesToDeallocate;
  {
    // Avoiding deadlock situation, where one thread is in the process of
    // enqueueing (with a locked mutex) a currently blocked task that waits for
    // another thread which is stuck at attempting cleanup.
    WriteLockT Lock = acquireWriteLock();
    if (Lock.owns_lock()) {
      auto FinishedCmd = static_cast<Command *>(FinishedEvent->getCommand());
      // The command might have been cleaned up (and set to nullptr) by another
      // thread
      if (FinishedCmd)
        MGraphBuilder.cleanupFinishedCommands(FinishedCmd, StreamsToDeallocate,
                                              AuxResourcesToDeallocate);
    }
  }
  deallocateStreams(StreamsToDeallocate);
}

// PUBLIC
void Scheduler::removeMemoryObject(detail::SYCLMemObjI *MemObj) {
  // We are going to traverse a graph of finished commands. Gather stream
  // objects from these commands if any and deallocate buffers for these stream
  // objects, this is needed to guarantee that streamed data is printed and
  // resources are released.
  std::vector<std::shared_ptr<stream_impl>> StreamsToDeallocate;
  // Similar to streams, we also collect the auxiliary resources used by the
  // commands. Cleanup will make sure the commands do not own the resources
  // anymore, so we just need them to survive the graph lock then they can die
  // as they go out of scope.
  std::vector<std::shared_ptr<const void>> AuxResourcesToDeallocate;

  {
    MemObjRecord *Record = nullptr;

    std::vector<EventImplPtr> EventsToWait;
    {
      // This only needs a shared mutex as it only involves enqueueing and
      // awaiting for events
      ReadLockT Lock = acquireReadLock();

      Record = MGraphBuilder.getMemObjRecord(MemObj);
      if (!Record)
        // No operations were performed on the mem object
        return;

      EventsToWait = GraphProcessor::collectEventsForRecToFinish(Record);
    }

    for (const EventImplPtr &Event : EventsToWait)
      Scheduler::waitForEvent(Event);

    {
      WriteLockT Lock = acquireWriteLock();
      MGraphBuilder.decrementLeafCountersForRecord(Record);
      MGraphBuilder.cleanupCommandsForRecord(Record, StreamsToDeallocate,
                                             AuxResourcesToDeallocate);
      MGraphBuilder.removeRecordForMemObj(MemObj);
    }
  }
  deallocateStreams(StreamsToDeallocate);
}

// PUBLIC
EventImplPtr Scheduler::addHostAccessor(Requirement *Req) {
  std::vector<Command *> AuxiliaryCmds;
  EventImplPtr NewCmdEvent = nullptr;

  {
    WriteLockT Lock = acquireWriteLock();

    Command *NewCmd = MGraphBuilder.addHostAccessor(Req, AuxiliaryCmds);
    if (!NewCmd)
      return nullptr;
    NewCmdEvent = NewCmd->getEvent();
  }

  std::vector<Command *> ToCleanUp;
  {
    ReadLockT ReadLock = acquireReadLock();
    EnqueueResultT Res;
    bool Enqueued;

    for (Command *Cmd : AuxiliaryCmds) {
      std::vector<EventImplPtr> WaitList;
      Enqueued = GraphProcessor::enqueueCommand(Cmd, Res, ToCleanUp, WaitList);
      if (!Enqueued && EnqueueResultT::SyclEnqueueFailed == Res.MResult)
        throw runtime_error("Enqueue process failed.",
                            PI_ERROR_INVALID_OPERATION);
    }

    if (Command *NewCmd = static_cast<Command *>(NewCmdEvent->getCommand())) {
      std::vector<EventImplPtr> WaitList;
      Enqueued =
          GraphProcessor::enqueueCommand(NewCmd, Res, ToCleanUp, WaitList);
      if (!Enqueued && EnqueueResultT::SyclEnqueueFailed == Res.MResult)
        throw runtime_error("Enqueue process failed.",
                            PI_ERROR_INVALID_OPERATION);
    }
  }

  {
    WriteLockT Lock = acquireWriteLock();
    cleanupCommands(ToCleanUp);
  }
  return NewCmdEvent;
}

// PUBLIC
void Scheduler::releaseHostAccessor(Requirement *Req) {
  Command *const BlockedCmd = Req->MBlockedCmd;

  std::vector<Command *> ToCleanUp;
  {
    ReadLockT ReadLock = acquireReadLock();

    assert(BlockedCmd && "Can't find appropriate command to unblock");

    BlockedCmd->MEnqueueStatus = EnqueueResultT::SyclEnqueueReady;

    enqueueLeavesOfReq(Req, ToCleanUp);
  }
  {
    WriteLockT Lock = acquireWriteLock();
    cleanupCommands(ToCleanUp);
  }
}

// PUBLIC
void Scheduler::NotifyHostTaskCompletion(Command *Cmd, Command *BlockingCmd) {
  // Completing command's event along with unblocking enqueue readiness of
  // empty command may lead to quick deallocation of MThisCmd by some cleanup
  // process. Thus we'll copy deps prior to completing of event and unblocking
  // of empty command.
  // Also, it's possible to have record deallocated prior to enqueue process.
  // Thus we employ read-lock of graph.

  std::vector<Command *> ToCleanUp;
  {
    ReadLockT Lock = acquireReadLock();

    std::vector<DepDesc> Deps = Cmd->MDeps;

    // update self-event status
    Cmd->getEvent()->setComplete();

    BlockingCmd->MEnqueueStatus = EnqueueResultT::SyclEnqueueReady;

    for (const DepDesc &Dep : Deps)
      Scheduler::enqueueLeavesOfReq(Dep.MDepRequirement, ToCleanUp);
  }
  { 
    WriteLockT Lock = acquireWriteLock();
    cleanupCommands(ToCleanUp);
  }
}

// PUBLIC
void Scheduler::allocateStreamBuffers(stream_impl *Impl,
                                      size_t StreamBufferSize,
                                      size_t FlushBufferSize) {
  std::lock_guard<std::recursive_mutex> lock(StreamBuffersPoolMutex);
  StreamBuffersPool.insert(
      {Impl, new StreamBuffers(StreamBufferSize, FlushBufferSize)});
}



// ==================================
// PRIVATE
// ==================================

void Scheduler::enqueueLeavesOfReq(const Requirement *const Req,
                                   std::vector<Command *> &ToCleanUp) {
  MemObjRecord *Record = Req->MSYCLMemObj->MRecord.get();
  auto EnqueueLeaves = [&ToCleanUp](LeavesCollection &Leaves) {
    for (Command *Cmd : Leaves) {
      EnqueueResultT Res;
      std::vector<EventImplPtr> WaitList;
      bool Enqueued =
          GraphProcessor::enqueueCommand(Cmd, Res, ToCleanUp, WaitList);
      if (!Enqueued && EnqueueResultT::SyclEnqueueFailed == Res.MResult)
        throw runtime_error("Enqueue process failed.",
                            PI_ERROR_INVALID_OPERATION);
    }
  };

  EnqueueLeaves(Record->MReadLeaves);
  EnqueueLeaves(Record->MWriteLeaves);
}

void Scheduler::deallocateStreamBuffers(stream_impl *Impl) {
  std::lock_guard<std::recursive_mutex> lock(StreamBuffersPoolMutex);
  delete StreamBuffersPool[Impl];
  StreamBuffersPool.erase(Impl);
}

Scheduler::Scheduler() {
  sycl::device HostDevice =
      createSyclObjFromImpl<device>(device_impl::getHostDeviceImpl());
  sycl::context HostContext{HostDevice};
  DefaultHostQueue = QueueImplPtr(
      new queue_impl(detail::getSyclObjImpl(HostDevice),
                     detail::getSyclObjImpl(HostContext), /*AsyncHandler=*/{},
                     /*PropList=*/{}));
}

Scheduler::~Scheduler() {
  // By specification there are several possible sync points: buffer
  // destruction, wait() method of a queue or event. Stream doesn't introduce
  // any synchronization point. It is guaranteed that stream is flushed and
  // resources are released only if one of the listed sync points was used for
  // the kernel. Otherwise resources for stream will not be released, issue a
  // warning in this case.
  if (pi::trace(pi::TraceLevel::PI_TRACE_BASIC)) {
    std::lock_guard<std::recursive_mutex> lock(StreamBuffersPoolMutex);
    if (!StreamBuffersPool.empty())
      fprintf(
          stderr,
          "\nWARNING: Some commands may have not finished the execution and "
          "not all resources were released. Please be sure that all kernels "
          "have synchronization points.\n\n");
  }
  // There might be some commands scheduled for post enqueue cleanup that
  // haven't been freed because of the graph mutex being locked at the time,
  // clean them up now.
  { 
    WriteLockT Lock = acquireWriteLock();
    cleanupCommands({});
  }
}

MemObjRecord *Scheduler::getMemObjRecord(const Requirement *const Req) {
  return Req->MSYCLMemObj->MRecord.get();
}

void Scheduler::cleanupCommands(const std::vector<Command *> &Cmds) {
  for (Command *Cmd : Cmds)
    MGraphBuilder.cleanupCommand(Cmd);

  std::vector<Command *> DeferredCleanupCommands;
  {
    std::lock_guard<std::mutex> Lock{MDeferredCleanupMutex};
    std::swap(DeferredCleanupCommands, MDeferredCleanupCommands);
  }
  for (Command *Cmd : DeferredCleanupCommands)
    MGraphBuilder.cleanupCommand(Cmd);
}

} // namespace detail
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
