//===-- scheduler.cpp - SYCL Scheduler --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CL/sycl/detail/sycl_mem_obj_i.hpp"
#include <CL/sycl/device_selector.hpp>
#include <detail/global_handler.hpp>
#include <detail/queue_impl.hpp>
#include <detail/scheduler/scheduler.hpp>
#include <detail/scheduler/scheduler_helpers.hpp>
#include <detail/stream_impl.hpp>

#include <chrono>
#include <cstdio>
#include <memory>
#include <mutex>
#include <set>
#include <thread>
#include <vector>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

void Scheduler::waitForRecordToFinish(MemObjRecord *Record,
                                      ReadLockT &GraphReadLock) {
#ifdef XPTI_ENABLE_INSTRUMENTATION
  // Will contain the list of dependencies for the Release Command
  std::set<Command *> DepCommands;
#endif
  for (Command *Cmd : Record->MReadLeaves) {
    EnqueueResultT Res;
    bool Enqueued = GraphProcessor::enqueueCommand(Cmd, Res);
    if (!Enqueued && EnqueueResultT::SyclEnqueueFailed == Res.MResult)
      throw runtime_error("Enqueue process failed.", PI_INVALID_OPERATION);
#ifdef XPTI_ENABLE_INSTRUMENTATION
    // Capture the dependencies
    DepCommands.insert(Cmd);
#endif
    GraphProcessor::waitForEvent(Cmd->getEvent(), GraphReadLock);
  }
  for (Command *Cmd : Record->MWriteLeaves) {
    EnqueueResultT Res;
    bool Enqueued = GraphProcessor::enqueueCommand(Cmd, Res);
    if (!Enqueued && EnqueueResultT::SyclEnqueueFailed == Res.MResult)
      throw runtime_error("Enqueue process failed.", PI_INVALID_OPERATION);
#ifdef XPTI_ENABLE_INSTRUMENTATION
    DepCommands.insert(Cmd);
#endif
    GraphProcessor::waitForEvent(Cmd->getEvent(), GraphReadLock);
  }
  for (AllocaCommandBase *AllocaCmd : Record->MAllocaCommands) {
    Command *ReleaseCmd = AllocaCmd->getReleaseCmd();
    EnqueueResultT Res;
    bool Enqueued = GraphProcessor::enqueueCommand(ReleaseCmd, Res);
    if (!Enqueued && EnqueueResultT::SyclEnqueueFailed == Res.MResult)
      throw runtime_error("Enqueue process failed.", PI_INVALID_OPERATION);
#ifdef XPTI_ENABLE_INSTRUMENTATION
    // Report these dependencies to the Command so these dependencies can be
    // reported as edges
    ReleaseCmd->resolveReleaseDependencies(DepCommands);
#endif
    GraphProcessor::waitForEvent(ReleaseCmd->getEvent(), GraphReadLock);
  }
}

EventImplPtr Scheduler::addCG(std::unique_ptr<detail::CG> CommandGroup,
                              QueueImplPtr Queue) {
  EventImplPtr NewEvent = nullptr;
  const bool IsKernel = CommandGroup->getType() == CG::KERNEL;
  std::vector<Command *> AuxiliaryCmds;
  const bool IsHostKernel = CommandGroup->getType() == CG::RUN_ON_HOST_INTEL;
  vector_class<StreamImplPtr> Streams;

  if (IsKernel) {
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

  {
    WriteLockT Lock(MGraphLock, std::defer_lock);
    acquireWriteLock(Lock);

    Command *NewCmd = nullptr;
    switch (CommandGroup->getType()) {
    case CG::UPDATE_HOST:
      NewCmd = MGraphBuilder.addCGUpdateHost(std::move(CommandGroup),
                                             DefaultHostQueue, AuxiliaryCmds);
      break;
    case CG::CODEPLAY_HOST_TASK:
      NewCmd = MGraphBuilder.addCG(std::move(CommandGroup), DefaultHostQueue,
                                   AuxiliaryCmds);
      break;
    default:
      NewCmd = MGraphBuilder.addCG(std::move(CommandGroup), std::move(Queue),
                                   AuxiliaryCmds);
    }
    NewEvent = NewCmd->getEvent();
  }

  {
    ReadLockT Lock(MGraphLock);

    Command *NewCmd = static_cast<Command *>(NewEvent->getCommand());

    EnqueueResultT Res;
    bool Enqueued;

    auto CleanUp = [&]() {
      if (NewCmd && (NewCmd->MDeps.size() == 0 && NewCmd->MUsers.size() == 0)) {
        if (IsHostKernel)
          static_cast<ExecCGCommand *>(NewCmd)->releaseCG();

        NewEvent->setCommand(nullptr);
        delete NewCmd;
      }
    };

    for (Command *Cmd : AuxiliaryCmds) {
      Enqueued = GraphProcessor::enqueueCommand(Cmd, Res);
      try {
        if (!Enqueued && EnqueueResultT::SyclEnqueueFailed == Res.MResult)
          throw runtime_error("Auxiliary enqueue process failed.",
                              PI_INVALID_OPERATION);
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
        bool Enqueued = GraphProcessor::enqueueCommand(NewCmd, Res);
        if (!Enqueued && EnqueueResultT::SyclEnqueueFailed == Res.MResult)
          throw runtime_error("Enqueue process failed.", PI_INVALID_OPERATION);
      } catch (...) {
        // enqueueCommand() func and if statement above may throw an exception,
        // so destroy required resources to avoid memory leak
        CleanUp();
        std::rethrow_exception(std::current_exception());
      }

      // If there are no memory dependencies decouple and free the command.
      // Though, dismiss ownership of native kernel command group as it's
      // resources may be in use by backend and synchronization point here is
      // at native kernel execution finish.
      CleanUp();
    }
  }

  for (auto StreamImplPtr : Streams) {
    StreamImplPtr->flush();
  }

  return NewEvent;
}

EventImplPtr Scheduler::addCopyBack(Requirement *Req) {
  std::vector<Command *> AuxiliaryCmds;
  Command *NewCmd = nullptr;
  {
    WriteLockT Lock(MGraphLock, std::defer_lock);
    acquireWriteLock(Lock);
    NewCmd = MGraphBuilder.addCopyBack(Req, AuxiliaryCmds);
    // Command was not creted because there were no operations with
    // buffer.
    if (!NewCmd)
      return nullptr;
  }

  try {
    ReadLockT Lock(MGraphLock);
    EnqueueResultT Res;
    bool Enqueued;

    for (Command *Cmd : AuxiliaryCmds) {
      Enqueued = GraphProcessor::enqueueCommand(Cmd, Res);
      if (!Enqueued && EnqueueResultT::SyclEnqueueFailed == Res.MResult)
        throw runtime_error("Enqueue process failed.", PI_INVALID_OPERATION);
    }

    Enqueued = GraphProcessor::enqueueCommand(NewCmd, Res);
    if (!Enqueued && EnqueueResultT::SyclEnqueueFailed == Res.MResult)
      throw runtime_error("Enqueue process failed.", PI_INVALID_OPERATION);
  } catch (...) {
    NewCmd->getQueue()->reportAsyncException(std::current_exception());
  }
  return NewCmd->getEvent();
}

Scheduler &Scheduler::getInstance() {
  return GlobalHandler::instance().getScheduler();
}

std::vector<EventImplPtr> Scheduler::getWaitList(EventImplPtr Event) {
  ReadLockT Lock(MGraphLock);
  return GraphProcessor::getWaitList(std::move(Event));
}

void Scheduler::waitForEvent(EventImplPtr Event) {
  ReadLockT Lock(MGraphLock);
  // It's fine to leave the lock unlocked upon return from waitForEvent as
  // there's no more actions to do here with graph
  GraphProcessor::waitForEvent(std::move(Event), Lock, /*LockTheLock=*/false);
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

void Scheduler::cleanupFinishedCommands(EventImplPtr FinishedEvent) {
  // We are going to traverse a graph of finished commands. Gather stream
  // objects from these commands if any and deallocate buffers for these stream
  // objects, this is needed to guarantee that streamed data is printed and
  // resources are released.
  std::vector<std::shared_ptr<stream_impl>> StreamsToDeallocate;
  {
    // Avoiding deadlock situation, where one thread is in the process of
    // enqueueing (with a locked mutex) a currently blocked task that waits for
    // another thread which is stuck at attempting cleanup.
    WriteLockT Lock(MGraphLock, std::try_to_lock);
    if (Lock.owns_lock()) {
      auto FinishedCmd = static_cast<Command *>(FinishedEvent->getCommand());
      // The command might have been cleaned up (and set to nullptr) by another
      // thread
      if (FinishedCmd)
        MGraphBuilder.cleanupFinishedCommands(FinishedCmd, StreamsToDeallocate);
    }
  }
  deallocateStreams(StreamsToDeallocate);
}

void Scheduler::removeMemoryObject(detail::SYCLMemObjI *MemObj) {
  // We are going to traverse a graph of finished commands. Gather stream
  // objects from these commands if any and deallocate buffers for these stream
  // objects, this is needed to guarantee that streamed data is printed and
  // resources are released.
  std::vector<std::shared_ptr<stream_impl>> StreamsToDeallocate;

  {
    MemObjRecord *Record = nullptr;
    WriteLockT Lock(MGraphLock, std::defer_lock);

    {
      acquireWriteLock(Lock);

      Record = MGraphBuilder.getMemObjRecord(MemObj);
      if (!Record)
        // No operations were performed on the mem object
        return;

      Lock.unlock();
    }

    {
      // This only needs a shared mutex as it only involves enqueueing and
      // awaiting for events
      ReadLockT Lock(MGraphLock);
      waitForRecordToFinish(Record, Lock);
    }

    {
      acquireWriteLock(Lock);
      MGraphBuilder.decrementLeafCountersForRecord(Record);
      MGraphBuilder.cleanupCommandsForRecord(Record, StreamsToDeallocate);
      MGraphBuilder.removeRecordForMemObj(MemObj);
    }
  }
  deallocateStreams(StreamsToDeallocate);
}

EventImplPtr Scheduler::addHostAccessor(Requirement *Req) {
  std::vector<Command *> AuxiliaryCmds;
  Command *NewCmd = nullptr;

  {
    WriteLockT Lock(MGraphLock, std::defer_lock);
    acquireWriteLock(Lock);

    NewCmd = MGraphBuilder.addHostAccessor(Req, AuxiliaryCmds);
  }

  if (!NewCmd)
    return nullptr;

  {
    ReadLockT ReadLock(MGraphLock);
    EnqueueResultT Res;
    bool Enqueued;

    for (Command *Cmd : AuxiliaryCmds) {
      Enqueued = GraphProcessor::enqueueCommand(Cmd, Res);
      if (!Enqueued && EnqueueResultT::SyclEnqueueFailed == Res.MResult)
        throw runtime_error("Enqueue process failed.", PI_INVALID_OPERATION);
    }

    Enqueued = GraphProcessor::enqueueCommand(NewCmd, Res);
    if (!Enqueued && EnqueueResultT::SyclEnqueueFailed == Res.MResult)
      throw runtime_error("Enqueue process failed.", PI_INVALID_OPERATION);
  }

  return NewCmd->getEvent();
}

void Scheduler::releaseHostAccessor(Requirement *Req) {
  Command *const BlockedCmd = Req->MBlockedCmd;

  ReadLockT Lock(MGraphLock);

  assert(BlockedCmd && "Can't find appropriate command to unblock");

  BlockedCmd->MEnqueueStatus = EnqueueResultT::SyclEnqueueReady;

  enqueueLeavesOfReqUnlocked(Req);
}

// static
void Scheduler::enqueueLeavesOfReqUnlocked(const Requirement *const Req) {
  MemObjRecord *Record = Req->MSYCLMemObj->MRecord.get();
  auto EnqueueLeaves = [](LeavesCollection &Leaves) {
    for (Command *Cmd : Leaves) {
      EnqueueResultT Res;
      bool Enqueued = GraphProcessor::enqueueCommand(Cmd, Res);
      if (!Enqueued && EnqueueResultT::SyclEnqueueFailed == Res.MResult)
        throw runtime_error("Enqueue process failed.", PI_INVALID_OPERATION);
    }
  };

  EnqueueLeaves(Record->MReadLeaves);
  EnqueueLeaves(Record->MWriteLeaves);
}

void Scheduler::allocateStreamBuffers(stream_impl *Impl,
                                      size_t StreamBufferSize,
                                      size_t FlushBufferSize) {
  std::lock_guard<std::recursive_mutex> lock(StreamBuffersPoolMutex);
  StreamBuffersPool.insert(
      {Impl, new StreamBuffers(StreamBufferSize, FlushBufferSize)});
}

void Scheduler::deallocateStreamBuffers(stream_impl *Impl) {
  std::lock_guard<std::recursive_mutex> lock(StreamBuffersPoolMutex);
  delete StreamBuffersPool[Impl];
  StreamBuffersPool.erase(Impl);
}

Scheduler::Scheduler() {
  sycl::device HostDevice;
  DefaultHostQueue = QueueImplPtr(
      new queue_impl(detail::getSyclObjImpl(HostDevice), /*AsyncHandler=*/{},
                     /*PropList=*/{}));
}

Scheduler::~Scheduler() {
  DefaultHostQueue->stopThreadPool();
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
}

void Scheduler::acquireWriteLock(WriteLockT &Lock) {
#ifdef _WIN32
  // Avoiding deadlock situation for MSVC. std::shared_timed_mutex specification
  // does not specify a priority for shared and exclusive accesses. It will be a
  // deadlock in MSVC's std::shared_timed_mutex implementation, if exclusive
  // access occurs after shared access.
  // TODO: after switching to C++17, change std::shared_timed_mutex to
  // std::shared_mutex and use std::lock_guard here both for Windows and Linux.
  while (!Lock.try_lock_for(std::chrono::milliseconds(10))) {
    // Without yield while loop acts like endless while loop and occupies the
    // whole CPU when multiple command groups are created in multiple host
    // threads
    std::this_thread::yield();
  }
#else
  // It is a deadlock on UNIX in implementation of lock and lock_shared, if
  // try_lock in the loop above will be executed, so using a single lock here
  Lock.lock();
#endif // _WIN32
}

MemObjRecord *Scheduler::getMemObjRecord(const Requirement *const Req) {
  return Req->MSYCLMemObj->MRecord.get();
}

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
