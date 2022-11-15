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

void Scheduler::waitForRecordToFinish(MemObjRecord *Record,
                                      ReadLockT &GraphReadLock) {
#ifdef XPTI_ENABLE_INSTRUMENTATION
  // Will contain the list of dependencies for the Release Command
  std::set<Command *> DepCommands;
#endif
  std::vector<Command *> ToCleanUp;
  for (Command *Cmd : Record->MReadLeaves) {
    EnqueueResultT Res;
    bool Enqueued = GraphProcessor::enqueueCommand(Cmd, Res, ToCleanUp);
    if (!Enqueued && EnqueueResultT::SyclEnqueueFailed == Res.MResult)
      throw runtime_error("Enqueue process failed.",
                          PI_ERROR_INVALID_OPERATION);
#ifdef XPTI_ENABLE_INSTRUMENTATION
    // Capture the dependencies
    DepCommands.insert(Cmd);
#endif
    GraphProcessor::waitForEvent(Cmd->getEvent(), GraphReadLock, ToCleanUp);
  }
  for (Command *Cmd : Record->MWriteLeaves) {
    EnqueueResultT Res;
    bool Enqueued = GraphProcessor::enqueueCommand(Cmd, Res, ToCleanUp);
    if (!Enqueued && EnqueueResultT::SyclEnqueueFailed == Res.MResult)
      throw runtime_error("Enqueue process failed.",
                          PI_ERROR_INVALID_OPERATION);
#ifdef XPTI_ENABLE_INSTRUMENTATION
    DepCommands.insert(Cmd);
#endif
    GraphProcessor::waitForEvent(Cmd->getEvent(), GraphReadLock, ToCleanUp);
  }
  for (AllocaCommandBase *AllocaCmd : Record->MAllocaCommands) {
    Command *ReleaseCmd = AllocaCmd->getReleaseCmd();
    EnqueueResultT Res;
    bool Enqueued = GraphProcessor::enqueueCommand(ReleaseCmd, Res, ToCleanUp);
    if (!Enqueued && EnqueueResultT::SyclEnqueueFailed == Res.MResult)
      throw runtime_error("Enqueue process failed.",
                          PI_ERROR_INVALID_OPERATION);
#ifdef XPTI_ENABLE_INSTRUMENTATION
    // Report these dependencies to the Command so these dependencies can be
    // reported as edges
    ReleaseCmd->resolveReleaseDependencies(DepCommands);
#endif
    GraphProcessor::waitForEvent(ReleaseCmd->getEvent(), GraphReadLock,
                                 ToCleanUp);
  }
}

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

  {
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
  }

  std::vector<Command *> ToCleanUp;
  {
    ReadLockT Lock = acquireReadLock();

    Command *NewCmd = static_cast<Command *>(NewEvent->getCommand());

    EnqueueResultT Res;
    bool Enqueued;

    auto CleanUp = [&]() {
      if (NewCmd && (NewCmd->MDeps.size() == 0 && NewCmd->MUsers.size() == 0)) {
        NewEvent->setCommand(nullptr);
        delete NewCmd;
      }
    };

    for (Command *Cmd : AuxiliaryCmds) {
      Enqueued = GraphProcessor::enqueueCommand(Cmd, Res, ToCleanUp);
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

      // Marking the event associated with the last command (NewCmd) as user
      // visible, in order to calculate subimission time, specifically for
      // level-zero plugin.
      if (!Queue->is_host() &&
          Queue->getPlugin().getBackend() == backend::ext_oneapi_level_zero) {
        bool isUserVisible = true;
        Queue->getPlugin().call<PiApiKind::piSetEventProperty>(
            &NewEvent->getHandleRef(), IS_USER_VISIBLE, 0, &isUserVisible);
      }

      EnqueueResultT Res;
      try {
        bool Enqueued = GraphProcessor::enqueueCommand(NewCmd, Res, ToCleanUp);
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
  }
  cleanupCommands(ToCleanUp);

  for (auto StreamImplPtr : Streams) {
    StreamImplPtr->flush(NewEvent);
  }

  return NewEvent;
}

EventImplPtr Scheduler::addCopyBack(Requirement *Req) {
  std::vector<Command *> AuxiliaryCmds;
  Command *NewCmd = nullptr;
  {
    WriteLockT Lock = acquireWriteLock();
    NewCmd = MGraphBuilder.addCopyBack(Req, AuxiliaryCmds);
    // Command was not creted because there were no operations with
    // buffer.
    if (!NewCmd)
      return nullptr;
  }

  std::vector<Command *> ToCleanUp;
  try {
    ReadLockT Lock = acquireReadLock();
    EnqueueResultT Res;
    bool Enqueued;

    for (Command *Cmd : AuxiliaryCmds) {
      Enqueued = GraphProcessor::enqueueCommand(Cmd, Res, ToCleanUp);
      if (!Enqueued && EnqueueResultT::SyclEnqueueFailed == Res.MResult)
        throw runtime_error("Enqueue process failed.",
                            PI_ERROR_INVALID_OPERATION);
    }

    Enqueued = GraphProcessor::enqueueCommand(NewCmd, Res, ToCleanUp);
    if (!Enqueued && EnqueueResultT::SyclEnqueueFailed == Res.MResult)
      throw runtime_error("Enqueue process failed.",
                          PI_ERROR_INVALID_OPERATION);
  } catch (...) {
    NewCmd->getQueue()->reportAsyncException(std::current_exception());
  }
  EventImplPtr NewEvent = NewCmd->getEvent();
  cleanupCommands(ToCleanUp);
  return NewEvent;
}

Scheduler &Scheduler::getInstance() {
  return GlobalHandler::instance().getScheduler();
}

void Scheduler::waitForEvent(const EventImplPtr &Event) {
  ReadLockT Lock = acquireReadLock();
  // It's fine to leave the lock unlocked upon return from waitForEvent as
  // there's no more actions to do here with graph
  std::vector<Command *> ToCleanUp;
  GraphProcessor::waitForEvent(std::move(Event), Lock, ToCleanUp,
                               /*LockTheLock=*/false);
  cleanupCommands(ToCleanUp);
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

void Scheduler::cleanupFinishedCommands(const EventImplPtr &FinishedEvent) {
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
    WriteLockT Lock(MGraphLock, std::try_to_lock);
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

    {
      // This only needs a shared mutex as it only involves enqueueing and
      // awaiting for events
      ReadLockT Lock = acquireReadLock();

      Record = MGraphBuilder.getMemObjRecord(MemObj);
      if (!Record)
        // No operations were performed on the mem object
        return;

      waitForRecordToFinish(Record, Lock);
    }

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
    ReadLockT Lock = acquireReadLock();
    EnqueueResultT Res;
    bool Enqueued;

    for (Command *Cmd : AuxiliaryCmds) {
      Enqueued = GraphProcessor::enqueueCommand(Cmd, Res, ToCleanUp);
      if (!Enqueued && EnqueueResultT::SyclEnqueueFailed == Res.MResult)
        throw runtime_error("Enqueue process failed.",
                            PI_ERROR_INVALID_OPERATION);
    }

    if (Command *NewCmd = static_cast<Command *>(NewCmdEvent->getCommand())) {
      Enqueued = GraphProcessor::enqueueCommand(NewCmd, Res, ToCleanUp);
      if (!Enqueued && EnqueueResultT::SyclEnqueueFailed == Res.MResult)
        throw runtime_error("Enqueue process failed.",
                            PI_ERROR_INVALID_OPERATION);
    }
  }

  cleanupCommands(ToCleanUp);
  return NewCmdEvent;
}

void Scheduler::releaseHostAccessor(Requirement *Req) {
  Command *const BlockedCmd = Req->MBlockedCmd;

  std::vector<Command *> ToCleanUp;
  {
    ReadLockT Lock = acquireReadLock();

    assert(BlockedCmd && "Can't find appropriate command to unblock");

    BlockedCmd->MEnqueueStatus = EnqueueResultT::SyclEnqueueReady;

    enqueueLeavesOfReqUnlocked(Req, ToCleanUp);
  }
  cleanupCommands(ToCleanUp);
}

void Scheduler::enqueueLeavesOfReqUnlocked(const Requirement *const Req,
                                           std::vector<Command *> &ToCleanUp) {
  MemObjRecord *Record = Req->MSYCLMemObj->MRecord.get();
  auto EnqueueLeaves = [&ToCleanUp](LeavesCollection &Leaves) {
    for (Command *Cmd : Leaves) {
      EnqueueResultT Res;
      bool Enqueued = GraphProcessor::enqueueCommand(Cmd, Res, ToCleanUp);
      if (!Enqueued && EnqueueResultT::SyclEnqueueFailed == Res.MResult)
        throw runtime_error("Enqueue process failed.",
                            PI_ERROR_INVALID_OPERATION);
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
  cleanupCommands({});
}

MemObjRecord *Scheduler::getMemObjRecord(const Requirement *const Req) {
  return Req->MSYCLMemObj->MRecord.get();
}

void Scheduler::cleanupCommands(const std::vector<Command *> &Cmds) {
  if (Cmds.empty())
  {
    std::lock_guard<std::mutex> Lock{MDeferredCleanupMutex};
    if (MDeferredCleanupCommands.empty())
      return;
  }

  WriteLockT Lock(MGraphLock, std::try_to_lock);
  // In order to avoid deadlocks related to blocked commands, defer cleanup if
  // the lock wasn't acquired.
  if (Lock.owns_lock()) {
    for (Command *Cmd : Cmds) {
      MGraphBuilder.cleanupCommand(Cmd);
    }
    std::vector<Command *> DeferredCleanupCommands;
    {
      std::lock_guard<std::mutex> Lock{MDeferredCleanupMutex};
      std::swap(DeferredCleanupCommands, MDeferredCleanupCommands);
    }
    for (Command *Cmd : DeferredCleanupCommands) {
      MGraphBuilder.cleanupCommand(Cmd);
    }

  } else {
    std::lock_guard<std::mutex> Lock{MDeferredCleanupMutex};
    MDeferredCleanupCommands.insert(MDeferredCleanupCommands.end(),
                                    Cmds.begin(), Cmds.end());
  }
}

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
      Scheduler::enqueueLeavesOfReqUnlocked(Dep.MDepRequirement, ToCleanUp);
  }
  cleanupCommands(ToCleanUp);
}

} // namespace detail
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
