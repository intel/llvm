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

bool Scheduler::checkLeavesCompletion(MemObjRecord *Record) {
  for (Command *Cmd : Record->MReadLeaves) {
    Tracer t("Record->MReadLeaves isCompleted, Cmd = " +
             std::to_string(reinterpret_cast<long long>(Cmd)));
    if (!Cmd->getEvent()->isCompleted())
      return false;
  }
  for (Command *Cmd : Record->MWriteLeaves) {
    Tracer t("Record->MWriteLeaves isCompleted, Cmd = " +
             std::to_string(reinterpret_cast<long long>(Cmd)));
    if (!Cmd->getEvent()->isCompleted())
      return false;
  }
  return true;
}

void Scheduler::waitForRecordToFinish(MemObjRecord *Record,
                                      ReadLockT &GraphReadLock) {
#ifdef XPTI_ENABLE_INSTRUMENTATION
  // Will contain the list of dependencies for the Release Command
  std::set<Command *> DepCommands;
#endif
  std::vector<Command *> ToCleanUp;
  for (Command *Cmd : Record->MReadLeaves) {
    {
      Tracer t("Record->MReadLeaves enqueueCommand, Cmd = " +
               std::to_string(reinterpret_cast<long long>(Cmd)));

      EnqueueResultT Res;
      bool Enqueued = GraphProcessor::enqueueCommand(Cmd, Res, ToCleanUp);
      if (!Enqueued && EnqueueResultT::SyclEnqueueFailed == Res.MResult)
        throw runtime_error("Enqueue process failed.",
                            PI_ERROR_INVALID_OPERATION);
    }
#ifdef XPTI_ENABLE_INSTRUMENTATION
    // Capture the dependencies
    DepCommands.insert(Cmd);
#endif
    Tracer t("Record->MReadLeaves waitForEvent, Cmd = " +
             std::to_string(reinterpret_cast<long long>(Cmd)));

    GraphProcessor::waitForEvent(Cmd->getEvent(), GraphReadLock, ToCleanUp);
  }
  for (Command *Cmd : Record->MWriteLeaves) {
    {
      Tracer t("Record->MWriteLeaves enqueueCommand, Cmd = " +
               std::to_string(reinterpret_cast<long long>(Cmd)));

      EnqueueResultT Res;
      bool Enqueued = GraphProcessor::enqueueCommand(Cmd, Res, ToCleanUp);
      if (!Enqueued && EnqueueResultT::SyclEnqueueFailed == Res.MResult)
        throw runtime_error("Enqueue process failed.",
                            PI_ERROR_INVALID_OPERATION);
    }
#ifdef XPTI_ENABLE_INSTRUMENTATION
    DepCommands.insert(Cmd);
#endif
    Tracer t("Record->MWriteLeaves waitForEvent, Cmd = " +
             std::to_string(reinterpret_cast<long long>(Cmd)));

    GraphProcessor::waitForEvent(Cmd->getEvent(), GraphReadLock, ToCleanUp);
  }
  for (AllocaCommandBase *AllocaCmd : Record->MAllocaCommands) {
    Command *ReleaseCmd = AllocaCmd->getReleaseCmd();
    {
      EnqueueResultT Res;
      Tracer t("Record->releasecmd enqueueCommand, ReleaseCmd = " +
               std::to_string(reinterpret_cast<long long>(ReleaseCmd)));
      bool Enqueued =
          GraphProcessor::enqueueCommand(ReleaseCmd, Res, ToCleanUp);
      if (!Enqueued && EnqueueResultT::SyclEnqueueFailed == Res.MResult)
        throw runtime_error("Enqueue process failed.",
                            PI_ERROR_INVALID_OPERATION);
    }
#ifdef XPTI_ENABLE_INSTRUMENTATION
    // Report these dependencies to the Command so these dependencies can be
    // reported as edges
    ReleaseCmd->resolveReleaseDependencies(DepCommands);
#endif
    Tracer t("Record->releasecmd waitForEvent, ReleaseCmd = " +
             std::to_string(reinterpret_cast<long long>(ReleaseCmd)));
    GraphProcessor::waitForEvent(ReleaseCmd->getEvent(), GraphReadLock,
                                 ToCleanUp);
  }
}

EventImplPtr Scheduler::addCG(std::unique_ptr<detail::CG> CommandGroup,
                              QueueImplPtr Queue) {
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
    WriteLockT Lock(MGraphLock, std::defer_lock);
    acquireWriteLock(Lock);

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
    ReadLockT Lock(MGraphLock);

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

  std::vector<Command *> ToCleanUp;
  try {
    ReadLockT Lock(MGraphLock);
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

void Scheduler::waitForEvent(EventImplPtr Event) {
  ReadLockT Lock(MGraphLock);
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
    MemObjRecord *Record = MGraphBuilder.getMemObjRecord(MemObj);
    if (!Record)
      // No operations were performed on the mem object
      return;

    {
      // This only needs a shared mutex as it only involves enqueueing and
      // awaiting for events
      ReadLockT Lock(MGraphLock);
      waitForRecordToFinish(Record, Lock);
    }
    {
      WriteLockT Lock(MGraphLock, std::defer_lock);
      acquireWriteLock(Lock);
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
    WriteLockT Lock(MGraphLock, std::defer_lock);
    acquireWriteLock(Lock);

    Command *NewCmd = MGraphBuilder.addHostAccessor(Req, AuxiliaryCmds);
    if (!NewCmd)
      return nullptr;
    NewCmdEvent = NewCmd->getEvent();
  }

  std::vector<Command *> ToCleanUp;
  {
    ReadLockT ReadLock(MGraphLock);
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
    ReadLockT Lock(MGraphLock);

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
  // Please be aware that releaseResources should be called before deletion of
  // Scheduler. Otherwise there can be the case when objects Scheduler keeps as
  // fields may need Scheduler for their release and they work with Scheduler
  // via GlobalHandler::getScheduler that will create new Scheduler object.
  // Still keep it here but it should do almost nothing if releaseResources
  // called before.
  releaseResources();
}

void Scheduler::releaseResources() {
  // There might be some commands scheduled for post enqueue cleanup that
  // haven't been freed because of the graph mutex being locked at the time,
  // clean them up now.
  cleanupCommands({});
  DefaultHostQueue.reset();

  // We need loop since sometimes we may need new objects to be added to
  // deferred mem objects storage during cleanup. Known example is: we cleanup
  // existing deferred mem objects under write lock, during this process we
  // cleanup commands related to this record, command may have last reference to
  // queue_impl, ~queue_impl is called and buffer for assert (which is created
  // with size only so all confitions for deferred release are satisfied) is
  // added to deferred mem obj storage. So we may end up with leak.
  while (!isDeferredMemObjectsEmpty()) {
    Tracer t("cleanupDeferredMemObjects(BlockingT::BLOCKING)");
    cleanupDeferredMemObjects(BlockingT::BLOCKING);
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

void Scheduler::cleanupCommands(const std::vector<Command *> &Cmds) {
  cleanupDeferredMemObjects(BlockingT::NON_BLOCKING);
  if (Cmds.empty()) {
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
      Tracer t("cleanupCommand cmd = " +
               std::to_string(reinterpret_cast<long long>(Cmd)));
      MGraphBuilder.cleanupCommand(Cmd);
    }

  } else {
    std::lock_guard<std::mutex> Lock{MDeferredCleanupMutex};
    MDeferredCleanupCommands.insert(MDeferredCleanupCommands.end(),
                                    Cmds.begin(), Cmds.end());
  }
}

void Scheduler::deferMemObjRelease(const std::shared_ptr<SYCLMemObjI> &MemObj) {
  {
    std::lock_guard<std::mutex> Lock{MDeferredMemReleaseMutex};
    MDeferredMemObjRelease.push_back(MemObj);
  }
  cleanupDeferredMemObjects(BlockingT::NON_BLOCKING);
}

inline bool Scheduler::isDeferredMemObjectsEmpty() {
  std::lock_guard<std::mutex> Lock{MDeferredMemReleaseMutex};
  return MDeferredMemObjRelease.empty();
}

void Scheduler::cleanupDeferredMemObjects(BlockingT Blocking) {
  if (isDeferredMemObjectsEmpty())
    return;
  if (Blocking == BlockingT::BLOCKING) {
    std::vector<std::shared_ptr<SYCLMemObjI>> TempStorage;
    {
      std::lock_guard<std::mutex> LockDef{MDeferredMemReleaseMutex};
      MDeferredMemObjRelease.swap(TempStorage);
    }
    // if any objects in TempStorage exist - it is leaving scope and being
    // deleted
  }

  std::vector<std::shared_ptr<SYCLMemObjI>> ObjsReadyToRelease;
  {

    ReadLockT Lock = ReadLockT(MGraphLock, std::try_to_lock);
    if (Lock.owns_lock()) {
      // Not expected that Blocking == true will be used in parallel with
      // adding MemObj to storage, no such scenario.
      std::lock_guard<std::mutex> LockDef{MDeferredMemReleaseMutex};
      auto MemObjIt = MDeferredMemObjRelease.begin();
      while (MemObjIt != MDeferredMemObjRelease.end()) {
        MemObjRecord *Record = MGraphBuilder.getMemObjRecord((*MemObjIt).get());
        if (!checkLeavesCompletion(Record)) {
          MemObjIt++;
          continue;
        }
        ObjsReadyToRelease.push_back(*MemObjIt);
        MemObjIt = MDeferredMemObjRelease.erase(MemObjIt);
      }
    }
  }
  // if any ObjsReadyToRelease found - it is leaving scope and being deleted
}
} // namespace detail
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
