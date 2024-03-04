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
inline namespace _V1 {
namespace detail {

bool Scheduler::checkLeavesCompletion(MemObjRecord *Record) {
  for (Command *Cmd : Record->MReadLeaves) {
    if (!(Cmd->getType() == detail::Command::ALLOCA ||
          Cmd->getType() == detail::Command::ALLOCA_SUB_BUF) &&
        !Cmd->getEvent()->isCompleted())
      return false;
  }
  for (Command *Cmd : Record->MWriteLeaves) {
    if (!(Cmd->getType() == detail::Command::ALLOCA ||
          Cmd->getType() == detail::Command::ALLOCA_SUB_BUF) &&
        !Cmd->getEvent()->isCompleted())
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
    EnqueueResultT Res;
    bool Enqueued =
        GraphProcessor::enqueueCommand(Cmd, GraphReadLock, Res, ToCleanUp, Cmd);
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
    bool Enqueued =
        GraphProcessor::enqueueCommand(Cmd, GraphReadLock, Res, ToCleanUp, Cmd);
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
    bool Enqueued = GraphProcessor::enqueueCommand(ReleaseCmd, GraphReadLock,
                                                   Res, ToCleanUp, ReleaseCmd);
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

EventImplPtr Scheduler::addCG(
    std::unique_ptr<detail::CG> CommandGroup, const QueueImplPtr &Queue,
    sycl::detail::pi::PiExtCommandBuffer CommandBuffer,
    const std::vector<sycl::detail::pi::PiExtSyncPoint> &Dependencies) {
  EventImplPtr NewEvent = nullptr;
  const CG::CGTYPE Type = CommandGroup->getType();
  std::vector<Command *> AuxiliaryCmds;
  std::vector<StreamImplPtr> Streams;

  if (Type == CG::Kernel) {
    auto *CGExecKernelPtr = static_cast<CGExecKernel *>(CommandGroup.get());
    Streams = CGExecKernelPtr->getStreams();
    CGExecKernelPtr->clearStreams();
    // Stream's flush buffer memory is mainly initialized in stream's __init
    // method. However, this method is not available on host device.
    // Initializing stream's flush buffer on the host side in a separate task.
    if (Queue->is_host()) {
      for (const StreamImplPtr &Stream : Streams) {
        Stream->initStreamHost(Queue);
      }
    }
  }
  std::vector<std::shared_ptr<const void>> AuxiliaryResources;
  AuxiliaryResources = CommandGroup->getAuxiliaryResources();
  CommandGroup->clearAuxiliaryResources();

  bool ShouldEnqueue = true;
  {
    WriteLockT Lock = acquireWriteLock();

    Command *NewCmd = nullptr;
    switch (Type) {
    case CG::UpdateHost:
      NewCmd = MGraphBuilder.addCGUpdateHost(std::move(CommandGroup),
                                             DefaultHostQueue, AuxiliaryCmds);
      NewEvent = NewCmd->getEvent();
      break;
    case CG::CodeplayHostTask: {
      auto Result = MGraphBuilder.addCG(std::move(CommandGroup),
                                        DefaultHostQueue, AuxiliaryCmds);
      NewCmd = Result.NewCmd;
      NewEvent = Result.NewEvent;
      ShouldEnqueue = Result.ShouldEnqueue;
      break;
    }
    default:
      auto Result = MGraphBuilder.addCG(std::move(CommandGroup),
                                        std::move(Queue), AuxiliaryCmds,
                                        CommandBuffer, std::move(Dependencies));

      NewCmd = Result.NewCmd;
      NewEvent = Result.NewEvent;
      ShouldEnqueue = Result.ShouldEnqueue;
    }
    NewEvent->setSubmissionTime();
  }

  if (ShouldEnqueue) {
    enqueueCommandForCG(NewEvent, AuxiliaryCmds);

    for (const auto &StreamImplPtr : Streams) {
      StreamImplPtr->flush(NewEvent);
    }
  }

  if (!AuxiliaryResources.empty())
    registerAuxiliaryResources(NewEvent, std::move(AuxiliaryResources));

  return NewEvent;
}

void Scheduler::enqueueCommandForCG(EventImplPtr NewEvent,
                                    std::vector<Command *> &AuxiliaryCmds,
                                    BlockingT Blocking) {
  std::vector<Command *> ToCleanUp;
  {
    ReadLockT Lock = acquireReadLock();

    Command *NewCmd =
        (NewEvent) ? static_cast<Command *>(NewEvent->getCommand()) : nullptr;

    EnqueueResultT Res;
    bool Enqueued;

    auto CleanUp = [&]() {
      if (NewCmd && (NewCmd->MDeps.size() == 0 && NewCmd->MUsers.size() == 0)) {
        if (NewEvent) {
          NewEvent->setCommand(nullptr);
        }
        delete NewCmd;
      }
    };

    for (Command *Cmd : AuxiliaryCmds) {
      Enqueued = GraphProcessor::enqueueCommand(Cmd, Lock, Res, ToCleanUp, Cmd,
                                                Blocking);
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
        bool Enqueued = GraphProcessor::enqueueCommand(
            NewCmd, Lock, Res, ToCleanUp, NewCmd, Blocking);
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
      Enqueued = GraphProcessor::enqueueCommand(Cmd, Lock, Res, ToCleanUp, Cmd);
      if (!Enqueued && EnqueueResultT::SyclEnqueueFailed == Res.MResult)
        throw runtime_error("Enqueue process failed.",
                            PI_ERROR_INVALID_OPERATION);
    }

    Enqueued =
        GraphProcessor::enqueueCommand(NewCmd, Lock, Res, ToCleanUp, NewCmd);
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

bool Scheduler::isInstanceAlive() {
  return GlobalHandler::instance().isSchedulerAlive();
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

bool Scheduler::removeMemoryObject(detail::SYCLMemObjI *MemObj,
                                   bool StrictLock) {
  MemObjRecord *Record = MGraphBuilder.getMemObjRecord(MemObj);
  if (!Record)
    // No operations were performed on the mem object
    return true;

  {
    // This only needs a shared mutex as it only involves enqueueing and
    // awaiting for events
    ReadLockT Lock = StrictLock ? ReadLockT(MGraphLock)
                                : ReadLockT(MGraphLock, std::try_to_lock);
    if (!Lock.owns_lock())
      return false;
    waitForRecordToFinish(Record, Lock);
  }
  {
    WriteLockT Lock = StrictLock ? acquireWriteLock()
                                 : WriteLockT(MGraphLock, std::try_to_lock);
    if (!Lock.owns_lock())
      return false;
    MGraphBuilder.decrementLeafCountersForRecord(Record);
    MGraphBuilder.cleanupCommandsForRecord(Record);
    MGraphBuilder.removeRecordForMemObj(MemObj);
  }
  return true;
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
      Enqueued = GraphProcessor::enqueueCommand(Cmd, Lock, Res, ToCleanUp, Cmd);
      if (!Enqueued && EnqueueResultT::SyclEnqueueFailed == Res.MResult)
        throw runtime_error("Enqueue process failed.",
                            PI_ERROR_INVALID_OPERATION);
    }

    if (Command *NewCmd = static_cast<Command *>(NewCmdEvent->getCommand())) {
      Enqueued =
          GraphProcessor::enqueueCommand(NewCmd, Lock, Res, ToCleanUp, NewCmd);
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

    enqueueLeavesOfReqUnlocked(Req, Lock, ToCleanUp);
  }
  cleanupCommands(ToCleanUp);
}

void Scheduler::enqueueLeavesOfReqUnlocked(const Requirement *const Req,
                                           ReadLockT &GraphReadLock,
                                           std::vector<Command *> &ToCleanUp) {
  MemObjRecord *Record = Req->MSYCLMemObj->MRecord.get();
  auto EnqueueLeaves = [&ToCleanUp, &GraphReadLock](LeavesCollection &Leaves) {
    for (Command *Cmd : Leaves) {
      EnqueueResultT Res;
      bool Enqueued = GraphProcessor::enqueueCommand(Cmd, GraphReadLock, Res,
                                                     ToCleanUp, Cmd);
      if (!Enqueued && EnqueueResultT::SyclEnqueueFailed == Res.MResult)
        throw runtime_error("Enqueue process failed.",
                            PI_ERROR_INVALID_OPERATION);
    }
  };

  EnqueueLeaves(Record->MReadLeaves);
  EnqueueLeaves(Record->MWriteLeaves);
}

void Scheduler::enqueueUnblockedCommands(
    const std::vector<EventImplPtr> &ToEnqueue, ReadLockT &GraphReadLock,
    std::vector<Command *> &ToCleanUp) {
  for (auto &Event : ToEnqueue) {
    Command *Cmd = static_cast<Command *>(Event->getCommand());
    if (!Cmd)
      continue;
    EnqueueResultT Res;
    bool Enqueued =
        GraphProcessor::enqueueCommand(Cmd, GraphReadLock, Res, ToCleanUp, Cmd);
    if (!Enqueued && EnqueueResultT::SyclEnqueueFailed == Res.MResult)
      throw runtime_error("Enqueue process failed.",
                          PI_ERROR_INVALID_OPERATION);
  }
}

Scheduler::Scheduler() {
  sycl::device HostDevice =
      createSyclObjFromImpl<device>(device_impl::getHostDeviceImpl());
  sycl::context HostContext{HostDevice};
  DefaultHostQueue = QueueImplPtr(
      new queue_impl(detail::getSyclObjImpl(HostDevice),
                     detail::getSyclObjImpl(HostContext), /*AsyncHandler=*/{},
                     /*PropList=*/{sycl::property::queue::enable_profiling()}));
}

Scheduler::~Scheduler() { DefaultHostQueue.reset(); }

void Scheduler::releaseResources(BlockingT Blocking) {
  //  There might be some commands scheduled for post enqueue cleanup that
  //  haven't been freed because of the graph mutex being locked at the time,
  //  clean them up now.
  cleanupCommands({});

  cleanupAuxiliaryResources(Blocking);
  // We need loop since sometimes we may need new objects to be added to
  // deferred mem objects storage during cleanup. Known example is: we cleanup
  // existing deferred mem objects under write lock, during this process we
  // cleanup commands related to this record, command may have last reference to
  // queue_impl, ~queue_impl is called and buffer for assert (which is created
  // with size only so all confitions for deferred release are satisfied) is
  // added to deferred mem obj storage. So we may end up with leak.
  do {
    cleanupDeferredMemObjects(Blocking);
  } while (Blocking == BlockingT::BLOCKING && !isDeferredMemObjectsEmpty());
}

MemObjRecord *Scheduler::getMemObjRecord(const Requirement *const Req) {
  return Req->MSYCLMemObj->MRecord.get();
}

void Scheduler::cleanupCommands(const std::vector<Command *> &Cmds) {
  cleanupAuxiliaryResources(BlockingT::NON_BLOCKING);
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
      MGraphBuilder.cleanupCommand(Cmd);
    }

  } else {
    std::lock_guard<std::mutex> Lock{MDeferredCleanupMutex};
    // Full cleanup for fusion placeholder commands is handled by the entry
    // points for fusion (start_fusion, ...). To avoid double free or access to
    // objects after their lifetime, fusion commands should therefore never be
    // added to the deferred command list.
    std::copy_if(Cmds.begin(), Cmds.end(),
                 std::back_inserter(MDeferredCleanupCommands),
                 [](const Command *Cmd) {
                   return Cmd->getType() != Command::CommandType::FUSION;
                 });
  }
}

void Scheduler::NotifyHostTaskCompletion(Command *Cmd) {
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
    // Host tasks are cleaned up upon completion rather than enqueuing.
    if (Cmd->MLeafCounter == 0) {
      ToCleanUp.push_back(Cmd);
      Cmd->MMarkedForCleanup = true;
    }

    {
      std::lock_guard<std::mutex> Guard(Cmd->MBlockedUsersMutex);
      // update self-event status
      Cmd->getEvent()->setComplete();
    }
    Scheduler::enqueueUnblockedCommands(Cmd->MBlockedUsers, Lock, ToCleanUp);
  }
  cleanupCommands(ToCleanUp);
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
    // Lock is needed for checkLeavesCompletion - if walks through Record leaves
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
  auto ReleaseCandidateIt = ObjsReadyToRelease.begin();
  while (ReleaseCandidateIt != ObjsReadyToRelease.end()) {
    if (!removeMemoryObject(ReleaseCandidateIt->get(), false))
      break;
    ReleaseCandidateIt = ObjsReadyToRelease.erase(ReleaseCandidateIt);
  }
  if (!ObjsReadyToRelease.empty()) {
    std::lock_guard<std::mutex> LockDef{MDeferredMemReleaseMutex};
    MDeferredMemObjRelease.insert(
        MDeferredMemObjRelease.end(),
        std::make_move_iterator(ObjsReadyToRelease.begin()),
        std::make_move_iterator(ObjsReadyToRelease.end()));
  }
}

static void registerAuxiliaryResourcesNoLock(
    std::unordered_map<EventImplPtr, std::vector<std::shared_ptr<const void>>>
        &AuxiliaryResources,
    const EventImplPtr &Event,
    std::vector<std::shared_ptr<const void>> &&Resources) {
  std::vector<std::shared_ptr<const void>> &StoredResources =
      AuxiliaryResources[Event];
  StoredResources.insert(StoredResources.end(),
                         std::make_move_iterator(Resources.begin()),
                         std::make_move_iterator(Resources.end()));
}

void Scheduler::takeAuxiliaryResources(const EventImplPtr &Dst,
                                       const EventImplPtr &Src) {
  std::unique_lock<std::mutex> Lock{MAuxiliaryResourcesMutex};
  auto Iter = MAuxiliaryResources.find(Src);
  if (Iter == MAuxiliaryResources.end()) {
    return;
  }
  registerAuxiliaryResourcesNoLock(MAuxiliaryResources, Dst,
                                   std::move(Iter->second));
  MAuxiliaryResources.erase(Iter);
}

void Scheduler::registerAuxiliaryResources(
    EventImplPtr &Event, std::vector<std::shared_ptr<const void>> Resources) {
  std::unique_lock<std::mutex> Lock{MAuxiliaryResourcesMutex};
  registerAuxiliaryResourcesNoLock(MAuxiliaryResources, Event,
                                   std::move(Resources));
}

void Scheduler::cleanupAuxiliaryResources(BlockingT Blocking) {
  std::unique_lock<std::mutex> Lock{MAuxiliaryResourcesMutex};
  for (auto It = MAuxiliaryResources.begin();
       It != MAuxiliaryResources.end();) {
    const EventImplPtr &Event = It->first;
    if (Blocking == BlockingT::BLOCKING) {
      Event->waitInternal();
      It = MAuxiliaryResources.erase(It);
    } else if (Event->isCompleted())
      It = MAuxiliaryResources.erase(It);
    else
      ++It;
  }
}

void Scheduler::startFusion(QueueImplPtr Queue) {
  WriteLockT Lock = acquireWriteLock();
  WriteLockT FusionMapLock = acquireFusionWriteLock();
  MGraphBuilder.startFusion(Queue);
}

void Scheduler::cleanUpCmdFusion(sycl::detail::queue_impl *Queue) {
  // No graph lock, we might be called because the graph builder is releasing
  // resources.
  WriteLockT FusionMapLock = acquireFusionWriteLock();
  MGraphBuilder.cleanUpCmdFusion(Queue);
}

void Scheduler::cancelFusion(QueueImplPtr Queue) {
  std::vector<Command *> ToEnqueue;
  {
    WriteLockT Lock = acquireWriteLock();
    WriteLockT FusionMapLock = acquireFusionWriteLock();
    MGraphBuilder.cancelFusion(Queue, ToEnqueue);
  }
  enqueueCommandForCG(nullptr, ToEnqueue);
}

EventImplPtr Scheduler::completeFusion(QueueImplPtr Queue,
                                       const property_list &PropList) {
  std::vector<Command *> ToEnqueue;
  EventImplPtr FusedEvent;
  {
    WriteLockT Lock = acquireWriteLock();
    WriteLockT FusionMapLock = acquireFusionWriteLock();
    FusedEvent = MGraphBuilder.completeFusion(Queue, ToEnqueue, PropList);
  }
  enqueueCommandForCG(nullptr, ToEnqueue);

  return FusedEvent;
}

bool Scheduler::isInFusionMode(QueueIdT queue) {
  ReadLockT Lock = acquireFusionReadLock();
  return MGraphBuilder.isInFusionMode(queue);
}

void Scheduler::printFusionWarning(const std::string &Message) {
  if (detail::SYCLConfig<detail::SYCL_RT_WARNING_LEVEL>::get() > 0) {
    std::cerr << "WARNING: " << Message << "\n";
  }
}

KernelFusionCommand *Scheduler::isPartOfActiveFusion(Command *Cmd) {
  auto CmdType = Cmd->getType();
  switch (CmdType) {
  case Command::FUSION: {
    auto *FusionCmd = static_cast<KernelFusionCommand *>(Cmd);
    return (FusionCmd->isActive()) ? FusionCmd : nullptr;
  }
  case Command::RUN_CG: {
    auto *CGCmd = static_cast<ExecCGCommand *>(Cmd);
    return (CGCmd->MFusionCmd && CGCmd->MFusionCmd->isActive())
               ? CGCmd->MFusionCmd
               : nullptr;
  }
  default:
    return nullptr;
  }
}

} // namespace detail
} // namespace _V1
} // namespace sycl
