//===-- graph_builder.cpp - SYCL Graph Builder ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "detail/config.hpp"
#include <CL/sycl/access/access.hpp>
#include <CL/sycl/detail/memory_manager.hpp>
#include <CL/sycl/exception.hpp>
#include <detail/context_impl.hpp>
#include <detail/event_impl.hpp>
#include <detail/queue_impl.hpp>
#include <detail/scheduler/scheduler.hpp>

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <queue>
#include <set>
#include <vector>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

/// Checks whether two requirements overlap or not.
///
/// This information can be used to prove that executing two kernels that
/// work on different parts of the memory object in parallel is legal.
// TODO merge with LeavesCollection's version of doOverlap (see
// leaves_collection.cpp).
static bool doOverlap(const Requirement *LHS, const Requirement *RHS) {
  return (LHS->MOffsetInBytes + LHS->MAccessRange.size() * LHS->MElemSize >=
          RHS->MOffsetInBytes) ||
         (RHS->MOffsetInBytes + RHS->MAccessRange.size() * RHS->MElemSize >=
          LHS->MOffsetInBytes);
}

static bool sameCtx(const ContextImplPtr &LHS, const ContextImplPtr &RHS) {
  // Consider two different host contexts to be the same to avoid additional
  // allocation on the host
  return LHS == RHS || (LHS->is_host() && RHS->is_host());
}

/// Checks if current requirement is requirement for sub buffer.
static bool IsSuitableSubReq(const Requirement *Req) {
  return Req->MIsSubBuffer;
}

/// Checks if the required access mode is allowed under the current one.
static bool isAccessModeAllowed(access::mode Required, access::mode Current) {
  switch (Current) {
  case access::mode::read:
    return (Required == Current);
  case access::mode::write:
    assert(false && "Write only access is expected to be mapped as read_write");
    return (Required == Current || Required == access::mode::discard_write);
  case access::mode::read_write:
  case access::mode::atomic:
  case access::mode::discard_write:
  case access::mode::discard_read_write:
    return true;
  }
  assert(false);
  return false;
}

/// Combines two access modes into a single one that allows both.
static access::mode combineAccessModes(access::mode A, access::mode B) {
  if (A == B)
    return A;

  if (A == access::mode::discard_write &&
      (B == access::mode::discard_read_write || B == access::mode::write))
    return B;

  if (B == access::mode::discard_write &&
      (A == access::mode::discard_read_write || A == access::mode::write))
    return A;

  return access::mode::read_write;
}

Scheduler::GraphBuilder::GraphBuilder() {
  if (const char *EnvVarCStr = SYCLConfig<SYCL_PRINT_EXECUTION_GRAPH>::get()) {
    std::string GraphPrintOpts(EnvVarCStr);
    bool EnableAlways = GraphPrintOpts.find("always") != std::string::npos;

    if (GraphPrintOpts.find("before_addCG") != std::string::npos ||
        EnableAlways)
      MPrintOptionsArray[BeforeAddCG] = true;
    if (GraphPrintOpts.find("after_addCG") != std::string::npos || EnableAlways)
      MPrintOptionsArray[AfterAddCG] = true;
    if (GraphPrintOpts.find("before_addCopyBack") != std::string::npos ||
        EnableAlways)
      MPrintOptionsArray[BeforeAddCopyBack] = true;
    if (GraphPrintOpts.find("after_addCopyBack") != std::string::npos ||
        EnableAlways)
      MPrintOptionsArray[AfterAddCopyBack] = true;
    if (GraphPrintOpts.find("before_addHostAcc") != std::string::npos ||
        EnableAlways)
      MPrintOptionsArray[BeforeAddHostAcc] = true;
    if (GraphPrintOpts.find("after_addHostAcc") != std::string::npos ||
        EnableAlways)
      MPrintOptionsArray[AfterAddHostAcc] = true;
  }
}

static bool markNodeAsVisited(Command *Cmd, std::vector<Command *> &Visited) {
  assert(Cmd && "Cmd can't be nullptr");
  if (Cmd->MMarks.MVisited)
    return false;
  Cmd->MMarks.MVisited = true;
  Visited.push_back(Cmd);
  return true;
}

static void unmarkVisitedNodes(std::vector<Command *> &Visited) {
  for (Command *Cmd : Visited)
    Cmd->MMarks.MVisited = false;
}

static void handleVisitedNodes(std::vector<Command *> &Visited) {
  for (Command *Cmd : Visited) {
    if (Cmd->MMarks.MToBeDeleted) {
      Cmd->getEvent()->setCommand(nullptr);
      delete Cmd;
    } else
      Cmd->MMarks.MVisited = false;
  }
}

static void printDotRecursive(std::fstream &Stream,
                              std::vector<Command *> &Visited, Command *Cmd) {
  if (!markNodeAsVisited(Cmd, Visited))
    return;
  for (Command *User : Cmd->MUsers) {
    if (User)
      printDotRecursive(Stream, Visited, User);
  }
  Cmd->printDot(Stream);
}

void Scheduler::GraphBuilder::printGraphAsDot(const char *ModeName) {
  static size_t Counter = 0;
  std::string ModeNameStr(ModeName);
  std::string FileName =
      "graph_" + std::to_string(Counter) + ModeNameStr + ".dot";

  Counter++;

  std::fstream Stream(FileName, std::ios::out);
  Stream << "strict digraph {" << std::endl;

  MVisitedCmds.clear();

  for (SYCLMemObjI *MemObject : MMemObjs)
    for (Command *AllocaCmd : MemObject->MRecord->MAllocaCommands)
      printDotRecursive(Stream, MVisitedCmds, AllocaCmd);

  Stream << "}" << std::endl;

  unmarkVisitedNodes(MVisitedCmds);
}

MemObjRecord *Scheduler::GraphBuilder::getMemObjRecord(SYCLMemObjI *MemObject) {
  return MemObject->MRecord.get();
}

MemObjRecord *Scheduler::GraphBuilder::getOrInsertMemObjRecord(
    const QueueImplPtr &Queue, const Requirement *Req,
    std::vector<Command *> &ToEnqueue) {
  SYCLMemObjI *MemObject = Req->MSYCLMemObj;
  MemObjRecord *Record = getMemObjRecord(MemObject);

  if (nullptr != Record)
    return Record;

  const size_t LeafLimit = 8;
  LeavesCollection::AllocateDependencyF AllocateDependency =
      [this](Command *Dependant, Command *Dependency, MemObjRecord *Record,
             LeavesCollection::EnqueueListT &ToEnqueue) {
        // Add the old leaf as a dependency for the new one by duplicating one
        // of the requirements for the current record
        DepDesc Dep = findDepForRecord(Dependant, Record);
        Dep.MDepCommand = Dependency;
        std::vector<Command *> ToCleanUp;
        Command *ConnectionCmd = Dependant->addDep(Dep, ToCleanUp);
        if (ConnectionCmd)
          ToEnqueue.push_back(ConnectionCmd);

        --(Dependency->MLeafCounter);
        if (Dependency->MLeafCounter == 0 &&
            Dependency->isSuccessfullyEnqueued() &&
            Dependency->supportsPostEnqueueCleanup())
          ToCleanUp.push_back(Dependency);
        for (Command *Cmd : ToCleanUp)
          cleanupCommand(Cmd);
      };

  const ContextImplPtr &InteropCtxPtr = Req->MSYCLMemObj->getInteropContext();
  if (InteropCtxPtr) {
    // The memory object has been constructed using interoperability constructor
    // which means that there is already an allocation(cl_mem) in some context.
    // Registering this allocation in the SYCL graph.

    std::vector<sycl::device> Devices =
        InteropCtxPtr->get_info<info::context::devices>();
    assert(Devices.size() != 0);
    DeviceImplPtr Dev = detail::getSyclObjImpl(Devices[0]);

    // Since all the Scheduler commands require queue but we have only context
    // here, we need to create a dummy queue bound to the context and one of the
    // devices from the context.
    QueueImplPtr InteropQueuePtr{new detail::queue_impl{
        Dev, InteropCtxPtr, /*AsyncHandler=*/{}, /*PropertyList=*/{}}};

    MemObject->MRecord.reset(
        new MemObjRecord{InteropCtxPtr, LeafLimit, AllocateDependency});
    getOrCreateAllocaForReq(MemObject->MRecord.get(), Req, InteropQueuePtr,
                            ToEnqueue);
  } else
    MemObject->MRecord.reset(new MemObjRecord{Queue->getContextImplPtr(),
                                              LeafLimit, AllocateDependency});

  MMemObjs.push_back(MemObject);
  return MemObject->MRecord.get();
}

void Scheduler::GraphBuilder::updateLeaves(const std::set<Command *> &Cmds,
                                           MemObjRecord *Record,
                                           access::mode AccessMode,
                                           std::vector<Command *> &ToCleanUp) {

  const bool ReadOnlyReq = AccessMode == access::mode::read;
  if (ReadOnlyReq)
    return;

  for (Command *Cmd : Cmds) {
    bool WasLeaf = Cmd->MLeafCounter > 0;
    Cmd->MLeafCounter -= Record->MReadLeaves.remove(Cmd);
    Cmd->MLeafCounter -= Record->MWriteLeaves.remove(Cmd);
    if (WasLeaf && Cmd->MLeafCounter == 0 && Cmd->isSuccessfullyEnqueued() &&
        Cmd->supportsPostEnqueueCleanup()) {
      ToCleanUp.push_back(Cmd);
    }
  }
}

void Scheduler::GraphBuilder::addNodeToLeaves(
    MemObjRecord *Record, Command *Cmd, access::mode AccessMode,
    std::vector<Command *> &ToEnqueue) {
  LeavesCollection &Leaves{AccessMode == access::mode::read
                               ? Record->MReadLeaves
                               : Record->MWriteLeaves};
  if (Leaves.push_back(Cmd, ToEnqueue))
    ++Cmd->MLeafCounter;
}

UpdateHostRequirementCommand *Scheduler::GraphBuilder::insertUpdateHostReqCmd(
    MemObjRecord *Record, Requirement *Req, const QueueImplPtr &Queue,
    std::vector<Command *> &ToEnqueue) {
  AllocaCommandBase *AllocaCmd =
      findAllocaForReq(Record, Req, Queue->getContextImplPtr());
  assert(AllocaCmd && "There must be alloca for requirement!");
  UpdateHostRequirementCommand *UpdateCommand =
      new UpdateHostRequirementCommand(Queue, *Req, AllocaCmd, &Req->MData);
  // Need copy of requirement because after host accessor destructor call
  // dependencies become invalid if requirement is stored by pointer.
  const Requirement *StoredReq = UpdateCommand->getRequirement();

  std::set<Command *> Deps =
      findDepsForReq(Record, Req, Queue->getContextImplPtr());
  std::vector<Command *> ToCleanUp;
  for (Command *Dep : Deps) {
    Command *ConnCmd =
        UpdateCommand->addDep(DepDesc{Dep, StoredReq, AllocaCmd}, ToCleanUp);
    if (ConnCmd)
      ToEnqueue.push_back(ConnCmd);
  }
  updateLeaves(Deps, Record, Req->MAccessMode, ToCleanUp);
  addNodeToLeaves(Record, UpdateCommand, Req->MAccessMode, ToEnqueue);
  for (Command *Cmd : ToCleanUp)
    cleanupCommand(Cmd);
  return UpdateCommand;
}

// Takes linked alloca commands. Makes AllocaCmdDst command active using map
// or unmap operation.
static Command *insertMapUnmapForLinkedCmds(AllocaCommandBase *AllocaCmdSrc,
                                            AllocaCommandBase *AllocaCmdDst,
                                            access::mode MapMode) {
  assert(AllocaCmdSrc->MLinkedAllocaCmd == AllocaCmdDst &&
         "Expected linked alloca commands");
  assert(AllocaCmdSrc->MIsActive &&
         "Expected source alloca command to be active");

  if (AllocaCmdSrc->getQueue()->is_host()) {
    UnMapMemObject *UnMapCmd = new UnMapMemObject(
        AllocaCmdDst, *AllocaCmdDst->getRequirement(),
        &AllocaCmdSrc->MMemAllocation, AllocaCmdDst->getQueue());

    std::swap(AllocaCmdSrc->MIsActive, AllocaCmdDst->MIsActive);

    return UnMapCmd;
  }

  MapMemObject *MapCmd = new MapMemObject(
      AllocaCmdSrc, *AllocaCmdSrc->getRequirement(),
      &AllocaCmdDst->MMemAllocation, AllocaCmdSrc->getQueue(), MapMode);

  std::swap(AllocaCmdSrc->MIsActive, AllocaCmdDst->MIsActive);

  return MapCmd;
}

Command *Scheduler::GraphBuilder::insertMemoryMove(
    MemObjRecord *Record, Requirement *Req, const QueueImplPtr &Queue,
    std::vector<Command *> &ToEnqueue) {

  AllocaCommandBase *AllocaCmdDst =
      getOrCreateAllocaForReq(Record, Req, Queue, ToEnqueue);
  if (!AllocaCmdDst)
    throw runtime_error("Out of host memory", PI_OUT_OF_HOST_MEMORY);

  std::set<Command *> Deps =
      findDepsForReq(Record, Req, Queue->getContextImplPtr());
  Deps.insert(AllocaCmdDst);
  // Get parent allocation of sub buffer to perform full copy of whole buffer
  if (IsSuitableSubReq(Req)) {
    if (AllocaCmdDst->getType() == Command::CommandType::ALLOCA_SUB_BUF)
      AllocaCmdDst =
          static_cast<AllocaSubBufCommand *>(AllocaCmdDst)->getParentAlloca();
  }

  AllocaCommandBase *AllocaCmdSrc =
      findAllocaForReq(Record, Req, Record->MCurContext);
  if (!AllocaCmdSrc && IsSuitableSubReq(Req)) {
    // Since no alloca command for the sub buffer requirement was found in the
    // current context, need to find a parent alloca command for it (it must be
    // there)
    auto IsSuitableAlloca = [Record](AllocaCommandBase *AllocaCmd) {
      bool Res = sameCtx(AllocaCmd->getQueue()->getContextImplPtr(),
                         Record->MCurContext) &&
                 // Looking for a parent buffer alloca command
                 AllocaCmd->getType() == Command::CommandType::ALLOCA;
      return Res;
    };
    const auto It =
        std::find_if(Record->MAllocaCommands.begin(),
                     Record->MAllocaCommands.end(), IsSuitableAlloca);
    AllocaCmdSrc = (Record->MAllocaCommands.end() != It) ? *It : nullptr;
  }
  if (!AllocaCmdSrc)
    throw runtime_error("Cannot find buffer allocation", PI_INVALID_VALUE);
  // Get parent allocation of sub buffer to perform full copy of whole buffer
  if (IsSuitableSubReq(Req)) {
    if (AllocaCmdSrc->getType() == Command::CommandType::ALLOCA_SUB_BUF)
      AllocaCmdSrc =
          static_cast<AllocaSubBufCommand *>(AllocaCmdSrc)->getParentAlloca();
    else if (AllocaCmdSrc->getSYCLMemObj() != Req->MSYCLMemObj)
      assert(false && "Inappropriate alloca command.");
  }

  Command *NewCmd = nullptr;

  if (AllocaCmdSrc->MLinkedAllocaCmd == AllocaCmdDst) {
    // Map write only as read-write
    access::mode MapMode = Req->MAccessMode;
    if (MapMode == access::mode::write)
      MapMode = access::mode::read_write;
    NewCmd = insertMapUnmapForLinkedCmds(AllocaCmdSrc, AllocaCmdDst, MapMode);
    Record->MHostAccess = MapMode;
  } else {

    if ((Req->MAccessMode == access::mode::discard_write) ||
        (Req->MAccessMode == access::mode::discard_read_write)) {
      Record->MCurContext = Queue->getContextImplPtr();
      return nullptr;
    } else {
      // Full copy of buffer is needed to avoid loss of data that may be caused
      // by copying specific range from host to device and backwards.
      NewCmd =
          new MemCpyCommand(*AllocaCmdSrc->getRequirement(), AllocaCmdSrc,
                            *AllocaCmdDst->getRequirement(), AllocaCmdDst,
                            AllocaCmdSrc->getQueue(), AllocaCmdDst->getQueue());
    }
  }
  std::vector<Command *> ToCleanUp;
  for (Command *Dep : Deps) {
    Command *ConnCmd = NewCmd->addDep(
        DepDesc{Dep, NewCmd->getRequirement(), AllocaCmdDst}, ToCleanUp);
    if (ConnCmd)
      ToEnqueue.push_back(ConnCmd);
  }
  updateLeaves(Deps, Record, access::mode::read_write, ToCleanUp);
  addNodeToLeaves(Record, NewCmd, access::mode::read_write, ToEnqueue);
  for (Command *Cmd : ToCleanUp)
    cleanupCommand(Cmd);
  Record->MCurContext = Queue->getContextImplPtr();
  return NewCmd;
}

Command *Scheduler::GraphBuilder::remapMemoryObject(
    MemObjRecord *Record, Requirement *Req, AllocaCommandBase *HostAllocaCmd,
    std::vector<Command *> &ToEnqueue) {
  assert(HostAllocaCmd->getQueue()->is_host() &&
         "Host alloca command expected");
  assert(HostAllocaCmd->MIsActive && "Active alloca command expected");

  AllocaCommandBase *LinkedAllocaCmd = HostAllocaCmd->MLinkedAllocaCmd;
  assert(LinkedAllocaCmd && "Linked alloca command expected");

  std::set<Command *> Deps = findDepsForReq(Record, Req, Record->MCurContext);

  UnMapMemObject *UnMapCmd = new UnMapMemObject(
      LinkedAllocaCmd, *LinkedAllocaCmd->getRequirement(),
      &HostAllocaCmd->MMemAllocation, LinkedAllocaCmd->getQueue());

  // Map write only as read-write
  access::mode MapMode = Req->MAccessMode;
  if (MapMode == access::mode::write)
    MapMode = access::mode::read_write;
  MapMemObject *MapCmd = new MapMemObject(
      LinkedAllocaCmd, *LinkedAllocaCmd->getRequirement(),
      &HostAllocaCmd->MMemAllocation, LinkedAllocaCmd->getQueue(), MapMode);

  std::vector<Command *> ToCleanUp;
  for (Command *Dep : Deps) {
    Command *ConnCmd = UnMapCmd->addDep(
        DepDesc{Dep, UnMapCmd->getRequirement(), LinkedAllocaCmd}, ToCleanUp);
    if (ConnCmd)
      ToEnqueue.push_back(ConnCmd);
  }

  Command *ConnCmd = MapCmd->addDep(
      DepDesc{UnMapCmd, MapCmd->getRequirement(), HostAllocaCmd}, ToCleanUp);
  if (ConnCmd)
    ToEnqueue.push_back(ConnCmd);

  updateLeaves(Deps, Record, access::mode::read_write, ToCleanUp);
  addNodeToLeaves(Record, MapCmd, access::mode::read_write, ToEnqueue);
  for (Command *Cmd : ToCleanUp)
    cleanupCommand(Cmd);
  Record->MHostAccess = MapMode;
  return MapCmd;
}

// The function adds copy operation of the up to date'st memory to the memory
// pointed by Req.
Command *
Scheduler::GraphBuilder::addCopyBack(Requirement *Req,
                                     std::vector<Command *> &ToEnqueue) {
  QueueImplPtr HostQueue = Scheduler::getInstance().getDefaultHostQueue();
  SYCLMemObjI *MemObj = Req->MSYCLMemObj;
  MemObjRecord *Record = getMemObjRecord(MemObj);
  if (Record && MPrintOptionsArray[BeforeAddCopyBack])
    printGraphAsDot("before_addCopyBack");

  // Do nothing if there were no or only read operations with the memory object.
  if (nullptr == Record || !Record->MMemModified)
    return nullptr;

  std::set<Command *> Deps =
      findDepsForReq(Record, Req, HostQueue->getContextImplPtr());
  AllocaCommandBase *SrcAllocaCmd =
      findAllocaForReq(Record, Req, Record->MCurContext);

  auto MemCpyCmdUniquePtr = std::make_unique<MemCpyCommandHost>(
      *SrcAllocaCmd->getRequirement(), SrcAllocaCmd, *Req, &Req->MData,
      SrcAllocaCmd->getQueue(), std::move(HostQueue));

  if (!MemCpyCmdUniquePtr)
    throw runtime_error("Out of host memory", PI_OUT_OF_HOST_MEMORY);

  MemCpyCommandHost *MemCpyCmd = MemCpyCmdUniquePtr.release();

  std::vector<Command *> ToCleanUp;
  for (Command *Dep : Deps) {
    Command *ConnCmd = MemCpyCmd->addDep(
        DepDesc{Dep, MemCpyCmd->getRequirement(), SrcAllocaCmd}, ToCleanUp);
    if (ConnCmd)
      ToEnqueue.push_back(ConnCmd);
  }

  updateLeaves(Deps, Record, Req->MAccessMode, ToCleanUp);
  addNodeToLeaves(Record, MemCpyCmd, Req->MAccessMode, ToEnqueue);
  for (Command *Cmd : ToCleanUp)
    cleanupCommand(Cmd);
  if (MPrintOptionsArray[AfterAddCopyBack])
    printGraphAsDot("after_addCopyBack");
  return MemCpyCmd;
}

// The function implements SYCL host accessor logic: host accessor
// should provide access to the buffer in user space.
Command *
Scheduler::GraphBuilder::addHostAccessor(Requirement *Req,
                                         std::vector<Command *> &ToEnqueue) {

  const QueueImplPtr &HostQueue = getInstance().getDefaultHostQueue();

  MemObjRecord *Record = getOrInsertMemObjRecord(HostQueue, Req, ToEnqueue);
  if (MPrintOptionsArray[BeforeAddHostAcc])
    printGraphAsDot("before_addHostAccessor");
  markModifiedIfWrite(Record, Req);

  AllocaCommandBase *HostAllocaCmd =
      getOrCreateAllocaForReq(Record, Req, HostQueue, ToEnqueue);

  if (sameCtx(HostAllocaCmd->getQueue()->getContextImplPtr(),
              Record->MCurContext)) {
    if (!isAccessModeAllowed(Req->MAccessMode, Record->MHostAccess))
      remapMemoryObject(Record, Req, HostAllocaCmd, ToEnqueue);
  } else
    insertMemoryMove(Record, Req, HostQueue, ToEnqueue);

  Command *UpdateHostAccCmd =
      insertUpdateHostReqCmd(Record, Req, HostQueue, ToEnqueue);

  // Need empty command to be blocked until host accessor is destructed
  EmptyCommand *EmptyCmd =
      addEmptyCmd<Requirement>(UpdateHostAccCmd, {Req}, HostQueue,
                               Command::BlockReason::HostAccessor, ToEnqueue);

  Req->MBlockedCmd = EmptyCmd;

  if (MPrintOptionsArray[AfterAddHostAcc])
    printGraphAsDot("after_addHostAccessor");

  return UpdateHostAccCmd;
}

Command *Scheduler::GraphBuilder::addCGUpdateHost(
    std::unique_ptr<detail::CG> CommandGroup, QueueImplPtr HostQueue,
    std::vector<Command *> &ToEnqueue) {

  auto UpdateHost = static_cast<CGUpdateHost *>(CommandGroup.get());
  Requirement *Req = UpdateHost->getReqToUpdate();

  MemObjRecord *Record = getOrInsertMemObjRecord(HostQueue, Req, ToEnqueue);
  return insertMemoryMove(Record, Req, HostQueue, ToEnqueue);
}

/// Start the search for the record from list of "leaf" commands and check if
/// the examined command can be executed in parallel with the new one with
/// regard to the memory object. If it can, then continue searching through
/// dependencies of that command. There are several rules used:
///
/// 1. New and examined commands only read -> can bypass
/// 2. New and examined commands has non-overlapping requirements -> can bypass
/// 3. New and examined commands have different contexts -> cannot bypass
std::set<Command *>
Scheduler::GraphBuilder::findDepsForReq(MemObjRecord *Record,
                                        const Requirement *Req,
                                        const ContextImplPtr &Context) {
  std::set<Command *> RetDeps;
  std::vector<Command *> Visited;
  const bool ReadOnlyReq = Req->MAccessMode == access::mode::read;

  std::vector<Command *> ToAnalyze{Record->MWriteLeaves.toVector()};

  if (!ReadOnlyReq) {
    std::vector<Command *> V{Record->MReadLeaves.toVector()};

    ToAnalyze.insert(ToAnalyze.begin(), V.begin(), V.end());
  }

  while (!ToAnalyze.empty()) {
    Command *DepCmd = ToAnalyze.back();
    ToAnalyze.pop_back();

    std::vector<Command *> NewAnalyze;

    for (const DepDesc &Dep : DepCmd->MDeps) {
      if (Dep.MDepRequirement->MSYCLMemObj != Req->MSYCLMemObj)
        continue;

      bool CanBypassDep = false;
      // If both only read
      CanBypassDep |=
          Dep.MDepRequirement->MAccessMode == access::mode::read && ReadOnlyReq;

      // If not overlap
      CanBypassDep |= !doOverlap(Dep.MDepRequirement, Req);

      // Going through copying memory between contexts is not supported.
      if (Dep.MDepCommand)
        CanBypassDep &=
            sameCtx(Context, Dep.MDepCommand->getQueue()->getContextImplPtr());

      if (!CanBypassDep) {
        RetDeps.insert(DepCmd);
        // No need to analyze deps of examining command as it's dependency
        // itself.
        NewAnalyze.clear();
        break;
      }

      if (markNodeAsVisited(Dep.MDepCommand, Visited))
        NewAnalyze.push_back(Dep.MDepCommand);
    }
    ToAnalyze.insert(ToAnalyze.end(), NewAnalyze.begin(), NewAnalyze.end());
  }
  unmarkVisitedNodes(Visited);
  return RetDeps;
}

// A helper function for finding a command dependency on a specific memory
// object
DepDesc Scheduler::GraphBuilder::findDepForRecord(Command *Cmd,
                                                  MemObjRecord *Record) {
  for (const DepDesc &DD : Cmd->MDeps) {
    if (getMemObjRecord(DD.MDepRequirement->MSYCLMemObj) == Record) {
      return DD;
    }
  }
  assert(false && "No dependency found for a leaf of the record");
  return {nullptr, nullptr, nullptr};
}

// The function searches for the alloca command matching context and
// requirement.
AllocaCommandBase *
Scheduler::GraphBuilder::findAllocaForReq(MemObjRecord *Record,
                                          const Requirement *Req,
                                          const ContextImplPtr &Context) {
  auto IsSuitableAlloca = [&Context, Req](AllocaCommandBase *AllocaCmd) {
    bool Res = sameCtx(AllocaCmd->getQueue()->getContextImplPtr(), Context);
    if (IsSuitableSubReq(Req)) {
      const Requirement *TmpReq = AllocaCmd->getRequirement();
      Res &= AllocaCmd->getType() == Command::CommandType::ALLOCA_SUB_BUF;
      Res &= TmpReq->MOffsetInBytes == Req->MOffsetInBytes;
      Res &= TmpReq->MSYCLMemObj->getSize() == Req->MSYCLMemObj->getSize();
    }
    return Res;
  };
  const auto It = std::find_if(Record->MAllocaCommands.begin(),
                               Record->MAllocaCommands.end(), IsSuitableAlloca);
  return (Record->MAllocaCommands.end() != It) ? *It : nullptr;
}

static bool checkHostUnifiedMemory(const ContextImplPtr &Ctx) {
  if (const char *HUMConfig = SYCLConfig<SYCL_HOST_UNIFIED_MEMORY>::get()) {
    if (std::strcmp(HUMConfig, "0") == 0)
      return Ctx->is_host();
    if (std::strcmp(HUMConfig, "1") == 0)
      return true;
  }
  for (const device &Device : Ctx->getDevices()) {
    if (!Device.get_info<info::device::host_unified_memory>())
      return false;
  }
  return true;
}

// The function searches for the alloca command matching context and
// requirement. If none exists, new allocation command is created.
// Note, creation of new allocation command can lead to the current context
// (Record->MCurContext) change.
AllocaCommandBase *Scheduler::GraphBuilder::getOrCreateAllocaForReq(
    MemObjRecord *Record, const Requirement *Req, QueueImplPtr Queue,
    std::vector<Command *> &ToEnqueue) {

  AllocaCommandBase *AllocaCmd =
      findAllocaForReq(Record, Req, Queue->getContextImplPtr());

  if (!AllocaCmd) {
    std::vector<Command *> ToCleanUp;
    if (IsSuitableSubReq(Req)) {
      // Get parent requirement. It's hard to get right parents' range
      // so full parent requirement has range represented in bytes
      range<3> ParentRange{Req->MSYCLMemObj->getSize(), 1, 1};
      Requirement ParentRequirement(/*Offset*/ {0, 0, 0}, ParentRange,
                                    ParentRange, access::mode::read_write,
                                    Req->MSYCLMemObj, /*Dims*/ 1,
                                    /*Working with bytes*/ sizeof(char));

      auto *ParentAlloca =
          getOrCreateAllocaForReq(Record, &ParentRequirement, Queue, ToEnqueue);
      AllocaCmd = new AllocaSubBufCommand(Queue, *Req, ParentAlloca, ToEnqueue,
                                          ToCleanUp);
    } else {

      const Requirement FullReq(/*Offset*/ {0, 0, 0}, Req->MMemoryRange,
                                Req->MMemoryRange, access::mode::read_write,
                                Req->MSYCLMemObj, Req->MDims, Req->MElemSize,
                                0 /*ReMOffsetInBytes*/, false /*MIsSubBuffer*/);
      // Can reuse user data for the first allocation. Do so if host unified
      // memory is supported regardless of the access mode (the pointer will be
      // reused). For devices without host unified memory the initialization
      // will be performed as a write operation.
      // TODO the case where the first alloca is made with a discard mode and
      // the user pointer is read-only is still not handled: it leads to
      // unnecessary copy on devices with unified host memory support.
      const bool HostUnifiedMemory =
          checkHostUnifiedMemory(Queue->getContextImplPtr());
      // TODO casting is required here to get the necessary information
      // without breaking ABI, replace with the next major version.
      auto *MemObj = static_cast<SYCLMemObjT *>(Req->MSYCLMemObj);
      const bool InitFromUserData = Record->MAllocaCommands.empty() &&
                                    (HostUnifiedMemory || MemObj->isInterop());
      AllocaCommandBase *LinkedAllocaCmd = nullptr;

      // For the first allocation on a device without host unified memory we
      // might need to also create a host alloca right away in order to perform
      // the initial memory write.
      if (Record->MAllocaCommands.empty()) {
        if (!HostUnifiedMemory &&
            Req->MAccessMode != access::mode::discard_write &&
            Req->MAccessMode != access::mode::discard_read_write) {
          // There's no need to make a host allocation if the buffer is not
          // initialized with user data.
          if (MemObj->hasUserDataPtr()) {
            QueueImplPtr DefaultHostQueue =
                Scheduler::getInstance().getDefaultHostQueue();
            AllocaCommand *HostAllocaCmd = new AllocaCommand(
                DefaultHostQueue, FullReq, true /* InitFromUserData */,
                nullptr /* LinkedAllocaCmd */);
            Record->MAllocaCommands.push_back(HostAllocaCmd);
            Record->MWriteLeaves.push_back(HostAllocaCmd, ToEnqueue);
            ++(HostAllocaCmd->MLeafCounter);
            Record->MCurContext = DefaultHostQueue->getContextImplPtr();
          }
        }
      } else {
        // If it is not the first allocation, try to setup a link
        // FIXME: Temporary limitation, linked alloca commands for an image is
        // not supported because map operation is not implemented for an image.
        if (Req->MSYCLMemObj->getType() == SYCLMemObjI::MemObjType::Buffer)
          // Current limitation is to setup link between current allocation and
          // new one. There could be situations when we could setup link with
          // "not" current allocation, but it will require memory copy.
          // Can setup link between cl and host allocations only
          if (Queue->is_host() != Record->MCurContext->is_host()) {
            // Linked commands assume that the host allocation is reused by the
            // plugin runtime and that can lead to unnecessary copy overhead on
            // devices that do not support host unified memory. Do not link the
            // allocations in this case.
            // However, if the user explicitly requests use of pinned host
            // memory, map/unmap operations are expected to work faster than
            // read/write from/to an artbitrary host pointer. Link such commands
            // regardless of host unified memory support.
            bool PinnedHostMemory = MemObj->has_property<
                sycl::ext::oneapi::property::buffer::use_pinned_host_memory>();

            bool HostUnifiedMemoryOnNonHostDevice =
                Queue->is_host() ? checkHostUnifiedMemory(Record->MCurContext)
                                 : HostUnifiedMemory;
            if (PinnedHostMemory || HostUnifiedMemoryOnNonHostDevice) {
              AllocaCommandBase *LinkedAllocaCmdCand =
                  findAllocaForReq(Record, Req, Record->MCurContext);

              // Cannot setup link if candidate is linked already
              if (LinkedAllocaCmdCand &&
                  !LinkedAllocaCmdCand->MLinkedAllocaCmd) {
                LinkedAllocaCmd = LinkedAllocaCmdCand;
              }
            }
          }
      }

      AllocaCmd =
          new AllocaCommand(Queue, FullReq, InitFromUserData, LinkedAllocaCmd);

      // Update linked command
      if (LinkedAllocaCmd) {
        Command *ConnCmd = AllocaCmd->addDep(
            DepDesc{LinkedAllocaCmd, AllocaCmd->getRequirement(),
                    LinkedAllocaCmd},
            ToCleanUp);
        if (ConnCmd)
          ToEnqueue.push_back(ConnCmd);
        LinkedAllocaCmd->MLinkedAllocaCmd = AllocaCmd;

        // To ensure that the leader allocation is removed first
        ConnCmd = AllocaCmd->getReleaseCmd()->addDep(
            DepDesc(LinkedAllocaCmd->getReleaseCmd(),
                    AllocaCmd->getRequirement(), LinkedAllocaCmd),
            ToCleanUp);
        if (ConnCmd)
          ToEnqueue.push_back(ConnCmd);

        // Device allocation takes ownership of the host ptr during
        // construction, host allocation doesn't. So, device allocation should
        // always be active here. Also if the "follower" command is a device one
        // we have to change current context to the device one.
        if (Queue->is_host()) {
          AllocaCmd->MIsActive = false;
        } else {
          LinkedAllocaCmd->MIsActive = false;
          Record->MCurContext = Queue->getContextImplPtr();

          std::set<Command *> Deps =
              findDepsForReq(Record, Req, Queue->getContextImplPtr());
          for (Command *Dep : Deps) {
            Command *ConnCmd = AllocaCmd->addDep(
                DepDesc{Dep, Req, LinkedAllocaCmd}, ToCleanUp);
            if (ConnCmd)
              ToEnqueue.push_back(ConnCmd);
          }
          updateLeaves(Deps, Record, Req->MAccessMode, ToCleanUp);
          addNodeToLeaves(Record, AllocaCmd, Req->MAccessMode, ToEnqueue);
        }
      }
    }

    Record->MAllocaCommands.push_back(AllocaCmd);
    Record->MWriteLeaves.push_back(AllocaCmd, ToEnqueue);
    ++(AllocaCmd->MLeafCounter);
    for (Command *Cmd : ToCleanUp)
      cleanupCommand(Cmd);
  }
  return AllocaCmd;
}

// The function sets MemModified flag in record if requirement has write access.
void Scheduler::GraphBuilder::markModifiedIfWrite(MemObjRecord *Record,
                                                  Requirement *Req) {
  switch (Req->MAccessMode) {
  case access::mode::write:
  case access::mode::read_write:
  case access::mode::discard_write:
  case access::mode::discard_read_write:
  case access::mode::atomic:
    Record->MMemModified = true;
    break;
  case access::mode::read:
    break;
  }
}

template <typename T>
typename detail::enable_if_t<
    std::is_same<typename std::remove_cv_t<T>, Requirement>::value,
    EmptyCommand *>
Scheduler::GraphBuilder::addEmptyCmd(Command *Cmd, const std::vector<T *> &Reqs,
                                     const QueueImplPtr &Queue,
                                     Command::BlockReason Reason,
                                     std::vector<Command *> &ToEnqueue,
                                     const bool AddDepsToLeaves) {
  EmptyCommand *EmptyCmd =
      new EmptyCommand(Scheduler::getInstance().getDefaultHostQueue());

  if (!EmptyCmd)
    throw runtime_error("Out of host memory", PI_OUT_OF_HOST_MEMORY);

  EmptyCmd->MIsBlockable = true;
  EmptyCmd->MEnqueueStatus = EnqueueResultT::SyclEnqueueBlocked;
  EmptyCmd->MBlockReason = Reason;

  for (T *Req : Reqs) {
    MemObjRecord *Record = getOrInsertMemObjRecord(Queue, Req, ToEnqueue);
    AllocaCommandBase *AllocaCmd =
        getOrCreateAllocaForReq(Record, Req, Queue, ToEnqueue);
    EmptyCmd->addRequirement(Cmd, AllocaCmd, Req);
  }
  // addRequirement above call addDep that already will add EmptyCmd as user for
  // Cmd no Reqs size check here so assume it is possible to have no Reqs passed
  if (!Reqs.size())
    Cmd->addUser(EmptyCmd);

  if (AddDepsToLeaves) {
    const std::vector<DepDesc> &Deps = Cmd->MDeps;
    std::vector<Command *> ToCleanUp;
    for (const DepDesc &Dep : Deps) {
      const Requirement *Req = Dep.MDepRequirement;
      MemObjRecord *Record = getMemObjRecord(Req->MSYCLMemObj);

      updateLeaves({Cmd}, Record, Req->MAccessMode, ToCleanUp);
      addNodeToLeaves(Record, EmptyCmd, Req->MAccessMode, ToEnqueue);
    }
    for (Command *Cmd : ToCleanUp)
      cleanupCommand(Cmd);
  }

  return EmptyCmd;
}

static bool isInteropHostTask(const std::unique_ptr<ExecCGCommand> &Cmd) {
  if (Cmd->getCG().getType() != CG::CGTYPE::CodeplayHostTask)
    return false;

  const detail::CGHostTask &HT =
      static_cast<detail::CGHostTask &>(Cmd->getCG());

  return HT.MHostTask->isInteropTask();
}

static void combineAccessModesOfReqs(std::vector<Requirement *> &Reqs) {
  std::unordered_map<SYCLMemObjI *, access::mode> CombinedModes;
  bool HasDuplicateMemObjects = false;
  for (const Requirement *Req : Reqs) {
    auto Result = CombinedModes.insert(
        std::make_pair(Req->MSYCLMemObj, Req->MAccessMode));
    if (!Result.second) {
      Result.first->second =
          combineAccessModes(Result.first->second, Req->MAccessMode);
      HasDuplicateMemObjects = true;
    }
  }

  if (!HasDuplicateMemObjects)
    return;
  for (Requirement *Req : Reqs) {
    Req->MAccessMode = CombinedModes[Req->MSYCLMemObj];
  }
}

Command *
Scheduler::GraphBuilder::addCG(std::unique_ptr<detail::CG> CommandGroup,
                               QueueImplPtr Queue,
                               std::vector<Command *> &ToEnqueue) {
  std::vector<Requirement *> &Reqs = CommandGroup->MRequirements;
  const std::vector<detail::EventImplPtr> &Events = CommandGroup->MEvents;
  const CG::CGTYPE CGType = CommandGroup->getType();

  auto NewCmd = std::make_unique<ExecCGCommand>(std::move(CommandGroup), Queue);
  if (!NewCmd)
    throw runtime_error("Out of host memory", PI_OUT_OF_HOST_MEMORY);

  if (MPrintOptionsArray[BeforeAddCG])
    printGraphAsDot("before_addCG");

  // If there are multiple requirements for the same memory object, its
  // AllocaCommand creation will be dependent on the access mode of the first
  // requirement. Combine these access modes to take all of them into account.
  combineAccessModesOfReqs(Reqs);
  std::vector<Command *> ToCleanUp;
  for (Requirement *Req : Reqs) {
    MemObjRecord *Record = nullptr;
    AllocaCommandBase *AllocaCmd = nullptr;

    bool isSameCtx = false;

    {
      const QueueImplPtr &QueueForAlloca =
          isInteropHostTask(NewCmd)
              ? static_cast<detail::CGHostTask &>(NewCmd->getCG()).MQueue
              : Queue;

      Record = getOrInsertMemObjRecord(QueueForAlloca, Req, ToEnqueue);
      markModifiedIfWrite(Record, Req);

      AllocaCmd =
          getOrCreateAllocaForReq(Record, Req, QueueForAlloca, ToEnqueue);

      isSameCtx =
          sameCtx(QueueForAlloca->getContextImplPtr(), Record->MCurContext);
    }

    // If there is alloca command we need to check if the latest memory is in
    // required context.
    if (isSameCtx) {
      // If the memory is already in the required host context, check if the
      // required access mode is valid, remap if not.
      if (Record->MCurContext->is_host() &&
          !isAccessModeAllowed(Req->MAccessMode, Record->MHostAccess))
        remapMemoryObject(Record, Req, AllocaCmd, ToEnqueue);
    } else {
      // Cannot directly copy memory from OpenCL device to OpenCL device -
      // create two copies: device->host and host->device.
      bool NeedMemMoveToHost = false;
      auto MemMoveTargetQueue = Queue;

      if (isInteropHostTask(NewCmd)) {
        const detail::CGHostTask &HT =
            static_cast<detail::CGHostTask &>(NewCmd->getCG());

        if (HT.MQueue->getContextImplPtr() != Record->MCurContext) {
          NeedMemMoveToHost = true;
          MemMoveTargetQueue = HT.MQueue;
        }
      } else if (!Queue->is_host() && !Record->MCurContext->is_host())
        NeedMemMoveToHost = true;

      if (NeedMemMoveToHost)
        insertMemoryMove(Record, Req,
                         Scheduler::getInstance().getDefaultHostQueue(),
                         ToEnqueue);
      insertMemoryMove(Record, Req, MemMoveTargetQueue, ToEnqueue);
    }
    std::set<Command *> Deps =
        findDepsForReq(Record, Req, Queue->getContextImplPtr());

    for (Command *Dep : Deps) {
      Command *ConnCmd =
          NewCmd->addDep(DepDesc{Dep, Req, AllocaCmd}, ToCleanUp);
      if (ConnCmd)
        ToEnqueue.push_back(ConnCmd);
    }
  }

  // Set new command as user for dependencies and update leaves.
  // Node dependencies can be modified further when adding the node to leaves,
  // iterate over their copy.
  // FIXME employ a reference here to eliminate copying of a vector
  std::vector<DepDesc> Deps = NewCmd->MDeps;
  for (DepDesc &Dep : Deps) {
    const Requirement *Req = Dep.MDepRequirement;
    MemObjRecord *Record = getMemObjRecord(Req->MSYCLMemObj);
    updateLeaves({Dep.MDepCommand}, Record, Req->MAccessMode, ToCleanUp);
    addNodeToLeaves(Record, NewCmd.get(), Req->MAccessMode, ToEnqueue);
  }

  // Register all the events as dependencies
  for (detail::EventImplPtr e : Events) {
    if (Command *ConnCmd = NewCmd->addDep(e, ToCleanUp))
      ToEnqueue.push_back(ConnCmd);
  }

  if (CGType == CG::CGTYPE::CodeplayHostTask)
    NewCmd->MEmptyCmd =
        addEmptyCmd(NewCmd.get(), NewCmd->getCG().MRequirements, Queue,
                    Command::BlockReason::HostTask, ToEnqueue);

  if (MPrintOptionsArray[AfterAddCG])
    printGraphAsDot("after_addCG");

  for (Command *Cmd : ToCleanUp)
    cleanupCommand(Cmd);
  return NewCmd.release();
}

void Scheduler::GraphBuilder::decrementLeafCountersForRecord(
    MemObjRecord *Record) {
  for (Command *Cmd : Record->MReadLeaves) {
    --(Cmd->MLeafCounter);
    if (Cmd->MLeafCounter == 0 && Cmd->isSuccessfullyEnqueued() &&
        Cmd->supportsPostEnqueueCleanup())
      cleanupCommand(Cmd);
  }
  for (Command *Cmd : Record->MWriteLeaves) {
    --(Cmd->MLeafCounter);
    if (Cmd->MLeafCounter == 0 && Cmd->isSuccessfullyEnqueued() &&
        Cmd->supportsPostEnqueueCleanup())
      cleanupCommand(Cmd);
  }
}

void Scheduler::GraphBuilder::cleanupCommandsForRecord(
    MemObjRecord *Record,
    std::vector<std::shared_ptr<stream_impl>> &StreamsToDeallocate,
    std::vector<std::shared_ptr<const void>> &ReduResourcesToDeallocate) {
  std::vector<AllocaCommandBase *> &AllocaCommands = Record->MAllocaCommands;
  if (AllocaCommands.empty())
    return;

  assert(MCmdsToVisit.empty());
  MVisitedCmds.clear();

  // First, mark all allocas for deletion and their direct users for traversal
  // Dependencies of the users will be cleaned up during the traversal
  for (Command *AllocaCmd : AllocaCommands) {
    markNodeAsVisited(AllocaCmd, MVisitedCmds);

    for (Command *UserCmd : AllocaCmd->MUsers)
      // Linked alloca cmd may be in users of this alloca. We're not going to
      // visit it.
      if (UserCmd->getType() != Command::CommandType::ALLOCA)
        MCmdsToVisit.push(UserCmd);
      else
        markNodeAsVisited(UserCmd, MVisitedCmds);

    AllocaCmd->MMarks.MToBeDeleted = true;
    // These commands will be deleted later, clear users now to avoid
    // updating them during edge removal
    AllocaCmd->MUsers.clear();
  }

  // Make sure the Linked Allocas are marked visited by the previous walk.
  // Remove allocation commands from the users of their dependencies.
  for (AllocaCommandBase *AllocaCmd : AllocaCommands) {
    AllocaCommandBase *LinkedCmd = AllocaCmd->MLinkedAllocaCmd;

    if (LinkedCmd) {
      assert(LinkedCmd->MMarks.MVisited);
    }

    for (DepDesc &Dep : AllocaCmd->MDeps)
      if (Dep.MDepCommand)
        Dep.MDepCommand->MUsers.erase(AllocaCmd);
  }

  // Traverse the graph using BFS
  while (!MCmdsToVisit.empty()) {
    Command *Cmd = MCmdsToVisit.front();
    MCmdsToVisit.pop();

    if (!markNodeAsVisited(Cmd, MVisitedCmds))
      continue;

    // Collect stream objects for a visited command.
    if (Cmd->getType() == Command::CommandType::RUN_CG) {
      auto ExecCmd = static_cast<ExecCGCommand *>(Cmd);

      // Transfer ownership of stream implementations.
      std::vector<std::shared_ptr<stream_impl>> Streams = ExecCmd->getStreams();
      ExecCmd->clearStreams();
      StreamsToDeallocate.insert(StreamsToDeallocate.end(), Streams.begin(),
                                 Streams.end());

      // Transfer ownership of auxiliary resources.
      std::vector<std::shared_ptr<const void>> ReduResources =
          ExecCmd->getAuxiliaryResources();
      ExecCmd->clearAuxiliaryResources();
      ReduResourcesToDeallocate.insert(ReduResourcesToDeallocate.end(),
                                       ReduResources.begin(),
                                       ReduResources.end());
    }

    for (Command *UserCmd : Cmd->MUsers)
      if (UserCmd->getType() != Command::CommandType::ALLOCA)
        MCmdsToVisit.push(UserCmd);

    // Delete all dependencies on any allocations being removed
    // Track which commands should have their users updated
    std::map<Command *, bool> ShouldBeUpdated;
    auto NewEnd = std::remove_if(
        Cmd->MDeps.begin(), Cmd->MDeps.end(), [&](const DepDesc &Dep) {
          if (std::find(AllocaCommands.begin(), AllocaCommands.end(),
                        Dep.MAllocaCmd) != AllocaCommands.end()) {
            ShouldBeUpdated.insert({Dep.MDepCommand, true});
            return true;
          }
          ShouldBeUpdated[Dep.MDepCommand] = false;
          return false;
        });
    Cmd->MDeps.erase(NewEnd, Cmd->MDeps.end());

    // Update users of removed dependencies
    for (auto DepCmdIt : ShouldBeUpdated) {
      if (!DepCmdIt.second)
        continue;
      DepCmdIt.first->MUsers.erase(Cmd);
    }

    // If all dependencies have been removed this way, mark the command for
    // deletion
    if (Cmd->MDeps.empty()) {
      Cmd->MUsers.clear();
      // Do not delete the node if it's scheduled for post-enqueue cleanup to
      // avoid double free.
      if (!Cmd->MPostEnqueueCleanup)
        Cmd->MMarks.MToBeDeleted = true;
    }
  }

  handleVisitedNodes(MVisitedCmds);
}

void Scheduler::GraphBuilder::cleanupCommand(Command *Cmd) {
  if (SYCLConfig<SYCL_DISABLE_POST_ENQUEUE_CLEANUP>::get())
    return;
  assert(Cmd->MLeafCounter == 0 && Cmd->isSuccessfullyEnqueued());
  Command::CommandType CmdT = Cmd->getType();

  assert(CmdT != Command::ALLOCA && CmdT != Command::ALLOCA_SUB_BUF);
  assert(CmdT != Command::RELEASE);
  assert(CmdT != Command::RUN_CG ||
         (static_cast<ExecCGCommand *>(Cmd))->getCG().getType() !=
             CG::CGTYPE::CodeplayHostTask);
#ifndef NDEBUG
  if (CmdT == Command::RUN_CG) {
    auto *ExecCGCmd = static_cast<ExecCGCommand *>(Cmd);
    if (ExecCGCmd->getCG().getType() == CG::CGTYPE::Kernel) {
      auto *ExecKernelCG = static_cast<CGExecKernel *>(&ExecCGCmd->getCG());
      assert(!ExecKernelCG->hasStreams());
      assert(!ExecKernelCG->hasAuxiliaryResources());
    }
  }
#endif
  (void)CmdT;

  for (Command *UserCmd : Cmd->MUsers) {
    for (DepDesc &Dep : UserCmd->MDeps) {
      // Link the users of the command to the alloca command(s) instead
      if (Dep.MDepCommand == Cmd) {
        // ... unless the user is the alloca itself.
        if (Dep.MAllocaCmd == UserCmd) {
          Dep.MDepCommand = nullptr;
        } else {
          Dep.MDepCommand = Dep.MAllocaCmd;
          Dep.MDepCommand->MUsers.insert(UserCmd);
        }
      }
    }
  }
  // Update dependency users
  for (DepDesc &Dep : Cmd->MDeps) {
    Command *DepCmd = Dep.MDepCommand;
    DepCmd->MUsers.erase(Cmd);
  }

  Cmd->getEvent()->setCommand(nullptr);
  delete Cmd;
}

void Scheduler::GraphBuilder::cleanupFinishedCommands(
    Command *FinishedCmd,
    std::vector<std::shared_ptr<stream_impl>> &StreamsToDeallocate,
    std::vector<std::shared_ptr<const void>> &ReduResourcesToDeallocate) {
  assert(MCmdsToVisit.empty());
  MCmdsToVisit.push(FinishedCmd);
  MVisitedCmds.clear();

  // Traverse the graph using BFS
  while (!MCmdsToVisit.empty()) {
    Command *Cmd = MCmdsToVisit.front();
    MCmdsToVisit.pop();

    if (!markNodeAsVisited(Cmd, MVisitedCmds))
      continue;

    // Collect stream objects for a visited command.
    if (Cmd->getType() == Command::CommandType::RUN_CG) {
      auto ExecCmd = static_cast<ExecCGCommand *>(Cmd);

      // Transfer ownership of stream implementations.
      std::vector<std::shared_ptr<stream_impl>> Streams = ExecCmd->getStreams();
      ExecCmd->clearStreams();
      StreamsToDeallocate.insert(StreamsToDeallocate.end(), Streams.begin(),
                                 Streams.end());

      // Transfer ownership of auxiliary resources.
      std::vector<std::shared_ptr<const void>> ReduResources =
          ExecCmd->getAuxiliaryResources();
      ExecCmd->clearAuxiliaryResources();
      ReduResourcesToDeallocate.insert(ReduResourcesToDeallocate.end(),
                                       ReduResources.begin(),
                                       ReduResources.end());
    }

    for (const DepDesc &Dep : Cmd->MDeps) {
      if (Dep.MDepCommand)
        MCmdsToVisit.push(Dep.MDepCommand);
    }

    // Do not clean up the node if it is a leaf for any memory object
    if (Cmd->MLeafCounter > 0)
      continue;
    // Do not clean up allocation commands
    Command::CommandType CmdT = Cmd->getType();
    if (CmdT == Command::ALLOCA || CmdT == Command::ALLOCA_SUB_BUF)
      continue;

    for (Command *UserCmd : Cmd->MUsers) {
      for (DepDesc &Dep : UserCmd->MDeps) {
        // Link the users of the command to the alloca command(s) instead
        if (Dep.MDepCommand == Cmd) {
          Dep.MDepCommand = Dep.MAllocaCmd;
          Dep.MDepCommand->MUsers.insert(UserCmd);
        }
      }
    }
    // Update dependency users
    for (DepDesc &Dep : Cmd->MDeps) {
      Command *DepCmd = Dep.MDepCommand;
      DepCmd->MUsers.erase(Cmd);
    }

    // Isolate the node instead of deleting it if it's scheduled for
    // post-enqueue cleanup to avoid double free.
    if (Cmd->MPostEnqueueCleanup) {
      Cmd->MDeps.clear();
      Cmd->MUsers.clear();
    } else {
      Cmd->MMarks.MToBeDeleted = true;
    }
  }
  handleVisitedNodes(MVisitedCmds);
}

void Scheduler::GraphBuilder::removeRecordForMemObj(SYCLMemObjI *MemObject) {
  const auto It = std::find_if(
      MMemObjs.begin(), MMemObjs.end(),
      [MemObject](const SYCLMemObjI *Obj) { return Obj == MemObject; });
  if (It != MMemObjs.end())
    MMemObjs.erase(It);
  MemObject->MRecord.reset();
}

// Make Cmd depend on DepEvent from different context. Connection is performed
// via distinct ConnectCmd with host task command group on host queue. Cmd will
// depend on ConnectCmd's host event.
// DepEvent may not have a command associated with it in at least two cases:
//  - the command was deleted upon cleanup process;
//  - DepEvent is user event.
// In both of these cases the only thing we can do is to make ConnectCmd depend
// on DepEvent.
// Otherwise, when there is a command associated with DepEvent, we make
// ConnectCmd depend on on this command. If there is valid, i.e. non-nil,
// requirement in Dep we make ConnectCmd depend on DepEvent's command with this
// requirement.
// Optionality of Dep is set by Dep.MDepCommand equal to nullptr.
Command *Scheduler::GraphBuilder::connectDepEvent(
    Command *const Cmd, EventImplPtr DepEvent, const DepDesc &Dep,
    std::vector<Command *> &ToCleanUp) {
  assert(Cmd->getWorkerContext() != DepEvent->getContextImpl());

  // construct Host Task type command manually and make it depend on DepEvent
  ExecCGCommand *ConnectCmd = nullptr;

  try {
    std::unique_ptr<detail::HostTask> HT(new detail::HostTask);
    std::unique_ptr<detail::CG> ConnectCG(new detail::CGHostTask(
        std::move(HT), /* Queue = */ {}, /* Context = */ {}, /* Args = */ {},
        /* ArgsStorage = */ {}, /* AccStorage = */ {},
        /* SharedPtrStorage = */ {}, /* Requirements = */ {},
        /* DepEvents = */ {DepEvent}, CG::CodeplayHostTask,
        /* Payload */ {}));
    ConnectCmd = new ExecCGCommand(
        std::move(ConnectCG), Scheduler::getInstance().getDefaultHostQueue());
  } catch (const std::bad_alloc &) {
    throw runtime_error("Out of host memory", PI_OUT_OF_HOST_MEMORY);
  }

  EmptyCommand *EmptyCmd = nullptr;

  if (Dep.MDepRequirement) {
    // make ConnectCmd depend on requirement
    // Dismiss the result here as it's not a connection now,
    // 'cause ConnectCmd is host one
    (void)ConnectCmd->addDep(Dep, ToCleanUp);
    assert(reinterpret_cast<Command *>(DepEvent->getCommand()) ==
           Dep.MDepCommand);
    // add user to Dep.MDepCommand is already performed beyond this if branch

    // ConnectCmd is added as dependency to Cmd
    // We build the following structure Cmd->EmptyCmd/ConnectCmd->DepCmd
    // No need to add ConnectCmd to leaves buffer since it is a dependency
    // for command Cmd that will be added there

    std::vector<Command *> ToEnqueue;
    const std::vector<const Requirement *> Reqs(1, Dep.MDepRequirement);
    EmptyCmd = addEmptyCmd(ConnectCmd, Reqs,
                           Scheduler::getInstance().getDefaultHostQueue(),
                           Command::BlockReason::HostTask, ToEnqueue, false);
    assert(ToEnqueue.size() == 0);

    // Depend Cmd on empty command
    {
      DepDesc CmdDep = Dep;
      CmdDep.MDepCommand = EmptyCmd;

      // Dismiss the result here as it's not a connection now,
      // 'cause EmptyCmd is host one
      (void)Cmd->addDep(CmdDep, ToCleanUp);
    }
  } else {
    // It is required condition in another a path and addUser will be set in
    // addDep
    if (Command *DepCmd = reinterpret_cast<Command *>(DepEvent->getCommand()))
      DepCmd->addUser(ConnectCmd);

    std::vector<Command *> ToEnqueue;
    EmptyCmd = addEmptyCmd<Requirement>(
        ConnectCmd, {}, Scheduler::getInstance().getDefaultHostQueue(),
        Command::BlockReason::HostTask, ToEnqueue);
    assert(ToEnqueue.size() == 0);

    // There is no requirement thus, empty command will only depend on
    // ConnectCmd via its event.
    // Dismiss the result here as it's not a connection now,
    // 'cause ConnectCmd is host one.
    (void)EmptyCmd->addDep(ConnectCmd->getEvent(), ToCleanUp);
    (void)ConnectCmd->addDep(DepEvent, ToCleanUp);

    // Depend Cmd on empty command
    // Dismiss the result here as it's not a connection now,
    // 'cause EmptyCmd is host one
    (void)Cmd->addDep(EmptyCmd->getEvent(), ToCleanUp);
    // Added by addDep in another path
    EmptyCmd->addUser(Cmd);
  }

  ConnectCmd->MEmptyCmd = EmptyCmd;

  return ConnectCmd;
}

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
