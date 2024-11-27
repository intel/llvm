//===-- graph_builder.cpp - SYCL Graph Builder ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "detail/config.hpp"
#include <detail/context_impl.hpp>
#include <detail/event_impl.hpp>
#include <detail/graph_impl.hpp>
#include <detail/memory_manager.hpp>
#include <detail/queue_impl.hpp>
#include <detail/scheduler/scheduler.hpp>
#include <detail/sycl_mem_obj_t.hpp>
#include <sstream>
#include <sycl/access/access.hpp>
#include <sycl/exception.hpp>
#include <sycl/feature_test.hpp>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <queue>
#include <set>
#include <vector>

namespace sycl {
inline namespace _V1 {
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

/// Checks if current requirement is requirement for sub buffer.
static bool IsSuitableSubReq(const Requirement *Req) {
  return Req->MIsSubBuffer;
}

static bool isOnSameContext(const ContextImplPtr Context,
                            const QueueImplPtr &Queue) {
  // Covers case for host usage (nullptr == nullptr) and existing device
  // contexts comparison.
  return Context == queue_impl::getContext(Queue);
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

MemObjRecord *
Scheduler::GraphBuilder::getOrInsertMemObjRecord(const QueueImplPtr &Queue,
                                                 const Requirement *Req) {
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
        if (Dependency->readyForCleanup())
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
    std::vector<Command *> ToEnqueue;
    getOrCreateAllocaForReq(MemObject->MRecord.get(), Req, InteropQueuePtr,
                            ToEnqueue);
    assert(ToEnqueue.empty() && "Creation of the first alloca for a record "
                                "shouldn't lead to any enqueuing (no linked "
                                "alloca or exceeding the leaf limit).");
  } else
    MemObject->MRecord.reset(new MemObjRecord{queue_impl::getContext(Queue),
                                              LeafLimit, AllocateDependency});

  MMemObjs.push_back(MemObject);
  return MemObject->MRecord.get();
}

void Scheduler::GraphBuilder::updateLeaves(const std::set<Command *> &Cmds,
                                           MemObjRecord *Record,
                                           access::mode AccessMode,
                                           const MapOfDependentCmds &DependentCmdsOfNewCmd,
                                           const QueueImplPtr &Queue,
                                           std::vector<Command *> &ToCleanUp) {

  const bool ReadOnlyReq = AccessMode == access::mode::read;
  for (Command *Cmd : Cmds) {
    if (! ReadOnlyReq) {
      bool WasLeaf = Cmd->MLeafCounter > 0;
      Cmd->MLeafCounter -= Record->MReadLeaves.remove(Cmd);
      Cmd->MLeafCounter -= Record->MWriteLeaves.remove(Cmd);
      if (WasLeaf && Cmd->readyForCleanup()) {
        ToCleanUp.push_back(Cmd);
      }
    }

    detectDuplicates(Cmd, DependentCmdsOfNewCmd, ToCleanUp);

    // For in-order queue, we may cleanup all dependent command from our queue
    if (Queue && Queue->isInOrder() && Cmd->getQueue() == Queue &&
        Cmd->isCleanupSubject())
      commandToCleanup(Cmd, ToCleanUp);
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

void Scheduler::GraphBuilder::detectDuplicates(
    Command *DepCommand, const MapOfDependentCmds &DependentCmdsOfNewCmd,
    std::vector<Command *> &ToCleanUp) {
  if (!DepCommand->isCleanupSubject())
    return;
  // any dependence of DepCommand already covered by NewCmd
  bool Duplicate = std::all_of(DepCommand->MDeps.begin(), DepCommand->MDeps.end(),
    [&DependentCmdsOfNewCmd](const DepDesc &DepOfDep) {
        return DependentCmdsOfNewCmd.isMemObjExist(
          std::make_pair(DepOfDep.MDepRequirement->MSYCLMemObj, DepOfDep.MDepRequirement->MAccessMode));
    });
  if (Duplicate) {
    commandToCleanup(DepCommand, ToCleanUp);
  }
}

void Scheduler::GraphBuilder::commandToCleanup(
    Command *DepCommand, std::vector<Command *> &ToCleanUp) {
  for (const DepDesc &DepOfDep : DepCommand->MDeps) {
          MemObjRecord *Record = getMemObjRecord(DepOfDep.MDepRequirement->MSYCLMemObj);
          DepCommand->MLeafCounter -= Record->MReadLeaves.remove(DepCommand);
          DepCommand->MLeafCounter -= Record->MWriteLeaves.remove(DepCommand);
  }
  assert(!DepCommand->MLeafCounter && "A command before cleanup must have no leaves.");
  ToCleanUp.push_back(DepCommand);
}

UpdateHostRequirementCommand *Scheduler::GraphBuilder::insertUpdateHostReqCmd(
    MemObjRecord *Record, Requirement *Req, const QueueImplPtr &Queue,
    std::vector<Command *> &ToEnqueue) {
  auto Context = queue_impl::getContext(Queue);
  AllocaCommandBase *AllocaCmd = findAllocaForReq(Record, Req, Context);
  assert(AllocaCmd && "There must be alloca for requirement!");
  UpdateHostRequirementCommand *UpdateCommand =
      new UpdateHostRequirementCommand(Queue, *Req, AllocaCmd, &Req->MData);
  // Need copy of requirement because after host accessor destructor call
  // dependencies become invalid if requirement is stored by pointer.
  const Requirement *StoredReq = UpdateCommand->getRequirement();

  std::set<Command *> Deps = findDepsForReq(Record, Req, Context);
  std::vector<Command *> ToCleanUp;
  for (Command *Dep : Deps) {
    Command *ConnCmd =
        UpdateCommand->addDep(DepDesc{Dep, StoredReq, AllocaCmd}, ToCleanUp);
    if (ConnCmd)
      ToEnqueue.push_back(ConnCmd);
  }
  const MapOfDependentCmds DependentCmdsOfNewCmd(UpdateCommand->MDeps);
  updateLeaves(Deps, Record, Req->MAccessMode,
    DependentCmdsOfNewCmd, Queue, ToCleanUp);
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

  if (!AllocaCmdSrc->getQueue()) {
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
    throw exception(make_error_code(errc::memory_allocation),
                    "Out of host memory");

  auto Context = queue_impl::getContext(Queue);
  std::set<Command *> Deps = findDepsForReq(Record, Req, Context);
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
      bool Res = isOnSameContext(Record->MCurContext, AllocaCmd->getQueue()) &&
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
    throw exception(make_error_code(errc::runtime),
                    "Cannot find buffer allocation");
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
      Record->MCurContext = Context;
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
  const MapOfDependentCmds DependentCmdsOfNewCmd(NewCmd->MDeps);
  updateLeaves(Deps, Record, access::mode::read_write, DependentCmdsOfNewCmd, Queue, ToCleanUp);
  addNodeToLeaves(Record, NewCmd, access::mode::read_write, ToEnqueue);
  for (Command *Cmd : ToCleanUp)
    cleanupCommand(Cmd);
  Record->MCurContext = Context;
  return NewCmd;
}

Command *Scheduler::GraphBuilder::remapMemoryObject(
    MemObjRecord *Record, Requirement *Req, AllocaCommandBase *HostAllocaCmd,
    std::vector<Command *> &ToEnqueue) {
  assert(!HostAllocaCmd->getQueue() && "Host alloca command expected");
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

  const MapOfDependentCmds DependentCmdsOfNewCmd(UnMapCmd->MDeps);
  updateLeaves(Deps, Record, access::mode::read_write,
    DependentCmdsOfNewCmd, LinkedAllocaCmd->getQueue(), ToCleanUp);
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
  SYCLMemObjI *MemObj = Req->MSYCLMemObj;
  MemObjRecord *Record = getMemObjRecord(MemObj);
  if (Record && MPrintOptionsArray[BeforeAddCopyBack])
    printGraphAsDot("before_addCopyBack");

  // Do nothing if there were no or only read operations with the memory object.
  if (nullptr == Record || !Record->MMemModified)
    return nullptr;

  std::set<Command *> Deps = findDepsForReq(Record, Req, nullptr);
  AllocaCommandBase *SrcAllocaCmd =
      findAllocaForReq(Record, Req, Record->MCurContext);

  auto MemCpyCmdUniquePtr = std::make_unique<MemCpyCommandHost>(
      *SrcAllocaCmd->getRequirement(), SrcAllocaCmd, *Req, &Req->MData,
      SrcAllocaCmd->getQueue(), nullptr);

  if (!MemCpyCmdUniquePtr)
    throw exception(make_error_code(errc::memory_allocation),
                    "Out of host memory");

  MemCpyCommandHost *MemCpyCmd = MemCpyCmdUniquePtr.release();

  std::vector<Command *> ToCleanUp;
  for (Command *Dep : Deps) {
    Command *ConnCmd = MemCpyCmd->addDep(
        DepDesc{Dep, MemCpyCmd->getRequirement(), SrcAllocaCmd}, ToCleanUp);
    if (ConnCmd)
      ToEnqueue.push_back(ConnCmd);
  }

  const MapOfDependentCmds DependentCmdsOfNewCmd(MemCpyCmd->MDeps);
  updateLeaves(Deps, Record, Req->MAccessMode,
    DependentCmdsOfNewCmd, SrcAllocaCmd->getQueue(), ToCleanUp);
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

  if (Req->MAccessMode != sycl::access_mode::read) {
    auto SYCLMemObj = static_cast<detail::SYCLMemObjT *>(Req->MSYCLMemObj);
    SYCLMemObj->handleWriteAccessorCreation();
  }
  // Host accessor is not attached to any queue so no QueueImplPtr object to be
  // sent to getOrInsertMemObjRecord.
  MemObjRecord *Record = getOrInsertMemObjRecord(nullptr, Req);
  if (MPrintOptionsArray[BeforeAddHostAcc])
    printGraphAsDot("before_addHostAccessor");
  markModifiedIfWrite(Record, Req);

  AllocaCommandBase *HostAllocaCmd =
      getOrCreateAllocaForReq(Record, Req, nullptr, ToEnqueue);

  if (isOnSameContext(Record->MCurContext, HostAllocaCmd->getQueue())) {
    if (!isAccessModeAllowed(Req->MAccessMode, Record->MHostAccess)) {
      remapMemoryObject(Record, Req,
                        Req->MIsSubBuffer ? (static_cast<AllocaSubBufCommand *>(
                                                 HostAllocaCmd))
                                                ->getParentAlloca()
                                          : HostAllocaCmd,
                        ToEnqueue);
    }
  } else
    insertMemoryMove(Record, Req, nullptr, ToEnqueue);

  Command *UpdateHostAccCmd =
      insertUpdateHostReqCmd(Record, Req, nullptr, ToEnqueue);

  // Need empty command to be blocked until host accessor is destructed
  EmptyCommand *EmptyCmd = addEmptyCmd(
      UpdateHostAccCmd, {Req}, Command::BlockReason::HostAccessor, ToEnqueue);

  Req->MBlockedCmd = EmptyCmd;

  if (MPrintOptionsArray[AfterAddHostAcc])
    printGraphAsDot("after_addHostAccessor");

  return UpdateHostAccCmd;
}

Command *Scheduler::GraphBuilder::addCGUpdateHost(
    std::unique_ptr<detail::CG> CommandGroup,
    std::vector<Command *> &ToEnqueue) {

  auto UpdateHost = static_cast<CGUpdateHost *>(CommandGroup.get());
  Requirement *Req = UpdateHost->getReqToUpdate();

  MemObjRecord *Record = getOrInsertMemObjRecord(nullptr, Req);
  return insertMemoryMove(Record, Req, nullptr, ToEnqueue);
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
      if (Dep.MDepCommand) {
        auto DepQueue = Dep.MDepCommand->getQueue();
        CanBypassDep &= isOnSameContext(Context, DepQueue);
      }

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
AllocaCommandBase *Scheduler::GraphBuilder::findAllocaForReq(
    MemObjRecord *Record, const Requirement *Req, const ContextImplPtr &Context,
    bool AllowConst) {
  auto IsSuitableAlloca = [&Context, Req,
                           AllowConst](AllocaCommandBase *AllocaCmd) {
    bool Res = isOnSameContext(Context, AllocaCmd->getQueue());
    if (IsSuitableSubReq(Req)) {
      const Requirement *TmpReq = AllocaCmd->getRequirement();
      Res &= AllocaCmd->getType() == Command::CommandType::ALLOCA_SUB_BUF;
      Res &= TmpReq->MOffsetInBytes == Req->MOffsetInBytes;
      Res &= TmpReq->MAccessRange == Req->MAccessRange;
      Res &= AllowConst || !AllocaCmd->MIsConst;
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
      return Ctx == nullptr;
    if (std::strcmp(HUMConfig, "1") == 0)
      return true;
  }
  // host task & host accessor is covered with no device context but provide
  // required support.
  if (Ctx == nullptr)
    return true;

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
    MemObjRecord *Record, const Requirement *Req, const QueueImplPtr &Queue,
    std::vector<Command *> &ToEnqueue) {
  auto Context = queue_impl::getContext(Queue);
  AllocaCommandBase *AllocaCmd =
      findAllocaForReq(Record, Req, Context, /*AllowConst=*/false);

  if (!AllocaCmd) {
    std::vector<Command *> ToCleanUp;
    if (IsSuitableSubReq(Req)) {
      // Get parent requirement. It's hard to get right parents' range
      // so full parent requirement has range represented in bytes
      range<3> ParentRange{Req->MSYCLMemObj->getSizeInBytes(), 1, 1};
      Requirement ParentRequirement(
          /*Offset*/ {0, 0, 0}, ParentRange, ParentRange,
          access::mode::read_write, Req->MSYCLMemObj, /*Dims*/ 1,
          /*Working with bytes*/ sizeof(char), /*offset*/ size_t(0));

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
      const bool HostUnifiedMemory = checkHostUnifiedMemory(Context);
      SYCLMemObjI *MemObj = Req->MSYCLMemObj;
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
            AllocaCommand *HostAllocaCmd = new AllocaCommand(
                nullptr, FullReq, true /* InitFromUserData */,
                nullptr /* LinkedAllocaCmd */,
                MemObj->isHostPointerReadOnly() /* IsConst */);
            Record->MAllocaCommands.push_back(HostAllocaCmd);
            Record->MWriteLeaves.push_back(HostAllocaCmd, ToEnqueue);
            ++(HostAllocaCmd->MLeafCounter);
            Record->MCurContext = nullptr;
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
          if ((Context == nullptr) != (Record->MCurContext == nullptr)) {
            // Linked commands assume that the host allocation is reused by the
            // unified runtime and that can lead to unnecessary copy overhead on
            // devices that do not support host unified memory. Do not link the
            // allocations in this case.
            // However, if the user explicitly requests use of pinned host
            // memory, map/unmap operations are expected to work faster than
            // read/write from/to an artbitrary host pointer. Link such commands
            // regardless of host unified memory support.
            bool PinnedHostMemory = MemObj->usesPinnedHostMemory();

            bool HostUnifiedMemoryOnNonHostDevice =
                Queue == nullptr ? checkHostUnifiedMemory(Record->MCurContext)
                                 : HostUnifiedMemory;
            if (PinnedHostMemory || HostUnifiedMemoryOnNonHostDevice) {
              AllocaCommandBase *LinkedAllocaCmdCand = findAllocaForReq(
                  Record, Req, Record->MCurContext, /*AllowConst=*/false);

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
        if (Queue == nullptr) {
          AllocaCmd->MIsActive = false;
        } else {
          LinkedAllocaCmd->MIsActive = false;
          Record->MCurContext = Context;

          std::set<Command *> Deps = findDepsForReq(Record, Req, Context);
          for (Command *Dep : Deps) {
            Command *ConnCmd = AllocaCmd->addDep(
                DepDesc{Dep, Req, LinkedAllocaCmd}, ToCleanUp);
            if (ConnCmd)
              ToEnqueue.push_back(ConnCmd);
          }
          const MapOfDependentCmds DependentCmdsOfNewCmd(AllocaCmd->MDeps);
          updateLeaves(Deps, Record, Req->MAccessMode,
            DependentCmdsOfNewCmd, Queue, ToCleanUp);
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

EmptyCommand *Scheduler::GraphBuilder::addEmptyCmd(
    Command *Cmd, const std::vector<Requirement *> &Reqs,
    Command::BlockReason Reason, std::vector<Command *> &ToEnqueue) {
  EmptyCommand *EmptyCmd = new EmptyCommand();

  if (!EmptyCmd)
    throw exception(make_error_code(errc::memory_allocation),
                    "Out of host memory");

  EmptyCmd->MIsBlockable = true;
  EmptyCmd->MEnqueueStatus = EnqueueResultT::SyclEnqueueBlocked;
  EmptyCmd->MBlockReason = Reason;

  for (Requirement *Req : Reqs) {
    MemObjRecord *Record = getOrInsertMemObjRecord(nullptr, Req);
    AllocaCommandBase *AllocaCmd =
        getOrCreateAllocaForReq(Record, Req, nullptr, ToEnqueue);
    EmptyCmd->addRequirement(Cmd, AllocaCmd, Req);
  }
  // addRequirement above call addDep that already will add EmptyCmd as user for
  // Cmd no Reqs size check here so assume it is possible to have no Reqs passed
  if (!Reqs.size())
    Cmd->addUser(EmptyCmd);

  const std::vector<DepDesc> &Deps = Cmd->MDeps;
  std::vector<Command *> ToCleanUp;
  const MapOfDependentCmds DependentCmdsOfNewCmd(EmptyCmd->MDeps);
  for (const DepDesc &Dep : Deps) {
    const Requirement *Req = Dep.MDepRequirement;
    MemObjRecord *Record = getMemObjRecord(Req->MSYCLMemObj);

    updateLeaves({Cmd}, Record, Req->MAccessMode, DependentCmdsOfNewCmd, nullptr, ToCleanUp);
    addNodeToLeaves(Record, EmptyCmd, Req->MAccessMode, ToEnqueue);
  }
  for (Command *Cmd : ToCleanUp)
    cleanupCommand(Cmd);

  return EmptyCmd;
}

static bool isInteropHostTask(ExecCGCommand *Cmd) {
  if (Cmd->getCG().getType() != CGType::CodeplayHostTask)
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

Command *Scheduler::GraphBuilder::addCG(
    std::unique_ptr<detail::CG> CommandGroup, const QueueImplPtr &Queue,
    std::vector<Command *> &ToEnqueue, bool EventNeeded,
    ur_exp_command_buffer_handle_t CommandBuffer,
    const std::vector<ur_exp_command_buffer_sync_point_t> &Dependencies) {
  std::vector<Requirement *> &Reqs = CommandGroup->getRequirements();
  std::vector<detail::EventImplPtr> &Events = CommandGroup->getEvents();

  auto NewCmd = std::make_unique<ExecCGCommand>(std::move(CommandGroup), Queue,
                                                EventNeeded, CommandBuffer,
                                                std::move(Dependencies));

  if (!NewCmd)
    throw exception(make_error_code(errc::memory_allocation),
                    "Out of host memory");

  bool isInteropTask = isInteropHostTask(NewCmd.get());

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
          isInteropTask
              ? static_cast<detail::CGHostTask &>(NewCmd->getCG()).MQueue
              : Queue;

      Record = getOrInsertMemObjRecord(QueueForAlloca, Req);
      markModifiedIfWrite(Record, Req);

      AllocaCmd =
          getOrCreateAllocaForReq(Record, Req, QueueForAlloca, ToEnqueue);

      isSameCtx = isOnSameContext(Record->MCurContext, QueueForAlloca);
    }

    // If there is alloca command we need to check if the latest memory is in
    // required context.
    if (isSameCtx) {
      // If the memory is already in the required host context, check if the
      // required access mode is valid, remap if not.
      if (!Record->MCurContext &&
          !isAccessModeAllowed(Req->MAccessMode, Record->MHostAccess)) {
        remapMemoryObject(Record, Req,
                          Req->MIsSubBuffer
                              ? (static_cast<AllocaSubBufCommand *>(AllocaCmd))
                                    ->getParentAlloca()
                              : AllocaCmd,
                          ToEnqueue);
      }
    } else {
      // Cannot directly copy memory from OpenCL device to OpenCL device -
      // create two copies: device->host and host->device.
      bool NeedMemMoveToHost = false;
      auto MemMoveTargetQueue = Queue;

      if (isInteropTask) {
        const detail::CGHostTask &HT =
            static_cast<detail::CGHostTask &>(NewCmd->getCG());

        if (!isOnSameContext(Record->MCurContext, HT.MQueue)) {
          NeedMemMoveToHost = true;
          MemMoveTargetQueue = HT.MQueue;
        }
      } else if (Queue && Record->MCurContext)
        NeedMemMoveToHost = true;

      if (NeedMemMoveToHost)
        insertMemoryMove(Record, Req, nullptr, ToEnqueue);
      insertMemoryMove(Record, Req, MemMoveTargetQueue, ToEnqueue);
    }

    std::set<Command *> Deps =
        findDepsForReq(Record, Req, queue_impl::getContext(Queue));

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

  const MapOfDependentCmds DependentCmdsOfNewCmd(NewCmd->MDeps);

  std::vector<DepDesc> Deps = NewCmd->MDeps;
  for (DepDesc &Dep : Deps) {
    const Requirement *Req = Dep.MDepRequirement;
    MemObjRecord *Record = getMemObjRecord(Req->MSYCLMemObj);
    updateLeaves({Dep.MDepCommand}, Record, Req->MAccessMode,
      DependentCmdsOfNewCmd, Queue, ToCleanUp);
    addNodeToLeaves(Record, NewCmd.get(), Req->MAccessMode, ToEnqueue);
  }

  // Register all the events as dependencies
  for (const detail::EventImplPtr &e : Events) {
    if (Command *ConnCmd = NewCmd->addDep(e, ToCleanUp))
      ToEnqueue.push_back(ConnCmd);
    // If NewCmd depends on another command, and all dependences of that command
    // already covered by NewCmd, can move the cmd in ToCleanUp
    if (auto *Cmd = static_cast<Command *>(e->getCommand()))
      detectDuplicates(Cmd, DependentCmdsOfNewCmd, ToCleanUp);
  }

  if (MPrintOptionsArray[AfterAddCG])
    printGraphAsDot("after_addCG");

  for (Command *Cmd : ToCleanUp) {
    cleanupCommand(Cmd);
  }

  return NewCmd.release();
}

void Scheduler::GraphBuilder::decrementLeafCountersForRecord(
    MemObjRecord *Record) {
  for (Command *Cmd : Record->MReadLeaves) {
    --(Cmd->MLeafCounter);
    if (Cmd->readyForCleanup())
      cleanupCommand(Cmd);
  }
  for (Command *Cmd : Record->MWriteLeaves) {
    --(Cmd->MLeafCounter);
    if (Cmd->readyForCleanup())
      cleanupCommand(Cmd);
  }
}

void Scheduler::GraphBuilder::cleanupCommandsForRecord(MemObjRecord *Record) {
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
      if (!Cmd->MMarkedForCleanup)
        Cmd->MMarks.MToBeDeleted = true;
    }
  }

  handleVisitedNodes(MVisitedCmds);
}

void Scheduler::GraphBuilder::cleanupCommand(
    Command *Cmd, [[maybe_unused]] bool AllowUnsubmitted) {
  if (SYCLConfig<SYCL_DISABLE_POST_ENQUEUE_CLEANUP>::get()) {
    static bool DeprWarningPrinted = false;
    if (!DeprWarningPrinted) {
      std::cerr << "WARNING: The enviroment variable "
                   "SYCL_DISABLE_POST_ENQUEUE_CLEANUP is deprecated. Please "
                   "use SYCL_DISABLE_EXECUTION_GRAPH_CLEANUP instead.\n";
      DeprWarningPrinted = true;
    }
    return;
  }
  if (SYCLConfig<SYCL_DISABLE_EXECUTION_GRAPH_CLEANUP>::get())
    return;

  assert(Cmd->MLeafCounter == 0 &&
         (Cmd->isSuccessfullyEnqueued() || AllowUnsubmitted));
  Command::CommandType CmdT = Cmd->getType();

  assert(CmdT != Command::ALLOCA && CmdT != Command::ALLOCA_SUB_BUF);
  assert(CmdT != Command::RELEASE);
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
    Command *const Cmd, const EventImplPtr &DepEvent, const DepDesc &Dep,
    std::vector<Command *> &ToCleanUp) {
  assert(Cmd->getWorkerContext() != DepEvent->getContextImpl());

  // construct Host Task type command manually and make it depend on DepEvent
  ExecCGCommand *ConnectCmd = nullptr;

  try {
    std::shared_ptr<detail::HostTask> HT(new detail::HostTask);
    std::unique_ptr<detail::CG> ConnectCG(new detail::CGHostTask(
        std::move(HT), /* Queue = */ Cmd->getQueue(), /* Context = */ {},
        /* Args = */ {},
        detail::CG::StorageInitHelper(
            /* ArgsStorage = */ {}, /* AccStorage = */ {},
            /* SharedPtrStorage = */ {}, /* Requirements = */ {},
            /* DepEvents = */ {DepEvent}),
        CGType::CodeplayHostTask,
        /* Payload */ {}));
    ConnectCmd = new ExecCGCommand(std::move(ConnectCG), nullptr,
                                   /*EventNeeded=*/true);
  } catch (const std::bad_alloc &) {
    throw exception(make_error_code(errc::memory_allocation),
                    "Out of host memory");
  }

  if (Dep.MDepRequirement) {
    // make ConnectCmd depend on requirement
    // Dismiss the result here as it's not a connection now,
    // 'cause ConnectCmd is host one
    (void)ConnectCmd->addDep(Dep, ToCleanUp);
    assert(reinterpret_cast<Command *>(DepEvent->getCommand()) ==
           Dep.MDepCommand);
    // add user to Dep.MDepCommand is already performed beyond this if branch
    {
      DepDesc DepOnConnect = Dep;
      DepOnConnect.MDepCommand = ConnectCmd;

      // Dismiss the result here as it's not a connection now,
      // 'cause ConnectCmd is host one
      std::ignore = Cmd->addDep(DepOnConnect, ToCleanUp);
    }
  } else {
    // It is required condition in another a path and addUser will be set in
    // addDep
    if (Command *DepCmd = reinterpret_cast<Command *>(DepEvent->getCommand()))
      DepCmd->addUser(ConnectCmd);

    std::ignore = ConnectCmd->addDep(DepEvent, ToCleanUp);

    std::ignore = Cmd->addDep(ConnectCmd->getEvent(), ToCleanUp);

    ConnectCmd->addUser(Cmd);
  }

  return ConnectCmd;
}

Command *Scheduler::GraphBuilder::addCommandGraphUpdate(
    ext::oneapi::experimental::detail::exec_graph_impl *Graph,
    std::vector<std::shared_ptr<ext::oneapi::experimental::detail::node_impl>>
        Nodes,
    const QueueImplPtr &Queue, std::vector<Requirement *> Requirements,
    std::vector<detail::EventImplPtr> &Events,
    std::vector<Command *> &ToEnqueue) {
  auto NewCmd =
      std::make_unique<UpdateCommandBufferCommand>(Queue, Graph, Nodes);
  // If there are multiple requirements for the same memory object, its
  // AllocaCommand creation will be dependent on the access mode of the first
  // requirement. Combine these access modes to take all of them into account.
  combineAccessModesOfReqs(Requirements);
  std::vector<Command *> ToCleanUp;
  for (Requirement *Req : Requirements) {
    MemObjRecord *Record = nullptr;
    AllocaCommandBase *AllocaCmd = nullptr;

    bool isSameCtx = false;

    {

      Record = getOrInsertMemObjRecord(Queue, Req);
      markModifiedIfWrite(Record, Req);

      AllocaCmd = getOrCreateAllocaForReq(Record, Req, Queue, ToEnqueue);

      isSameCtx = isOnSameContext(Record->MCurContext, Queue);
    }

    if (!isSameCtx) {
      // Cannot directly copy memory from OpenCL device to OpenCL device -
      // create two copies: device->host and host->device.
      bool NeedMemMoveToHost = false;
      auto MemMoveTargetQueue = Queue;

      if (Queue && Record->MCurContext)
        NeedMemMoveToHost = true;

      if (NeedMemMoveToHost)
        insertMemoryMove(Record, Req, nullptr, ToEnqueue);
      insertMemoryMove(Record, Req, MemMoveTargetQueue, ToEnqueue);
    }
    std::set<Command *> Deps =
        findDepsForReq(Record, Req, queue_impl::getContext(Queue));

    for (Command *Dep : Deps) {
      if (Dep != NewCmd.get()) {
        Command *ConnCmd =
            NewCmd->addDep(DepDesc{Dep, Req, AllocaCmd}, ToCleanUp);
        if (ConnCmd)
          ToEnqueue.push_back(ConnCmd);
      }
    }
  }

  const MapOfDependentCmds DependentCmdsOfNewCmd(NewCmd->MDeps);
  // Set new command as user for dependencies and update leaves.
  // Node dependencies can be modified further when adding the node to leaves,
  // iterate over their copy.
  // FIXME employ a reference here to eliminate copying of a vector
  std::vector<DepDesc> Deps = NewCmd->MDeps;
  for (DepDesc &Dep : Deps) {
    const Requirement *Req = Dep.MDepRequirement;
    MemObjRecord *Record = getMemObjRecord(Req->MSYCLMemObj);
    updateLeaves({Dep.MDepCommand}, Record, Req->MAccessMode,
      DependentCmdsOfNewCmd, Queue, ToCleanUp);
    addNodeToLeaves(Record, NewCmd.get(), Req->MAccessMode, ToEnqueue);
  }

  // Register all the events as dependencies
  for (detail::EventImplPtr e : Events) {
    if (e->getCommand() &&
        e->getCommand() == static_cast<Command *>(NewCmd.get())) {
      continue;
    }
    if (Command *ConnCmd = NewCmd->addDep(e, ToCleanUp))
      ToEnqueue.push_back(ConnCmd);

    // If NewCmd depends on another command, and all dependences of that command
    // already covered by NewCmd, can move the cmd in ToCleanUp
    if (auto *Cmd = static_cast<Command *>(e->getCommand()))
      detectDuplicates(Cmd, DependentCmdsOfNewCmd, ToCleanUp);
  }

  if (MPrintOptionsArray[AfterAddCG])
    printGraphAsDot("after_addCG");

  for (Command *Cmd : ToCleanUp) {
    cleanupCommand(Cmd);
  }

  return NewCmd.release();
}

} // namespace detail
} // namespace _V1
} // namespace sycl
