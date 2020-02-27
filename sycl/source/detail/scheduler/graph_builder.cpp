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
#include <fstream>
#include <map>
#include <memory>
#include <queue>
#include <set>
#include <vector>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

// The function checks whether two requirements overlaps or not. This
// information can be used to prove that executing two kernels that
// work on different parts of the memory object in parallel is legal.
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

// The function checks if current requirement is requirement for sub buffer
static bool IsSuitableSubReq(const Requirement *Req) {
  return Req->MIsSubBuffer;
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

static void printDotRecursive(std::fstream &Stream,
                              std::set<Command *> &Visited, Command *Cmd) {
  if (!Visited.insert(Cmd).second)
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

  std::set<Command *> Visited;

  for (SYCLMemObjI *MemObject : MMemObjs)
    for (Command *AllocaCmd : MemObject->MRecord->MAllocaCommands)
      printDotRecursive(Stream, Visited, AllocaCmd);

  Stream << "}" << std::endl;
}

// Returns record for the memory objects passed, nullptr if doesn't exist.
MemObjRecord *Scheduler::GraphBuilder::getMemObjRecord(SYCLMemObjI *MemObject) {
  return MemObject->MRecord.get();
}

// Returns record for the memory object requirement refers to, if doesn't
// exist, creates new one.
MemObjRecord *
Scheduler::GraphBuilder::getOrInsertMemObjRecord(const QueueImplPtr &Queue,
                                                 Requirement *Req) {
  SYCLMemObjI *MemObject = Req->MSYCLMemObj;
  MemObjRecord *Record = getMemObjRecord(MemObject);

  if (nullptr != Record)
    return Record;

  const size_t LeafLimit = 8;
  MemObject->MRecord.reset(
      new MemObjRecord{Queue->getContextImplPtr(), LeafLimit});

  MMemObjs.push_back(MemObject);
  return MemObject->MRecord.get();
}

// Helper function which removes all values in Cmds from Leaves
void Scheduler::GraphBuilder::updateLeaves(const std::set<Command *> &Cmds,
                                           MemObjRecord *Record,
                                           access::mode AccessMode) {

  const bool ReadOnlyReq = AccessMode == access::mode::read;
  if (ReadOnlyReq)
    return;

  for (Command *Cmd : Cmds) {
    auto NewEnd = std::remove(Record->MReadLeaves.begin(),
                              Record->MReadLeaves.end(), Cmd);
    Cmd->MLeafCounter -= std::distance(NewEnd, Record->MReadLeaves.end());
    Record->MReadLeaves.erase(NewEnd, Record->MReadLeaves.end());

    NewEnd = std::remove(Record->MWriteLeaves.begin(),
                         Record->MWriteLeaves.end(), Cmd);
    Cmd->MLeafCounter -= std::distance(NewEnd, Record->MWriteLeaves.end());
    Record->MWriteLeaves.erase(NewEnd, Record->MWriteLeaves.end());
  }
}

void Scheduler::GraphBuilder::addNodeToLeaves(MemObjRecord *Record,
                                              Command *Cmd,
                                              access::mode AccessMode) {
  CircularBuffer<Command *> &Leaves{AccessMode == access::mode::read
                                        ? Record->MReadLeaves
                                        : Record->MWriteLeaves};
  if (Leaves.full()) {
    Command *OldLeaf = Leaves.front();
    // TODO this is a workaround for duplicate leaves, remove once fixed
    if (OldLeaf == Cmd)
      return;
    // Add the old leaf as a dependency for the new one by duplicating one of
    // the requirements for the current record
    DepDesc Dep = findDepForRecord(Cmd, Record);
    Dep.MDepCommand = OldLeaf;
    Cmd->addDep(Dep);
    OldLeaf->addUser(Cmd);
    --(OldLeaf->MLeafCounter);
  }
  Leaves.push_back(Cmd);
  ++(Cmd->MLeafCounter);
}

UpdateHostRequirementCommand *Scheduler::GraphBuilder::insertUpdateHostReqCmd(
    MemObjRecord *Record, Requirement *Req, const QueueImplPtr &Queue) {
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
  for (Command *Dep : Deps) {
    UpdateCommand->addDep(DepDesc{Dep, StoredReq, AllocaCmd});
    Dep->addUser(UpdateCommand);
  }
  updateLeaves(Deps, Record, Req->MAccessMode);
  addNodeToLeaves(Record, UpdateCommand, Req->MAccessMode);
  return UpdateCommand;
}

// Takes linked alloca commands. Makes AllocaCmdDst command active using map
// or unmap operation.
static Command *insertMapUnmapForLinkedCmds(AllocaCommandBase *AllocaCmdSrc,
                                            AllocaCommandBase *AllocaCmdDst) {
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

  MapMemObject *MapCmd =
      new MapMemObject(AllocaCmdSrc, *AllocaCmdSrc->getRequirement(),
                       &AllocaCmdDst->MMemAllocation, AllocaCmdSrc->getQueue());

  std::swap(AllocaCmdSrc->MIsActive, AllocaCmdDst->MIsActive);

  return MapCmd;
}

Command *Scheduler::GraphBuilder::insertMemoryMove(MemObjRecord *Record,
                                                   Requirement *Req,
                                                   const QueueImplPtr &Queue) {

  AllocaCommandBase *AllocaCmdDst = getOrCreateAllocaForReq(Record, Req, Queue);
  if (!AllocaCmdDst)
    throw runtime_error("Out of host memory");

  std::set<Command *> Deps =
      findDepsForReq(Record, Req, Queue->getContextImplPtr());
  Deps.insert(AllocaCmdDst);
  // Get parent allocation of sub buffer to perform full copy of whole buffer
  if (IsSuitableSubReq(Req)) {
    if (AllocaCmdDst->getType() == Command::CommandType::ALLOCA_SUB_BUF)
      AllocaCmdDst =
          static_cast<AllocaSubBufCommand *>(AllocaCmdDst)->getParentAlloca();
    else
      assert(
          !"Inappropriate alloca command. AllocaSubBufCommand was expected.");
  }

  AllocaCommandBase *AllocaCmdSrc =
      findAllocaForReq(Record, Req, Record->MCurContext);
  if (!AllocaCmdSrc && IsSuitableSubReq(Req)) {
    // Since no alloca command for the sub buffer requirement was found in the
    // current context, need to find a parent alloca command for it (it must be
    // there)
    auto IsSuitableAlloca = [Record, Req](AllocaCommandBase *AllocaCmd) {
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
    throw runtime_error("Cannot find buffer allocation");
  // Get parent allocation of sub buffer to perform full copy of whole buffer
  if (IsSuitableSubReq(Req)) {
    if (AllocaCmdSrc->getType() == Command::CommandType::ALLOCA_SUB_BUF)
      AllocaCmdSrc =
          static_cast<AllocaSubBufCommand *>(AllocaCmdSrc)->getParentAlloca();
    else if (AllocaCmdSrc->getSYCLMemObj() != Req->MSYCLMemObj)
      assert(!"Inappropriate alloca command.");
  }

  Command *NewCmd = nullptr;

  if (AllocaCmdSrc->MLinkedAllocaCmd == AllocaCmdDst) {
    NewCmd = insertMapUnmapForLinkedCmds(AllocaCmdSrc, AllocaCmdDst);
  } else {

    // Full copy of buffer is needed to avoid loss of data that may be caused
    // by copying specific range from host to device and backwards.
    NewCmd =
        new MemCpyCommand(*AllocaCmdSrc->getRequirement(), AllocaCmdSrc,
                          *AllocaCmdDst->getRequirement(), AllocaCmdDst,
                          AllocaCmdSrc->getQueue(), AllocaCmdDst->getQueue());
  }

  for (Command *Dep : Deps) {
    NewCmd->addDep(DepDesc{Dep, NewCmd->getRequirement(), AllocaCmdDst});
    Dep->addUser(NewCmd);
  }
  updateLeaves(Deps, Record, access::mode::read_write);
  addNodeToLeaves(Record, NewCmd, access::mode::read_write);
  Record->MCurContext = Queue->getContextImplPtr();
  return NewCmd;
}

// The function adds copy operation of the up to date'st memory to the memory
// pointed by Req.
Command *Scheduler::GraphBuilder::addCopyBack(Requirement *Req) {

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

  std::unique_ptr<MemCpyCommandHost> MemCpyCmdUniquePtr(new MemCpyCommandHost(
      *SrcAllocaCmd->getRequirement(), SrcAllocaCmd, *Req, &Req->MData,
      SrcAllocaCmd->getQueue(), std::move(HostQueue)));

  if (!MemCpyCmdUniquePtr)
    throw runtime_error("Out of host memory");

  MemCpyCommandHost *MemCpyCmd = MemCpyCmdUniquePtr.release();
  for (Command *Dep : Deps) {
    MemCpyCmd->addDep(DepDesc{Dep, MemCpyCmd->getRequirement(), SrcAllocaCmd});
    Dep->addUser(MemCpyCmd);
  }

  updateLeaves(Deps, Record, Req->MAccessMode);
  addNodeToLeaves(Record, MemCpyCmd, Req->MAccessMode);
  if (MPrintOptionsArray[AfterAddCopyBack])
    printGraphAsDot("after_addCopyBack");
  return MemCpyCmd;
}

// The function implements SYCL host accessor logic: host accessor
// should provide access to the buffer in user space.
Command *Scheduler::GraphBuilder::addHostAccessor(Requirement *Req,
                                                const bool destructor) {

  const QueueImplPtr &HostQueue = getInstance().getDefaultHostQueue();

  MemObjRecord *Record = getOrInsertMemObjRecord(HostQueue, Req);
  if (MPrintOptionsArray[BeforeAddHostAcc])
    printGraphAsDot("before_addHostAccessor");
  markModifiedIfWrite(Record, Req);

  AllocaCommandBase *HostAllocaCmd =
      getOrCreateAllocaForReq(Record, Req, HostQueue);

  if (!sameCtx(HostAllocaCmd->getQueue()->getContextImplPtr(),
               Record->MCurContext))
    insertMemoryMove(Record, Req, HostQueue);

  Command *UpdateHostAccCmd = insertUpdateHostReqCmd(Record, Req, HostQueue);

  // Need empty command to be blocked until host accessor is destructed
  EmptyCommand *EmptyCmd = new EmptyCommand(HostQueue, *Req);
  EmptyCmd->addDep(
      DepDesc{UpdateHostAccCmd, EmptyCmd->getRequirement(), HostAllocaCmd});
  UpdateHostAccCmd->addUser(EmptyCmd);

  EmptyCmd->MIsBlockable = true;
  EmptyCmd->MCanEnqueue = false;
  EmptyCmd->MBlockReason = "A Buffer is locked by the host accessor";

  updateLeaves({UpdateHostAccCmd}, Record, Req->MAccessMode);
  addNodeToLeaves(Record, EmptyCmd, Req->MAccessMode);

  Req->MBlockedCmd = EmptyCmd;

  if (MPrintOptionsArray[AfterAddHostAcc])
    printGraphAsDot("after_addHostAccessor");

  return UpdateHostAccCmd;
}

Command *Scheduler::GraphBuilder::addCGUpdateHost(
    std::unique_ptr<detail::CG> CommandGroup, QueueImplPtr HostQueue) {

  CGUpdateHost *UpdateHost = (CGUpdateHost *)CommandGroup.get();
  Requirement *Req = UpdateHost->getReqToUpdate();

  MemObjRecord *Record = getOrInsertMemObjRecord(HostQueue, Req);
  return insertMemoryMove(Record, Req, HostQueue);
}

// The functions finds dependencies for the requirement. It starts searching
// from list of "leaf" commands for the record and check if the examining
// command can be executed in parallel with new one with regard to the memory
// object. If can, then continue searching through dependencies of that
// command. There are several rules used:
//
// 1. New and examined commands only read -> can bypass
// 2. New and examined commands has non-overlapping requirements -> can bypass
// 3. New and examined commands has different contexts -> cannot bypass
std::set<Command *>
Scheduler::GraphBuilder::findDepsForReq(MemObjRecord *Record, Requirement *Req,
                                        const ContextImplPtr &Context) {
  std::set<Command *> RetDeps;
  std::set<Command *> Visited;
  const bool ReadOnlyReq = Req->MAccessMode == access::mode::read;

  std::vector<Command *> ToAnalyze{Record->MWriteLeaves.begin(),
                                   Record->MWriteLeaves.end()};

  if (!ReadOnlyReq)
    ToAnalyze.insert(ToAnalyze.begin(), Record->MReadLeaves.begin(),
                     Record->MReadLeaves.end());

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

      if (Visited.insert(Dep.MDepCommand).second)
        NewAnalyze.push_back(Dep.MDepCommand);
    }
    ToAnalyze.insert(ToAnalyze.end(), NewAnalyze.begin(), NewAnalyze.end());
  }
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
    MemObjRecord *Record, Requirement *Req, const ContextImplPtr &Context) {
  auto IsSuitableAlloca = [&Context, Req](AllocaCommandBase *AllocaCmd) {
    bool Res = sameCtx(AllocaCmd->getQueue()->getContextImplPtr(), Context);
    if (IsSuitableSubReq(Req)) {
      const Requirement *TmpReq = AllocaCmd->getRequirement();
      Res &= TmpReq->MOffsetInBytes == Req->MOffsetInBytes;
      Res &= TmpReq->MSYCLMemObj->getSize() == Req->MSYCLMemObj->getSize();
    }
    return Res;
  };
  const auto It = std::find_if(Record->MAllocaCommands.begin(),
                               Record->MAllocaCommands.end(), IsSuitableAlloca);
  return (Record->MAllocaCommands.end() != It) ? *It : nullptr;
}

// The function searches for the alloca command matching context and
// requirement. If none exists, new allocation command is created.
// Note, creation of new allocation command can lead to the current context
// (Record->MCurContext) change.
AllocaCommandBase *Scheduler::GraphBuilder::getOrCreateAllocaForReq(
    MemObjRecord *Record, Requirement *Req, QueueImplPtr Queue) {

  AllocaCommandBase *AllocaCmd =
      findAllocaForReq(Record, Req, Queue->getContextImplPtr());

  if (!AllocaCmd) {
    if (IsSuitableSubReq(Req)) {
      // Get parent requirement. It's hard to get right parents' range
      // so full parent requirement has range represented in bytes
      range<3> ParentRange{Req->MSYCLMemObj->getSize(), 1, 1};
      Requirement ParentRequirement(/*Offset*/ {0, 0, 0}, ParentRange,
                                    ParentRange, access::mode::read_write,
                                    Req->MSYCLMemObj, /*Dims*/ 1,
                                    /*Working with bytes*/ sizeof(char));

      auto *ParentAlloca =
          getOrCreateAllocaForReq(Record, &ParentRequirement, Queue);
      AllocaCmd = new AllocaSubBufCommand(Queue, *Req, ParentAlloca);
      updateLeaves(findDepsForReq(Record, Req, Queue->getContextImplPtr()),
                   Record, access::mode::read_write);
    } else {

      const Requirement FullReq(/*Offset*/ {0, 0, 0}, Req->MMemoryRange,
                                Req->MMemoryRange, access::mode::read_write,
                                Req->MSYCLMemObj, Req->MDims, Req->MElemSize);
      // Can reuse user data for the first allocation
      const bool InitFromUserData = Record->MAllocaCommands.empty();

      AllocaCommandBase *LinkedAllocaCmd = nullptr;
      // If it is not the first allocation, try to setup a link
      // FIXME: Temporary limitation, linked alloca commands for an image is not
      // supported because map operation is not implemented for an image.
      if (!Record->MAllocaCommands.empty() &&
          Req->MSYCLMemObj->getType() == SYCLMemObjI::MemObjType::BUFFER)
        // Current limitation is to setup link between current allocation and
        // new one. There could be situations when we could setup link with
        // "not" current allocation, but it will require memory copy.
        // Can setup link between cl and host allocations only
        if (Queue->is_host() != Record->MCurContext->is_host()) {

          AllocaCommandBase *LinkedAllocaCmdCand =
              findAllocaForReq(Record, Req, Record->MCurContext);

          // Cannot setup link if candidate is linked already
          if (LinkedAllocaCmdCand && !LinkedAllocaCmdCand->MLinkedAllocaCmd)
            LinkedAllocaCmd = LinkedAllocaCmdCand;
        }

      AllocaCmd =
          new AllocaCommand(Queue, FullReq, InitFromUserData, LinkedAllocaCmd);

      // Update linked command
      if (LinkedAllocaCmd) {
        AllocaCmd->addDep(DepDesc{LinkedAllocaCmd, AllocaCmd->getRequirement(),
                                  LinkedAllocaCmd});
        LinkedAllocaCmd->addUser(AllocaCmd);
        LinkedAllocaCmd->MLinkedAllocaCmd = AllocaCmd;

        // To ensure that the leader allocation is removed first
        AllocaCmd->getReleaseCmd()->addDep(
            DepDesc(LinkedAllocaCmd->getReleaseCmd(), AllocaCmd->getRequirement(),
                    LinkedAllocaCmd));

        // Device allocation takes ownership of the host ptr during
        // construction, host allocation doesn't. So, device allocation should
        // always be active here. Also if the "follower" command is a device one
        // we have to change current context to the device one.
        if (Queue->is_host()) {
          AllocaCmd->MIsActive = false;
        } else {
          LinkedAllocaCmd->MIsActive = false;
          Record->MCurContext = Queue->getContextImplPtr();
        }
      }
    }

    Record->MAllocaCommands.push_back(AllocaCmd);
    Record->MWriteLeaves.push_back(AllocaCmd);
    ++(AllocaCmd->MLeafCounter);
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
  case access::mode::read:
    break;
  }
}

Command *
Scheduler::GraphBuilder::addCG(std::unique_ptr<detail::CG> CommandGroup,
                               QueueImplPtr Queue) {
  const std::vector<Requirement *> &Reqs = CommandGroup->MRequirements;
  const std::vector<detail::EventImplPtr> &Events = CommandGroup->MEvents;
  std::unique_ptr<ExecCGCommand> NewCmd(
      new ExecCGCommand(std::move(CommandGroup), Queue));
  if (!NewCmd)
    throw runtime_error("Out of host memory");

  if (MPrintOptionsArray[BeforeAddCG])
    printGraphAsDot("before_addCG");

  for (Requirement *Req : Reqs) {
    MemObjRecord *Record = getOrInsertMemObjRecord(Queue, Req);
    markModifiedIfWrite(Record, Req);

    AllocaCommandBase *AllocaCmd = getOrCreateAllocaForReq(Record, Req, Queue);
    // If there is alloca command we need to check if the latest memory is in
    // required context.
    if (!sameCtx(Queue->getContextImplPtr(), Record->MCurContext)) {
      // Cannot directly copy memory from OpenCL device to OpenCL device -
      // create two copies: device->host and host->device.
      if (!Queue->is_host() && !Record->MCurContext->is_host())
        insertMemoryMove(Record, Req,
                         Scheduler::getInstance().getDefaultHostQueue());
      insertMemoryMove(Record, Req, Queue);
    }
    std::set<Command *> Deps =
        findDepsForReq(Record, Req, Queue->getContextImplPtr());

    for (Command *Dep : Deps)
      NewCmd->addDep(DepDesc{Dep, Req, AllocaCmd});
  }

  // Set new command as user for dependencies and update leaves.
  for (DepDesc &Dep : NewCmd->MDeps) {
    Dep.MDepCommand->addUser(NewCmd.get());
    const Requirement *Req = Dep.MDepRequirement;
    MemObjRecord *Record = getMemObjRecord(Req->MSYCLMemObj);
    updateLeaves({Dep.MDepCommand}, Record, Req->MAccessMode);
    addNodeToLeaves(Record, NewCmd.get(), Req->MAccessMode);
  }

  // Register all the events as dependencies
  for (detail::EventImplPtr e : Events) {
    NewCmd->addDep(e);
  }

  if (MPrintOptionsArray[AfterAddCG])
    printGraphAsDot("after_addCG");
  return NewCmd.release();
}

void Scheduler::GraphBuilder::decrementLeafCountersForRecord(
    MemObjRecord *Record) {
  for (Command *Cmd : Record->MReadLeaves) {
    --(Cmd->MLeafCounter);
  }
  for (Command *Cmd : Record->MWriteLeaves) {
    --(Cmd->MLeafCounter);
  }
}

void Scheduler::GraphBuilder::cleanupCommandsForRecord(MemObjRecord *Record) {
  std::vector<AllocaCommandBase *> &AllocaCommands = Record->MAllocaCommands;
  if (AllocaCommands.empty())
    return;

  std::queue<Command *> ToVisit;
  std::set<Command *> Visited;
  std::vector<Command *> CmdsToDelete;
  // First, mark all allocas for deletion and their direct users for traversal
  // Dependencies of the users will be cleaned up during the traversal
  for (Command *AllocaCmd : AllocaCommands) {
    Visited.insert(AllocaCmd);
    for (Command *UserCmd : AllocaCmd->MUsers)
      ToVisit.push(UserCmd);
    CmdsToDelete.push_back(AllocaCmd);
    // These commands will be deleted later, clear users now to avoid
    // updating them during edge removal
    AllocaCmd->MUsers.clear();
  }

  // Traverse the graph using BFS
  while (!ToVisit.empty()) {
    Command *Cmd = ToVisit.front();
    ToVisit.pop();

    if (!Visited.insert(Cmd).second)
      continue;

    for (Command *UserCmd : Cmd->MUsers)
      ToVisit.push(UserCmd);

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
      CmdsToDelete.push_back(Cmd);
      Cmd->MUsers.clear();
    }
  }

  for (Command *Cmd : CmdsToDelete) {
    Cmd->getEvent()->setCommand(nullptr);
    delete Cmd;
  }
}

void Scheduler::GraphBuilder::cleanupFinishedCommands(Command *FinishedCmd) {
  std::queue<Command *> CmdsToVisit({FinishedCmd});
  std::set<Command *> Visited;

  // Traverse the graph using BFS
  while (!CmdsToVisit.empty()) {
    Command *Cmd = CmdsToVisit.front();
    CmdsToVisit.pop();

    if (!Visited.insert(Cmd).second)
      continue;

    for (const DepDesc &Dep : Cmd->MDeps) {
      if (Dep.MDepCommand)
        CmdsToVisit.push(Dep.MDepCommand);
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
    Cmd->getEvent()->setCommand(nullptr);
    delete Cmd;
  }
}

void Scheduler::GraphBuilder::removeRecordForMemObj(SYCLMemObjI *MemObject) {
  const auto It = std::find_if(
      MMemObjs.begin(), MMemObjs.end(),
      [MemObject](const SYCLMemObjI *Obj) { return Obj == MemObject; });
  if (It != MMemObjs.end())
    MMemObjs.erase(It);
  MemObject->MRecord.reset();
}

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
