//===-- graph_builder.cpp - SYCL Graph Builder ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/access/access.hpp>
#include "detail/config.hpp"
#include <CL/sycl/detail/context_impl.hpp>
#include <CL/sycl/detail/event_impl.hpp>
#include <CL/sycl/detail/memory_manager.hpp>
#include <CL/sycl/detail/queue_impl.hpp>
#include <CL/sycl/detail/scheduler/scheduler.hpp>
#include <CL/sycl/exception.hpp>

#include <cstdlib>
#include <fstream>
#include <memory>
#include <queue>
#include <set>
#include <vector>

__SYCL_INLINE namespace cl {
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

  MemObject->MRecord.reset(new MemObjRecord{/*MAllocaCommands*/ {},
                                            /*MReadLeaves*/ {},
                                            /*MWriteLeaves*/ {},
                                            Queue->get_context_impl(),
                                            /*MMemModified*/ false});

  MMemObjs.push_back(MemObject);
  return MemObject->MRecord.get();
}

// Helper function which removes all values in Cmds from Leaves
void Scheduler::GraphBuilder::UpdateLeaves(const std::set<Command *> &Cmds,
                                           MemObjRecord *Record,
                                           access::mode AccessMode) {

  const bool ReadOnlyReq = AccessMode == access::mode::read;
  if (ReadOnlyReq)
    return;

  for (const Command *Cmd : Cmds) {
    auto NewEnd = std::remove(Record->MReadLeaves.begin(),
                              Record->MReadLeaves.end(), Cmd);
    Record->MReadLeaves.erase(NewEnd, Record->MReadLeaves.end());

    NewEnd = std::remove(Record->MWriteLeaves.begin(),
                         Record->MWriteLeaves.end(), Cmd);
    Record->MWriteLeaves.erase(NewEnd, Record->MWriteLeaves.end());
  }
}

void Scheduler::GraphBuilder::AddNodeToLeaves(MemObjRecord *Record,
                                              Command *Cmd,
                                              access::mode AccessMode) {
  if (AccessMode == access::mode::read)
    Record->MReadLeaves.push_back(Cmd);
  else
    Record->MWriteLeaves.push_back(Cmd);
}

UpdateHostRequirementCommand *Scheduler::GraphBuilder::insertUpdateHostReqCmd(
    MemObjRecord *Record, Requirement *Req, const QueueImplPtr &Queue) {
  AllocaCommandBase *AllocaCmd =
      findAllocaForReq(Record, Req, Queue->get_context_impl());
  assert(AllocaCmd && "There must be alloca for requirement!");
  UpdateHostRequirementCommand *UpdateCommand =
      new UpdateHostRequirementCommand(Queue, *Req, AllocaCmd, &Req->MData);
  // Need copy of requirement because after host accessor destructor call
  // dependencies become invalid if requirement is stored by pointer.
  const Requirement *StoredReq = UpdateCommand->getRequirement();

  std::set<Command *> Deps =
      findDepsForReq(Record, Req, Queue->get_context_impl());
  for (Command *Dep : Deps) {
    UpdateCommand->addDep(DepDesc{Dep, StoredReq, AllocaCmd});
    Dep->addUser(UpdateCommand);
  }
  UpdateLeaves(Deps, Record, Req->MAccessMode);
  AddNodeToLeaves(Record, UpdateCommand, Req->MAccessMode);
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
      findDepsForReq(Record, Req, Queue->get_context_impl());
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
  if (!AllocaCmdSrc)
    throw runtime_error("Cannot find buffer allocation");
  // Get parent allocation of sub buffer to perform full copy of whole buffer
  if (IsSuitableSubReq(Req)) {
    if (AllocaCmdSrc->getType() == Command::CommandType::ALLOCA_SUB_BUF)
      AllocaCmdSrc =
          static_cast<AllocaSubBufCommand *>(AllocaCmdSrc)->getParentAlloca();
    else
      assert(
          !"Inappropriate alloca command. AllocaSubBufCommand was expected.");
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
  UpdateLeaves(Deps, Record, access::mode::read_write);
  AddNodeToLeaves(Record, NewCmd, access::mode::read_write);
  Record->MCurContext = Queue->get_context_impl();
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
      findDepsForReq(Record, Req, HostQueue->get_context_impl());
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

  UpdateLeaves(Deps, Record, Req->MAccessMode);
  AddNodeToLeaves(Record, MemCpyCmd, Req->MAccessMode);
  if (MPrintOptionsArray[AfterAddCopyBack])
    printGraphAsDot("after_addCopyBack");
  return MemCpyCmd;
}

// The function implements SYCL host accessor logic: host accessor
// should provide access to the buffer in user space.
Command *Scheduler::GraphBuilder::addHostAccessor(Requirement *Req) {

  const QueueImplPtr &HostQueue = getInstance().getDefaultHostQueue();

  MemObjRecord *Record = getOrInsertMemObjRecord(HostQueue, Req);
  if (MPrintOptionsArray[BeforeAddHostAcc])
    printGraphAsDot("before_addHostAccessor");
  markModifiedIfWrite(Record, Req);

  AllocaCommandBase *HostAllocaCmd =
      getOrCreateAllocaForReq(Record, Req, HostQueue);

  if (!sameCtx(HostAllocaCmd->getQueue()->get_context_impl(),
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

  UpdateLeaves({UpdateHostAccCmd}, Record, Req->MAccessMode);
  AddNodeToLeaves(Record, EmptyCmd, Req->MAccessMode);

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

  std::vector<Command *> ToAnalyze;

  ToAnalyze = Record->MWriteLeaves;

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
            sameCtx(Context, Dep.MDepCommand->getQueue()->get_context_impl());

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

// The function searches for the alloca command matching context and
// requirement.
AllocaCommandBase *Scheduler::GraphBuilder::findAllocaForReq(
    MemObjRecord *Record, Requirement *Req, const ContextImplPtr &Context) {
  auto IsSuitableAlloca = [&Context, Req](AllocaCommandBase *AllocaCmd) {
    bool Res = sameCtx(AllocaCmd->getQueue()->get_context_impl(), Context);
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
      findAllocaForReq(Record, Req, Queue->get_context_impl());

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
      UpdateLeaves(findDepsForReq(Record, Req, Queue->get_context_impl()),
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
          Record->MCurContext = Queue->get_context_impl();
        }
      }
    }

    Record->MAllocaCommands.push_back(AllocaCmd);
    Record->MWriteLeaves.push_back(AllocaCmd);
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
    if (!sameCtx(Queue->get_context_impl(), Record->MCurContext)) {
      // Cannot directly copy memory from OpenCL device to OpenCL device -
      // create two copies: device->host and host->device.
      if (!Queue->is_host() && !Record->MCurContext->is_host())
        insertMemoryMove(Record, Req,
                         Scheduler::getInstance().getDefaultHostQueue());
      insertMemoryMove(Record, Req, Queue);
    }
    std::set<Command *> Deps =
        findDepsForReq(Record, Req, Queue->get_context_impl());

    for (Command *Dep : Deps)
      NewCmd->addDep(DepDesc{Dep, Req, AllocaCmd});
  }

  // Set new command as user for dependencies and update leaves.
  for (DepDesc &Dep : NewCmd->MDeps) {
    Dep.MDepCommand->addUser(NewCmd.get());
    const Requirement *Req = Dep.MDepRequirement;
    MemObjRecord *Record = getMemObjRecord(Req->MSYCLMemObj);
    UpdateLeaves({Dep.MDepCommand}, Record, Req->MAccessMode);
    AddNodeToLeaves(Record, NewCmd.get(), Req->MAccessMode);
  }

  // Register all the events as dependencies
  for (detail::EventImplPtr e : Events) {
    NewCmd->addDep(e);
  }

  if (MPrintOptionsArray[AfterAddCG])
    printGraphAsDot("after_addCG");
  return NewCmd.release();
}

void Scheduler::GraphBuilder::cleanupCommandsForRecord(MemObjRecord *Record) {
  if (Record->MAllocaCommands.empty())
    return;

  std::queue<Command *> RemoveQueue;
  std::set<Command *> Visited;

  // TODO: release commands need special handling here as they are not reachable
  // from alloca commands

  for (AllocaCommandBase *AllocaCmd : Record->MAllocaCommands) {
    if (Visited.find(AllocaCmd) == Visited.end())
      RemoveQueue.push(AllocaCmd);
    // Use BFS to find and process all users of removal candidate
    while (!RemoveQueue.empty()) {
      Command *CandidateCommand = RemoveQueue.front();
      RemoveQueue.pop();

      if (Visited.insert(CandidateCommand).second) {
        for (Command *UserCmd : CandidateCommand->MUsers) {
          // As candidate command is about to be freed, we need
          // to remove it from dependency list of other commands.
          auto NewEnd =
              std::remove_if(UserCmd->MDeps.begin(), UserCmd->MDeps.end(),
                             [CandidateCommand](const DepDesc &Dep) {
                               return Dep.MDepCommand == CandidateCommand;
                             });
          UserCmd->MDeps.erase(NewEnd, UserCmd->MDeps.end());

          // Commands that have no unsatisfied dependencies can be executed
          // and are good candidates for clean up.
          if (UserCmd->MDeps.empty())
            RemoveQueue.push(UserCmd);
        }
        CandidateCommand->getEvent()->setCommand(nullptr);
        delete CandidateCommand;
      }
    }
  }
}

void Scheduler::GraphBuilder::cleanupCommands(bool CleanupReleaseCommands) {
  // TODO: Implement.
}

void Scheduler::GraphBuilder::removeRecordForMemObj(SYCLMemObjI *MemObject) {
  const auto It = std::find_if(
      MMemObjs.begin(), MMemObjs.end(),
      [MemObject](const SYCLMemObjI *Obj) { return Obj == MemObject; });
  if (It != MMemObjs.end())
    MMemObjs.erase(It);
  MemObject->MRecord.reset(nullptr);
}

} // namespace detail
} // namespace sycl
} // namespace cl
