//===-- graph_builder.cpp - SYCL Graph Builder ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/access/access.hpp>
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

namespace cl {
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

// The function checks if current requirement is requirement for sub buffer
static bool IsSuitableSubReq(const Requirement *Req) {
  return Req->MIsSubBuffer;
}

Scheduler::GraphBuilder::GraphBuilder() {
  if (const char *EnvVarCStr = std::getenv("SYCL_PRINT_EXECUTION_GRAPH")) {
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
MemObjRecord *
Scheduler::GraphBuilder::getMemObjRecord(SYCLMemObjI *MemObject) {
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
                                            /*MReadLeafs*/ {},
                                            /*MWriteLeafs*/ {},
                                            Queue->get_context_impl(),
                                            /*MMemModified*/ false});

  MMemObjs.push_back(MemObject);
  return MemObject->MRecord.get();
}

// Helper function which removes all values in Cmds from Leafs
void Scheduler::GraphBuilder::UpdateLeafs(const std::set<Command *> &Cmds,
                                          MemObjRecord *Record,
                                          access::mode AccessMode) {

  const bool ReadOnlyReq = AccessMode == access::mode::read;
  if (ReadOnlyReq)
    return;

  for (const Command *Cmd : Cmds) {
    auto NewEnd =
        std::remove(Record->MReadLeafs.begin(), Record->MReadLeafs.end(), Cmd);
    Record->MReadLeafs.erase(NewEnd, Record->MReadLeafs.end());

    NewEnd = std::remove(Record->MWriteLeafs.begin(), Record->MWriteLeafs.end(),
                         Cmd);
    Record->MWriteLeafs.erase(NewEnd, Record->MWriteLeafs.end());
  }
}

void Scheduler::GraphBuilder::AddNodeToLeafs(MemObjRecord *Record, Command *Cmd,
                                             access::mode AccessMode) {
  if (AccessMode == access::mode::read)
    Record->MReadLeafs.push_back(Cmd);
  else
    Record->MWriteLeafs.push_back(Cmd);
}

UpdateHostRequirementCommand *Scheduler::GraphBuilder::insertUpdateHostReqCmd(
    MemObjRecord *Record, Requirement *Req, const QueueImplPtr &Queue) {
  AllocaCommandBase *AllocaCmd =
      findAllocaForReq(Record, Req, Queue->get_context_impl());
  assert(AllocaCmd && "There must be alloca for requirement!");
  UpdateHostRequirementCommand *UpdateCommand =
      new UpdateHostRequirementCommand(Queue, Req, AllocaCmd);
  // Need copy of requirement because after host accessor destructor call
  // dependencies become invalid if requirement is stored by pointer.
  Requirement *StoredReq = UpdateCommand->getStoredRequirement();

  std::set<Command *> Deps = findDepsForReq(Record, Req, Queue);
  for (Command *Dep : Deps) {
    UpdateCommand->addDep(DepDesc{Dep, StoredReq, AllocaCmd});
    Dep->addUser(UpdateCommand);
  }
  UpdateLeafs(Deps, Record, Req->MAccessMode);
  AddNodeToLeafs(Record, UpdateCommand, Req->MAccessMode);
  return UpdateCommand;
}

MemCpyCommand *
Scheduler::GraphBuilder::insertMemCpyCmd(MemObjRecord *Record, Requirement *Req,
                                         const QueueImplPtr &Queue,
                                         bool UseExclusiveQueue) {
  std::set<Command *> Deps = findDepsForReq(Record, Req, Queue);
  QueueImplPtr SrcQueue = (*Deps.begin())->getQueue();

  AllocaCommandBase *AllocaCmdDst = getOrCreateAllocaForReq(Record, Req, Queue);
  if (!AllocaCmdDst)
    throw runtime_error("Out of host memory");
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
      getOrCreateAllocaForReq(Record, Req, SrcQueue);
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

  // Full copy of buffer is needed to avoid loss of data that may be caused
  // by copying specific range form host to device and backwards.
  MemCpyCommand *MemCpyCmd = new MemCpyCommand(
      *AllocaCmdSrc->getAllocationReq(), AllocaCmdSrc,
      *AllocaCmdDst->getAllocationReq(), AllocaCmdDst, AllocaCmdSrc->getQueue(),
      AllocaCmdDst->getQueue(), UseExclusiveQueue);

  for (Command *Dep : Deps) {
    MemCpyCmd->addDep(DepDesc{Dep, &MemCpyCmd->MDstReq, AllocaCmdDst});
    Dep->addUser(MemCpyCmd);
  }
  UpdateLeafs(Deps, Record, access::mode::read_write);
  AddNodeToLeafs(Record, MemCpyCmd, access::mode::read_write);
  Record->MCurContext = Queue->get_context_impl();
  return MemCpyCmd;
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

  std::set<Command *> Deps = findDepsForReq(Record, Req, HostQueue);
  AllocaCommandBase *SrcAllocaCmd =
      findAllocaForReq(Record, Req, Record->MCurContext);

  std::unique_ptr<MemCpyCommandHost> MemCpyCmdUniquePtr(new MemCpyCommandHost(
      *SrcAllocaCmd->getAllocationReq(), SrcAllocaCmd, Req,
      SrcAllocaCmd->getQueue(), std::move(HostQueue)));

  if (!MemCpyCmdUniquePtr)
    throw runtime_error("Out of host memory");

  MemCpyCommandHost *MemCpyCmd = MemCpyCmdUniquePtr.release();
  for (Command *Dep : Deps) {
    MemCpyCmd->addDep(DepDesc{Dep, &MemCpyCmd->MDstReq, SrcAllocaCmd});
    Dep->addUser(MemCpyCmd);
  }

  UpdateLeafs(Deps, Record, Req->MAccessMode);
  AddNodeToLeafs(Record, MemCpyCmd, Req->MAccessMode);
  if (MPrintOptionsArray[AfterAddCopyBack])
    printGraphAsDot("after_addCopyBack");
  return MemCpyCmd;
}

// The function implements SYCL host accessor logic: host accessor
// should provide access to the buffer in user space, then during
// destruction the memory should be written back(if access mode is not read
// only) to the memory object. No operations with buffer allowed during host
// accessor lifetime.
Command *Scheduler::GraphBuilder::addHostAccessor(Requirement *Req,
                                                  EventImplPtr &RetEvent) {
  QueueImplPtr HostQueue = Scheduler::getInstance().getDefaultHostQueue();
  MemObjRecord *Record = getOrInsertMemObjRecord(HostQueue, Req);
  if (MPrintOptionsArray[BeforeAddHostAcc])
    printGraphAsDot("before_addHostAccessor");
  markModifiedIfWrite(Record, Req);
  QueueImplPtr SrcQueue;

  std::set<Command *> Deps = findDepsForReq(Record, Req, HostQueue);
  // If we got freshly new Record, there is no point in searching
  // for allocas on device.
  if (Record->MAllocaCommands.empty())
    SrcQueue = HostQueue;
  else
    SrcQueue = (*Deps.begin())->getQueue();

  AllocaCommandBase *SrcAllocaCmd =
      getOrCreateAllocaForReq(Record, Req, SrcQueue);
  Requirement *SrcReq = SrcAllocaCmd->getAllocationReq();
  if (SrcQueue->is_host()) {
    UpdateHostRequirementCommand *UpdateCmd =
        insertUpdateHostReqCmd(Record, Req, SrcQueue);
    RetEvent = UpdateCmd->getEvent();
    Record->MCurContext = SrcQueue->get_context_impl();
    if (MPrintOptionsArray[AfterAddHostAcc])
      printGraphAsDot("after_addHostAccessor");
    return UpdateCmd;
  }

  // Prepare "user" event that will block second operation(unmap of copy) until
  // host accessor is destructed.
  ContextImplPtr SrcContext = SrcQueue->get_context_impl();
  Req->BlockingEvent.reset(new detail::event_impl());
  Req->BlockingEvent->setContextImpl(SrcContext);
  RT::PiEvent &Event = Req->BlockingEvent->getHandleRef();
  RT::PiResult Error = PI_SUCCESS;
  PI_CALL((Event = RT::piEventCreate(
      SrcContext->getHandleRef(), &Error), Error));

  // In case of memory is 1 dimensional and located on OpenCL device we
  // can use map/unmap operation.
  // TODO: Implement mapping/unmapping for images
  if (!SrcQueue->is_host() && Req->MDims == 1 &&
      Req->MAccessRange == Req->MMemoryRange &&
      Req->MSYCLMemObj->getType() == detail::SYCLMemObjI::MemObjType::BUFFER) {

    std::unique_ptr<MapMemObject> MapCmdUniquePtr(
        new MapMemObject(*SrcReq, SrcAllocaCmd, Req, SrcQueue));

    /*
    [SYCL] Use exclusive queues for blocked commands.

    SYCL host accessor must wait in c'tor until the memory it provides
    access to is available on the host and should write memory back on
    destruction. No operations with the memory object are allowed
    during lifetime of the host accessor.

    In order to implement host accessor logic SYCL RT enqueues two tasks:
    read from device to host/map and write from host to device/unmap.
    The read/map operation should be completed during host accessor
    construction while write/unmap should be blocked until host accessor
    is destructed.

    To achieve blocking of write/unmap operation SYCL RT blocks it on
    user event then unblock during host accessor destruction.

    For the code:
    {
      ...
      Q.submit(...); // <== 1 Kernel
      auto HostAcc = Buf.get_access<...>(); // <== Host acc creation
      Q.submit(...); // <== 2 Kernel
    } // <== Host acc desctruction

    We generate the following graph(arrows represent dependencies)

              +-------------+
              |  1 Kernel   |
              +-------------+
                    ^
                    |
              +-------------+
              |  Read/Map   |  <== This task should be completed
              +-------------+      during host acc creation
                    ^
                    |
              +-------------+
              | Write/Unmap |  <== This is blocked by user event
              +-------------+      Can be completed after host acc
                    ^              desctruction
                    |
              +-------------+
              |  2 Kernel   |
              +-------------+

    And the following content in OpenCL command queue:

        +----------------------------------------------+
    Q1: | 1 Kernel | Read/Map | Write/Unmap | 2 Kernel |
        +----------------------------------------------+
                                      ^
                                      |
       This is blocked by user event -+

    This works fine, but for example below problems can happen:

    For the code:
    {
      ...
      Q.submit(...); // <== 1 Kernel
      auto HostAcc1 = Buf1.get_access<...>(); // <== Host acc 1 creation
      auto HostAcc2 = Buf2.get_access<...>(); // <== Host acc 2 creation
      Q.submit(...); // <== 2 Kernel
    } // <== Host acc 1 and 2 desctruction

    We generate the following graph(arrows represent dependencies)

                      +-------------+
                      |  1 Kernel   |
                      +-------------+
                            ^
            +---------------+---------------+
      +-------------+                +-------------+
      |  Read/Map 1 |                |  Read/Map 2 |  <== This task should be
      +-------------+                +-------------+       completed during host
            ^                               ^              acc creation
            |                               |
      +-------------+                +-------------+
      |Write/Unmap 1|                |Write/Unmap 2|  <== This is blocked by
      +-------------+                +-------------+      user event. Can be
            ^                               ^             completed after host
            +---------------+---------------+             accdesctruction
                      +-------------+
                      |  2 Kernel   |
                      +-------------+

    And the following content in OpenCL command queue:

        +-------------------------------------------------------------------+
    Q1: | 1K | Read/Map 1 | Write/Unmap 1 | Read/Map 2 | Write/Unmap 2 | 2K |
        +-------------------------------------------------------------------+
                                      ^                       ^
                                      |                       |
       This is blocked by user event -+-----------------------+

    In the situation above there is "Write/Unmap 1" command already in
    command queue which is blocked by user event and cannot be executed
    and "Read/Map 2" command which is enqueued after "Write/Unmap 1" but
    we should wait for the completion of this command before exiting
    construction of the second host accessor.

    Such cases is not supported in some OpenCL implementations. They
    assume that the commands the are submitted before one user waits on
    eventually completes.

    This patch workarounds problem by using separate(exclusive) queues for
    such tasks while still using one(common) queue for all other tasks.

    So, for the second example SYCL RT creates 3 OpenCL queues, where
    second and third queues are used for "Write/Unmap 1" and "Write/Unmap 2"
    command respectively:

        +-----------------------------------------------+
    Q1: | 1 Kernel | Read/Map 1 | Read/Map 2 | 2 Kernel |
        +-----------------------------------------------+

        +---------------+
    Q2: | Write/Unmap 1 |<----+
        +---------------+     |
                              |-----This is blocked by user event
        +---------------+     |
    Q3: | Write/Unmap 2 |<----+
        +---------------+
    */

    std::unique_ptr<UnMapMemObject> UnMapCmdUniquePtr(new UnMapMemObject(
        *SrcReq, SrcAllocaCmd, Req, SrcQueue, /*UseExclusiveQueue*/ true));

    if (!MapCmdUniquePtr || !UnMapCmdUniquePtr)
      throw runtime_error("Out of host memory");

    MapMemObject *MapCmd = MapCmdUniquePtr.release();
    for (Command *Dep : Deps) {
      MapCmd->addDep(DepDesc{Dep, &MapCmd->MDstReq, SrcAllocaCmd});
      Dep->addUser(MapCmd);
    }

    Command *UnMapCmd = UnMapCmdUniquePtr.release();
    UnMapCmd->addDep(DepDesc{MapCmd, &MapCmd->MDstReq, SrcAllocaCmd});
    MapCmd->addUser(UnMapCmd);

    UpdateLeafs(Deps, Record, Req->MAccessMode);
    AddNodeToLeafs(Record, UnMapCmd, Req->MAccessMode);

    UnMapCmd->addDep(Req->BlockingEvent);

    RetEvent = MapCmd->getEvent();
    if (MPrintOptionsArray[AfterAddHostAcc])
      printGraphAsDot("after_addHostAccessor");
    return UnMapCmd;
  }

  // In other cases insert two mem copy operations.
  MemCpyCommand *DevToHostCmd = insertMemCpyCmd(Record, Req, HostQueue);
  DevToHostCmd->setAccessorToUpdate(Req);
  Command *HostToDevCmd =
      insertMemCpyCmd(Record, Req, SrcQueue, /*UseExclusiveQueue*/ true);
  HostToDevCmd->addDep(Req->BlockingEvent);

  RetEvent = DevToHostCmd->getEvent();
  if (MPrintOptionsArray[AfterAddHostAcc])
    printGraphAsDot("after_addHostAccessor");
  return HostToDevCmd;
}

Command *Scheduler::GraphBuilder::addCGUpdateHost(
    std::unique_ptr<detail::CG> CommandGroup, QueueImplPtr HostQueue) {
  // Dummy implementation of update host logic, just copy memory to the host
  // device. We could avoid copying if there is no allocation of host memory.

  CGUpdateHost *UpdateHost = (CGUpdateHost *)CommandGroup.get();
  Requirement *Req = UpdateHost->getReqToUpdate();

  MemObjRecord *Record = getOrInsertMemObjRecord(HostQueue, Req);
  return insertMemCpyCmd(Record, Req, HostQueue);
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
                                        QueueImplPtr Queue) {
  sycl::context Context = Queue->get_context();
  std::set<Command *> RetDeps;
  std::set<Command *> Visited;
  const bool ReadOnlyReq = Req->MAccessMode == access::mode::read;

  std::vector<Command *> ToAnalyze;

  ToAnalyze = Record->MWriteLeafs;

  if (!ReadOnlyReq)
    ToAnalyze.insert(ToAnalyze.begin(), Record->MReadLeafs.begin(),
                     Record->MReadLeafs.end());

  while (!ToAnalyze.empty()) {
    Command *DepCmd = ToAnalyze.back();
    ToAnalyze.pop_back();

    std::vector<Command *> NewAnalyze;

    for (const DepDesc &Dep : DepCmd->MDeps) {
      if (Dep.MReq->MSYCLMemObj != Req->MSYCLMemObj)
        continue;

      bool CanBypassDep = false;
      // If both only read
      CanBypassDep |=
          Dep.MReq->MAccessMode == access::mode::read && ReadOnlyReq;

      // If not overlap
      CanBypassDep |= !doOverlap(Dep.MReq, Req);

      // Going through copying memory between contexts is not supported.
      if (Dep.MDepCommand)
        CanBypassDep &= Context == Dep.MDepCommand->getQueue()->get_context();

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
    bool Res = AllocaCmd->getQueue()->get_context_impl() == Context;
    if (IsSuitableSubReq(Req)) {
      auto TmpReq = AllocaCmd->getAllocationReq();
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
// requirement. If none exists, new command will be created.
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
      UpdateLeafs(findDepsForReq(Record, Req, Queue), Record,
                  access::mode::read_write);
    } else {
      Requirement FullReq(/*Offset*/ {0, 0, 0}, Req->MMemoryRange,
                          Req->MMemoryRange, access::mode::read_write,
                          Req->MSYCLMemObj, Req->MDims, Req->MElemSize);
      AllocaCmd = new AllocaCommand(Queue, FullReq);
    }

    Record->MAllocaCommands.push_back(AllocaCmd);
    Record->MWriteLeafs.push_back(AllocaCmd);
  }
  return AllocaCmd;
}

// The function sets MemModified flag in record if requirement has write access.
void Scheduler::GraphBuilder::markModifiedIfWrite(
    MemObjRecord *Record, Requirement *Req) {
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

    // If there is alloca command we need to check if the latest memory is in
    // required context.
    if (Record->MAllocaCommands.size()) {
      if (Queue->get_context_impl() != Record->MCurContext) {
        // Cannot directly copy memory from OpenCL device to OpenCL device -
        // create two copies: device->host and host->device.
        if (!Queue->is_host() && !Record->MCurContext->is_host())
          insertMemCpyCmd(Record, Req,
                          Scheduler::getInstance().getDefaultHostQueue());
        insertMemCpyCmd(Record, Req, Queue);
      }
    }
    AllocaCommandBase *AllocaCmd = getOrCreateAllocaForReq(Record, Req, Queue);
    std::set<Command *> Deps = findDepsForReq(Record, Req, Queue);

    for (Command *Dep : Deps)
      NewCmd->addDep(DepDesc{Dep, Req, AllocaCmd});
  }

  // Set new command as user for dependencies and update leafs.
  for (DepDesc &Dep : NewCmd->MDeps) {
    Dep.MDepCommand->addUser(NewCmd.get());
    Requirement *Req = Dep.MReq;
    MemObjRecord *Record = getMemObjRecord(Req->MSYCLMemObj);
    UpdateLeafs({Dep.MDepCommand}, Record, Req->MAccessMode);
    AddNodeToLeafs(Record, NewCmd.get(), Req->MAccessMode);
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
          auto NewEnd = std::remove_if(UserCmd->MDeps.begin(),
                  UserCmd->MDeps.end(), [CandidateCommand] (const DepDesc &Dep) {
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
  const auto It = std::find_if(MMemObjs.begin(), MMemObjs.end(),
                                 [MemObject](const SYCLMemObjI *Obj) {
                                   return Obj == MemObject;
                                 });
  if (It != MMemObjs.end())
    MMemObjs.erase(It);
  MemObject->MRecord.reset(nullptr);
}

} // namespace detail
} // namespace sycl
} // namespace cl
