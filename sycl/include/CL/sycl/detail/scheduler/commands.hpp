//==-------------- commands.hpp - SYCL standard header file ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <atomic>
#include <memory>
#include <vector>

#include <CL/sycl/access/access.hpp>
#include <CL/sycl/detail/accessor_impl.hpp>
#include <CL/sycl/detail/cg.hpp>

__SYCL_INLINE namespace cl {
namespace sycl {
namespace detail {

class queue_impl;
class event_impl;
class context_impl;

using QueueImplPtr = std::shared_ptr<detail::queue_impl>;
using EventImplPtr = std::shared_ptr<detail::event_impl>;
using ContextImplPtr = std::shared_ptr<detail::context_impl>;

class Command;
class AllocaCommand;
class AllocaCommandBase;
class ReleaseCommand;

enum BlockingT { NON_BLOCKING = 0, BLOCKING };

// The struct represents the result of command enqueueing
struct EnqueueResultT {
  enum ResultT { SUCCESS, BLOCKED, FAILED };
  EnqueueResultT(ResultT Result = SUCCESS, Command *Cmd = nullptr,
                 cl_int ErrCode = CL_SUCCESS)
      : MResult(Result), MCmd(Cmd), MErrCode(ErrCode) {}
  // Indicates result of enqueueing
  ResultT MResult;
  // Pointer to the command failed to enqueue
  Command *MCmd;
  // Error code which is set when enqueueing fails
  cl_int MErrCode;
};

// DepDesc represents dependency between two commands
struct DepDesc {
  DepDesc(Command *DepCommand, const Requirement *Req,
          AllocaCommandBase *AllocaCmd)
      : MDepCommand(DepCommand), MDepRequirement(Req), MAllocaCmd(AllocaCmd) {}

  friend bool operator<(const DepDesc &Lhs, const DepDesc &Rhs) {
    return std::tie(Lhs.MDepRequirement, Lhs.MDepCommand) <
           std::tie(Rhs.MDepRequirement, Rhs.MDepCommand);
  }

  // The actual dependency command.
  Command *MDepCommand = nullptr;
  // Requirement for the dependency.
  const Requirement *MDepRequirement = nullptr;
  // Allocation command for the memory object we have requirement for.
  // Used to simplify searching for memory handle.
  AllocaCommandBase *MAllocaCmd = nullptr;
};

// The Command represents some action that needs to be performed on one or more
// memory objects. The command has vector of Depdesc objects that represent
// dependencies of the command. It has vector of pointer to commands that depend
// on the command. It has pointer to sycl::queue object. And has event that is
// associated with the command.
class Command {
public:
  enum CommandType {
    RUN_CG,
    COPY_MEMORY,
    ALLOCA,
    ALLOCA_SUB_BUF,
    RELEASE,
    MAP_MEM_OBJ,
    UNMAP_MEM_OBJ,
    UPDATE_REQUIREMENT,
    EMPTY_TASK
  };

  Command(CommandType Type, QueueImplPtr Queue);

  void addDep(DepDesc NewDep) {
    if (NewDep.MDepCommand)
      MDepsEvents.push_back(NewDep.MDepCommand->getEvent());
    MDeps.push_back(NewDep);
  }

  void addDep(EventImplPtr Event) { MDepsEvents.push_back(std::move(Event)); }

  void addUser(Command *NewUser) { MUsers.push_back(NewUser); }

  // Return type of the command, e.g. Allocate, MemoryCopy.
  CommandType getType() const { return MType; }

  // The method checks if the command is enqueued, waits for it to be unblocked
  // if "Blocking" argument is true, then calls enqueueImp.
  // Returns true if the command is enqueued. Sets EnqueueResult to the specific
  // status otherwise.
  bool enqueue(EnqueueResultT &EnqueueResult, BlockingT Blocking);

  bool isFinished();

  bool isEnqueued() const { return MEnqueued; }

  std::shared_ptr<queue_impl> getQueue() const { return MQueue; }

  std::shared_ptr<event_impl> getEvent() const { return MEvent; }

  virtual void printDot(std::ostream &Stream) const = 0;

  virtual const Requirement *getRequirement() const {
    assert(!"Internal Error. The command has no stored requirement");
    return nullptr;
  }

  virtual ~Command() = default;

protected:
  EventImplPtr MEvent;
  QueueImplPtr MQueue;
  std::vector<EventImplPtr> MDepsEvents;

  void waitForEvents(QueueImplPtr Queue, std::vector<RT::PiEvent> &RawEvents,
                     RT::PiEvent &Event);
  std::vector<RT::PiEvent> prepareEvents(ContextImplPtr Context);

  // Private interface. Derived classes should implement this method.
  virtual cl_int enqueueImp() = 0;

  // The type of the command
  CommandType MType;
  // Indicates whether the command is enqueued or not
  std::atomic<bool> MEnqueued;
  // Mutex used to protect enqueueing from race conditions
  std::mutex MEnqueueMtx;

public:
  // Contains list of dependencies(edges)
  std::vector<DepDesc> MDeps;
  // Contains list of commands that depend on the command
  std::vector<Command *> MUsers;
  // Indicates whether the command can be blocked from enqueueing
  bool MIsBlockable = false;
  // Indicates whether the command is blocked from enqueueing
  std::atomic<bool> MCanEnqueue;

  const char *MBlockReason = "Unknown";
};

// The command does nothing during enqueue. The task can be used to implement
// lock in the graph, or to merge several nodes into one.
class EmptyCommand : public Command {
public:
  EmptyCommand(QueueImplPtr Queue, Requirement Req)
      : Command(CommandType::EMPTY_TASK, std::move(Queue)),
        MRequirement(std::move(Req)) {}

  void printDot(std::ostream &Stream) const final;
  const Requirement *getRequirement() const final { return &MRequirement; }

private:
  cl_int enqueueImp() final { return CL_SUCCESS; }

  Requirement MRequirement;
};

// The command enqueues release instance of memory allocated on Host or
// underlying framework.
class ReleaseCommand : public Command {
public:
  ReleaseCommand(QueueImplPtr Queue, AllocaCommandBase *AllocaCmd)
      : Command(CommandType::RELEASE, std::move(Queue)), MAllocaCmd(AllocaCmd) {
  }

  void printDot(std::ostream &Stream) const final;

private:
  cl_int enqueueImp() final;

  // Command which allocates memory release command should dealocate
  AllocaCommandBase *MAllocaCmd = nullptr;
};

class AllocaCommandBase : public Command {
public:
  AllocaCommandBase(CommandType Type, QueueImplPtr Queue, Requirement Req,
                    AllocaCommandBase *LinkedAllocaCmd)
      : Command(Type, Queue), MLinkedAllocaCmd(LinkedAllocaCmd),
        MIsLeaderAlloca(nullptr == LinkedAllocaCmd), MReleaseCmd(Queue, this),
        MRequirement(std::move(Req)) {
    MRequirement.MAccessMode = access::mode::read_write;
  }

  ReleaseCommand *getReleaseCmd() { return &MReleaseCmd; }

  SYCLMemObjI *getSYCLMemObj() const { return MRequirement.MSYCLMemObj; }

  void *getMemAllocation() const { return MMemAllocation; }

  const Requirement *getRequirement() const final { return &MRequirement; }

  void *MMemAllocation = nullptr;

  // Alloca command linked with current command.
  // Device and host alloca commands can be linked, so they may share the same
  // memory. Only one allocation from a pair can be accessed at a time. Alloca
  // commands associated with such allocation is "active". In order to switch
  // "active" status between alloca commands map/unmap operations are used.
  AllocaCommandBase *MLinkedAllocaCmd = nullptr;
  // Indicates that current alloca is active one.
  bool MIsActive = true;

  // Indicates that the command owns memory allocation in case of connected
  // alloca command
  bool MIsLeaderAlloca = true;

protected:
  ReleaseCommand MReleaseCmd;
  Requirement MRequirement;
};

// The command enqueues allocation of instance of memory object on Host or
// underlying framework.
class AllocaCommand : public AllocaCommandBase {
public:
  AllocaCommand(QueueImplPtr Queue, Requirement Req,
                bool InitFromUserData = true,
                AllocaCommandBase *LinkedAllocaCmd = nullptr)
      : AllocaCommandBase(CommandType::ALLOCA, std::move(Queue), std::move(Req),
                          LinkedAllocaCmd),
        MInitFromUserData(InitFromUserData) {
    addDep(DepDesc(nullptr, getRequirement(), this));
  }

  void printDot(std::ostream &Stream) const final;

private:
  cl_int enqueueImp() final;

  // The flag indicates that alloca should try to reuse pointer provided by the
  // user during memory object construction
  bool MInitFromUserData = false;
};

class AllocaSubBufCommand : public AllocaCommandBase {
public:
  AllocaSubBufCommand(QueueImplPtr Queue, Requirement Req,
                      AllocaCommandBase *ParentAlloca)
      : AllocaCommandBase(CommandType::ALLOCA_SUB_BUF, std::move(Queue),
                          std::move(Req),
                          /*LinkedAllocaCmd*/ nullptr),
        MParentAlloca(ParentAlloca) {
    addDep(DepDesc(MParentAlloca, getRequirement(), MParentAlloca));
  }

  void printDot(std::ostream &Stream) const final;
  AllocaCommandBase *getParentAlloca() { return MParentAlloca; }

private:
  cl_int enqueueImp() final;

  AllocaCommandBase *MParentAlloca = nullptr;
};

class MapMemObject : public Command {
public:
  MapMemObject(AllocaCommandBase *SrcAllocaCmd, Requirement Req, void **DstPtr,
               QueueImplPtr Queue);

  void printDot(std::ostream &Stream) const final;
  const Requirement *getRequirement() const final { return &MSrcReq; }

private:
  cl_int enqueueImp() final;

  AllocaCommandBase *MSrcAllocaCmd = nullptr;
  Requirement MSrcReq;
  void **MDstPtr = nullptr;
};

class UnMapMemObject : public Command {
public:
  UnMapMemObject(AllocaCommandBase *DstAllocaCmd, Requirement Req,
                 void **SrcPtr, QueueImplPtr Queue);

  void printDot(std::ostream &Stream) const final;
  const Requirement *getRequirement() const final { return &MDstReq; }

private:
  cl_int enqueueImp() final;

  AllocaCommandBase *MDstAllocaCmd = nullptr;
  Requirement MDstReq;
  void **MSrcPtr = nullptr;
};

// The command enqueues memory copy between two instances of memory object.
class MemCpyCommand : public Command {
public:
  MemCpyCommand(Requirement SrcReq, AllocaCommandBase *SrcAllocaCmd,
                Requirement DstReq, AllocaCommandBase *DstAllocaCmd,
                QueueImplPtr SrcQueue, QueueImplPtr DstQueue);

  void printDot(std::ostream &Stream) const final;
  const Requirement *getRequirement() const final { return &MDstReq; }

private:
  cl_int enqueueImp() final;

  QueueImplPtr MSrcQueue;
  Requirement MSrcReq;
  AllocaCommandBase *MSrcAllocaCmd = nullptr;
  Requirement MDstReq;
  AllocaCommandBase *MDstAllocaCmd = nullptr;
};

// The command enqueues memory copy between two instances of memory object.
class MemCpyCommandHost : public Command {
public:
  MemCpyCommandHost(Requirement SrcReq, AllocaCommandBase *SrcAllocaCmd,
                    Requirement DstReq, void **DstPtr, QueueImplPtr SrcQueue,
                    QueueImplPtr DstQueue);

  void printDot(std::ostream &Stream) const final;
  const Requirement *getRequirement() const final { return &MDstReq; }

private:
  cl_int enqueueImp() final;

  QueueImplPtr MSrcQueue;
  Requirement MSrcReq;
  AllocaCommandBase *MSrcAllocaCmd = nullptr;
  Requirement MDstReq;
  void **MDstPtr = nullptr;
};

// The command enqueues execution of kernel or explicit memory operation.
class ExecCGCommand : public Command {
public:
  ExecCGCommand(std::unique_ptr<detail::CG> CommandGroup, QueueImplPtr Queue)
      : Command(CommandType::RUN_CG, std::move(Queue)),
        MCommandGroup(std::move(CommandGroup)) {}

  void flushStreams();

  void printDot(std::ostream &Stream) const final;

private:
  cl_int enqueueImp() final;

  AllocaCommandBase *getAllocaForReq(Requirement *Req);

  std::unique_ptr<detail::CG> MCommandGroup;
};

class UpdateHostRequirementCommand : public Command {
public:
  UpdateHostRequirementCommand(QueueImplPtr Queue, Requirement Req,
                               AllocaCommandBase *SrcAllocaCmd, void **DstPtr)
      : Command(CommandType::UPDATE_REQUIREMENT, std::move(Queue)),
        MSrcAllocaCmd(SrcAllocaCmd), MDstReq(std::move(Req)), MDstPtr(DstPtr) {}

  void printDot(std::ostream &Stream) const final;
  const Requirement *getRequirement() const final { return &MDstReq; }

private:
  cl_int enqueueImp() final;

  AllocaCommandBase *MSrcAllocaCmd = nullptr;
  Requirement MDstReq;
  void **MDstPtr = nullptr;
};

} // namespace detail
} // namespace sycl
} // namespace cl
