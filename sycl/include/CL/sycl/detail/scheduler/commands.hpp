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

#include <CL/sycl/detail/accessor_impl.hpp>
#include <CL/sycl/detail/cg.hpp>

namespace cl {
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

// DepDesc represents dependency between two commands
struct DepDesc {
  DepDesc(Command *DepCommand, Requirement *Req, AllocaCommandBase *AllocaCmd)
      : MDepCommand(DepCommand), MReq(Req), MAllocaCmd(AllocaCmd) {}

  friend bool operator<(const DepDesc &Lhs, const DepDesc &Rhs) {
    return std::tie(Lhs.MReq, Lhs.MDepCommand) <
           std::tie(Rhs.MReq, Rhs.MDepCommand);
  }

  // The actual dependency command.
  Command *MDepCommand = nullptr;
  // Requirement for the dependency.
  Requirement *MReq = nullptr;
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

  Command(CommandType Type, QueueImplPtr Queue, bool UseExclusiveQueue = false);

  void addDep(DepDesc NewDep) {
    if (NewDep.MDepCommand)
      MDepsEvents.push_back(NewDep.MDepCommand->getEvent());
    MDeps.push_back(NewDep);
  }

  void addDep(EventImplPtr Event) { MDepsEvents.push_back(std::move(Event)); }

  void addUser(Command *NewUser) { MUsers.push_back(NewUser); }

  // Return type of the command, e.g. Allocate, MemoryCopy.
  CommandType getType() const { return MType; }

  // The method checks if the command is enqueued, call enqueueImp if not and
  // returns CL_SUCCESS on success.
  cl_int enqueue();

  bool isFinished();

  bool isEnqueued() const { return MEnqueued; }

  std::shared_ptr<queue_impl> getQueue() const { return MQueue; }

  std::shared_ptr<event_impl> getEvent() const { return MEvent; }

  virtual ~Command() = default;

  virtual void printDot(std::ostream &Stream) const = 0;

protected:
  EventImplPtr MEvent;
  QueueImplPtr MQueue;
  std::vector<EventImplPtr> MDepsEvents;

  void waitForEvents(QueueImplPtr Queue, std::vector<RT::PiEvent> &RawEvents,
                     RT::PiEvent &Event);
  std::vector<RT::PiEvent> prepareEvents(ContextImplPtr Context);

  bool MUseExclusiveQueue = false;

  // Private interface. Derived classes should implement this method.
  virtual cl_int enqueueImp() = 0;

public:
  std::vector<DepDesc> MDeps;
  std::vector<Command *> MUsers;

private:
  CommandType MType;
  std::atomic<bool> MEnqueued;
};

// The command does nothing during enqueue. The task can be used to implement
// lock in the graph, or to merge several nodes into one.
class EmptyCommand : public Command {
public:
  EmptyCommand(QueueImplPtr Queue, Requirement *Req)
      : Command(CommandType::EMPTY_TASK, std::move(Queue)),
        MStoredRequirement(*Req) {}

  Requirement *getStoredRequirement() { return &MStoredRequirement; }

private:
  cl_int enqueueImp() override { return CL_SUCCESS; }
  void printDot(std::ostream &Stream) const override;

  Requirement MStoredRequirement;
};

// The command enqueues release instance of memory allocated on Host or
// underlying framework.
class ReleaseCommand : public Command {
public:
  ReleaseCommand(QueueImplPtr Queue, AllocaCommandBase *AllocaCmd)
      : Command(CommandType::RELEASE, std::move(Queue)), MAllocaCmd(AllocaCmd) {
  }

  void printDot(std::ostream &Stream) const override;

private:
  cl_int enqueueImp() override;

  AllocaCommandBase *MAllocaCmd = nullptr;
};

class AllocaCommandBase : public Command {
public:
  AllocaCommandBase(CommandType Type, QueueImplPtr Queue, Requirement Req)
      : Command(Type, Queue), MReleaseCmd(Queue, this), MReq(std::move(Req)) {
    MReq.MAccessMode = access::mode::read_write;
  }

  ReleaseCommand *getReleaseCmd() { return &MReleaseCmd; }

  SYCLMemObjI *getSYCLMemObj() const { return MReq.MSYCLMemObj; }

  void *getMemAllocation() const { return MMemAllocation; }

  Requirement *getAllocationReq() { return &MReq; }

protected:
  ReleaseCommand MReleaseCmd;
  Requirement MReq;
  void *MMemAllocation = nullptr;
};

// The command enqueues allocation of instance of memory object on Host or
// underlying framework.
class AllocaCommand : public AllocaCommandBase {
public:
  AllocaCommand(QueueImplPtr Queue, Requirement Req,
                bool InitFromUserData = true)
      : AllocaCommandBase(CommandType::ALLOCA, std::move(Queue), Req),
        MInitFromUserData(InitFromUserData) {
    addDep(DepDesc(nullptr, &MReq, this));
  }

  void printDot(std::ostream &Stream) const override;

private:
  cl_int enqueueImp() override final;

  bool MInitFromUserData = false;
};

class AllocaSubBufCommand : public AllocaCommandBase {
public:
  AllocaSubBufCommand(QueueImplPtr Queue, Requirement Req,
                      AllocaCommandBase *ParentAlloca)
      : AllocaCommandBase(CommandType::ALLOCA_SUB_BUF, std::move(Queue),
                          std::move(Req)),
        MParentAlloca(ParentAlloca) {
    addDep(DepDesc(MParentAlloca, &MReq, MParentAlloca));
  }

  void printDot(std::ostream &Stream) const override;
  AllocaCommandBase *getParentAlloca() { return MParentAlloca; }

private:
  cl_int enqueueImp() override final;

  AllocaCommandBase *MParentAlloca;
};

class MapMemObject : public Command {
public:
  MapMemObject(Requirement SrcReq, AllocaCommandBase *SrcAlloca,
               Requirement *DstAcc, QueueImplPtr Queue);

  Requirement MSrcReq;
  AllocaCommandBase *MSrcAlloca = nullptr;
  Requirement *MDstAcc = nullptr;
  Requirement MDstReq;

  void printDot(std::ostream &Stream) const override;

private:
  cl_int enqueueImp() override;
};

class UnMapMemObject : public Command {
public:
  UnMapMemObject(Requirement SrcReq, AllocaCommandBase *SrcAlloca,
                 Requirement *DstAcc, QueueImplPtr Queue,
                 bool UseExclusiveQueue = false);

  void printDot(std::ostream &Stream) const override;

private:
  cl_int enqueueImp() override;

  Requirement MSrcReq;
  AllocaCommandBase *MSrcAlloca = nullptr;
  Requirement *MDstAcc = nullptr;
};

// The command enqueues memory copy between two instances of memory object.
class MemCpyCommand : public Command {
public:
  MemCpyCommand(Requirement SrcReq, AllocaCommandBase *SrcAlloca,
                Requirement DstReq, AllocaCommandBase *DstAlloca,
                QueueImplPtr SrcQueue, QueueImplPtr DstQueue,
                bool UseExclusiveQueue = false);

  QueueImplPtr MSrcQueue;
  Requirement MSrcReq;
  AllocaCommandBase *MSrcAlloca = nullptr;
  Requirement MDstReq;
  AllocaCommandBase *MDstAlloca = nullptr;
  Requirement *MAccToUpdate = nullptr;

  void setAccessorToUpdate(Requirement *AccToUpdate) {
    MAccToUpdate = AccToUpdate;
  }

  void printDot(std::ostream &Stream) const override;

private:
  cl_int enqueueImp() override;
};

// The command enqueues memory copy between two instances of memory object.
class MemCpyCommandHost : public Command {
public:
  MemCpyCommandHost(Requirement SrcReq, AllocaCommandBase *SrcAlloca,
                    Requirement *DstAcc, QueueImplPtr SrcQueue,
                    QueueImplPtr DstQueue);

  QueueImplPtr MSrcQueue;
  Requirement MSrcReq;
  AllocaCommandBase *MSrcAlloca = nullptr;
  Requirement MDstReq;
  Requirement *MDstAcc = nullptr;

  void printDot(std::ostream &Stream) const override;

private:
  cl_int enqueueImp() override;
};

// The command enqueues execution of kernel or explicit memory operation.
class ExecCGCommand : public Command {
public:
  ExecCGCommand(std::unique_ptr<detail::CG> CommandGroup, QueueImplPtr Queue)
      : Command(CommandType::RUN_CG, std::move(Queue)),
        MCommandGroup(std::move(CommandGroup)) {}

  void flushStreams();

  void printDot(std::ostream &Stream) const override;

private:
  // Implementation of enqueueing of ExecCGCommand.
  cl_int enqueueImp() override;

  AllocaCommandBase *getAllocaForReq(Requirement *Req);

  std::unique_ptr<detail::CG> MCommandGroup;
};

class UpdateHostRequirementCommand : public Command {
public:
  UpdateHostRequirementCommand(QueueImplPtr Queue, Requirement *Req,
                               AllocaCommandBase *AllocaForReq)
      : Command(CommandType::UPDATE_REQUIREMENT, std::move(Queue)),
        MReqToUpdate(Req), MAllocaForReq(AllocaForReq),
        MStoredRequirement(*Req) {}

  Requirement *getStoredRequirement() { return &MStoredRequirement; }

private:
  cl_int enqueueImp() override;
  void printDot(std::ostream &Stream) const override;

  Requirement *MReqToUpdate = nullptr;
  AllocaCommandBase *MAllocaForReq = nullptr;
  Requirement MStoredRequirement;
};

} // namespace detail
} // namespace sycl
} // namespace cl
