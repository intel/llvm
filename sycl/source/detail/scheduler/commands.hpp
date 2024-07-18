//==-------------- commands.hpp - SYCL standard header file ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <atomic>
#include <cstdint>
#include <deque>
#include <memory>
#include <optional>
#include <set>
#include <unordered_set>
#include <vector>

#include <detail/accessor_impl.hpp>
#include <detail/cg.hpp>
#include <detail/event_impl.hpp>
#include <detail/program_manager/program_manager.hpp>
#include <sycl/access/access.hpp>

namespace sycl {
inline namespace _V1 {

namespace ext::oneapi::experimental::detail {
class exec_graph_impl;
class node_impl;
} // namespace ext::oneapi::experimental::detail
namespace detail {

#ifdef XPTI_ENABLE_INSTRUMENTATION
void emitInstrumentationGeneral(uint32_t StreamID, uint64_t InstanceID,
                                xpti_td *TraceEvent, uint16_t Type,
                                const void *Addr);
#endif

class queue_impl;
class event_impl;
class context_impl;
class DispatchHostTask;

using QueueImplPtr = std::shared_ptr<detail::queue_impl>;
using EventImplPtr = std::shared_ptr<detail::event_impl>;
using ContextImplPtr = std::shared_ptr<detail::context_impl>;
using StreamImplPtr = std::shared_ptr<detail::stream_impl>;

class Command;
class AllocaCommand;
class AllocaCommandBase;
class ReleaseCommand;
class ExecCGCommand;
class EmptyCommand;

enum BlockingT { NON_BLOCKING = 0, BLOCKING };

/// Result of command enqueueing.
struct EnqueueResultT {
  enum ResultT {
    SyclEnqueueReady,
    SyclEnqueueSuccess,
    SyclEnqueueBlocked,
    SyclEnqueueFailed
  };
  EnqueueResultT(ResultT Result = SyclEnqueueSuccess, Command *Cmd = nullptr,
                 pi_int32 ErrCode = PI_SUCCESS)
      : MResult(Result), MCmd(Cmd), MErrCode(ErrCode) {}
  /// Indicates the result of enqueueing.
  ResultT MResult;
  /// Pointer to the command which failed to enqueue.
  Command *MCmd;
  /// Error code which is set when enqueueing fails.
  pi_int32 MErrCode;
};

/// Dependency between two commands.
struct DepDesc {
  DepDesc(Command *DepCommand, const Requirement *Req,
          AllocaCommandBase *AllocaCmd)
      : MDepCommand(DepCommand), MDepRequirement(Req), MAllocaCmd(AllocaCmd) {}

  friend bool operator<(const DepDesc &Lhs, const DepDesc &Rhs) {
    return std::tie(Lhs.MDepRequirement, Lhs.MDepCommand) <
           std::tie(Rhs.MDepRequirement, Rhs.MDepCommand);
  }

  /// The actual dependency command.
  Command *MDepCommand = nullptr;
  /// Requirement for the dependency.
  const Requirement *MDepRequirement = nullptr;
  /// Allocation command for the memory object we have requirement for.
  /// Used to simplify searching for memory handle.
  AllocaCommandBase *MAllocaCmd = nullptr;
};

/// The Command class represents some action that needs to be performed on one
/// or more memory objects. The Command has a vector of DepDesc objects that
/// represent dependencies of the command. It has a vector of pointers to
/// commands that depend on the command. It has a pointer to a \ref queue object
/// and an event that is associated with the command.
///
/// \ingroup sycl_graph
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
    EMPTY_TASK,
    HOST_TASK,
    FUSION,
    EXEC_CMD_BUFFER,
    UPDATE_CMD_BUFFER
  };

  Command(CommandType Type, QueueImplPtr Queue,
          sycl::detail::pi::PiExtCommandBuffer CommandBuffer = nullptr,
          const std::vector<sycl::detail::pi::PiExtSyncPoint> &SyncPoints = {});

  /// \param NewDep dependency to be added
  /// \param ToCleanUp container for commands that can be cleaned up.
  /// \return an optional connection cmd to enqueue
  [[nodiscard]] Command *addDep(DepDesc NewDep,
                                std::vector<Command *> &ToCleanUp);

  /// \param NewDep dependency to be added
  /// \param ToCleanUp container for commands that can be cleaned up.
  /// \return an optional connection cmd to enqueue
  [[nodiscard]] Command *addDep(EventImplPtr Event,
                                std::vector<Command *> &ToCleanUp);

  void addUser(Command *NewUser) { MUsers.insert(NewUser); }

  /// \return type of the command, e.g. Allocate, MemoryCopy.
  CommandType getType() const { return MType; }

  /// Checks if the command is enqueued, and calls enqueueImp.
  ///
  /// \param EnqueueResult is set to the specific status if enqueue failed.
  /// \param Blocking if this argument is true, function will wait for the
  ///        command to be unblocked before calling enqueueImp.
  /// \param ToCleanUp container for commands that can be cleaned up.
  /// \return true if the command is enqueued.
  virtual bool enqueue(EnqueueResultT &EnqueueResult, BlockingT Blocking,
                       std::vector<Command *> &ToCleanUp);

  bool isFinished();

  bool isSuccessfullyEnqueued() const {
    return MEnqueueStatus == EnqueueResultT::SyclEnqueueSuccess;
  }

  // Shows that command could not be enqueued, now it may be true for empty task
  // only
  bool isEnqueueBlocked() const {
    return MIsBlockable && MEnqueueStatus == EnqueueResultT::SyclEnqueueBlocked;
  }
  // Shows that command could be enqueued, but is blocking enqueue of all
  // commands depending on it. Regular usage - host task.
  bool isBlocking() const { return isHostTask() && !MEvent->isCompleted(); }

  void addBlockedUserUnique(const EventImplPtr &NewUser) {
    if (std::find(MBlockedUsers.begin(), MBlockedUsers.end(), NewUser) !=
        MBlockedUsers.end())
      return;
    MBlockedUsers.push_back(NewUser);
  }

  const QueueImplPtr &getQueue() const { return MQueue; }

  const EventImplPtr &getEvent() const { return MEvent; }

  // Methods needed to support SYCL instrumentation

  /// Proxy method which calls emitInstrumentationData.
  void emitInstrumentationDataProxy();
  /// Instrumentation method which emits telemetry data.
  virtual void emitInstrumentationData() = 0;
  /// Looks at all the dependencies for the release command and enables
  /// instrumentation to report these dependencies as edges.
  void resolveReleaseDependencies(std::set<Command *> &list);
  /// Creates an edge event when the dependency is a command.
  void emitEdgeEventForCommandDependence(
      Command *Cmd, void *ObjAddr, bool IsCommand,
      std::optional<access::mode> AccMode = std::nullopt);
  /// Creates an edge event when the dependency is an event.
  void emitEdgeEventForEventDependence(Command *Cmd,
                                       sycl::detail::pi::PiEvent &EventAddr);
  /// Creates a signal event with the enqueued kernel event handle.
  void emitEnqueuedEventSignal(sycl::detail::pi::PiEvent &PiEventAddr);
  /// Create a trace event of node_create type; this must be guarded by a
  /// check for xptiTraceEnabled().
  /// Post Condition: MTraceEvent will be set to the event created.
  /// \param MAddress  The address to use to create the payload.
  uint64_t makeTraceEventProlog(void *MAddress);
  /// If prolog has been run, run epilog; this must be guarded by a check for
  /// xptiTraceEnabled().
  void makeTraceEventEpilog();
  /// Emits an event of Type.
  void emitInstrumentation(uint16_t Type, const char *Txt = nullptr);

  // End Methods needed to support SYCL instrumentation

  virtual void printDot(std::ostream &Stream) const = 0;

  virtual const Requirement *getRequirement() const {
    assert(false && "Internal Error. The command has no stored requirement");
    return nullptr;
  }

  virtual ~Command() { MEvent->cleanDepEventsThroughOneLevel(); }

  const char *getBlockReason() const;

  /// Get the context of the queue this command will be submitted to. Could
  /// differ from the context of MQueue for memory copy commands.
  virtual ContextImplPtr getWorkerContext() const;

  /// Returns true iff the command produces a PI event on non-host devices.
  virtual bool producesPiEvent() const;

  /// Returns true iff this command can be freed by post enqueue cleanup.
  virtual bool supportsPostEnqueueCleanup() const;

  /// Returns true iff this command is ready to be submitted for cleanup.
  virtual bool readyForCleanup() const;

  /// Collect PI events from EventImpls and filter out some of them in case of
  /// in order queue
  std::vector<sycl::detail::pi::PiEvent>
  getPiEvents(const std::vector<EventImplPtr> &EventImpls) const;
  /// Collect PI events from EventImpls and filter out some of them in case of
  /// in order queue. Does blocking enqueue if event is expected to produce pi
  /// event but has empty native handle.
  std::vector<sycl::detail::pi::PiEvent>
  getPiEventsBlocking(const std::vector<EventImplPtr> &EventImpls) const;

  bool isHostTask() const;

  bool isFusable() const;

protected:
  QueueImplPtr MQueue;
  EventImplPtr MEvent;
  QueueImplPtr MWorkerQueue;

  /// Dependency events prepared for waiting by backend.
  /// See processDepEvent for details.
  std::vector<EventImplPtr> &MPreparedDepsEvents;
  std::vector<EventImplPtr> &MPreparedHostDepsEvents;

  void waitForEvents(QueueImplPtr Queue, std::vector<EventImplPtr> &RawEvents,
                     sycl::detail::pi::PiEvent &Event);

  void waitForPreparedHostEvents() const;

  /// Perform glueing of events from different contexts
  /// \param DepEvent event this commands should depend on
  /// \param Dep optional DepDesc to perform connection of events properly
  /// \param ToCleanUp container for commands that can be cleaned up.
  /// \return returns an optional connection command to enqueue
  ///
  /// Glueing (i.e. connecting) will be performed if and only if DepEvent is
  /// not from host context and its context doesn't match to context of this
  /// command. Context of this command is fetched via getWorkerContext().
  ///
  /// Optionality of Dep is set by Dep.MDepCommand not equal to nullptr.
  [[nodiscard]] Command *processDepEvent(EventImplPtr DepEvent,
                                         const DepDesc &Dep,
                                         std::vector<Command *> &ToCleanUp);

  /// Private interface. Derived classes should implement this method.
  virtual pi_int32 enqueueImp() = 0;

  /// The type of the command.
  CommandType MType;
  /// Mutex used to protect enqueueing from race conditions
  std::mutex MEnqueueMtx;

  friend class DispatchHostTask;

public:
  const std::vector<EventImplPtr> &getPreparedHostDepsEvents() const {
    return MPreparedHostDepsEvents;
  }

  const std::vector<EventImplPtr> &getPreparedDepsEvents() const {
    return MPreparedDepsEvents;
  }

  // XPTI instrumentation. Copy code location details to the internal struct.
  // Memory is allocated in this method and released in destructor.
  void copySubmissionCodeLocation();

  /// Clear all dependency events This should only be used if a command is about
  /// to be deleted without being executed before that. As of now, the only
  /// valid use case for this function is in kernel fusion, where the fused
  /// kernel commands are replaced by the fused command without ever being
  /// executed.
  void clearAllDependencies() {
    MPreparedDepsEvents.clear();
    MPreparedHostDepsEvents.clear();
    MDeps.clear();
  }

  /// Contains list of dependencies(edges)
  std::vector<DepDesc> MDeps;
  /// Contains list of commands that depend on the command.
  std::unordered_set<Command *> MUsers;
  /// Indicates whether the command can be blocked from enqueueing.
  bool MIsBlockable = false;
  /// Counts the number of memory objects this command is a leaf for.
  unsigned MLeafCounter = 0;

  struct Marks {
    /// Used for marking the node as visited during graph traversal.
    bool MVisited = false;
    /// Used for marking the node for deletion during cleanup.
    bool MToBeDeleted = false;
  };
  /// Used for marking the node during graph traversal.
  Marks MMarks;

  enum class BlockReason : int { HostAccessor = 0, HostTask };

  // Only have reasonable value while MIsBlockable is true
  BlockReason MBlockReason;

  /// Describes the status of the command.
  std::atomic<EnqueueResultT::ResultT> MEnqueueStatus;

  // All member variables defined here are needed for the SYCL instrumentation
  // layer. Do not guard these variables below with XPTI_ENABLE_INSTRUMENTATION
  // to ensure we have the same object layout when the macro in the library and
  // SYCL app are not the same.

  /// The event for node_create and task_begin.
  void *MTraceEvent = nullptr;
  /// The stream under which the traces are emitted.
  ///
  /// Stream ids are positive integers and we set it to an invalid value.
  int32_t MStreamID = -1;
  /// Reserved for storing the object address such as SPIR-V or memory object
  /// address.
  void *MAddress = nullptr;
  /// Buffer to build the address string.
  std::string MAddressString;
  /// Buffer to build the command node type.
  std::string MCommandNodeType;
  /// Buffer to build the command end-user understandable name.
  std::string MCommandName;
  /// Flag to indicate if makeTraceEventProlog() has been run.
  bool MTraceEventPrologComplete = false;
  /// Flag to indicate if this is the first time we are seeing this payload.
  bool MFirstInstance = false;
  /// Instance ID tracked for the command.
  uint64_t MInstanceID = 0;
  /// Represents code location of command submission to SYCL API, assigned with
  /// the valid value only if command execution is async (host task) or delayed
  /// (blocked by host task).
  code_location MSubmissionCodeLocation;
  /// Introduces string to handle memory management since code_location struct
  /// works with raw char arrays.
  std::string MSubmissionFileName;
  std::string MSubmissionFunctionName;

  // This flag allows to control whether event should be set complete
  // after successfull enqueue of command. Event is considered as "host" event
  // if there is no backend representation of event (i.e. getHandleRef() return
  // reference to nullptr value). By default the flag is set to true due to most
  // of host operations are synchronous. The only asynchronous operation
  // currently is host-task.
  bool MShouldCompleteEventIfPossible = true;

  /// Indicates that the node will be freed by graph cleanup. Such nodes should
  /// be ignored by other cleanup mechanisms (e.g. during memory object
  /// removal).
  bool MMarkedForCleanup = false;

  /// Contains list of commands that depends on the host command explicitly (by
  /// depends_on). Not involved in the cleanup process since it is one-way link
  /// and does not hold resources.
  /// Using EventImplPtr since enqueueUnblockedCommands and event.wait may
  /// intersect with command enqueue.
  std::vector<EventImplPtr> MBlockedUsers;
  std::mutex MBlockedUsersMutex;

protected:
  /// Gets the command buffer (if any) associated with this command.
  sycl::detail::pi::PiExtCommandBuffer getCommandBuffer() const {
    return MCommandBuffer;
  }

  /// CommandBuffer which will be used to submit to instead of the queue, if
  /// set.
  sycl::detail::pi::PiExtCommandBuffer MCommandBuffer;
  /// List of sync points for submissions to a command buffer.
  std::vector<sycl::detail::pi::PiExtSyncPoint> MSyncPointDeps;
};

/// The empty command does nothing during enqueue. The task can be used to
/// implement lock in the graph, or to merge several nodes into one.
class EmptyCommand : public Command {
public:
  EmptyCommand();

  void printDot(std::ostream &Stream) const final;
  const Requirement *getRequirement() const final { return &MRequirements[0]; }
  void addRequirement(Command *DepCmd, AllocaCommandBase *AllocaCmd,
                      const Requirement *Req);

  void emitInstrumentationData() override;

  bool producesPiEvent() const final;

private:
  pi_int32 enqueueImp() final;

  // Employing deque here as it allows to push_back/emplace_back without
  // invalidation of pointer or reference to stored data item regardless of
  // iterator invalidation.
  std::deque<Requirement> MRequirements;
};

/// The release command enqueues release of a memory object instance allocated
/// on Host or underlying framework.
class ReleaseCommand : public Command {
public:
  ReleaseCommand(QueueImplPtr Queue, AllocaCommandBase *AllocaCmd);

  void printDot(std::ostream &Stream) const final;
  void emitInstrumentationData() override;
  bool producesPiEvent() const final;
  bool supportsPostEnqueueCleanup() const final;
  bool readyForCleanup() const final;

private:
  pi_int32 enqueueImp() final;

  /// Command which allocates memory release command should dealocate.
  AllocaCommandBase *MAllocaCmd = nullptr;
};

/// Base class for memory allocation commands.
class AllocaCommandBase : public Command {
public:
  AllocaCommandBase(CommandType Type, QueueImplPtr Queue, Requirement Req,
                    AllocaCommandBase *LinkedAllocaCmd, bool IsConst);

  ReleaseCommand *getReleaseCmd() { return &MReleaseCmd; }

  SYCLMemObjI *getSYCLMemObj() const { return MRequirement.MSYCLMemObj; }

  virtual void *getMemAllocation() const = 0;

  const Requirement *getRequirement() const final { return &MRequirement; }

  void emitInstrumentationData() override;

  bool producesPiEvent() const final;

  bool supportsPostEnqueueCleanup() const final;

  bool readyForCleanup() const final;

  void *MMemAllocation = nullptr;

  /// Alloca command linked with current command.
  /// Device and host alloca commands can be linked, so they may share the same
  /// memory. Only one allocation from a pair can be accessed at a time. Alloca
  /// commands associated with such allocation is "active". In order to switch
  /// "active" status between alloca commands map/unmap operations are used.
  AllocaCommandBase *MLinkedAllocaCmd = nullptr;
  /// Indicates that current alloca is active one.
  bool MIsActive = true;

  /// Indicates that the command owns memory allocation in case of connected
  /// alloca command.
  bool MIsLeaderAlloca = true;
  // Indicates that the data in this allocation must not be modified
  bool MIsConst = false;

protected:
  Requirement MRequirement;
  ReleaseCommand MReleaseCmd;
};

/// The alloca command enqueues allocation of instance of memory object on Host
/// or underlying framework.
class AllocaCommand : public AllocaCommandBase {
public:
  AllocaCommand(QueueImplPtr Queue, Requirement Req,
                bool InitFromUserData = true,
                AllocaCommandBase *LinkedAllocaCmd = nullptr,
                bool IsConst = false);

  void *getMemAllocation() const final { return MMemAllocation; }
  void printDot(std::ostream &Stream) const final;
  void emitInstrumentationData() override;

private:
  pi_int32 enqueueImp() final;

  /// The flag indicates that alloca should try to reuse pointer provided by
  /// the user during memory object construction.
  bool MInitFromUserData = false;
};

/// The AllocaSubBuf command enqueues creation of sub-buffer of memory object.
class AllocaSubBufCommand : public AllocaCommandBase {
public:
  AllocaSubBufCommand(QueueImplPtr Queue, Requirement Req,
                      AllocaCommandBase *ParentAlloca,
                      std::vector<Command *> &ToEnqueue,
                      std::vector<Command *> &ToCleanUp);

  void *getMemAllocation() const final;
  void printDot(std::ostream &Stream) const final;
  AllocaCommandBase *getParentAlloca() { return MParentAlloca; }
  void emitInstrumentationData() override;

private:
  pi_int32 enqueueImp() final;

  AllocaCommandBase *MParentAlloca = nullptr;
};

/// The map command enqueues mapping of device memory onto host memory.
class MapMemObject : public Command {
public:
  MapMemObject(AllocaCommandBase *SrcAllocaCmd, Requirement Req, void **DstPtr,
               QueueImplPtr Queue, access::mode MapMode);

  void printDot(std::ostream &Stream) const final;
  const Requirement *getRequirement() const final { return &MSrcReq; }
  void emitInstrumentationData() override;

private:
  pi_int32 enqueueImp() final;

  AllocaCommandBase *MSrcAllocaCmd = nullptr;
  Requirement MSrcReq;
  void **MDstPtr = nullptr;
  access::mode MMapMode;
};

/// The unmap command removes mapping of host memory onto device memory.
class UnMapMemObject : public Command {
public:
  UnMapMemObject(AllocaCommandBase *DstAllocaCmd, Requirement Req,
                 void **SrcPtr, QueueImplPtr Queue);

  void printDot(std::ostream &Stream) const final;
  const Requirement *getRequirement() const final { return &MDstReq; }
  void emitInstrumentationData() override;
  bool producesPiEvent() const final;

private:
  pi_int32 enqueueImp() final;

  AllocaCommandBase *MDstAllocaCmd = nullptr;
  Requirement MDstReq;
  void **MSrcPtr = nullptr;
};

/// The mem copy command enqueues memory copy between two instances of memory
/// object.
class MemCpyCommand : public Command {
public:
  MemCpyCommand(Requirement SrcReq, AllocaCommandBase *SrcAllocaCmd,
                Requirement DstReq, AllocaCommandBase *DstAllocaCmd,
                QueueImplPtr SrcQueue, QueueImplPtr DstQueue);

  void printDot(std::ostream &Stream) const final;
  const Requirement *getRequirement() const final { return &MDstReq; }
  void emitInstrumentationData() final;
  ContextImplPtr getWorkerContext() const final;
  bool producesPiEvent() const final;

private:
  pi_int32 enqueueImp() final;

  QueueImplPtr MSrcQueue;
  Requirement MSrcReq;
  AllocaCommandBase *MSrcAllocaCmd = nullptr;
  Requirement MDstReq;
  AllocaCommandBase *MDstAllocaCmd = nullptr;
};

/// The mem copy host command enqueues memory copy between two instances of
/// memory object.
class MemCpyCommandHost : public Command {
public:
  MemCpyCommandHost(Requirement SrcReq, AllocaCommandBase *SrcAllocaCmd,
                    Requirement DstReq, void **DstPtr, QueueImplPtr SrcQueue,
                    QueueImplPtr DstQueue);

  void printDot(std::ostream &Stream) const final;
  const Requirement *getRequirement() const final { return &MDstReq; }
  void emitInstrumentationData() final;
  ContextImplPtr getWorkerContext() const final;

private:
  pi_int32 enqueueImp() final;

  QueueImplPtr MSrcQueue;
  Requirement MSrcReq;
  AllocaCommandBase *MSrcAllocaCmd = nullptr;
  Requirement MDstReq;
  void **MDstPtr = nullptr;
};

pi_int32
enqueueReadWriteHostPipe(const QueueImplPtr &Queue, const std::string &PipeName,
                         bool blocking, void *ptr, size_t size,
                         std::vector<sycl::detail::pi::PiEvent> &RawEvents,
                         const detail::EventImplPtr &OutEventImpl, bool read);

void enqueueImpKernel(
    const QueueImplPtr &Queue, NDRDescT &NDRDesc, std::vector<ArgDesc> &Args,
    const std::shared_ptr<detail::kernel_bundle_impl> &KernelBundleImplPtr,
    const std::shared_ptr<detail::kernel_impl> &MSyclKernel,
    const std::string &KernelName,
    std::vector<sycl::detail::pi::PiEvent> &RawEvents,
    const detail::EventImplPtr &Event,
    const std::function<void *(Requirement *Req)> &getMemAllocationFunc,
    sycl::detail::pi::PiKernelCacheConfig KernelCacheConfig,
    bool KernelIsCooperative, const bool KernelUsesClusterLaunch);

class KernelFusionCommand;

/// The exec CG command enqueues execution of kernel or explicit memory
/// operation.
class ExecCGCommand : public Command {
public:
  ExecCGCommand(
      std::unique_ptr<detail::CG> CommandGroup, QueueImplPtr Queue,
      bool EventNeeded,
      sycl::detail::pi::PiExtCommandBuffer CommandBuffer = nullptr,
      const std::vector<sycl::detail::pi::PiExtSyncPoint> &Dependencies = {});

  std::vector<std::shared_ptr<const void>> getAuxiliaryResources() const;

  void clearAuxiliaryResources();

  void printDot(std::ostream &Stream) const final;
  void emitInstrumentationData() final;
  std::string_view getTypeString() const;

  detail::CG &getCG() const { return *MCommandGroup; }

  // MEmptyCmd is only employed if this command refers to host-task.
  // The mechanism of lookup for single EmptyCommand amongst users of
  // host-task-representing command is unreliable. This unreliability roots in
  // the cleanup process.
  EmptyCommand *MEmptyCmd = nullptr;

  // MFusionCommand is employed to mark a CG command as part of a kernel fusion
  // and allows to refer back to the corresponding KernelFusionCommand if
  // necessary.
  KernelFusionCommand *MFusionCmd = nullptr;

  // MEventNeeded is true if the command needs to produce a valid event. The
  // implementation may elect to not produce events (native or SYCL) if this
  // is false.
  bool MEventNeeded = true;

  bool producesPiEvent() const final;

  bool supportsPostEnqueueCleanup() const final;

  bool readyForCleanup() const final;

private:
  pi_int32 enqueueImp() final;
  pi_int32 enqueueImpCommandBuffer();
  pi_int32 enqueueImpQueue();

  AllocaCommandBase *getAllocaForReq(Requirement *Req);

  std::unique_ptr<detail::CG> MCommandGroup;

  friend class Command;
};

// For XPTI instrumentation only.
// Method used to emit data in cases when we do not create node in graph.
// Very close to ExecCGCommand::emitInstrumentationData content.
#ifdef XPTI_ENABLE_INSTRUMENTATION
std::pair<xpti_td *, uint64_t> emitKernelInstrumentationData(
    int32_t StreamID, const std::shared_ptr<detail::kernel_impl> &SyclKernel,
    const detail::code_location &CodeLoc, const std::string &SyclKernelName,
    const QueueImplPtr &Queue, const NDRDescT &NDRDesc,
    const std::shared_ptr<detail::kernel_bundle_impl> &KernelBundleImplPtr,
    std::vector<ArgDesc> &CGArgs);
#endif

class UpdateHostRequirementCommand : public Command {
public:
  UpdateHostRequirementCommand(QueueImplPtr Queue, Requirement Req,
                               AllocaCommandBase *SrcAllocaCmd, void **DstPtr);

  void printDot(std::ostream &Stream) const final;
  const Requirement *getRequirement() const final { return &MDstReq; }
  void emitInstrumentationData() final;

private:
  pi_int32 enqueueImp() final;

  AllocaCommandBase *MSrcAllocaCmd = nullptr;
  Requirement MDstReq;
  void **MDstPtr = nullptr;
};

/// The KernelFusionCommand is placed in the execution graph together with the
/// individual kernels of the fusion list to control kernel fusion.
class KernelFusionCommand : public Command {
public:
  enum class FusionStatus { ACTIVE, CANCELLED, COMPLETE, DELETED };

  explicit KernelFusionCommand(QueueImplPtr Queue);

  void printDot(std::ostream &Stream) const final;
  void emitInstrumentationData() final;
  bool producesPiEvent() const final;

  std::vector<Command *> &auxiliaryCommands();

  void addToFusionList(ExecCGCommand *Kernel);

  std::vector<ExecCGCommand *> &getFusionList();

  ///
  /// Set the status of this fusion command to \p Status. This function should
  /// only be called under the protection of the scheduler write-lock.
  void setFusionStatus(FusionStatus Status);

  /// Reset the queue. This can be required as the command is held in order
  /// to maintain events alive, however this prevent the normal destruction of
  /// the queue.
  void resetQueue();

  bool isActive() const { return MStatus == FusionStatus::ACTIVE; }

  bool readyForDeletion() const { return MStatus == FusionStatus::DELETED; }

private:
  pi_int32 enqueueImp() final;

  std::vector<ExecCGCommand *> MFusionList;

  std::vector<Command *> MAuxiliaryCommands;

  FusionStatus MStatus;
};

class UpdateCommandBufferCommand : public Command {
public:
  explicit UpdateCommandBufferCommand(
      QueueImplPtr Queue,
      ext::oneapi::experimental::detail::exec_graph_impl *Graph,
      std::vector<std::shared_ptr<ext::oneapi::experimental::detail::node_impl>>
          Nodes);

  void printDot(std::ostream &Stream) const final;
  void emitInstrumentationData() final;
  bool producesPiEvent() const final;

private:
  pi_int32 enqueueImp() final;

  ext::oneapi::experimental::detail::exec_graph_impl *MGraph;
  std::vector<std::shared_ptr<ext::oneapi::experimental::detail::node_impl>>
      MNodes;
};

// Enqueues a given kernel to a PiExtCommandBuffer
pi_int32 enqueueImpCommandBufferKernel(
    context Ctx, DeviceImplPtr DeviceImpl,
    sycl::detail::pi::PiExtCommandBuffer CommandBuffer,
    const CGExecKernel &CommandGroup,
    std::vector<sycl::detail::pi::PiExtSyncPoint> &SyncPoints,
    sycl::detail::pi::PiExtSyncPoint *OutSyncPoint,
    sycl::detail::pi::PiExtCommandBufferCommand *OutCommand,
    const std::function<void *(Requirement *Req)> &getMemAllocationFunc);

// Sets arguments for a given kernel and device based on the argument type.
// Refactored from SetKernelParamsAndLaunch to allow it to be used in the graphs
// extension.
void SetArgBasedOnType(
    const detail::plugin &Plugin, sycl::detail::pi::PiKernel Kernel,
    const std::shared_ptr<device_image_impl> &DeviceImageImpl,
    const std::function<void *(Requirement *Req)> &getMemAllocationFunc,
    const sycl::context &Context, detail::ArgDesc &Arg, size_t NextTrueIndex);

void applyFuncOnFilteredArgs(
    const KernelArgMask *EliminatedArgMask, std::vector<ArgDesc> &Args,
    std::function<void(detail::ArgDesc &Arg, int NextTrueIndex)> Func);

void ReverseRangeDimensionsForKernel(NDRDescT &NDR);

} // namespace detail
} // namespace _V1
} // namespace sycl
