//==-------------- scheduler.hpp - SYCL standard header file ---------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <detail/scheduler/commands.hpp>
#include <detail/scheduler/leaves_collection.hpp>
#include <detail/sycl_mem_obj_i.hpp>
#include <sycl/detail/cg.hpp>

#include <cstddef>
#include <memory>
#include <queue>
#include <set>
#include <shared_mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

/// \defgroup sycl_graph DPC++ Execution Graph
///
/// SYCL, unlike OpenCL, provides a programming model in which the user doesn't
/// need to manage dependencies between kernels and memory explicitly. The DPC++
/// Runtime must ensure correct execution with respect to the order commands are
/// submitted.
///
/// This document describes the part of the DPC++ Runtime that is responsible
/// for building and processing dependency graph.
///
/// ## A couple of words about DPC++ and SYCL execution and memory model
///
/// The SYCL framework defines command group (\ref CG) as an entity that
/// represents minimal execution block. The command group is submitted to SYCL
/// queue and consists of a kernel or an explicit memory operation, and their
/// requirements. The SYCL queue defines the device and context using which the
/// kernel should be executed.
///
/// The commands that contain explicit memory operations include copy, fill,
/// update_host and other operations. It's up to implementation how to define
/// these operations.
///
/// The relative order of command groups submission defines the order in which
/// kernels must be executed if their memory requirements intersect. For
/// example, if a command group A writes to a buffer X, command group B reads
/// from X, then the scheduled execution order of A and B will be the same as
/// their dynamic submission order (matches program order if submitted from the
/// same host thread).
///
/// Memory requirements are requests to SYCL memory objects, such as buffer and
/// image. SYCL memory objects are not bound to any specific context or device,
/// it's SYCL responsibility to allocate and/or copy memory to the target
/// context to achieve correct execution.
///
/// Refer to SYCL Specification 1.2.1 sections 3.4 and 3.5 to find more
/// information about SYCL execution and memory model.
///
/// ### Example of DPC++ application
///
/// \code{.cpp}
/// {
///   // Creating SYCL CPU and GPU queues
///   sycl::queue CPU_Queue = ...;
///   sycl::queue GPU_Queue = ...;
///
///   // Creating 3 SYCL buffers
///   auto BufferA = ...; // Buffer is initialized with host memory.
///   auto BufferB = ...;
///   auto BufferC = ...;
///
///   // "Copy command group" section
///   // Request processing explicit copy operation on CPU
///   // The copy operation reads from BufferA and writes to BufferB
///
///   CPU_Queue.submit([&](handler &CGH) {
///     auto A = BufferA.get_access<read>(CGH);
///     auto B = BufferB.get_access<write>(CGH);
///     CGH.copy(A, B);
///   });
///
///   // "Multi command group" section
///   // Request processing multi kernel on GPU
///   // The kernel reads from BufferB, multiplies by 4 and writes result to
///   // BufferC
///
///   GPU_Queue.submit([&](handler &CGH) {
///     auto B = BufferB.get_access<read>(CGH);
///     auto C = BufferC.get_access<write>(CGH);
///     CGH.parallel_for<class multi>(range<1>{N}, [=](id<1> Index) {
///       C[Index] = B[Index] * 4;
///     });
///   });
///
///   // "Host accessor creation" section
///   // Request the latest data of BufferC for the moment
///   // This is a synchronization point, which means that the DPC++ RT blocks
///   // on creation of the accessor until requested data is available.
///   auto C = BufferC.get_access<read>();
/// }
/// \endcode
///
/// In the example above the DPC++ RT does the following:
///
/// 1. **Copy command group**.
///    The DPC++ RT allocates memory for BufferA and BufferB on CPU then
///    executes an explicit copy operation on CPU.
///
/// 2. **Multi command group**
///    DPC++ RT allocates memory for BufferC and BufferB on GPU and copy
///    content of BufferB from CPU to GPU, then execute "multi" kernel on GPU.
///
/// 3. **Host accessor creation**
///    DPC++ RT allocates(it's possible to reuse already allocated memory)
///    memory available for user for BufferC then copy content of BufferC from
///    GPU to this memory.
///
/// So, the example above will be converted to the following OpenCL pseudo code
/// \code{.cpp}
/// // Initialization(not related to the Scheduler)
/// Platform = clGetPlatforms(...);
/// DeviceCPU = clGetDevices(CL_DEVICE_TYPE_CPU, ...);
/// DeviceGPU = clGetDevices(CL_DEVICE_TYPE_GPU, ...);
/// ContextCPU = clCreateContext(DeviceCPU, ...)
/// ContextGPU = clCreateContext(DeviceGPU, ...)
/// QueueCPU = clCreateCommandQueue(ContextCPU, DeviceCPU, ...);
/// QueueGPU = clCreateCommandQueue(ContextGPU, DeviceGPU, ...);
///
/// // Copy command group:
/// BufferACPU = clCreateBuffer(ContextCPU, CL_MEM_USE_HOST_PTR, ...);
/// BufferBCPU = clCreateBuffer(ContextCPU, ...);
/// CopyEvent = clEnqueueCopyBuffer(QueueCPU, BufferACPU, BufferBCPU, ...)
///
/// // Multi command group:
/// ReadBufferEvent =
///    clEnqueueReadBuffer(QueueCPU, BufferBCPU, HostPtr, CopyEvent, ...);
/// BufferBGPU = clCreateBuffer(ContextGPU, ...);
///
/// UserEvent = clCreateUserEvent(ContextCPU);
/// clSetEventCallback(ReadBufferEvent, event_completion_callback,
/// /*data=*/UserEvent);
///
/// WriteBufferEvent = clEnqueueWriteBuffer(QueueGPU, BufferBGPU, HostPtr,
/// UserEvent, ...); BufferCGPU = clCreateBuffer(ContextGPU, ...); ProgramGPU =
/// clCreateProgramWithIL(ContextGPU, ...); clBuildProgram(ProgramGPU);
/// MultiKernel = clCreateKernel("multi");
/// clSetKernelArg(MultiKernel, BufferBGPU, ...);
/// clSetKernelArg(MultiKernel, BufferCGPU, ...);
/// MultiEvent =
///    clEnqueueNDRangeKernel(QueueGPU, MultiKernel, WriteBufferEvent, ...);
///
/// // Host accessor creation:
/// clEnqueueMapBuffer(QueueGPU, BufferCGPU, BLOCKING_MAP, MultiEvent, ...);
///
/// // Releasing mem objects during SYCL buffers destruction.
/// clReleaseBuffer(BufferACPU);
/// clReleaseBuffer(BufferBCPU);
/// clReleaseBuffer(BufferBGPU);
/// clReleaseBuffer(BufferCGPU);
///
/// // Release(not related to the Scheduler)
/// clReleaseKernel(MultiKernel);
/// clReleaseProgram(ProgramGPU);
/// clReleaseContext(ContextGPU);
/// clReleaseContext(ContextCPU);
/// \endcode

// For testing purposes
class MockScheduler;

namespace sycl {
inline namespace _V1 {
namespace detail {
class queue_impl;
class event_impl;
class context_impl;
class DispatchHostTask;

using ContextImplPtr = std::shared_ptr<detail::context_impl>;
using EventImplPtr = std::shared_ptr<detail::event_impl>;
using QueueImplPtr = std::shared_ptr<detail::queue_impl>;
using StreamImplPtr = std::shared_ptr<detail::stream_impl>;

using QueueIdT = std::hash<std::shared_ptr<detail::queue_impl>>::result_type;
using CommandPtr = std::unique_ptr<Command>;
using FusionList = std::unique_ptr<KernelFusionCommand>;
using FusionMap = std::unordered_map<QueueIdT, FusionList>;

/// Memory Object Record
///
/// The MemObjRecord is used in command groups (todo better desc).
/// There must be a single MemObjRecord for each SYCL memory object.
///
/// \ingroup sycl_graph
struct MemObjRecord {
  MemObjRecord(ContextImplPtr Ctx, std::size_t LeafLimit,
               LeavesCollection::AllocateDependencyF AllocateDependency)
      : MReadLeaves{this, LeafLimit, AllocateDependency},
        MWriteLeaves{this, LeafLimit, AllocateDependency}, MCurContext{Ctx} {}

  // Contains all allocation commands for the memory object.
  std::vector<AllocaCommandBase *> MAllocaCommands;

  // Contains latest read only commands working with memory object.
  LeavesCollection MReadLeaves;

  // Contains latest write commands working with memory object.
  LeavesCollection MWriteLeaves;

  // The context which has the latest state of the memory object.
  ContextImplPtr MCurContext;

  // The mode this object can be accessed with from the host context.
  // Valid only if the current context is host.
  access::mode MHostAccess = access::mode::read_write;

  // The flag indicates that the content of the memory object was/will be
  // modified. Used while deciding if copy back needed.
  bool MMemModified = false;
};

/// DPC++ graph scheduler class.
///
/// \section sched_overview Scheduler Overview
///
/// The Scheduler is a part of DPC++ RT which ensures correct execution of
/// command groups. To achieve this Scheduler manages acyclic dependency graph
/// (which can have independent sub-graphs) that consists of several types of
/// nodes that represent specific commands:
///
/// 1. Allocate memory.
///    The command represents memory allocation operation. There can be
///    multiple allocations for a single SYCL memory object.
/// 2. Release memory.
///    The command represents memory release operation.
/// 3. Execute command group.
///    The command represents \ref CG "Command Group" (kernel) execution
///    operation.
/// 4. Copy memory.
///    The command represents memory copy operation between two memory
///    allocations of a single memory object.
///
/// As the main input Scheduler takes a command group and returns an event
/// representing it, so it can be waited on later. When a new
/// command group comes, Scheduler adds one or more nodes to the graph
/// depending on the command groups' requirements. For example, if a new
/// command group is submitted to the SYCL context which has the latest data
/// for all the requirements, Scheduler adds a new "Execute command group"
/// command making it dependent on all commands affecting new command group's
/// requirements. But if one of the requirements has no up-to-date instance in
/// the context which the command group is submitted to, Scheduler
/// additionally inserts copy memory command (together with allocate memory
/// command if needed).
///
/// A simple graph looks like:
//
//    +----------+     +----------+     +----------+
//    |          |     |          |     |          |
//    | Allocate |<----| Execute  |<----| Execute  |
//    |          |     |          |     |          |
//    +----------+     +----------+     +----------+
//
/// \dot
/// digraph G {
///   rankdir="LR";
///   Execute1 [label = "Execute"];
///   Execute2 [label = "Execute"];
///   Allocate;
///   Allocate -> Execute2 [dir = back];
///   Execute2 -> Execute1 [dir = back];
/// }
/// \enddot
///
/// Where nodes represent commands and edges represent dependencies between
/// them. There are three commands connected by arrows which mean that before
/// executing second command group the first one must be executed. Also before
/// executing the first command group memory allocation must be performed.
///
/// At some point Scheduler enqueues commands to the underlying devices. To do
/// this, Scheduler performs topological sort to get the order in which commands
/// should be enqueued. For example, the following graph (D depends on B and C,
/// B and C depends on A) will be enqueued in the following order:
/// \code{.cpp}
///   EventA = Enqueue(A, /*Deps=*/{});
///   EventB = Enqueue(B, /*Deps=*/{EventA});
///   EventC = Enqueue(C, /*Deps=*/{EventA});
///   EventD = Enqueue(D, /*Deps=*/{EventB, EventC});
/// \endcode
///
//                             +----------+
//                             |          |
//                             |    D     |
//                             |          |
//                             +----------+
//                            /            \
//                           /              \
//                          v                v
//                      +----------+     +----------+
//                      |          |     |          |
//                      |     B    |     |    C     |
//                      |          |     |          |
//                      +----------+     +----------+
//                             \            /
//                              \          /
//                               v        v
//                              +----------+
//                              |          |
//                              |    A     |
//                              |          |
//                              +----------+
/// \dot
/// digraph G {
///   D -> B;
///   D -> C;
///   C -> A;
///   B -> A;
/// }
/// \enddot
///
/// \section sched_impl Implementation details
///
/// The Scheduler is split up into two parts: graph builder and graph processor.
///
/// To build dependencies, Scheduler needs to memorize memory objects and
/// commands that modify them.
///
/// To detect that two command groups access the same memory object and create
/// a dependency between them, Scheduler needs to store information about
/// the memory object.
///
/// \subsection sched_thread_safety Thread safety
///
/// To ensure thread safe execution of methods, Scheduler provides access to the
/// graph that's guarded by a read-write mutex (analog of shared mutex from
/// C++17).
///
/// A read-write mutex allows concurrent access to read-only operations, while
/// write operations require exclusive access.
///
/// All the methods of GraphBuilder lock the mutex in write mode because these
/// methods can modify the graph.
/// Methods of GraphProcessor lock the mutex in read mode as they are not
/// modifying the graph.
///
/// \subsection shced_err_handling Error handling
///
/// There are two sources of errors that needs to be handled in Scheduler:
/// 1. errors that happen during command enqueue process
/// 2. the error that happend during command execution.
///
/// If an error occurs during command enqueue process, the Command::enqueue
/// method returns the faulty command. Scheduler then reschedules the command
/// and all dependent commands (if any).
///
/// An error with command processing can happen in underlying runtime, in this
/// case Scheduler is notified asynchronously (using callback mechanism) what
/// triggers rescheduling.
///
/// \ingroup sycl_graph
class Scheduler {
public:
  /// Registers a command group, and adds it to the dependency graph.
  ///
  /// It's called by SYCL's queue.submit.
  ///
  /// \param CommandGroup is a unique_ptr to a command group to be added.
  /// \param Queue Queue that is registering the command-group.
  /// \param CommandBuffer Optional command buffer to enqueue to instead of
  /// directly to the queue.
  /// \param Dependencies Optional list of dependency
  /// sync points when enqueuing to a command buffer.
  /// \return an event object to wait on for command group completion.
  EventImplPtr
  addCG(std::unique_ptr<detail::CG> CommandGroup, const QueueImplPtr &Queue,
        sycl::detail::pi::PiExtCommandBuffer CommandBuffer = nullptr,
        const std::vector<sycl::detail::pi::PiExtSyncPoint> &Dependencies = {});

  /// Registers a command group, that copies most recent memory to the memory
  /// pointed by the requirement.
  ///
  /// \param Req is a requirement that points to the memory where data is
  /// needed.
  /// \return an event object to wait on for copy finish.
  EventImplPtr addCopyBack(Requirement *Req);

  /// Waits for the event.
  ///
  /// This operation is blocking. For eager execution mode this method invokes
  /// corresponding function of device API.
  ///
  /// \param Event is a pointer to event to wait on.
  void waitForEvent(const EventImplPtr &Event);

  /// Removes buffer from the graph.
  ///
  /// The lifetime of memory object descriptor begins when the first command
  /// group that uses the memory object is submitted and ends when
  /// "removeMemoryObject(...)" method is called which means there will be no
  /// command group that uses the memory object. When removeMemoryObject is
  /// called Scheduler will enqueue and wait on all release commands associated
  /// with the memory object, which effectively guarantees that all commands
  /// accessing the memory object are complete and then the resources allocated
  /// for the memory object are freed. Then all the commands affecting the
  /// memory object are removed.
  ///
  /// This member function is used by \ref buffer and \ref image.
  ///
  /// \param MemObj is a memory object that points to the buffer being removed.
  /// \param StrictLock WA, is a flag used to identify if strict read and write
  /// lock are allowed or not. Default value is always applied in buffer_impl
  /// destructor. StrictLock == false is introduced for
  /// cleanupDeferredMemObjects to avoid blocking mem object release that may
  /// lead to dead lock. \return WA, true if all release action completed and we
  /// could delete memory object, false otherwise, most possible reason to
  /// receive false - fail to obtain write lock.
  bool removeMemoryObject(detail::SYCLMemObjI *MemObj, bool StrictLock = true);

  /// Adds nodes to the graph, that update the requirement with the pointer
  /// to the host memory.
  ///
  /// Assumes the host pointer contains the latest data. New operations with
  /// the same memory object that have side effects are blocked until
  /// releaseHostAccessor(Requirement *Req) is callled.
  ///
  /// \param Req is the requirement to be updated.
  /// \return an event which indicates when these nodes are completed
  /// and host accessor is ready for use.
  EventImplPtr addHostAccessor(Requirement *Req);

  /// Unblocks operations with the memory object.
  ///
  /// \param Req is a requirement that points to the memory object being
  /// unblocked.
  void releaseHostAccessor(Requirement *Req);

  /// \return an instance of the scheduler object.
  static Scheduler &getInstance();
  /// \return true if an instance of the scheduler object exists.
  static bool isInstanceAlive();

  QueueImplPtr getDefaultHostQueue() { return DefaultHostQueue; }

  const QueueImplPtr &getDefaultHostQueue() const { return DefaultHostQueue; }

  static MemObjRecord *getMemObjRecord(const Requirement *const Req);

  void deferMemObjRelease(const std::shared_ptr<detail::SYCLMemObjI> &MemObj);

  void startFusion(QueueImplPtr Queue);

  void cleanUpCmdFusion(sycl::detail::queue_impl *Queue);

  void cancelFusion(QueueImplPtr Queue);

  EventImplPtr completeFusion(QueueImplPtr Queue, const property_list &);

  bool isInFusionMode(QueueIdT Queue);

  Scheduler();
  ~Scheduler();
  void releaseResources(BlockingT Blocking = BlockingT::BLOCKING);
  bool isDeferredMemObjectsEmpty();

  void enqueueCommandForCG(EventImplPtr NewEvent,
                           std::vector<Command *> &AuxilaryCmds,
                           BlockingT Blocking = NON_BLOCKING);

protected:
  using RWLockT = std::shared_timed_mutex;
  using ReadLockT = std::shared_lock<RWLockT>;
  using WriteLockT = std::unique_lock<RWLockT>;

  /// Provides exclusive access to std::shared_timed_mutex object with deadlock
  /// avoidance
  WriteLockT acquireWriteLock() {
#ifdef _WIN32
    WriteLockT Lock(MGraphLock, std::defer_lock);
    while (!Lock.try_lock_for(std::chrono::milliseconds(10))) {
      // Without yield while loop acts like endless while loop and occupies the
      // whole CPU when multiple command groups are created in multiple host
      // threads
      std::this_thread::yield();
    }
#else
    WriteLockT Lock(MGraphLock);
    // It is a deadlock on UNIX in implementation of lock and lock_shared, if
    // try_lock in the loop above will be executed, so using a single lock here
#endif // _WIN32
    return Lock;
  }

  /// Provides exclusive access to std::shared_timed_mutex object with deadlock
  /// avoidance to the Fusion map
  WriteLockT acquireFusionWriteLock() {
#ifdef _WIN32
    WriteLockT Lock(MFusionMapLock, std::defer_lock);
    while (!Lock.try_lock_for(std::chrono::milliseconds(10))) {
      // Without yield while loop acts like endless while loop and occupies the
      // whole CPU when multiple command groups are created in multiple host
      // threads
      std::this_thread::yield();
    }
#else
    WriteLockT Lock(MFusionMapLock);
    // It is a deadlock on UNIX in implementation of lock and lock_shared, if
    // try_lock in the loop above will be executed, so using a single lock here
#endif // _WIN32
    return Lock;
  }

  /// Provides shared access to std::shared_timed_mutex object with deadlock
  /// avoidance
  ReadLockT acquireReadLock() { return ReadLockT{MGraphLock}; }

  /// Provides shared access to std::shared_timed_mutex object with deadlock
  /// avoidance to the Fusion map
  ReadLockT acquireFusionReadLock() { return ReadLockT{MFusionMapLock}; }

  void cleanupCommands(const std::vector<Command *> &Cmds);

  void NotifyHostTaskCompletion(Command *Cmd);

  static void enqueueLeavesOfReqUnlocked(const Requirement *const Req,
                                         ReadLockT &GraphReadLock,
                                         std::vector<Command *> &ToCleanUp);

  static void
  enqueueUnblockedCommands(const std::vector<EventImplPtr> &CmdsToEnqueue,
                           ReadLockT &GraphReadLock,
                           std::vector<Command *> &ToCleanUp);

  // May lock graph with read and write modes during execution.
  void cleanupDeferredMemObjects(BlockingT Blocking);

  // POD struct to convey some additional information from GraphBuilder::addCG
  // to the Scheduler to support kernel fusion.
  struct GraphBuildResult {
    Command *NewCmd;
    EventImplPtr NewEvent;
    bool ShouldEnqueue;
  };

  /// Assign \p Src's auxiliary resources to \p Dst.
  void takeAuxiliaryResources(const EventImplPtr &Dst, const EventImplPtr &Src);
  void registerAuxiliaryResources(
      EventImplPtr &Event, std::vector<std::shared_ptr<const void>> Resources);
  void cleanupAuxiliaryResources(BlockingT Blocking);

  /// Graph builder class.
  ///
  /// The graph builder provides means to change an existing graph (e.g. add
  /// or remove edges/nodes).
  ///
  /// \ingroup sycl_graph
  class GraphBuilder {
  public:
    GraphBuilder();

    /// Registers \ref CG "command group" and adds it to the dependency graph.
    ///
    /// \sa queue::submit, Scheduler::addCG
    /// \param CommandBuffer Optional command buffer to enqueue to instead of
    /// directly to the queue.
    /// \param Dependencies Optional list of dependency
    /// sync points when enqueuing to a command buffer.
    ///
    /// \return a command that represents command group execution and a bool
    /// indicating whether this command should be enqueued to the graph
    /// processor right away or not.
    GraphBuildResult addCG(
        std::unique_ptr<detail::CG> CommandGroup, const QueueImplPtr &Queue,
        std::vector<Command *> &ToEnqueue,
        sycl::detail::pi::PiExtCommandBuffer CommandBuffer = nullptr,
        const std::vector<sycl::detail::pi::PiExtSyncPoint> &Dependencies = {});

    /// Registers a \ref CG "command group" that updates host memory to the
    /// latest state.
    ///
    /// \return a command that represents command group execution.
    Command *addCGUpdateHost(std::unique_ptr<detail::CG> CommandGroup,
                             const QueueImplPtr &HostQueue,
                             std::vector<Command *> &ToEnqueue);

    /// Enqueues a command to update memory to the latest state.
    ///
    /// \param Req is a requirement, that describes memory object.
    Command *addCopyBack(Requirement *Req, std::vector<Command *> &ToEnqueue);

    /// Enqueues a command to create a host accessor.
    ///
    /// \param Req points to memory being accessed.
    Command *addHostAccessor(Requirement *Req,
                             std::vector<Command *> &ToEnqueue);

    /// [Provisional] Optimizes the whole graph.
    void optimize();

    /// [Provisional] Optimizes subgraph that consists of command associated
    /// with Event passed and its dependencies.
    void optimize(const EventImplPtr &Event);

    void cleanupCommand(Command *Cmd, bool AllowUnsubmitted = false);

    /// Reschedules the command passed using Queue provided.
    ///
    /// This can lead to rescheduling of all dependent commands. This can be
    /// used when the user provides a "secondary" queue to the submit method
    /// which may be used when the command fails to enqueue/execute in the
    /// primary queue.
    void rescheduleCommand(Command *Cmd, const QueueImplPtr &Queue);

    /// \return a pointer to the corresponding memory object record for the
    /// SYCL memory object provided, or nullptr if it does not exist.
    MemObjRecord *getMemObjRecord(SYCLMemObjI *MemObject);

    /// \return a pointer to MemObjRecord for pointer to memory object. If the
    /// record is not found, nullptr is returned.
    MemObjRecord *getOrInsertMemObjRecord(const QueueImplPtr &Queue,
                                          const Requirement *Req,
                                          std::vector<Command *> &ToEnqueue);

    /// Decrements leaf counters for all leaves of the record.
    void decrementLeafCountersForRecord(MemObjRecord *Record);

    /// Removes commands that use the given MemObjRecord from the graph.
    void cleanupCommandsForRecord(MemObjRecord *Record);

    /// Removes the MemObjRecord for the memory object passed.
    void removeRecordForMemObj(SYCLMemObjI *MemObject);

    /// Adds new command to leaves if needed.
    void addNodeToLeaves(MemObjRecord *Record, Command *Cmd,
                         access::mode AccessMode,
                         std::vector<Command *> &ToEnqueue);

    /// Removes commands from leaves.
    void updateLeaves(const std::set<Command *> &Cmds, MemObjRecord *Record,
                      access::mode AccessMode,
                      std::vector<Command *> &ToCleanUp);

    /// Perform connection of events in multiple contexts
    /// \param Cmd dependant command
    /// \param DepEvent event to depend on
    /// \param Dep optional DepDesc to perform connection properly
    /// \param ToCleanUp container for commands that can be cleaned up due to
    /// their removal from leaves
    /// \returns the connecting command which is to be enqueued
    ///
    /// Optionality of Dep is set by Dep.MDepCommand equal to nullptr.
    Command *connectDepEvent(Command *const Cmd, const EventImplPtr &DepEvent,
                             const DepDesc &Dep,
                             std::vector<Command *> &ToCleanUp);

    void startFusion(QueueImplPtr Queue);

    /// Clean up the internal fusion commands held for the given queue.
    /// @param Queue the queue for which to remove the fusion commands.
    void cleanUpCmdFusion(sycl::detail::queue_impl *Queue);

    void cancelFusion(QueueImplPtr Queue, std::vector<Command *> &ToEnqueue);

    EventImplPtr completeFusion(QueueImplPtr Queue,
                                std::vector<Command *> &ToEnqueue,
                                const property_list &);

    bool isInFusionMode(QueueIdT queue);

    std::vector<SYCLMemObjI *> MMemObjs;

  private:
    /// Inserts the command required to update the memory object state in the
    /// context.
    ///
    /// Copy/map/unmap operations can be inserted depending on the source and
    /// destination.
    ///
    /// \param Record is a memory object that needs to be updated.
    /// \param Req is a Requirement describing destination.
    /// \param Queue is a queue that is bound to target context.
    Command *insertMemoryMove(MemObjRecord *Record, Requirement *Req,
                              const QueueImplPtr &Queue,
                              std::vector<Command *> &ToEnqueue);

    // Inserts commands required to remap the memory object to its current host
    // context so that the required access mode becomes valid.
    Command *remapMemoryObject(MemObjRecord *Record, Requirement *Req,
                               AllocaCommandBase *HostAllocaCmd,
                               std::vector<Command *> &ToEnqueue);

    UpdateHostRequirementCommand *
    insertUpdateHostReqCmd(MemObjRecord *Record, Requirement *Req,
                           const QueueImplPtr &Queue,
                           std::vector<Command *> &ToEnqueue);

    /// Finds dependencies for the requirement.
    std::set<Command *> findDepsForReq(MemObjRecord *Record,
                                       const Requirement *Req,
                                       const ContextImplPtr &Context);

    EmptyCommand *addEmptyCmd(Command *Cmd,
                              const std::vector<Requirement *> &Req,
                              const QueueImplPtr &Queue,
                              Command::BlockReason Reason,
                              std::vector<Command *> &ToEnqueue,
                              const bool AddDepsToLeaves = true);

    void createGraphForCommand(Command *NewCmd, CG &CG, bool isInteropTask,
                               std::vector<Requirement *> &Reqs,
                               const std::vector<detail::EventImplPtr> &Events,
                               QueueImplPtr Queue,
                               std::vector<Command *> &ToEnqueue);

  protected:
    /// Finds a command dependency corresponding to the record.
    DepDesc findDepForRecord(Command *Cmd, MemObjRecord *Record);

    /// Searches for suitable alloca in memory record.
    AllocaCommandBase *findAllocaForReq(MemObjRecord *Record,
                                        const Requirement *Req,
                                        const ContextImplPtr &Context,
                                        bool AllowConst = true);

    friend class Command;

  private:
    friend class ::MockScheduler;

    /// Searches for suitable alloca in memory record.
    ///
    /// If none found, creates new one.
    AllocaCommandBase *
    getOrCreateAllocaForReq(MemObjRecord *Record, const Requirement *Req,
                            const QueueImplPtr &Queue,
                            std::vector<Command *> &ToEnqueue);

    void markModifiedIfWrite(MemObjRecord *Record, Requirement *Req);

    FusionMap::iterator findFusionList(QueueIdT Id) {
      return MFusionMap.find(Id);
    }

    void removeNodeFromGraph(Command *Node, std::vector<Command *> &ToEnqueue);

    /// Used to track commands that need to be visited during graph
    /// traversal.
    std::queue<Command *> MCmdsToVisit;
    /// Used to track commands that have been visited during graph traversal.
    std::vector<Command *> MVisitedCmds;

    /// Used to track queues that are in fusion mode and the
    /// command-groups/kernels submitted for fusion.
    FusionMap MFusionMap;

    /// Prints contents of graph to text file in DOT format
    ///
    /// \param ModeName is a stringified printing mode name to be used
    /// in the result file name.
    void printGraphAsDot(const char *ModeName);
    enum PrintOptions {
      BeforeAddCG = 0,
      AfterAddCG,
      BeforeAddCopyBack,
      AfterAddCopyBack,
      BeforeAddHostAcc,
      AfterAddHostAcc,
      AfterFusionComplete,
      AfterFusionCancel,
      Size
    };
    std::array<bool, PrintOptions::Size> MPrintOptionsArray{false};
  };

  /// Graph Processor provides interfaces for enqueueing commands and their
  /// dependencies to the underlying runtime.
  ///
  /// Member functions of this class do not modify the graph.
  ///
  /// \section sched_enqueue Command enqueueing
  ///
  /// Commands are enqueued whenever they come to the Scheduler. Each command
  /// has enqueue method which takes vector of events that represents
  /// dependencies and returns event which represents the command.
  /// GraphProcessor performs topological sort to get the order in which
  /// commands have to be enqueued. Then it enqueues each command, passing a
  /// vector of events that this command needs to wait on. If an error happens
  /// during command enqueue, the whole process is stopped, the faulty command
  /// is propagated back to the Scheduler.
  ///
  /// The command with dependencies that belong to a context different from its
  /// own can't be enqueued directly (limitation of OpenCL runtime).
  /// Instead, for each dependency, a proxy event is created in the target
  /// context and linked using OpenCL callback mechanism with original one.
  /// For example, the following SYCL code:
  ///
  /// \code{.cpp}
  ///   // The ContextA and ContextB are different OpenCL contexts
  ///   sycl::queue Q1(ContextA);
  ///   sycl::queue Q2(ContextB);
  ///
  ///   Q1.submit(Task1);
  ///
  ///   Q2.submit(Task2);
  /// \endcode
  ///
  /// is translated to the following OCL API calls:
  ///
  /// \code{.cpp}
  ///   void event_completion_callback(void *data) {
  ///     // Change status of event to complete.
  ///     clSetEventStatus((cl_event *)data, CL_COMPLETE); // Scope of Context2
  ///   }
  ///
  ///  // Enqueue TASK1
  ///  EventTask1 = clEnqueueNDRangeKernel(Q1, TASK1, ..); // Scope of Context1
  ///  // Read memory to host
  ///  ReadMem = clEnqueueReadBuffer(A, .., /*Deps=*/EventTask1); // Scope of
  ///  // Context1
  ///
  ///  // Create user event with initial status "not completed".
  ///  UserEvent = clCreateUserEvent(Context2); // Scope of Context2
  ///  // Ask OpenCL to call callback with UserEvent as data when "read memory
  ///  // to host" operation is completed
  ///  clSetEventCallback(ReadMem, event_completion_callback,
  ///                     /*data=*/UserEvent); // Scope of Context1
  ///
  ///  // Enqueue write memory from host, block it on user event
  ///  // It will be unblocked when we change UserEvent status to completed in
  ///  // callback.
  ///  WriteMem =
  ///     clEnqueueWriteBuffer(A, .., /*Dep=*/UserEvent); // Scope of Context2
  ///  // Enqueue TASK2
  ///  EventTask2 =
  ///     clEnqueueNDRangeKernel(TASK, .., /*Dep=*/WriteMem); // Scope of
  ///     // Context2
  /// \endcode
  ///
  /// The alternative approach that has been considered is to have separate
  /// dispatcher thread that would wait for all events from the Context other
  /// then target Context to complete and then enqueue command with dependencies
  /// from target Context only. Alternative approach makes code significantly
  /// more complex and can hurt performance on CPU device vs chosen approach
  /// with callbacks.
  ///
  /// \ingroup sycl_graph
  class GraphProcessor {
  public:
    /// Waits for the command, associated with Event passed, is completed.
    /// \param GraphReadLock read-lock which is already acquired for reading
    /// \param ToCleanUp container for commands that can be cleaned up.
    /// \param LockTheLock selects if graph lock should be locked upon return
    ///
    /// The function may unlock and lock GraphReadLock as needed. Upon return
    /// the lock is left in locked state if and only if LockTheLock is true.
    static void waitForEvent(const EventImplPtr &Event,
                             ReadLockT &GraphReadLock,
                             std::vector<Command *> &ToCleanUp,
                             bool LockTheLock = true);

    /// Enqueues the command and all its dependencies.
    ///
    /// \param EnqueueResult is set to specific status if enqueue failed.
    /// \param ToCleanUp container for commands that can be cleaned up.
    /// \param RootCommand top level command enqueueCommand is called for.
    /// \return true if the command is successfully enqueued.
    ///
    /// The function may unlock and lock GraphReadLock as needed. Upon return
    /// the lock is left in locked state.
    static bool enqueueCommand(Command *Cmd, ReadLockT &GraphReadLock,
                               EnqueueResultT &EnqueueResult,
                               std::vector<Command *> &ToCleanUp,
                               Command *RootCommand,
                               BlockingT Blocking = NON_BLOCKING);

    /// Check if successfully enqueued command is expected to be blocking for
    /// the dependent commands before its completion.
    ///
    /// \param Cmd command that is currently enqueued.
    /// \param EnqueueResult enqueue status to be updated if command is
    /// blocking.
    /// \param RootCommand top level command enqueueCommand is called
    /// for. If Cmd == RootCommand we should not report command as blocking as
    /// no dependencies to be blocked yet.
    /// \param Blocking mode for enqueue, we do not report command as blocking
    /// if it is true and allow to wait for command completion in enqueueImp.
    /// \return true if we should continue enqueue process.
    static bool handleBlockingCmd(Command *Cmd, EnqueueResultT &EnqueueResult,
                                  Command *RootCommand, BlockingT Blocking);
  };

  /// This function waits on all of the graph leaves which somehow use the
  /// memory object which is represented by \c Record. The function is called
  /// upon destruction of memory buffer.
  /// \param Record memory record to await graph leaves of to finish
  /// \param GraphReadLock locked graph read lock
  ///
  /// GraphReadLock will be unlocked/locked as needed. Upon return from the
  /// function, GraphReadLock will be left in locked state.
  void waitForRecordToFinish(MemObjRecord *Record, ReadLockT &GraphReadLock);
  bool checkLeavesCompletion(MemObjRecord *Record);

  GraphBuilder MGraphBuilder;
  RWLockT MGraphLock;
  RWLockT MFusionMapLock;

  std::vector<Command *> MDeferredCleanupCommands;
  std::mutex MDeferredCleanupMutex;

  std::vector<std::shared_ptr<SYCLMemObjI>> MDeferredMemObjRelease;
  std::mutex MDeferredMemReleaseMutex;

  std::unordered_map<EventImplPtr, std::vector<std::shared_ptr<const void>>>
      MAuxiliaryResources;
  std::mutex MAuxiliaryResourcesMutex;

  QueueImplPtr DefaultHostQueue;

  friend class Command;
  friend class DispatchHostTask;
  friend class queue_impl;
  friend class event_impl;
  friend class ::MockScheduler;

private:
  static void printFusionWarning(const std::string &Message);

  static KernelFusionCommand *isPartOfActiveFusion(Command *Cmd);
};

} // namespace detail
} // namespace _V1
} // namespace sycl
