//===----------- commands.cpp - SYCL commands -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/error_handling/error_handling.hpp>

#include <CL/sycl/access/access.hpp>
#include <CL/sycl/backend_types.hpp>
#include <CL/sycl/detail/cg_types.hpp>
#include <CL/sycl/detail/cl.h>
#include <CL/sycl/detail/kernel_desc.hpp>
#include <CL/sycl/detail/memory_manager.hpp>
#include <CL/sycl/program.hpp>
#include <CL/sycl/sampler.hpp>
#include <detail/context_impl.hpp>
#include <detail/event_impl.hpp>
#include <detail/kernel_bundle_impl.hpp>
#include <detail/kernel_impl.hpp>
#include <detail/kernel_info.hpp>
#include <detail/program_impl.hpp>
#include <detail/program_manager/program_manager.hpp>
#include <detail/queue_impl.hpp>
#include <detail/sampler_impl.hpp>
#include <detail/scheduler/commands.hpp>
#include <detail/scheduler/scheduler.hpp>
#include <detail/stream_impl.hpp>
#include <detail/xpti_registry.hpp>

#include <cassert>
#include <optional>
#include <string>
#include <vector>

#ifdef __has_include
#if __has_include(<cxxabi.h>)
#define __SYCL_ENABLE_GNU_DEMANGLING
#include <cstdlib>
#include <cxxabi.h>
#include <memory>
#endif
#endif

#ifdef XPTI_ENABLE_INSTRUMENTATION
#include "xpti/xpti_trace_framework.hpp"
#include <detail/xpti_registry.hpp>
#endif

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

#ifdef XPTI_ENABLE_INSTRUMENTATION
// Global graph for the application
extern xpti::trace_event_data_t *GSYCLGraphEvent;
#endif

#ifdef __SYCL_ENABLE_GNU_DEMANGLING
struct DemangleHandle {
  char *p;
  DemangleHandle(char *ptr) : p(ptr) {}
  ~DemangleHandle() { std::free(p); }
};
static std::string demangleKernelName(std::string Name) {
  int Status = -1; // some arbitrary value to eliminate the compiler warning
  DemangleHandle result(abi::__cxa_demangle(Name.c_str(), NULL, NULL, &Status));
  return (Status == 0) ? result.p : Name;
}
#else
static std::string demangleKernelName(std::string Name) { return Name; }
#endif

static std::string deviceToString(device Device) {
  if (Device.is_host())
    return "HOST";
  else if (Device.is_cpu())
    return "CPU";
  else if (Device.is_gpu())
    return "GPU";
  else if (Device.is_accelerator())
    return "ACCELERATOR";
  else
    return "UNKNOWN";
}

#ifdef XPTI_ENABLE_INSTRUMENTATION
static size_t deviceToID(const device &Device) {
  if (Device.is_host())
    return 0;
  else
    return reinterpret_cast<size_t>(getSyclObjImpl(Device)->getHandleRef());
}
#endif

static std::string accessModeToString(access::mode Mode) {
  switch (Mode) {
  case access::mode::read:
    return "read";
  case access::mode::write:
    return "write";
  case access::mode::read_write:
    return "read_write";
  case access::mode::discard_write:
    return "discard_write";
  case access::mode::discard_read_write:
    return "discard_read_write";
  default:
    return "unknown";
  }
}

#ifdef XPTI_ENABLE_INSTRUMENTATION
// Using the command group type to create node types for the asynchronous task
// graph modeling
static std::string commandToNodeType(Command::CommandType Type) {
  switch (Type) {
  case Command::CommandType::RUN_CG:
    return "command_group_node";
  case Command::CommandType::COPY_MEMORY:
    return "memory_transfer_node";
  case Command::CommandType::ALLOCA:
    return "memory_allocation_node";
  case Command::CommandType::ALLOCA_SUB_BUF:
    return "sub_buffer_creation_node";
  case Command::CommandType::RELEASE:
    return "memory_deallocation_node";
  case Command::CommandType::MAP_MEM_OBJ:
    return "memory_transfer_node";
  case Command::CommandType::UNMAP_MEM_OBJ:
    return "memory_transfer_node";
  case Command::CommandType::UPDATE_REQUIREMENT:
    return "host_acc_create_buffer_lock_node";
  case Command::CommandType::EMPTY_TASK:
    return "host_acc_destroy_buffer_release_node";
  default:
    return "unknown_node";
  }
}

// Using the names being generated and the string are subject to change to
// something more meaningful to end-users as this will be visible in analysis
// tools that subscribe to this data
static std::string commandToName(Command::CommandType Type) {
  switch (Type) {
  case Command::CommandType::RUN_CG:
    return "Command Group Action";
  case Command::CommandType::COPY_MEMORY:
    return "Memory Transfer (Copy)";
  case Command::CommandType::ALLOCA:
    return "Memory Allocation";
  case Command::CommandType::ALLOCA_SUB_BUF:
    return "Sub Buffer Creation";
  case Command::CommandType::RELEASE:
    return "Memory Deallocation";
  case Command::CommandType::MAP_MEM_OBJ:
    return "Memory Transfer (Map)";
  case Command::CommandType::UNMAP_MEM_OBJ:
    return "Memory Transfer (Unmap)";
  case Command::CommandType::UPDATE_REQUIREMENT:
    return "Host Accessor Creation/Buffer Lock";
  case Command::CommandType::EMPTY_TASK:
    return "Host Accessor Destruction/Buffer Lock Release";
  default:
    return "Unknown Action";
  }
}
#endif

static std::vector<RT::PiEvent>
getPiEvents(const std::vector<EventImplPtr> &EventImpls) {
  std::vector<RT::PiEvent> RetPiEvents;
  for (auto &EventImpl : EventImpls) {
    if (EventImpl->getHandleRef() != nullptr)
      RetPiEvents.push_back(EventImpl->getHandleRef());
  }

  return RetPiEvents;
}

static void flushCrossQueueDeps(const std::vector<EventImplPtr> &EventImpls,
                                const QueueImplPtr &Queue) {
  for (auto &EventImpl : EventImpls) {
    EventImpl->flushIfNeeded(Queue);
  }
}

class DispatchHostTask {
  ExecCGCommand *MThisCmd;
  std::vector<interop_handle::ReqToMem> MReqToMem;

  pi_result waitForEvents() const {
    std::map<const detail::plugin *, std::vector<EventImplPtr>>
        RequiredEventsPerPlugin;

    for (const EventImplPtr &Event : MThisCmd->MPreparedDepsEvents) {
      const detail::plugin &Plugin = Event->getPlugin();
      RequiredEventsPerPlugin[&Plugin].push_back(Event);
    }

    // wait for dependency device events
    // FIXME Current implementation of waiting for events will make the thread
    // 'sleep' until all of dependency events are complete. We need a bit more
    // sophisticated waiting mechanism to allow to utilize this thread for any
    // other available job and resume once all required events are ready.
    for (auto &PluginWithEvents : RequiredEventsPerPlugin) {
      std::vector<RT::PiEvent> RawEvents = getPiEvents(PluginWithEvents.second);
      try {
        PluginWithEvents.first->call<PiApiKind::piEventsWait>(RawEvents.size(),
                                                              RawEvents.data());
      } catch (const sycl::exception &E) {
        CGHostTask &HostTask = static_cast<CGHostTask &>(MThisCmd->getCG());
        HostTask.MQueue->reportAsyncException(std::current_exception());
        return (pi_result)E.get_cl_code();
      } catch (...) {
        CGHostTask &HostTask = static_cast<CGHostTask &>(MThisCmd->getCG());
        HostTask.MQueue->reportAsyncException(std::current_exception());
        return PI_ERROR_UNKNOWN;
      }
    }

    // Wait for dependency host events.
    // Host events can't throw exceptions so don't try to catch it.
    for (const EventImplPtr &Event : MThisCmd->MPreparedHostDepsEvents) {
      Event->waitInternal();
    }

    return PI_SUCCESS;
  }

public:
  DispatchHostTask(ExecCGCommand *ThisCmd,
                   std::vector<interop_handle::ReqToMem> ReqToMem)
      : MThisCmd{ThisCmd}, MReqToMem(std::move(ReqToMem)) {}

  void operator()() const {
    assert(MThisCmd->getCG().getType() == CG::CGTYPE::CodeplayHostTask);

    CGHostTask &HostTask = static_cast<CGHostTask &>(MThisCmd->getCG());

    pi_result WaitResult = waitForEvents();
    if (WaitResult != PI_SUCCESS) {
      std::exception_ptr EPtr = std::make_exception_ptr(sycl::runtime_error(
          std::string("Couldn't wait for host-task's dependencies"),
          WaitResult));
      HostTask.MQueue->reportAsyncException(EPtr);

      // reset host-task's lambda and quit
      HostTask.MHostTask.reset();
      return;
    }

    try {
      // we're ready to call the user-defined lambda now
      if (HostTask.MHostTask->isInteropTask()) {
        interop_handle IH{MReqToMem, HostTask.MQueue,
                          HostTask.MQueue->getDeviceImplPtr(),
                          HostTask.MQueue->getContextImplPtr()};

        HostTask.MHostTask->call(IH);
      } else
        HostTask.MHostTask->call();
    } catch (...) {
      HostTask.MQueue->reportAsyncException(std::current_exception());
    }

    HostTask.MHostTask.reset();

    // unblock user empty command here
    EmptyCommand *EmptyCmd = MThisCmd->MEmptyCmd;
    assert(EmptyCmd && "No empty command found");

    // Completing command's event along with unblocking enqueue readiness of
    // empty command may lead to quick deallocation of MThisCmd by some cleanup
    // process. Thus we'll copy deps prior to completing of event and unblocking
    // of empty command.
    // Also, it's possible to have record deallocated prior to enqueue process.
    // Thus we employ read-lock of graph.
    std::vector<Command *> ToCleanUp;
    Scheduler &Sched = Scheduler::getInstance();
    {
      Scheduler::ReadLockT Lock(Sched.MGraphLock);

      std::vector<DepDesc> Deps = MThisCmd->MDeps;

      // update self-event status
      MThisCmd->MEvent->setComplete();

      EmptyCmd->MEnqueueStatus = EnqueueResultT::SyclEnqueueReady;

      EventImplPtr EmptyCmdEvent = EmptyCmd->getEvent();
      const std::unordered_set<EventImplPtr> &CmdsToEnqueue =
          EmptyCmd->getBlockedUsers();
      // If we have blocked users empty command will be enqueued as one of
      // dependency otherwise we have to enqueue it manually
      if (CmdsToEnqueue.empty())
        Scheduler::enqueueUnblockedCommands(
            EmptyCmdEvent, std::unordered_set<EventImplPtr>{EmptyCmdEvent},
            ToCleanUp);
      else
        Scheduler::enqueueUnblockedCommands(EmptyCmdEvent, CmdsToEnqueue,
                                            ToCleanUp);

      for (const DepDesc &Dep : Deps)
        Scheduler::enqueueLeavesOfReqUnlocked(Dep.MDepRequirement, ToCleanUp);
    }
    Sched.cleanupCommands(ToCleanUp);
  }
};

void Command::waitForPreparedHostEvents() const {
  for (const EventImplPtr &HostEvent : MPreparedHostDepsEvents)
    HostEvent->waitInternal();
}

void Command::waitForEvents(QueueImplPtr Queue,
                            std::vector<EventImplPtr> &EventImpls,
                            RT::PiEvent &Event) {

  if (!EventImpls.empty()) {
    if (Queue->is_host()) {
      // Host queue can wait for events from different contexts, i.e. it may
      // contain events with different contexts in its MPreparedDepsEvents.
      // OpenCL 2.1 spec says that clWaitForEvents will return
      // CL_INVALID_CONTEXT if events specified in the list do not belong to
      // the same context. Thus we split all the events into per-context map.
      // An example. We have two queues for the same CPU device: Q1, Q2. Thus
      // we will have two different contexts for the same CPU device: C1, C2.
      // Also we have default host queue. This queue is accessible via
      // Scheduler. Now, let's assume we have three different events: E1(C1),
      // E2(C1), E3(C2). Also, we have an EmptyCommand which is to be executed
      // on host queue. The command's MPreparedDepsEvents will contain all three
      // events (E1, E2, E3). Now, if piEventsWait is called for all three
      // events we'll experience failure with CL_INVALID_CONTEXT 'cause these
      // events refer to different contexts.
      std::map<context_impl *, std::vector<EventImplPtr>>
          RequiredEventsPerContext;

      for (const EventImplPtr &Event : EventImpls) {
        ContextImplPtr Context = Event->getContextImpl();
        assert(Context.get() &&
               "Only non-host events are expected to be waited for here");
        RequiredEventsPerContext[Context.get()].push_back(Event);
      }

      for (auto &CtxWithEvents : RequiredEventsPerContext) {
        std::vector<RT::PiEvent> RawEvents = getPiEvents(CtxWithEvents.second);
        CtxWithEvents.first->getPlugin().call<PiApiKind::piEventsWait>(
            RawEvents.size(), RawEvents.data());
      }
    } else {
#ifndef NDEBUG
      for (const EventImplPtr &Event : EventImpls)
        assert(Event->getContextImpl().get() &&
               "Only non-host events are expected to be waited for here");
#endif

      std::vector<RT::PiEvent> RawEvents = getPiEvents(EventImpls);
      flushCrossQueueDeps(EventImpls, getWorkerQueue());
      const detail::plugin &Plugin = Queue->getPlugin();
      Plugin.call<PiApiKind::piEnqueueEventsWait>(
          Queue->getHandleRef(), RawEvents.size(), &RawEvents[0], &Event);
    }
  }
}

/// It is safe to bind MPreparedDepsEvents and MPreparedHostDepsEvents
/// references to event_impl class members because Command
/// should not outlive the event connected to it.
Command::Command(CommandType Type, QueueImplPtr Queue)
    : MQueue(std::move(Queue)),
      MEvent(std::make_shared<detail::event_impl>(MQueue)),
      MPreparedDepsEvents(MEvent->getPreparedDepsEvents()),
      MPreparedHostDepsEvents(MEvent->getPreparedHostDepsEvents()),
      MType(Type) {
  MSubmittedQueue = MQueue;
  MEvent->setCommand(this);
  MEvent->setContextImpl(MQueue->getContextImplPtr());
  MEnqueueStatus = EnqueueResultT::SyclEnqueueReady;

#ifdef XPTI_ENABLE_INSTRUMENTATION
  if (!xptiTraceEnabled())
    return;
  // Obtain the stream ID so all commands can emit traces to that stream
  MStreamID = xptiRegisterStream(SYCL_STREAM_NAME);
#endif
}

void Command::emitInstrumentationDataProxy() {
#ifdef XPTI_ENABLE_INSTRUMENTATION
  emitInstrumentationData();
#endif
}

/// Method takes in void * for the address as adding a template function to
/// the command group object maybe undesirable.
/// @param Cmd The command object of the source of the edge
/// @param ObjAddr The address that defines the edge dependency; it is the event
/// address when the edge is for an event and a memory object address if it is
/// due to an accessor
/// @param Prefix Contains "event" if the dependency is an edge and contains the
/// access mode to the buffer if it is due to an accessor
/// @param IsCommand True if the dependency has a command object as the source,
/// false otherwise
void Command::emitEdgeEventForCommandDependence(
    Command *Cmd, void *ObjAddr, bool IsCommand,
    std::optional<access::mode> AccMode) {
#ifdef XPTI_ENABLE_INSTRUMENTATION
  // Bail early if either the source or the target node for the given dependency
  // is undefined or NULL
  if (!(xptiTraceEnabled() && MTraceEvent && Cmd && Cmd->MTraceEvent))
    return;

  // If all the information we need for creating an edge event is available,
  // then go ahead with creating it; if not, bail early!
  xpti::utils::StringHelper SH;
  std::string AddressStr = SH.addressAsString<void *>(ObjAddr);
  std::string Prefix = AccMode ? accessModeToString(AccMode.value()) : "Event";
  std::string TypeString = SH.nameWithAddressString(Prefix, AddressStr);
  // Create an edge with the dependent buffer address for which a command
  // object has been created as one of the properties of the edge
  xpti::payload_t Payload(TypeString.c_str(), MAddress);
  uint64_t EdgeInstanceNo;
  xpti_td *EdgeEvent =
      xptiMakeEvent(TypeString.c_str(), &Payload, xpti::trace_graph_event,
                    xpti_at::active, &EdgeInstanceNo);
  if (EdgeEvent) {
    xpti_td *SrcEvent = static_cast<xpti_td *>(Cmd->MTraceEvent);
    xpti_td *TgtEvent = static_cast<xpti_td *>(MTraceEvent);
    EdgeEvent->source_id = SrcEvent->unique_id;
    EdgeEvent->target_id = TgtEvent->unique_id;
    if (IsCommand) {
      xpti::addMetadata(EdgeEvent, "access_mode",
                        static_cast<int>(AccMode.value()));
      xpti::addMetadata(EdgeEvent, "memory_object",
                        reinterpret_cast<size_t>(ObjAddr));
    } else {
      xpti::addMetadata(EdgeEvent, "event", reinterpret_cast<size_t>(ObjAddr));
    }
    xptiNotifySubscribers(MStreamID, xpti::trace_edge_create,
                          detail::GSYCLGraphEvent, EdgeEvent, EdgeInstanceNo,
                          nullptr);
  }
  // General comment - None of these are serious errors as the instrumentation
  // layer MUST be tolerant of errors. If we need to let the end user know, we
  // throw exceptions in the future
#endif
}

/// Creates an edge when the dependency is due to an event.
/// @param Cmd The command object of the source of the edge
/// @param PiEventAddr The address that defines the edge dependency, which in
/// this case is an event
void Command::emitEdgeEventForEventDependence(Command *Cmd,
                                              RT::PiEvent &PiEventAddr) {
#ifdef XPTI_ENABLE_INSTRUMENTATION
  // If we have failed to create an event to represent the Command, then we
  // cannot emit an edge event. Bail early!
  if (!(xptiTraceEnabled() && MTraceEvent))
    return;

  if (Cmd && Cmd->MTraceEvent) {
    // If the event is associated with a command, we use this command's trace
    // event as the source of edge, hence modeling the control flow
    emitEdgeEventForCommandDependence(Cmd, (void *)PiEventAddr, false);
    return;
  }
  if (PiEventAddr) {
    xpti::utils::StringHelper SH;
    std::string AddressStr = SH.addressAsString<RT::PiEvent>(PiEventAddr);
    // This is the case when it is a OCL event enqueued by the user or another
    // event is registered by the runtime as a dependency The dependency on
    // this occasion is an OCL event; so we build a virtual node in the graph
    // with the event as the metadata for the node
    std::string NodeName = SH.nameWithAddressString("virtual_node", AddressStr);
    // Node name is "virtual_node[<event_addr>]"
    xpti::payload_t VNPayload(NodeName.c_str(), MAddress);
    uint64_t VNodeInstanceNo;
    xpti_td *NodeEvent =
        xptiMakeEvent(NodeName.c_str(), &VNPayload, xpti::trace_graph_event,
                      xpti_at::active, &VNodeInstanceNo);
    // Emit the virtual node first
    xpti::addMetadata(NodeEvent, "kernel_name", NodeName);
    xptiNotifySubscribers(MStreamID, xpti::trace_node_create,
                          detail::GSYCLGraphEvent, NodeEvent, VNodeInstanceNo,
                          nullptr);
    // Create a new event for the edge
    std::string EdgeName = SH.nameWithAddressString("Event", AddressStr);
    xpti::payload_t EdgePayload(EdgeName.c_str(), MAddress);
    uint64_t EdgeInstanceNo;
    xpti_td *EdgeEvent =
        xptiMakeEvent(EdgeName.c_str(), &EdgePayload, xpti::trace_graph_event,
                      xpti_at::active, &EdgeInstanceNo);
    if (EdgeEvent && NodeEvent) {
      // Source node represents the event and this event needs to be completed
      // before target node can execute
      xpti_td *TgtEvent = static_cast<xpti_td *>(MTraceEvent);
      EdgeEvent->source_id = NodeEvent->unique_id;
      EdgeEvent->target_id = TgtEvent->unique_id;
      xpti::addMetadata(EdgeEvent, "event",
                        reinterpret_cast<size_t>(PiEventAddr));
      xptiNotifySubscribers(MStreamID, xpti::trace_edge_create,
                            detail::GSYCLGraphEvent, EdgeEvent, EdgeInstanceNo,
                            nullptr);
    }
    return;
  }
#endif
}

uint64_t Command::makeTraceEventProlog(void *MAddress) {
  uint64_t CommandInstanceNo = 0;
#ifdef XPTI_ENABLE_INSTRUMENTATION
  if (!xptiTraceEnabled())
    return CommandInstanceNo;

  MTraceEventPrologComplete = true;
  // Setup the member variables with information needed for event notification
  MCommandNodeType = commandToNodeType(MType);
  MCommandName = commandToName(MType);
  xpti::utils::StringHelper SH;
  MAddressString = SH.addressAsString<void *>(MAddress);
  std::string CommandString =
      SH.nameWithAddressString(MCommandName, MAddressString);

  xpti::payload_t p(CommandString.c_str(), MAddress);
  xpti_td *CmdTraceEvent =
      xptiMakeEvent(CommandString.c_str(), &p, xpti::trace_graph_event,
                    xpti_at::active, &CommandInstanceNo);
  MInstanceID = CommandInstanceNo;
  if (CmdTraceEvent) {
    MTraceEvent = (void *)CmdTraceEvent;
    // If we are seeing this event again, then the instance ID will be greater
    // than 1; in this case, we must skip sending a notification to create a
    // node as this node has already been created. We return this value so the
    // epilog method can be called selectively.
    MFirstInstance = (CommandInstanceNo == 1);
  }
#endif
  return CommandInstanceNo;
}

void Command::makeTraceEventEpilog() {
#ifdef XPTI_ENABLE_INSTRUMENTATION
  if (!(xptiTraceEnabled() && MTraceEvent))
    return;
  assert(MTraceEventPrologComplete);
  xptiNotifySubscribers(MStreamID, xpti::trace_node_create,
                        detail::GSYCLGraphEvent,
                        static_cast<xpti_td *>(MTraceEvent), MInstanceID,
                        static_cast<const void *>(MCommandNodeType.c_str()));
#endif
}

Command *Command::processDepEvent(EventImplPtr DepEvent, const DepDesc &Dep,
                                  std::vector<Command *> &ToCleanUp) {
  const QueueImplPtr &WorkerQueue = getWorkerQueue();
  const ContextImplPtr &WorkerContext = WorkerQueue->getContextImplPtr();

  // 1. Async work is not supported for host device.
  // 2. Some types of commands do not produce PI events after they are enqueued
  // (e.g. alloca). Note that we can't check the pi event to make that
  // distinction since the command might still be unenqueued at this point.
  bool PiEventExpected =
      !DepEvent->is_host() || getType() == CommandType::HOST_TASK;
  auto *DepCmd = static_cast<Command *>(DepEvent->getCommand());

  if (DepCmd) {
    PiEventExpected &= DepCmd->producesPiEvent();

    // MBlockingExplicitDeps used to avoid graph depth search for empty command
    // to add new command as user to it. It is trade-off between average perf
    // and scenario influence and memory usage. if
    // BlockingCmdEvent->getCommand() returns nullptr - task is completed and
    // event is signalled, not blocking any more so no copy.
    for (auto &BlockingCmdEvent : DepCmd->MBlockingExplicitDeps) {
      if (EmptyCommand *BlockingCmd =
              static_cast<EmptyCommand *>(BlockingCmdEvent->getCommand())) {
        MBlockingExplicitDeps.insert(BlockingCmdEvent);
        BlockingCmd->removeBlockedUser(DepCmd->getEvent());
        BlockingCmd->addBlockedUser(this->MEvent);
      }
    }
  }

  if (!PiEventExpected) {
    // call to waitInternal() is in waitForPreparedHostEvents() as it's called
    // from enqueue process functions

    // Explicit user is a notification approach of host task completion only
    if (DepCmd && DepCmd->getType() == CommandType::EMPTY_TASK) {
      MBlockingExplicitDeps.insert(DepCmd->getEvent());
      EmptyCommand *EmptyCmd = static_cast<EmptyCommand *>(DepCmd);
      EmptyCmd->addBlockedUser(this->MEvent);
    }

    MPreparedHostDepsEvents.push_back(DepEvent);
    return nullptr;
  }

  Command *ConnectionCmd = nullptr;

  // Do not add redundant event dependencies for in-order queues.
  if (Dep.MDepCommand && Dep.MDepCommand->getWorkerQueue() == WorkerQueue &&
      WorkerQueue->has_property<property::queue::in_order>() &&
      getType() != CommandType::HOST_TASK)
    return nullptr;

  ContextImplPtr DepEventContext = DepEvent->getContextImpl();
  // If contexts don't match we'll connect them using host task
  if (DepEventContext != WorkerContext && !WorkerContext->is_host()) {
    Scheduler::GraphBuilder &GB = Scheduler::getInstance().MGraphBuilder;
    ConnectionCmd = GB.connectDepEvent(this, DepEvent, Dep, ToCleanUp);
  } else
    MPreparedDepsEvents.push_back(std::move(DepEvent));

  return ConnectionCmd;
}

const ContextImplPtr &Command::getWorkerContext() const {
  return MQueue->getContextImplPtr();
}

const QueueImplPtr &Command::getWorkerQueue() const { return MQueue; }

bool Command::producesPiEvent() const { return true; }

bool Command::supportsPostEnqueueCleanup() const {
  // Isolated commands are cleaned up separately
  return !MUsers.empty() || !MDeps.empty();
}

Command *Command::addDep(DepDesc NewDep, std::vector<Command *> &ToCleanUp) {
  Command *ConnectionCmd = nullptr;

  if (NewDep.MDepCommand) {
    ConnectionCmd =
        processDepEvent(NewDep.MDepCommand->getEvent(), NewDep, ToCleanUp);
  }
  // ConnectionCmd insertion builds the following dependency structure:
  // this -> emptyCmd (for ConnectionCmd) -> ConnectionCmd -> NewDep
  // that means that this and NewDep are already dependent
  if (!ConnectionCmd) {
    MDeps.push_back(NewDep);
    if (NewDep.MDepCommand)
      NewDep.MDepCommand->addUser(this);
  }

#ifdef XPTI_ENABLE_INSTRUMENTATION
  emitEdgeEventForCommandDependence(NewDep.MDepCommand,
                                    (void *)NewDep.MDepRequirement->MSYCLMemObj,
                                    true, NewDep.MDepRequirement->MAccessMode);
#endif

  return ConnectionCmd;
}

Command *Command::addDep(EventImplPtr Event,
                         std::vector<Command *> &ToCleanUp) {
#ifdef XPTI_ENABLE_INSTRUMENTATION
  // We need this for just the instrumentation, so guarding it will prevent
  // unused variable warnings when instrumentation is turned off
  Command *Cmd = (Command *)Event->getCommand();
  RT::PiEvent &PiEventAddr = Event->getHandleRef();
  // Now make an edge for the dependent event
  emitEdgeEventForEventDependence(Cmd, PiEventAddr);
#endif

  return processDepEvent(std::move(Event), DepDesc{nullptr, nullptr, nullptr},
                         ToCleanUp);
}

void Command::emitEnqueuedEventSignal(RT::PiEvent &PiEventAddr) {
#ifdef XPTI_ENABLE_INSTRUMENTATION
  if (!(xptiTraceEnabled() && MTraceEvent && PiEventAddr))
    return;
  // Asynchronous call, so send a signal with the event information as
  // user_data
  xptiNotifySubscribers(MStreamID, xpti::trace_signal, detail::GSYCLGraphEvent,
                        static_cast<xpti_td *>(MTraceEvent), MInstanceID,
                        (void *)PiEventAddr);
#endif
}

void Command::emitInstrumentation(uint16_t Type, const char *Txt) {
#ifdef XPTI_ENABLE_INSTRUMENTATION
  if (!(xptiTraceEnabled() && MTraceEvent))
    return;
  // Trace event notifier that emits a Type event
  xptiNotifySubscribers(MStreamID, Type, detail::GSYCLGraphEvent,
                        static_cast<xpti_td *>(MTraceEvent), MInstanceID,
                        static_cast<const void *>(Txt));
#endif
}

bool Command::enqueue(EnqueueResultT &EnqueueResult, BlockingT Blocking,
                      std::vector<Command *> &ToCleanUp) {
  // Exit if already enqueued
  if (MEnqueueStatus == EnqueueResultT::SyclEnqueueSuccess)
    return true;

  // If the command is blocked from enqueueing
  if (MIsBlockable && MEnqueueStatus == EnqueueResultT::SyclEnqueueBlocked) {
    // Exit if enqueue type is not blocking
    if (!Blocking) {
      EnqueueResult = EnqueueResultT(EnqueueResultT::SyclEnqueueBlocked, this);
      return false;
    }
    static bool ThrowOnBlock = getenv("SYCL_THROW_ON_BLOCK") != nullptr;
    if (ThrowOnBlock)
      throw sycl::runtime_error(
          std::string("Waiting for blocked command. Block reason: ") +
              std::string(getBlockReason()),
          PI_INVALID_OPERATION);

#ifdef XPTI_ENABLE_INSTRUMENTATION
    // Scoped trace event notifier that emits a barrier begin and barrier end
    // event, which models the barrier while enqueuing along with the blocked
    // reason, as determined by the scheduler
    std::string Info = "enqueue.barrier[";
    Info += std::string(getBlockReason()) + "]";
    emitInstrumentation(xpti::trace_barrier_begin, Info.c_str());
#endif

    // Wait if blocking
    while (MEnqueueStatus == EnqueueResultT::SyclEnqueueBlocked)
      ;
#ifdef XPTI_ENABLE_INSTRUMENTATION
    emitInstrumentation(xpti::trace_barrier_end, Info.c_str());
#endif
  }

  std::lock_guard<std::mutex> Lock(MEnqueueMtx);

  // Exit if the command is already enqueued
  if (MEnqueueStatus == EnqueueResultT::SyclEnqueueSuccess)
    return true;

#ifdef XPTI_ENABLE_INSTRUMENTATION
  emitInstrumentation(xpti::trace_task_begin, nullptr);
#endif

  if (MEnqueueStatus == EnqueueResultT::SyclEnqueueFailed) {
    EnqueueResult = EnqueueResultT(EnqueueResultT::SyclEnqueueFailed, this);
    return false;
  }

  // Command status set to "failed" beforehand, so this command
  // has already been marked as "failed" if enqueueImp throws an exception.
  // This will avoid execution of the same failed command twice.
  MEnqueueStatus = EnqueueResultT::SyclEnqueueFailed;
  MShouldCompleteEventIfPossible = true;
  cl_int Res = enqueueImp();

  if (CL_SUCCESS != Res)
    EnqueueResult =
        EnqueueResultT(EnqueueResultT::SyclEnqueueFailed, this, Res);
  else {
    if (MShouldCompleteEventIfPossible &&
        (MEvent->is_host() || MEvent->getHandleRef() == nullptr))
      MEvent->setComplete();

    // Consider the command is successfully enqueued if return code is
    // CL_SUCCESS
    MEnqueueStatus = EnqueueResultT::SyclEnqueueSuccess;
    if (MLeafCounter == 0 && supportsPostEnqueueCleanup() &&
        !SYCLConfig<SYCL_DISABLE_POST_ENQUEUE_CLEANUP>::get()) {
      assert(!MPostEnqueueCleanup);
      MPostEnqueueCleanup = true;
      ToCleanUp.push_back(this);
    }
  }

  // Emit this correlation signal before the task end
  emitEnqueuedEventSignal(MEvent->getHandleRef());
#ifdef XPTI_ENABLE_INSTRUMENTATION
  emitInstrumentation(xpti::trace_task_end, nullptr);
#endif
  return MEnqueueStatus == EnqueueResultT::SyclEnqueueSuccess;
}

void Command::resolveReleaseDependencies(std::set<Command *> &DepList) {
#ifdef XPTI_ENABLE_INSTRUMENTATION
  assert(MType == CommandType::RELEASE && "Expected release command");
  if (!MTraceEvent)
    return;
  // The current command is the target node for all dependencies as the source
  // nodes have to be completed first before the current node can begin to
  // execute; these edges model control flow
  xpti_td *TgtTraceEvent = static_cast<xpti_td *>(MTraceEvent);
  // We have all the Commands that must be completed before the release command
  // can be enqueued; here we'll find the command that is an Alloca with the
  // same SYCLMemObject address and create a dependency line (edge) between them
  // in our sematic modeling
  for (auto &Item : DepList) {
    if (Item->MTraceEvent && Item->MAddress == MAddress) {
      xpti::utils::StringHelper SH;
      std::string AddressStr = SH.addressAsString<void *>(MAddress);
      std::string TypeString =
          "Edge:" + SH.nameWithAddressString(commandToName(MType), AddressStr);

      // Create an edge with the dependent buffer address being one of the
      // properties of the edge
      xpti::payload_t p(TypeString.c_str(), MAddress);
      uint64_t EdgeInstanceNo;
      xpti_td *EdgeEvent =
          xptiMakeEvent(TypeString.c_str(), &p, xpti::trace_graph_event,
                        xpti_at::active, &EdgeInstanceNo);
      if (EdgeEvent) {
        xpti_td *SrcTraceEvent = static_cast<xpti_td *>(Item->MTraceEvent);
        EdgeEvent->target_id = TgtTraceEvent->unique_id;
        EdgeEvent->source_id = SrcTraceEvent->unique_id;
        xpti::addMetadata(EdgeEvent, "memory_object",
                          reinterpret_cast<size_t>(MAddress));
        xptiNotifySubscribers(MStreamID, xpti::trace_edge_create,
                              detail::GSYCLGraphEvent, EdgeEvent,
                              EdgeInstanceNo, nullptr);
      }
    }
  }
#endif
}

const char *Command::getBlockReason() const {
  switch (MBlockReason) {
  case BlockReason::HostAccessor:
    return "A Buffer is locked by the host accessor";
  case BlockReason::HostTask:
    return "Blocked by host task";
  }

  return "Unknown block reason";
}

void EmptyCommand::addBlockedUser(const EventImplPtr &NewUser) {
  MBlockedUsers.insert(NewUser);
}
void EmptyCommand::removeBlockedUser(const EventImplPtr &User) {
  MBlockedUsers.erase(User);
}
const std::unordered_set<EventImplPtr> &EmptyCommand::getBlockedUsers() const {
  return MBlockedUsers;
}

AllocaCommandBase::AllocaCommandBase(CommandType Type, QueueImplPtr Queue,
                                     Requirement Req,
                                     AllocaCommandBase *LinkedAllocaCmd)
    : Command(Type, Queue), MLinkedAllocaCmd(LinkedAllocaCmd),
      MIsLeaderAlloca(nullptr == LinkedAllocaCmd), MRequirement(std::move(Req)),
      MReleaseCmd(Queue, this) {
  MRequirement.MAccessMode = access::mode::read_write;
  emitInstrumentationDataProxy();
}

void AllocaCommandBase::emitInstrumentationData() {
#ifdef XPTI_ENABLE_INSTRUMENTATION
  if (!xptiTraceEnabled())
    return;
  // Create a payload with the command name and an event using this payload to
  // emit a node_create
  MAddress = MRequirement.MSYCLMemObj;
  makeTraceEventProlog(MAddress);
  // Set the relevant meta data properties for this command
  if (MTraceEvent && MFirstInstance) {
    xpti_td *TE = static_cast<xpti_td *>(MTraceEvent);
    xpti::addMetadata(TE, "sycl_device", deviceToID(MQueue->get_device()));
    xpti::addMetadata(TE, "sycl_device_type",
                      deviceToString(MQueue->get_device()));
    xpti::addMetadata(TE, "sycl_device_name",
                      getSyclObjImpl(MQueue->get_device())->getDeviceName());
    xpti::addMetadata(TE, "memory_object", reinterpret_cast<size_t>(MAddress));
  }
#endif
}

bool AllocaCommandBase::producesPiEvent() const { return false; }

bool AllocaCommandBase::supportsPostEnqueueCleanup() const { return false; }

AllocaCommand::AllocaCommand(QueueImplPtr Queue, Requirement Req,
                             bool InitFromUserData,
                             AllocaCommandBase *LinkedAllocaCmd)
    : AllocaCommandBase(CommandType::ALLOCA, std::move(Queue), std::move(Req),
                        LinkedAllocaCmd),
      MInitFromUserData(InitFromUserData) {
  // Node event must be created before the dependent edge is added to this node,
  // so this call must be before the addDep() call.
  emitInstrumentationDataProxy();
  // "Nothing to depend on"
  std::vector<Command *> ToCleanUp;
  Command *ConnectionCmd =
      addDep(DepDesc(nullptr, getRequirement(), this), ToCleanUp);
  assert(ConnectionCmd == nullptr);
  assert(ToCleanUp.empty());
  (void)ConnectionCmd;
}

void AllocaCommand::emitInstrumentationData() {
#ifdef XPTI_ENABLE_INSTRUMENTATION
  if (!xptiTraceEnabled())
    return;

  // Only if it is the first event, we emit a node create event
  if (MFirstInstance) {
    makeTraceEventEpilog();
  }
#endif
}

cl_int AllocaCommand::enqueueImp() {
  waitForPreparedHostEvents();
  std::vector<EventImplPtr> EventImpls = MPreparedDepsEvents;

  RT::PiEvent &Event = MEvent->getHandleRef();

  void *HostPtr = nullptr;
  if (!MIsLeaderAlloca) {

    if (MQueue->is_host()) {
      // Do not need to make allocation if we have a linked device allocation
      Command::waitForEvents(MQueue, EventImpls, Event);

      return CL_SUCCESS;
    }
    HostPtr = MLinkedAllocaCmd->getMemAllocation();
  }
  // TODO: Check if it is correct to use std::move on stack variable and
  // delete it RawEvents below.
  MMemAllocation = MemoryManager::allocate(
      MQueue->getContextImplPtr(), getSYCLMemObj(), MInitFromUserData, HostPtr,
      std::move(EventImpls), Event);

  return CL_SUCCESS;
}

void AllocaCommand::printDot(std::ostream &Stream) const {
  Stream << "\"" << this << "\" [style=filled, fillcolor=\"#FFD28A\", label=\"";

  Stream << "ID = " << this << "\\n";
  Stream << "ALLOCA ON " << deviceToString(MQueue->get_device()) << "\\n";
  Stream << " MemObj : " << this->MRequirement.MSYCLMemObj << "\\n";
  Stream << " Link : " << this->MLinkedAllocaCmd << "\\n";
  Stream << "\"];" << std::endl;


  for (const auto &Dep : MDeps) {
    if (Dep.MDepCommand == nullptr)
      continue;
    Stream << "  \"" << this << "\" -> \"" << Dep.MDepCommand << "\""
           << " [ label = \"Access mode: "
           << accessModeToString(Dep.MDepRequirement->MAccessMode) << "\\n"
           << "MemObj: " << Dep.MDepRequirement->MSYCLMemObj << " \" ]"
           << std::endl;
  }
}

AllocaSubBufCommand::AllocaSubBufCommand(QueueImplPtr Queue, Requirement Req,
                                         AllocaCommandBase *ParentAlloca,
                                         std::vector<Command *> &ToEnqueue,
                                         std::vector<Command *> &ToCleanUp)
    : AllocaCommandBase(CommandType::ALLOCA_SUB_BUF, std::move(Queue),
                        std::move(Req),
                        /*LinkedAllocaCmd*/ nullptr),
      MParentAlloca(ParentAlloca) {
  // Node event must be created before the dependent edge
  // is added to this node, so this call must be before
  // the addDep() call.
  emitInstrumentationDataProxy();
  Command *ConnectionCmd = addDep(
      DepDesc(MParentAlloca, getRequirement(), MParentAlloca), ToCleanUp);
  if (ConnectionCmd)
    ToEnqueue.push_back(ConnectionCmd);
}

void AllocaSubBufCommand::emitInstrumentationData() {
#ifdef XPTI_ENABLE_INSTRUMENTATION
  if (!xptiTraceEnabled())
    return;

  // Only if it is the first event, we emit a node create event and any meta
  // data that is available for the command
  if (MFirstInstance) {
    xpti_td *TE = static_cast<xpti_td *>(MTraceEvent);
    xpti::addMetadata(TE, "offset", this->MRequirement.MOffsetInBytes);
    xpti::addMetadata(TE, "access_range_start",
                      this->MRequirement.MAccessRange[0]);
    xpti::addMetadata(TE, "access_range_end",
                      this->MRequirement.MAccessRange[1]);
    makeTraceEventEpilog();
  }
#endif
}

void *AllocaSubBufCommand::getMemAllocation() const {
  // In some cases parent`s memory allocation might change (e.g., after
  // map/unmap operations). If parent`s memory allocation changes, sub-buffer
  // memory allocation should be changed as well.
  if (MQueue->is_host()) {
    return static_cast<void *>(
        static_cast<char *>(MParentAlloca->getMemAllocation()) +
        MRequirement.MOffsetInBytes);
  }
  return MMemAllocation;
}

cl_int AllocaSubBufCommand::enqueueImp() {
  waitForPreparedHostEvents();
  std::vector<EventImplPtr> EventImpls = MPreparedDepsEvents;
  RT::PiEvent &Event = MEvent->getHandleRef();

  MMemAllocation = MemoryManager::allocateMemSubBuffer(
      MQueue->getContextImplPtr(), MParentAlloca->getMemAllocation(),
      MRequirement.MElemSize, MRequirement.MOffsetInBytes,
      MRequirement.MAccessRange, std::move(EventImpls), Event);

  XPTIRegistry::bufferAssociateNotification(MParentAlloca->getSYCLMemObj(),
                                            MMemAllocation);
  return CL_SUCCESS;
}

void AllocaSubBufCommand::printDot(std::ostream &Stream) const {
  Stream << "\"" << this << "\" [style=filled, fillcolor=\"#FFD28A\", label=\"";

  Stream << "ID = " << this << "\\n";
  Stream << "ALLOCA SUB BUF ON " << deviceToString(MQueue->get_device())
         << "\\n";
  Stream << " MemObj : " << this->MRequirement.MSYCLMemObj << "\\n";
  Stream << " Offset : " << this->MRequirement.MOffsetInBytes << "\\n";
  Stream << " Access range : " << this->MRequirement.MAccessRange[0] << "\\n";
  Stream << "\"];" << std::endl;

  for (const auto &Dep : MDeps) {
    if (Dep.MDepCommand == nullptr)
      continue;
    Stream << "  \"" << this << "\" -> \"" << Dep.MDepCommand << "\""
           << " [ label = \"Access mode: "
           << accessModeToString(Dep.MDepRequirement->MAccessMode) << "\\n"
           << "MemObj: " << Dep.MDepRequirement->MSYCLMemObj << " \" ]"
           << std::endl;
  }
}

ReleaseCommand::ReleaseCommand(QueueImplPtr Queue, AllocaCommandBase *AllocaCmd)
    : Command(CommandType::RELEASE, std::move(Queue)), MAllocaCmd(AllocaCmd) {
  emitInstrumentationDataProxy();
}

void ReleaseCommand::emitInstrumentationData() {
#ifdef XPTI_ENABLE_INSTRUMENTATION
  if (!xptiTraceEnabled())
    return;
  // Create a payload with the command name and an event using this payload to
  // emit a node_create
  MAddress = MAllocaCmd->getSYCLMemObj();
  makeTraceEventProlog(MAddress);

  if (MFirstInstance) {
    xpti_td *TE = static_cast<xpti_td *>(MTraceEvent);
    xpti::addMetadata(TE, "sycl_device", deviceToID(MQueue->get_device()));
    xpti::addMetadata(TE, "sycl_device_type",
                      deviceToString(MQueue->get_device()));
    xpti::addMetadata(TE, "sycl_device_name",
                      getSyclObjImpl(MQueue->get_device())->getDeviceName());
    xpti::addMetadata(TE, "allocation_type",
                      commandToName(MAllocaCmd->getType()));
    makeTraceEventEpilog();
  }
#endif
}

cl_int ReleaseCommand::enqueueImp() {
  waitForPreparedHostEvents();
  std::vector<EventImplPtr> EventImpls = MPreparedDepsEvents;
  std::vector<RT::PiEvent> RawEvents = getPiEvents(EventImpls);
  bool SkipRelease = false;

  // On host side we only allocate memory for full buffers.
  // Thus, deallocating sub buffers leads to double memory freeing.
  SkipRelease |= MQueue->is_host() && MAllocaCmd->getType() == ALLOCA_SUB_BUF;

  const bool CurAllocaIsHost = MAllocaCmd->getQueue()->is_host();
  bool NeedUnmap = false;
  if (MAllocaCmd->MLinkedAllocaCmd) {

    // When releasing one of the "linked" allocations special rules take place:
    // 1. Device allocation should always be released.
    // 2. Host allocation should be released if host allocation is "leader".
    // 3. Device alloca in the pair should be in active state in order to be
    //    correctly released.


    // There is no actual memory allocation if a host alloca command is created
    // being linked to a device allocation.
    SkipRelease |= CurAllocaIsHost && !MAllocaCmd->MIsLeaderAlloca;

    NeedUnmap |= CurAllocaIsHost == MAllocaCmd->MIsActive;
  }

  if (NeedUnmap) {
    const QueueImplPtr &Queue = CurAllocaIsHost
                                    ? MAllocaCmd->MLinkedAllocaCmd->getQueue()
                                    : MAllocaCmd->getQueue();

    EventImplPtr UnmapEventImpl(new event_impl(Queue));
    UnmapEventImpl->setContextImpl(Queue->getContextImplPtr());
    RT::PiEvent &UnmapEvent = UnmapEventImpl->getHandleRef();

    void *Src = CurAllocaIsHost
                    ? MAllocaCmd->getMemAllocation()
                    : MAllocaCmd->MLinkedAllocaCmd->getMemAllocation();

    void *Dst = !CurAllocaIsHost
                    ? MAllocaCmd->getMemAllocation()
                    : MAllocaCmd->MLinkedAllocaCmd->getMemAllocation();

    MemoryManager::unmap(MAllocaCmd->getSYCLMemObj(), Dst, Queue, Src,
                         RawEvents, UnmapEvent);

    std::swap(MAllocaCmd->MIsActive, MAllocaCmd->MLinkedAllocaCmd->MIsActive);
    EventImpls.clear();
    EventImpls.push_back(UnmapEventImpl);
  }
  RT::PiEvent &Event = MEvent->getHandleRef();
  if (SkipRelease)
    Command::waitForEvents(MQueue, EventImpls, Event);
  else {
    MemoryManager::release(
        MQueue->getContextImplPtr(), MAllocaCmd->getSYCLMemObj(),
        MAllocaCmd->getMemAllocation(), std::move(EventImpls), Event);
  }
  return CL_SUCCESS;
}

void ReleaseCommand::printDot(std::ostream &Stream) const {
  Stream << "\"" << this << "\" [style=filled, fillcolor=\"#FF827A\", label=\"";

  Stream << "ID = " << this << " ; ";
  Stream << "RELEASE ON " << deviceToString(MQueue->get_device()) << "\\n";
  Stream << " Alloca : " << MAllocaCmd << "\\n";
  Stream << " MemObj : " << MAllocaCmd->getSYCLMemObj() << "\\n";
  Stream << "\"];" << std::endl;

  for (const auto &Dep : MDeps) {
    Stream << "  \"" << this << "\" -> \"" << Dep.MDepCommand << "\""
           << " [ label = \"Access mode: "
           << accessModeToString(Dep.MDepRequirement->MAccessMode) << "\\n"
           << "MemObj: " << Dep.MDepRequirement->MSYCLMemObj << " \" ]"
           << std::endl;
  }
}

bool ReleaseCommand::producesPiEvent() const { return false; }

bool ReleaseCommand::supportsPostEnqueueCleanup() const { return false; }

MapMemObject::MapMemObject(AllocaCommandBase *SrcAllocaCmd, Requirement Req,
                           void **DstPtr, QueueImplPtr Queue,
                           access::mode MapMode)
    : Command(CommandType::MAP_MEM_OBJ, std::move(Queue)),
      MSrcAllocaCmd(SrcAllocaCmd), MSrcReq(std::move(Req)), MDstPtr(DstPtr),
      MMapMode(MapMode) {
  emitInstrumentationDataProxy();
}

void MapMemObject::emitInstrumentationData() {
#ifdef XPTI_ENABLE_INSTRUMENTATION
  if (!xptiTraceEnabled())
    return;
  // Create a payload with the command name and an event using this payload to
  // emit a node_create
  MAddress = MSrcAllocaCmd->getSYCLMemObj();
  makeTraceEventProlog(MAddress);

  if (MFirstInstance) {
    xpti_td *TE = static_cast<xpti_td *>(MTraceEvent);
    xpti::addMetadata(TE, "sycl_device", deviceToID(MQueue->get_device()));
    xpti::addMetadata(TE, "sycl_device_type",
                      deviceToString(MQueue->get_device()));
    xpti::addMetadata(TE, "sycl_device_name",
                      getSyclObjImpl(MQueue->get_device())->getDeviceName());
    xpti::addMetadata(TE, "memory_object", reinterpret_cast<size_t>(MAddress));
    makeTraceEventEpilog();
  }
#endif
}

cl_int MapMemObject::enqueueImp() {
  waitForPreparedHostEvents();
  std::vector<EventImplPtr> EventImpls = MPreparedDepsEvents;
  std::vector<RT::PiEvent> RawEvents = getPiEvents(EventImpls);
  flushCrossQueueDeps(EventImpls, getWorkerQueue());

  RT::PiEvent &Event = MEvent->getHandleRef();
  *MDstPtr = MemoryManager::map(
      MSrcAllocaCmd->getSYCLMemObj(), MSrcAllocaCmd->getMemAllocation(), MQueue,
      MMapMode, MSrcReq.MDims, MSrcReq.MMemoryRange, MSrcReq.MAccessRange,
      MSrcReq.MOffset, MSrcReq.MElemSize, std::move(RawEvents), Event);

  return CL_SUCCESS;
}

void MapMemObject::printDot(std::ostream &Stream) const {
  Stream << "\"" << this << "\" [style=filled, fillcolor=\"#77AFFF\", label=\"";

  Stream << "ID = " << this << " ; ";
  Stream << "MAP ON " << deviceToString(MQueue->get_device()) << "\\n";

  Stream << "\"];" << std::endl;

  for (const auto &Dep : MDeps) {
    Stream << "  \"" << this << "\" -> \"" << Dep.MDepCommand << "\""
           << " [ label = \"Access mode: "
           << accessModeToString(Dep.MDepRequirement->MAccessMode) << "\\n"
           << "MemObj: " << Dep.MDepRequirement->MSYCLMemObj << " \" ]"
           << std::endl;
  }
}

UnMapMemObject::UnMapMemObject(AllocaCommandBase *DstAllocaCmd, Requirement Req,
                               void **SrcPtr, QueueImplPtr Queue)
    : Command(CommandType::UNMAP_MEM_OBJ, std::move(Queue)),
      MDstAllocaCmd(DstAllocaCmd), MDstReq(std::move(Req)), MSrcPtr(SrcPtr) {
  emitInstrumentationDataProxy();
}

void UnMapMemObject::emitInstrumentationData() {
#ifdef XPTI_ENABLE_INSTRUMENTATION
  if (!xptiTraceEnabled())
    return;
  // Create a payload with the command name and an event using this payload to
  // emit a node_create
  MAddress = MDstAllocaCmd->getSYCLMemObj();
  makeTraceEventProlog(MAddress);

  if (MFirstInstance) {
    xpti_td *TE = static_cast<xpti_td *>(MTraceEvent);
    xpti::addMetadata(TE, "sycl_device", deviceToID(MQueue->get_device()));
    xpti::addMetadata(TE, "sycl_device_type",
                      deviceToString(MQueue->get_device()));
    xpti::addMetadata(TE, "sycl_device_name",
                      getSyclObjImpl(MQueue->get_device())->getDeviceName());
    xpti::addMetadata(TE, "memory_object", reinterpret_cast<size_t>(MAddress));
    makeTraceEventEpilog();
  }
#endif
}

bool UnMapMemObject::producesPiEvent() const {
  // TODO remove this workaround once the batching issue is addressed in Level
  // Zero plugin.
  // Consider the following scenario on Level Zero:
  // 1. Kernel A, which uses buffer A, is submitted to queue A.
  // 2. Kernel B, which uses buffer B, is submitted to queue B.
  // 3. queueA.wait().
  // 4. queueB.wait().
  // DPCPP runtime used to treat unmap/write commands for buffer A/B as host
  // dependencies (i.e. they were waited for prior to enqueueing any command
  // that's dependent on them). This allowed Level Zero plugin to detect that
  // each queue is idle on steps 1/2 and submit the command list right away.
  // This is no longer the case since we started passing these dependencies in
  // an event waitlist and Level Zero plugin attempts to batch these commands,
  // so the execution of kernel B starts only on step 4. This workaround
  // restores the old behavior in this case until this is resolved.
  return MQueue->getPlugin().getBackend() != backend::ext_oneapi_level_zero ||
         MEvent->getHandleRef() != nullptr;
}

cl_int UnMapMemObject::enqueueImp() {
  waitForPreparedHostEvents();
  std::vector<EventImplPtr> EventImpls = MPreparedDepsEvents;
  std::vector<RT::PiEvent> RawEvents = getPiEvents(EventImpls);
  flushCrossQueueDeps(EventImpls, getWorkerQueue());

  RT::PiEvent &Event = MEvent->getHandleRef();
  MemoryManager::unmap(MDstAllocaCmd->getSYCLMemObj(),
                       MDstAllocaCmd->getMemAllocation(), MQueue, *MSrcPtr,
                       std::move(RawEvents), Event);

  return CL_SUCCESS;
}

void UnMapMemObject::printDot(std::ostream &Stream) const {
  Stream << "\"" << this << "\" [style=filled, fillcolor=\"#EBC40F\", label=\"";

  Stream << "ID = " << this << " ; ";
  Stream << "UNMAP ON " << deviceToString(MQueue->get_device()) << "\\n";

  Stream << "\"];" << std::endl;

  for (const auto &Dep : MDeps) {
    Stream << "  \"" << this << "\" -> \"" << Dep.MDepCommand << "\""
           << " [ label = \"Access mode: "
           << accessModeToString(Dep.MDepRequirement->MAccessMode) << "\\n"
           << "MemObj: " << Dep.MDepRequirement->MSYCLMemObj << " \" ]"
           << std::endl;
  }
}

MemCpyCommand::MemCpyCommand(Requirement SrcReq,
                             AllocaCommandBase *SrcAllocaCmd,
                             Requirement DstReq,
                             AllocaCommandBase *DstAllocaCmd,
                             QueueImplPtr SrcQueue, QueueImplPtr DstQueue)
    : Command(CommandType::COPY_MEMORY, std::move(DstQueue)),
      MSrcQueue(SrcQueue), MSrcReq(std::move(SrcReq)),
      MSrcAllocaCmd(SrcAllocaCmd), MDstReq(std::move(DstReq)),
      MDstAllocaCmd(DstAllocaCmd) {
  if (!MSrcQueue->is_host())
    MEvent->setContextImpl(MSrcQueue->getContextImplPtr());

  emitInstrumentationDataProxy();
}

void MemCpyCommand::emitInstrumentationData() {
#ifdef XPTI_ENABLE_INSTRUMENTATION
  if (!xptiTraceEnabled())
    return;
  // Create a payload with the command name and an event using this payload to
  // emit a node_create
  MAddress = MSrcAllocaCmd->getSYCLMemObj();
  makeTraceEventProlog(MAddress);

  if (MFirstInstance) {
    xpti_td *CmdTraceEvent = static_cast<xpti_td *>(MTraceEvent);
    xpti::addMetadata(CmdTraceEvent, "sycl_device",
                      deviceToID(MQueue->get_device()));
    xpti::addMetadata(CmdTraceEvent, "sycl_device_type",
                      deviceToString(MQueue->get_device()));
    xpti::addMetadata(CmdTraceEvent, "sycl_device_name",
                      getSyclObjImpl(MQueue->get_device())->getDeviceName());
    xpti::addMetadata(CmdTraceEvent, "memory_object",
                      reinterpret_cast<size_t>(MAddress));
    xpti::addMetadata(CmdTraceEvent, "copy_from",
                      reinterpret_cast<size_t>(
                          getSyclObjImpl(MSrcQueue->get_device()).get()));
    xpti::addMetadata(
        CmdTraceEvent, "copy_to",
        reinterpret_cast<size_t>(getSyclObjImpl(MQueue->get_device()).get()));
    makeTraceEventEpilog();
  }
#endif
}

const ContextImplPtr &MemCpyCommand::getWorkerContext() const {
  return getWorkerQueue()->getContextImplPtr();
}

const QueueImplPtr &MemCpyCommand::getWorkerQueue() const {
  return MQueue->is_host() ? MSrcQueue : MQueue;
}

bool MemCpyCommand::producesPiEvent() const {
  // TODO remove this workaround once the batching issue is addressed in Level
  // Zero plugin.
  // Consider the following scenario on Level Zero:
  // 1. Kernel A, which uses buffer A, is submitted to queue A.
  // 2. Kernel B, which uses buffer B, is submitted to queue B.
  // 3. queueA.wait().
  // 4. queueB.wait().
  // DPCPP runtime used to treat unmap/write commands for buffer A/B as host
  // dependencies (i.e. they were waited for prior to enqueueing any command
  // that's dependent on them). This allowed Level Zero plugin to detect that
  // each queue is idle on steps 1/2 and submit the command list right away.
  // This is no longer the case since we started passing these dependencies in
  // an event waitlist and Level Zero plugin attempts to batch these commands,
  // so the execution of kernel B starts only on step 4. This workaround
  // restores the old behavior in this case until this is resolved.
  return MQueue->is_host() ||
         MQueue->getPlugin().getBackend() != backend::ext_oneapi_level_zero ||
         MEvent->getHandleRef() != nullptr;
}

cl_int MemCpyCommand::enqueueImp() {
  waitForPreparedHostEvents();
  std::vector<EventImplPtr> EventImpls = MPreparedDepsEvents;

  RT::PiEvent &Event = MEvent->getHandleRef();

  auto RawEvents = getPiEvents(EventImpls);
  flushCrossQueueDeps(EventImpls, getWorkerQueue());

  MemoryManager::copy(
      MSrcAllocaCmd->getSYCLMemObj(), MSrcAllocaCmd->getMemAllocation(),
      MSrcQueue, MSrcReq.MDims, MSrcReq.MMemoryRange, MSrcReq.MAccessRange,
      MSrcReq.MOffset, MSrcReq.MElemSize, MDstAllocaCmd->getMemAllocation(),
      MQueue, MDstReq.MDims, MDstReq.MMemoryRange, MDstReq.MAccessRange,
      MDstReq.MOffset, MDstReq.MElemSize, std::move(RawEvents), Event);

  return CL_SUCCESS;
}

void MemCpyCommand::printDot(std::ostream &Stream) const {
  Stream << "\"" << this << "\" [style=filled, fillcolor=\"#C7EB15\" label=\"";

  Stream << "ID = " << this << " ; ";
  Stream << "MEMCPY ON " << deviceToString(MQueue->get_device()) << "\\n";
  Stream << "From: " << MSrcAllocaCmd << " is host: " << MSrcQueue->is_host()
         << "\\n";
  Stream << "To: " << MDstAllocaCmd << " is host: " << MQueue->is_host()
         << "\\n";

  Stream << "\"];" << std::endl;

  for (const auto &Dep : MDeps) {
    Stream << "  \"" << this << "\" -> \"" << Dep.MDepCommand << "\""
           << " [ label = \"Access mode: "
           << accessModeToString(Dep.MDepRequirement->MAccessMode) << "\\n"
           << "MemObj: " << Dep.MDepRequirement->MSYCLMemObj << " \" ]"
           << std::endl;
  }
}

AllocaCommandBase *ExecCGCommand::getAllocaForReq(Requirement *Req) {
  for (const DepDesc &Dep : MDeps) {
    if (Dep.MDepRequirement == Req)
      return Dep.MAllocaCmd;
  }
  throw runtime_error("Alloca for command not found", PI_INVALID_OPERATION);
}

std::vector<StreamImplPtr> ExecCGCommand::getStreams() const {
  if (MCommandGroup->getType() == CG::Kernel)
    return ((CGExecKernel *)MCommandGroup.get())->getStreams();
  return {};
}

void ExecCGCommand::clearStreams() {
  if (MCommandGroup->getType() == CG::Kernel)
    ((CGExecKernel *)MCommandGroup.get())->clearStreams();
}

cl_int UpdateHostRequirementCommand::enqueueImp() {
  waitForPreparedHostEvents();
  std::vector<EventImplPtr> EventImpls = MPreparedDepsEvents;
  RT::PiEvent &Event = MEvent->getHandleRef();
  Command::waitForEvents(MQueue, EventImpls, Event);

  assert(MSrcAllocaCmd && "Expected valid alloca command");
  assert(MSrcAllocaCmd->getMemAllocation() && "Expected valid source pointer");
  assert(MDstPtr && "Expected valid target pointer");
  *MDstPtr = MSrcAllocaCmd->getMemAllocation();

  return CL_SUCCESS;
}

void UpdateHostRequirementCommand::printDot(std::ostream &Stream) const {
  Stream << "\"" << this << "\" [style=filled, fillcolor=\"#f1337f\", label=\"";

  Stream << "ID = " << this << "\\n";
  Stream << "UPDATE REQ ON " << deviceToString(MQueue->get_device()) << "\\n";
  bool IsReqOnBuffer =
      MDstReq.MSYCLMemObj->getType() == SYCLMemObjI::MemObjType::Buffer;
  Stream << "TYPE: " << (IsReqOnBuffer ? "Buffer" : "Image") << "\\n";
  if (IsReqOnBuffer)
    Stream << "Is sub buffer: " << std::boolalpha << MDstReq.MIsSubBuffer
           << "\\n";

  Stream << "\"];" << std::endl;

  for (const auto &Dep : MDeps) {
    Stream << "  \"" << this << "\" -> \"" << Dep.MDepCommand << "\""
           << " [ label = \"Access mode: "
           << accessModeToString(Dep.MDepRequirement->MAccessMode) << "\\n"
           << "MemObj: " << Dep.MAllocaCmd->getSYCLMemObj() << " \" ]"
           << std::endl;
  }
}

MemCpyCommandHost::MemCpyCommandHost(Requirement SrcReq,
                                     AllocaCommandBase *SrcAllocaCmd,
                                     Requirement DstReq, void **DstPtr,
                                     QueueImplPtr SrcQueue,
                                     QueueImplPtr DstQueue)
    : Command(CommandType::COPY_MEMORY, std::move(DstQueue)),
      MSrcQueue(SrcQueue), MSrcReq(std::move(SrcReq)),
      MSrcAllocaCmd(SrcAllocaCmd), MDstReq(std::move(DstReq)), MDstPtr(DstPtr) {
  if (!MSrcQueue->is_host())
    MEvent->setContextImpl(MSrcQueue->getContextImplPtr());

  emitInstrumentationDataProxy();
}

void MemCpyCommandHost::emitInstrumentationData() {
#ifdef XPTI_ENABLE_INSTRUMENTATION
  if (!xptiTraceEnabled())
    return;
  // Create a payload with the command name and an event using this payload to
  // emit a node_create
  MAddress = MSrcAllocaCmd->getSYCLMemObj();
  makeTraceEventProlog(MAddress);

  if (MFirstInstance) {
    xpti_td *CmdTraceEvent = static_cast<xpti_td *>(MTraceEvent);
    xpti::addMetadata(CmdTraceEvent, "sycl_device",
                      deviceToID(MQueue->get_device()));
    xpti::addMetadata(CmdTraceEvent, "sycl_device_type",
                      deviceToString(MQueue->get_device()));
    xpti::addMetadata(CmdTraceEvent, "sycl_device_name",
                      getSyclObjImpl(MQueue->get_device())->getDeviceName());
    xpti::addMetadata(CmdTraceEvent, "memory_object",
                      reinterpret_cast<size_t>(MAddress));
    xpti::addMetadata(CmdTraceEvent, "copy_from",
                      reinterpret_cast<size_t>(
                          getSyclObjImpl(MSrcQueue->get_device()).get()));
    xpti::addMetadata(
        CmdTraceEvent, "copy_to",
        reinterpret_cast<size_t>(getSyclObjImpl(MQueue->get_device()).get()));
    makeTraceEventEpilog();
  }
#endif
}

const ContextImplPtr &MemCpyCommandHost::getWorkerContext() const {
  return getWorkerQueue()->getContextImplPtr();
}

const QueueImplPtr &MemCpyCommandHost::getWorkerQueue() const {
  return MQueue->is_host() ? MSrcQueue : MQueue;
}

cl_int MemCpyCommandHost::enqueueImp() {
  const QueueImplPtr &Queue = getWorkerQueue();
  waitForPreparedHostEvents();
  std::vector<EventImplPtr> EventImpls = MPreparedDepsEvents;
  std::vector<RT::PiEvent> RawEvents = getPiEvents(EventImpls);

  RT::PiEvent &Event = MEvent->getHandleRef();
  // Omit copying if mode is discard one.
  // TODO: Handle this at the graph building time by, for example, creating
  // empty node instead of memcpy.
  if (MDstReq.MAccessMode == access::mode::discard_read_write ||
      MDstReq.MAccessMode == access::mode::discard_write) {
    Command::waitForEvents(Queue, EventImpls, Event);

    return CL_SUCCESS;
  }

  flushCrossQueueDeps(EventImpls, getWorkerQueue());
  MemoryManager::copy(
      MSrcAllocaCmd->getSYCLMemObj(), MSrcAllocaCmd->getMemAllocation(),
      MSrcQueue, MSrcReq.MDims, MSrcReq.MMemoryRange, MSrcReq.MAccessRange,
      MSrcReq.MOffset, MSrcReq.MElemSize, *MDstPtr, MQueue, MDstReq.MDims,
      MDstReq.MMemoryRange, MDstReq.MAccessRange, MDstReq.MOffset,
      MDstReq.MElemSize, std::move(RawEvents), Event);

  return CL_SUCCESS;
}

EmptyCommand::EmptyCommand(QueueImplPtr Queue)
    : Command(CommandType::EMPTY_TASK, std::move(Queue)) {
  emitInstrumentationDataProxy();
}

cl_int EmptyCommand::enqueueImp() {
  waitForPreparedHostEvents();
  waitForEvents(MQueue, MPreparedDepsEvents, MEvent->getHandleRef());

  return CL_SUCCESS;
}

void EmptyCommand::addRequirement(Command *DepCmd, AllocaCommandBase *AllocaCmd,
                                  const Requirement *Req) {
  const Requirement &ReqRef = *Req;
  MRequirements.emplace_back(ReqRef);
  const Requirement *const StoredReq = &MRequirements.back();

  // EmptyCommand is always host one, so we believe that result of addDep is nil
  std::vector<Command *> ToCleanUp;
  Command *Cmd = addDep(DepDesc{DepCmd, StoredReq, AllocaCmd}, ToCleanUp);
  assert(Cmd == nullptr && "Conection command should be null for EmptyCommand");
  assert(ToCleanUp.empty() && "addDep should add a command for cleanup only if "
                              "there's a connection command");
  (void)Cmd;
}

void EmptyCommand::emitInstrumentationData() {
#ifdef XPTI_ENABLE_INSTRUMENTATION
  if (!xptiTraceEnabled())
    return;
  // Create a payload with the command name and an event using this payload to
  // emit a node_create
  if (MRequirements.empty())
    return;

  Requirement &Req = *MRequirements.begin();

  MAddress = Req.MSYCLMemObj;
  makeTraceEventProlog(MAddress);

  if (MFirstInstance) {
    xpti_td *CmdTraceEvent = static_cast<xpti_td *>(MTraceEvent);
    xpti::addMetadata(CmdTraceEvent, "sycl_device",
                      deviceToID(MQueue->get_device()));
    xpti::addMetadata(CmdTraceEvent, "sycl_device_type",
                      deviceToString(MQueue->get_device()));
    xpti::addMetadata(CmdTraceEvent, "sycl_device_name",
                      getSyclObjImpl(MQueue->get_device())->getDeviceName());
    xpti::addMetadata(CmdTraceEvent, "memory_object",
                      reinterpret_cast<size_t>(MAddress));
    makeTraceEventEpilog();
  }
#endif
}

void EmptyCommand::printDot(std::ostream &Stream) const {
  Stream << "\"" << this << "\" [style=filled, fillcolor=\"#8d8f29\", label=\"";

  Stream << "ID = " << this << "\\n";
  Stream << "EMPTY NODE"
         << "\\n";

  Stream << "\"];" << std::endl;

  for (const auto &Dep : MDeps) {
    Stream << "  \"" << this << "\" -> \"" << Dep.MDepCommand << "\""
           << " [ label = \"Access mode: "
           << accessModeToString(Dep.MDepRequirement->MAccessMode) << "\\n"
           << "MemObj: " << Dep.MDepRequirement->MSYCLMemObj << " \" ]"
           << std::endl;
  }
}

bool EmptyCommand::producesPiEvent() const { return false; }

bool EmptyCommand::supportsPostEnqueueCleanup() const {
  // Even if it is isolated it is not cleaned up separately because not rendered
  // after addCG (only paired host task is enqueued there and analyzed be
  // released).
  return true;
}

void MemCpyCommandHost::printDot(std::ostream &Stream) const {
  Stream << "\"" << this << "\" [style=filled, fillcolor=\"#B6A2EB\", label=\"";

  Stream << "ID = " << this << "\\n";
  Stream << "MEMCPY HOST ON " << deviceToString(MQueue->get_device()) << "\\n";

  Stream << "\"];" << std::endl;

  for (const auto &Dep : MDeps) {
    Stream << "  \"" << this << "\" -> \"" << Dep.MDepCommand << "\""
           << " [ label = \"Access mode: "
           << accessModeToString(Dep.MDepRequirement->MAccessMode) << "\\n"
           << "MemObj: " << Dep.MDepRequirement->MSYCLMemObj << " \" ]"
           << std::endl;
  }
}

UpdateHostRequirementCommand::UpdateHostRequirementCommand(
    QueueImplPtr Queue, Requirement Req, AllocaCommandBase *SrcAllocaCmd,
    void **DstPtr)
    : Command(CommandType::UPDATE_REQUIREMENT, std::move(Queue)),
      MSrcAllocaCmd(SrcAllocaCmd), MDstReq(std::move(Req)), MDstPtr(DstPtr) {

  emitInstrumentationDataProxy();
}

void UpdateHostRequirementCommand::emitInstrumentationData() {
#ifdef XPTI_ENABLE_INSTRUMENTATION
  if (!xptiTraceEnabled())
    return;
  // Create a payload with the command name and an event using this payload to
  // emit a node_create
  MAddress = MSrcAllocaCmd->getSYCLMemObj();
  makeTraceEventProlog(MAddress);

  if (MFirstInstance) {
    xpti_td *CmdTraceEvent = static_cast<xpti_td *>(MTraceEvent);
    xpti::addMetadata(CmdTraceEvent, "sycl_device",
                      deviceToID(MQueue->get_device()));
    xpti::addMetadata(CmdTraceEvent, "sycl_device_type",
                      deviceToString(MQueue->get_device()));
    xpti::addMetadata(CmdTraceEvent, "sycl_device_name",
                      getSyclObjImpl(MQueue->get_device())->getDeviceName());
    xpti::addMetadata(CmdTraceEvent, "memory_object",
                      reinterpret_cast<size_t>(MAddress));
    makeTraceEventEpilog();
  }
#endif
}

static std::string cgTypeToString(detail::CG::CGTYPE Type) {
  switch (Type) {
  case detail::CG::Kernel:
    return "Kernel";
    break;
  case detail::CG::UpdateHost:
    return "update_host";
    break;
  case detail::CG::Fill:
    return "fill";
    break;
  case detail::CG::CopyAccToAcc:
    return "copy acc to acc";
    break;
  case detail::CG::CopyAccToPtr:
    return "copy acc to ptr";
    break;
  case detail::CG::CopyPtrToAcc:
    return "copy ptr to acc";
    break;
  case detail::CG::CopyUSM:
    return "copy usm";
    break;
  case detail::CG::FillUSM:
    return "fill usm";
    break;
  case detail::CG::PrefetchUSM:
    return "prefetch usm";
    break;
  case detail::CG::CodeplayHostTask:
    return "host task";
    break;
  default:
    return "unknown";
    break;
  }
}

ExecCGCommand::ExecCGCommand(std::unique_ptr<detail::CG> CommandGroup,
                             QueueImplPtr Queue)
    : Command(CommandType::RUN_CG, std::move(Queue)),
      MCommandGroup(std::move(CommandGroup)) {
  if (MCommandGroup->getType() == detail::CG::CodeplayHostTask) {
    MSubmittedQueue =
        static_cast<detail::CGHostTask *>(MCommandGroup.get())->MQueue;
    MEvent->setNeedsCleanupAfterWait(true);
  } else if (MCommandGroup->getType() == CG::CGTYPE::Kernel &&
             (static_cast<CGExecKernel *>(MCommandGroup.get()))->hasStreams())
    MEvent->setNeedsCleanupAfterWait(true);

  emitInstrumentationDataProxy();
}

void ExecCGCommand::emitInstrumentationData() {
#ifdef XPTI_ENABLE_INSTRUMENTATION
  if (!xptiTraceEnabled())
    return;
  // Create a payload with the command name and an event using this payload to
  // emit a node_create
  bool HasSourceInfo = false;
  std::string KernelName;
  std::optional<bool> FromSource;
  switch (MCommandGroup->getType()) {
  case detail::CG::Kernel: {
    auto KernelCG =
        reinterpret_cast<detail::CGExecKernel *>(MCommandGroup.get());

    if (KernelCG->MSyclKernel && KernelCG->MSyclKernel->isCreatedFromSource()) {
      FromSource = true;
      pi_kernel KernelHandle = KernelCG->MSyclKernel->getHandleRef();
      MAddress = KernelHandle;
      KernelName = MCommandGroup->MFunctionName;
    } else {
      FromSource = false;
      KernelName = demangleKernelName(KernelCG->getKernelName());
    }
  } break;
  default:
    KernelName = cgTypeToString(MCommandGroup->getType());
    break;
  }
  std::string CommandType = commandToNodeType(MType);
  //  Get source file, line number information from the CommandGroup object
  //  and create payload using name, address, and source info
  //
  //  On Windows, since the support for builtin functions is not available in
  //  MSVC, the MFileName, MLine will be set to nullptr and "0" respectively.
  //  Handle this condition explicitly here.
  xpti::payload_t Payload;
  if (!MCommandGroup->MFileName.empty()) {
    // File name has a valid string
    Payload =
        xpti::payload_t(KernelName.c_str(), MCommandGroup->MFileName.c_str(),
                        MCommandGroup->MLine, MCommandGroup->MColumn, MAddress);
    HasSourceInfo = true;
  } else if (MAddress) {
    // We have a valid function name and an address
    Payload = xpti::payload_t(KernelName.c_str(), MAddress);
  } else {
    // In any case, we will have a valid function name and we'll use that to
    // create the hash
    Payload = xpti::payload_t(KernelName.c_str());
  }

  uint64_t CGKernelInstanceNo;
  // Create event using the payload
  xpti_td *CmdTraceEvent =
      xptiMakeEvent("ExecCG", &Payload, xpti::trace_graph_event,
                    xpti::trace_activity_type_t::active, &CGKernelInstanceNo);

  if (CmdTraceEvent) {
    MInstanceID = CGKernelInstanceNo;
    MTraceEvent = (void *)CmdTraceEvent;
    // If we are seeing this event again, then the instance ID will be greater
    // than 1; in this case, we will skip sending a notification to create a
    // node as this node has already been created.
    if (CGKernelInstanceNo > 1)
      return;

    xpti::addMetadata(CmdTraceEvent, "sycl_device",
                      deviceToID(MQueue->get_device()));
    xpti::addMetadata(CmdTraceEvent, "sycl_device_type",
                      deviceToString(MQueue->get_device()));
    xpti::addMetadata(CmdTraceEvent, "sycl_device_name",
                      getSyclObjImpl(MQueue->get_device())->getDeviceName());
    if (!KernelName.empty()) {
      xpti::addMetadata(CmdTraceEvent, "kernel_name", KernelName);
    }
    if (FromSource.has_value()) {
      xpti::addMetadata(CmdTraceEvent, "from_source", FromSource.value());
    }
    if (HasSourceInfo) {
      xpti::addMetadata(CmdTraceEvent, "sym_function_name", KernelName);
      xpti::addMetadata(CmdTraceEvent, "sym_source_file_name",
                        MCommandGroup->MFileName);
      xpti::addMetadata(CmdTraceEvent, "sym_line_no", MCommandGroup->MLine);
      xpti::addMetadata(CmdTraceEvent, "sym_column_no", MCommandGroup->MColumn);
    }

    xptiNotifySubscribers(MStreamID, xpti::trace_node_create,
                          detail::GSYCLGraphEvent, CmdTraceEvent,
                          CGKernelInstanceNo,
                          static_cast<const void *>(CommandType.c_str()));
  }
#endif
}

void ExecCGCommand::printDot(std::ostream &Stream) const {
  Stream << "\"" << this << "\" [style=filled, fillcolor=\"#AFFF82\", label=\"";

  Stream << "ID = " << this << "\\n";
  Stream << "EXEC CG ON " << deviceToString(MQueue->get_device()) << "\\n";

  switch (MCommandGroup->getType()) {
  case detail::CG::Kernel: {
    auto KernelCG =
        reinterpret_cast<detail::CGExecKernel *>(MCommandGroup.get());
    Stream << "Kernel name: ";
    if (KernelCG->MSyclKernel && KernelCG->MSyclKernel->isCreatedFromSource())
      Stream << "created from source";
    else
      Stream << demangleKernelName(KernelCG->getKernelName());
    Stream << "\\n";
    break;
  }
  default:
    Stream << "CG type: " << cgTypeToString(MCommandGroup->getType()) << "\\n";
    break;
  }

  Stream << "\"];" << std::endl;

  for (const auto &Dep : MDeps) {
    Stream << "  \"" << this << "\" -> \"" << Dep.MDepCommand << "\""
           << " [ label = \"Access mode: "
           << accessModeToString(Dep.MDepRequirement->MAccessMode) << "\\n"
           << "MemObj: " << Dep.MDepRequirement->MSYCLMemObj << " \" ]"
           << std::endl;
  }
}

// SYCL has a parallel_for_work_group variant where the only NDRange
// characteristics set by a user is the number of work groups. This does not map
// to the OpenCL clEnqueueNDRangeAPI, which requires global work size to be set
// as well. This function determines local work size based on the device
// characteristics and the number of work groups requested by the user, then
// calculates the global work size.
// SYCL specification (from 4.8.5.3):
// The member function handler::parallel_for_work_group is parameterized by the
// number of work - groups, such that the size of each group is chosen by the
// runtime, or by the number of work - groups and number of work - items for
// users who need more control.
static void adjustNDRangePerKernel(NDRDescT &NDR, RT::PiKernel Kernel,
                                   const device_impl &DeviceImpl) {
  if (NDR.GlobalSize[0] != 0)
    return; // GlobalSize is set - no need to adjust
  // check the prerequisites:
  assert(NDR.NumWorkGroups[0] != 0 && NDR.LocalSize[0] == 0);
  // TODO might be good to cache this info together with the kernel info to
  // avoid get_kernel_work_group_info on every kernel run
  range<3> WGSize = get_kernel_device_specific_info<
      range<3>,
      cl::sycl::info::kernel_device_specific::compile_work_group_size>::
      get(Kernel, DeviceImpl.getHandleRef(), DeviceImpl.getPlugin());

  if (WGSize[0] == 0) {
    WGSize = {1, 1, 1};
  }
  NDR.set(NDR.Dims, nd_range<3>(NDR.NumWorkGroups * WGSize, WGSize));
}

// We have the following mapping between dimensions with SPIR-V builtins:
// 1D: id[0] -> x
// 2D: id[0] -> y, id[1] -> x
// 3D: id[0] -> z, id[1] -> y, id[2] -> x
// So in order to ensure the correctness we update all the kernel
// parameters accordingly.
// Initially we keep the order of NDRDescT as it provided by the user, this
// simplifies overall handling and do the reverse only when
// the kernel is enqueued.
static void ReverseRangeDimensionsForKernel(NDRDescT &NDR) {
  if (NDR.Dims > 1) {
    std::swap(NDR.GlobalSize[0], NDR.GlobalSize[NDR.Dims - 1]);
    std::swap(NDR.LocalSize[0], NDR.LocalSize[NDR.Dims - 1]);
    std::swap(NDR.GlobalOffset[0], NDR.GlobalOffset[NDR.Dims - 1]);
  }
}

static pi_result SetKernelParamsAndLaunch(
    const QueueImplPtr &Queue, std::vector<ArgDesc> &Args,
    const std::shared_ptr<device_image_impl> &DeviceImageImpl,
    RT::PiKernel Kernel, NDRDescT &NDRDesc, std::vector<RT::PiEvent> &RawEvents,
    RT::PiEvent *OutEvent,
    const ProgramManager::KernelArgMask &EliminatedArgMask,
    const std::function<void *(Requirement *Req)> &getMemAllocationFunc) {
  const detail::plugin &Plugin = Queue->getPlugin();

  auto setFunc = [&Plugin, Kernel, &DeviceImageImpl, &getMemAllocationFunc,
                  &Queue](detail::ArgDesc &Arg, size_t NextTrueIndex) {
    switch (Arg.MType) {
    case kernel_param_kind_t::kind_stream:
      break;
    case kernel_param_kind_t::kind_accessor: {
      Requirement *Req = (Requirement *)(Arg.MPtr);
      assert(getMemAllocationFunc != nullptr &&
             "The function should not be nullptr as we followed the path for "
             "which accessors are used");
      RT::PiMem MemArg = (RT::PiMem)getMemAllocationFunc(Req);
      if (Plugin.getBackend() == backend::opencl) {
        Plugin.call<PiApiKind::piKernelSetArg>(Kernel, NextTrueIndex,
                                               sizeof(RT::PiMem), &MemArg);
      } else {
        Plugin.call<PiApiKind::piextKernelSetArgMemObj>(Kernel, NextTrueIndex,
                                                        &MemArg);
      }
      break;
    }
    case kernel_param_kind_t::kind_std_layout: {
      Plugin.call<PiApiKind::piKernelSetArg>(Kernel, NextTrueIndex, Arg.MSize,
                                             Arg.MPtr);
      break;
    }
    case kernel_param_kind_t::kind_sampler: {
      sampler *SamplerPtr = (sampler *)Arg.MPtr;
      RT::PiSampler Sampler = detail::getSyclObjImpl(*SamplerPtr)
                                  ->getOrCreateSampler(Queue->get_context());
      Plugin.call<PiApiKind::piextKernelSetArgSampler>(Kernel, NextTrueIndex,
                                                       &Sampler);
      break;
    }
    case kernel_param_kind_t::kind_pointer: {
      Plugin.call<PiApiKind::piextKernelSetArgPointer>(Kernel, NextTrueIndex,
                                                       Arg.MSize, Arg.MPtr);
      break;
    }
    case kernel_param_kind_t::kind_specialization_constants_buffer: {
      if (Queue->is_host()) {
        throw cl::sycl::feature_not_supported(
            "SYCL2020 specialization constants are not yet supported on host "
            "device",
            PI_INVALID_OPERATION);
      }
      assert(DeviceImageImpl != nullptr);
      RT::PiMem SpecConstsBuffer = DeviceImageImpl->get_spec_const_buffer_ref();
      // Avoid taking an address of nullptr
      RT::PiMem *SpecConstsBufferArg =
          SpecConstsBuffer ? &SpecConstsBuffer : nullptr;
      Plugin.call<PiApiKind::piextKernelSetArgMemObj>(Kernel, NextTrueIndex,
                                                      SpecConstsBufferArg);
      break;
    }
    case kernel_param_kind_t::kind_invalid:
      throw runtime_error("Invalid kernel param kind", PI_INVALID_VALUE);
      break;
    }
  };

  if (EliminatedArgMask.empty()) {
    for (ArgDesc &Arg : Args) {
      setFunc(Arg, Arg.MIndex);
    }
  } else {
    // TODO this is not necessary as long as we can guarantee that the arguments
    // are already sorted (e. g. handle the sorting in handler if necessary due
    // to set_arg(...) usage).
    std::sort(Args.begin(), Args.end(), [](const ArgDesc &A, const ArgDesc &B) {
      return A.MIndex < B.MIndex;
    });
    int LastIndex = -1;
    size_t NextTrueIndex = 0;

    for (ArgDesc &Arg : Args) {
      // Handle potential gaps in set arguments (e. g. if some of them are set
      // on the user side).
      for (int Idx = LastIndex + 1; Idx < Arg.MIndex; ++Idx)
        if (!EliminatedArgMask[Idx])
          ++NextTrueIndex;
      LastIndex = Arg.MIndex;

      if (EliminatedArgMask[Arg.MIndex])
        continue;

      setFunc(Arg, NextTrueIndex);
      ++NextTrueIndex;
    }
  }

  adjustNDRangePerKernel(NDRDesc, Kernel, *(Queue->getDeviceImplPtr()));

  // Remember this information before the range dimensions are reversed
  const bool HasLocalSize = (NDRDesc.LocalSize[0] != 0);

  ReverseRangeDimensionsForKernel(NDRDesc);

  size_t RequiredWGSize[3] = {0, 0, 0};
  size_t *LocalSize = nullptr;

  if (HasLocalSize)
    LocalSize = &NDRDesc.LocalSize[0];
  else {
    Plugin.call<PiApiKind::piKernelGetGroupInfo>(
        Kernel, Queue->getDeviceImplPtr()->getHandleRef(),
        PI_KERNEL_GROUP_INFO_COMPILE_WORK_GROUP_SIZE, sizeof(RequiredWGSize),
        RequiredWGSize, /* param_value_size_ret = */ nullptr);

    const bool EnforcedLocalSize =
        (RequiredWGSize[0] != 0 || RequiredWGSize[1] != 0 ||
         RequiredWGSize[2] != 0);
    if (EnforcedLocalSize)
      LocalSize = RequiredWGSize;
  }

  pi_result Error = Plugin.call_nocheck<PiApiKind::piEnqueueKernelLaunch>(
      Queue->getHandleRef(), Kernel, NDRDesc.Dims, &NDRDesc.GlobalOffset[0],
      &NDRDesc.GlobalSize[0], LocalSize, RawEvents.size(),
      RawEvents.empty() ? nullptr : &RawEvents[0], OutEvent);
  return Error;
}

// The function initialize accessors and calls lambda.
// The function is used as argument to piEnqueueNativeKernel which requires
// that the passed function takes one void* argument.
void DispatchNativeKernel(void *Blob) {
  void **CastedBlob = (void **)Blob;

  std::vector<Requirement *> *Reqs =
      static_cast<std::vector<Requirement *> *>(CastedBlob[0]);

  std::unique_ptr<HostKernelBase> *HostKernel =
      static_cast<std::unique_ptr<HostKernelBase> *>(CastedBlob[1]);

  NDRDescT *NDRDesc = static_cast<NDRDescT *>(CastedBlob[2]);

  // Other value are pointer to the buffers.
  void **NextArg = CastedBlob + 3;
  for (detail::Requirement *Req : *Reqs)
    Req->MData = *(NextArg++);

  (*HostKernel)->call(*NDRDesc, nullptr);

  // The ownership of these objects have been passed to us, need to cleanup
  delete Reqs;
  delete HostKernel;
  delete NDRDesc;
}

cl_int enqueueImpKernel(
    const QueueImplPtr &Queue, NDRDescT &NDRDesc, std::vector<ArgDesc> &Args,
    const std::shared_ptr<detail::kernel_bundle_impl> &KernelBundleImplPtr,
    const std::shared_ptr<detail::kernel_impl> &MSyclKernel,
    const std::string &KernelName, const detail::OSModuleHandle &OSModuleHandle,
    std::vector<RT::PiEvent> &RawEvents, RT::PiEvent *OutEvent,
    const std::function<void *(Requirement *Req)> &getMemAllocationFunc) {

  // Run OpenCL kernel
  auto ContextImpl = Queue->getContextImplPtr();
  auto DeviceImpl = Queue->getDeviceImplPtr();
  RT::PiKernel Kernel = nullptr;
  std::mutex *KernelMutex = nullptr;
  RT::PiProgram Program = nullptr;

  std::shared_ptr<kernel_impl> SyclKernelImpl;
  std::shared_ptr<device_image_impl> DeviceImageImpl;

  // Use kernel_bundle if available unless it is interop.
  // Interop bundles can't be used in the first branch, because the kernels
  // in interop kernel bundles (if any) do not have kernel_id
  // and can therefore not be looked up, but since they are self-contained
  // they can simply be launched directly.
  if (KernelBundleImplPtr && !KernelBundleImplPtr->isInterop()) {
    kernel_id KernelID =
        detail::ProgramManager::getInstance().getSYCLKernelID(KernelName);
    kernel SyclKernel =
        KernelBundleImplPtr->get_kernel(KernelID, KernelBundleImplPtr);

    SyclKernelImpl = detail::getSyclObjImpl(SyclKernel);

    Kernel = SyclKernelImpl->getHandleRef();
    DeviceImageImpl = SyclKernelImpl->getDeviceImage();

    Program = DeviceImageImpl->get_program_ref();

    std::tie(Kernel, KernelMutex) =
        detail::ProgramManager::getInstance().getOrCreateKernel(
            KernelBundleImplPtr->get_context(), KernelName,
            /*PropList=*/{}, Program);
  } else if (nullptr != MSyclKernel) {
    assert(MSyclKernel->get_info<info::kernel::context>() ==
           Queue->get_context());
    Kernel = MSyclKernel->getHandleRef();

    auto SyclProg =
        detail::getSyclObjImpl(MSyclKernel->get_info<info::kernel::program>());
    Program = SyclProg->getHandleRef();
    if (SyclProg->is_cacheable()) {
      RT::PiKernel FoundKernel = nullptr;
      std::tie(FoundKernel, KernelMutex, std::ignore) =
          detail::ProgramManager::getInstance().getOrCreateKernel(
              OSModuleHandle, ContextImpl, DeviceImpl, KernelName,
              SyclProg.get());
      assert(FoundKernel == Kernel);
    }
  } else {
    std::tie(Kernel, KernelMutex, Program) =
        detail::ProgramManager::getInstance().getOrCreateKernel(
            OSModuleHandle, ContextImpl, DeviceImpl, KernelName, nullptr);
  }

  pi_result Error = PI_SUCCESS;
  ProgramManager::KernelArgMask EliminatedArgMask;
  if (nullptr == MSyclKernel || !MSyclKernel->isCreatedFromSource()) {
    EliminatedArgMask =
        detail::ProgramManager::getInstance().getEliminatedKernelArgMask(
            OSModuleHandle, Program, KernelName);
  }
  if (KernelMutex != nullptr) {
    // For cacheable kernels, we use per-kernel mutex
    std::lock_guard<std::mutex> Lock(*KernelMutex);
    Error = SetKernelParamsAndLaunch(Queue, Args, DeviceImageImpl, Kernel,
                                     NDRDesc, RawEvents, OutEvent,
                                     EliminatedArgMask, getMemAllocationFunc);
  } else {
    Error = SetKernelParamsAndLaunch(Queue, Args, DeviceImageImpl, Kernel,
                                     NDRDesc, RawEvents, OutEvent,
                                     EliminatedArgMask, getMemAllocationFunc);
  }

  if (PI_SUCCESS != Error) {
    // If we have got non-success error code, let's analyze it to emit nice
    // exception explaining what was wrong
    const device_impl &DeviceImpl = *(Queue->getDeviceImplPtr());
    return detail::enqueue_kernel_launch::handleError(Error, DeviceImpl, Kernel,
                                                      NDRDesc);
  }

  return PI_SUCCESS;
}

cl_int ExecCGCommand::enqueueImp() {
  if (getCG().getType() != CG::CGTYPE::CodeplayHostTask)
    waitForPreparedHostEvents();
  std::vector<EventImplPtr> EventImpls = MPreparedDepsEvents;
  auto RawEvents = getPiEvents(EventImpls);
  flushCrossQueueDeps(EventImpls, getWorkerQueue());

  RT::PiEvent *Event = (MQueue->has_discard_events_support() &&
                        MCommandGroup->MRequirements.size() == 0)
                           ? nullptr
                           : &MEvent->getHandleRef();
  switch (MCommandGroup->getType()) {

  case CG::CGTYPE::UpdateHost: {
    throw runtime_error("Update host should be handled by the Scheduler.",
                        PI_INVALID_OPERATION);
  }
  case CG::CGTYPE::CopyAccToPtr: {
    CGCopy *Copy = (CGCopy *)MCommandGroup.get();
    Requirement *Req = (Requirement *)Copy->getSrc();
    AllocaCommandBase *AllocaCmd = getAllocaForReq(Req);

    MemoryManager::copy(
        AllocaCmd->getSYCLMemObj(), AllocaCmd->getMemAllocation(), MQueue,
        Req->MDims, Req->MMemoryRange, Req->MAccessRange, Req->MOffset,
        Req->MElemSize, Copy->getDst(),
        Scheduler::getInstance().getDefaultHostQueue(), Req->MDims,
        Req->MAccessRange, Req->MAccessRange, /*DstOffset=*/{0, 0, 0},
        Req->MElemSize, std::move(RawEvents), MEvent->getHandleRef());

    return CL_SUCCESS;
  }
  case CG::CGTYPE::CopyPtrToAcc: {
    CGCopy *Copy = (CGCopy *)MCommandGroup.get();
    Requirement *Req = (Requirement *)(Copy->getDst());
    AllocaCommandBase *AllocaCmd = getAllocaForReq(Req);

    Scheduler::getInstance().getDefaultHostQueue();

    MemoryManager::copy(
        AllocaCmd->getSYCLMemObj(), Copy->getSrc(),
        Scheduler::getInstance().getDefaultHostQueue(), Req->MDims,
        Req->MAccessRange, Req->MAccessRange,
        /*SrcOffset*/ {0, 0, 0}, Req->MElemSize, AllocaCmd->getMemAllocation(),
        MQueue, Req->MDims, Req->MMemoryRange, Req->MAccessRange, Req->MOffset,
        Req->MElemSize, std::move(RawEvents), MEvent->getHandleRef());

    return CL_SUCCESS;
  }
  case CG::CGTYPE::CopyAccToAcc: {
    CGCopy *Copy = (CGCopy *)MCommandGroup.get();
    Requirement *ReqSrc = (Requirement *)(Copy->getSrc());
    Requirement *ReqDst = (Requirement *)(Copy->getDst());

    AllocaCommandBase *AllocaCmdSrc = getAllocaForReq(ReqSrc);
    AllocaCommandBase *AllocaCmdDst = getAllocaForReq(ReqDst);

    MemoryManager::copy(
        AllocaCmdSrc->getSYCLMemObj(), AllocaCmdSrc->getMemAllocation(), MQueue,
        ReqSrc->MDims, ReqSrc->MMemoryRange, ReqSrc->MAccessRange,
        ReqSrc->MOffset, ReqSrc->MElemSize, AllocaCmdDst->getMemAllocation(),
        MQueue, ReqDst->MDims, ReqDst->MMemoryRange, ReqDst->MAccessRange,
        ReqDst->MOffset, ReqDst->MElemSize, std::move(RawEvents),
        MEvent->getHandleRef());

    return CL_SUCCESS;
  }
  case CG::CGTYPE::Fill: {
    CGFill *Fill = (CGFill *)MCommandGroup.get();
    Requirement *Req = (Requirement *)(Fill->getReqToFill());
    AllocaCommandBase *AllocaCmd = getAllocaForReq(Req);

    MemoryManager::fill(
        AllocaCmd->getSYCLMemObj(), AllocaCmd->getMemAllocation(), MQueue,
        Fill->MPattern.size(), Fill->MPattern.data(), Req->MDims,
        Req->MMemoryRange, Req->MAccessRange, Req->MOffset, Req->MElemSize,
        std::move(RawEvents), MEvent->getHandleRef());

    return CL_SUCCESS;
  }
  case CG::CGTYPE::RunOnHostIntel: {
    CGExecKernel *HostTask = (CGExecKernel *)MCommandGroup.get();

    // piEnqueueNativeKernel takes arguments blob which is passes to user
    // function.
    // Need the following items to restore context in the host task.
    // Make a copy on heap to "dettach" from the command group as it can be
    // released before the host task completes.
    std::vector<void *> ArgsBlob(HostTask->MArgs.size() + 3);

    std::vector<Requirement *> *CopyReqs =
        new std::vector<Requirement *>(HostTask->MRequirements);

    // Not actually a copy, but move. Should be OK as it's not expected that
    // MHostKernel will be used elsewhere.
    std::unique_ptr<HostKernelBase> *CopyHostKernel =
        new std::unique_ptr<HostKernelBase>(std::move(HostTask->MHostKernel));

    NDRDescT *CopyNDRDesc = new NDRDescT(HostTask->MNDRDesc);

    ArgsBlob[0] = (void *)CopyReqs;
    ArgsBlob[1] = (void *)CopyHostKernel;
    ArgsBlob[2] = (void *)CopyNDRDesc;

    void **NextArg = ArgsBlob.data() + 3;

    if (MQueue->is_host()) {
      for (ArgDesc &Arg : HostTask->MArgs) {
        assert(Arg.MType == kernel_param_kind_t::kind_accessor);

        Requirement *Req = (Requirement *)(Arg.MPtr);
        AllocaCommandBase *AllocaCmd = getAllocaForReq(Req);

        *NextArg = AllocaCmd->getMemAllocation();
        NextArg++;
      }

      if (!RawEvents.empty()) {
        // Assuming that the events are for devices to the same Plugin.
        const detail::plugin &Plugin = EventImpls[0]->getPlugin();
        Plugin.call<PiApiKind::piEventsWait>(RawEvents.size(), &RawEvents[0]);
      }
      DispatchNativeKernel((void *)ArgsBlob.data());

      return CL_SUCCESS;
    }

    std::vector<pi_mem> Buffers;
    // piEnqueueNativeKernel requires additional array of pointers to args blob,
    // values that pointers point to are replaced with actual pointers to the
    // memory before execution of user function.
    std::vector<void *> MemLocs;

    for (ArgDesc &Arg : HostTask->MArgs) {
      assert(Arg.MType == kernel_param_kind_t::kind_accessor);

      Requirement *Req = (Requirement *)(Arg.MPtr);
      AllocaCommandBase *AllocaCmd = getAllocaForReq(Req);
      pi_mem MemArg = (pi_mem)AllocaCmd->getMemAllocation();

      Buffers.push_back(MemArg);
      MemLocs.push_back(NextArg);
      NextArg++;
    }
    const detail::plugin &Plugin = MQueue->getPlugin();
    pi_result Error = Plugin.call_nocheck<PiApiKind::piEnqueueNativeKernel>(
        MQueue->getHandleRef(), DispatchNativeKernel, (void *)ArgsBlob.data(),
        ArgsBlob.size() * sizeof(ArgsBlob[0]), Buffers.size(), Buffers.data(),
        const_cast<const void **>(MemLocs.data()), RawEvents.size(),
        RawEvents.empty() ? nullptr : RawEvents.data(), Event);

    switch (Error) {
    case PI_INVALID_OPERATION:
      throw cl::sycl::runtime_error(
          "Device doesn't support run_on_host_intel tasks.", Error);
    case PI_SUCCESS:
      return Error;
    default:
      throw cl::sycl::runtime_error(
          "Enqueueing run_on_host_intel task has failed.", Error);
    }
  }
  case CG::CGTYPE::Kernel: {
    CGExecKernel *ExecKernel = (CGExecKernel *)MCommandGroup.get();

    NDRDescT &NDRDesc = ExecKernel->MNDRDesc;
    std::vector<ArgDesc> &Args = ExecKernel->MArgs;

    if (MQueue->is_host() || (MQueue->getPlugin().getBackend() ==
                              backend::ext_intel_esimd_emulator)) {
      for (ArgDesc &Arg : Args)
        if (kernel_param_kind_t::kind_accessor == Arg.MType) {
          Requirement *Req = (Requirement *)(Arg.MPtr);
          AllocaCommandBase *AllocaCmd = getAllocaForReq(Req);
          Req->MData = AllocaCmd->getMemAllocation();
        }
      if (!RawEvents.empty()) {
        // Assuming that the events are for devices to the same Plugin.
        const detail::plugin &Plugin = EventImpls[0]->getPlugin();
        Plugin.call<PiApiKind::piEventsWait>(RawEvents.size(), &RawEvents[0]);
      }

      if (MQueue->is_host()) {
        ExecKernel->MHostKernel->call(NDRDesc,
                                      getEvent()->getHostProfilingInfo());
      } else {
        assert(MQueue->getPlugin().getBackend() ==
               backend::ext_intel_esimd_emulator);
        // Dims==0 for 'single_task() - void(void) type'
        uint32_t Dims = (Args.size() > 0) ? NDRDesc.Dims : 0;
        MQueue->getPlugin().call<PiApiKind::piEnqueueKernelLaunch>(
            nullptr,
            reinterpret_cast<pi_kernel>(ExecKernel->MHostKernel->getPtr()),
            Dims, &NDRDesc.GlobalOffset[0], &NDRDesc.GlobalSize[0],
            &NDRDesc.LocalSize[0], 0, nullptr, nullptr);
      }

      return CL_SUCCESS;
    }

    auto getMemAllocationFunc = [this](Requirement *Req) {
      AllocaCommandBase *AllocaCmd = getAllocaForReq(Req);
      return AllocaCmd->getMemAllocation();
    };

    const std::shared_ptr<detail::kernel_impl> &SyclKernel =
        ExecKernel->MSyclKernel;
    const std::string &KernelName = ExecKernel->MKernelName;
    const detail::OSModuleHandle &OSModuleHandle = ExecKernel->MOSModuleHandle;

    if (!Event) {
      // Kernel only uses assert if it's non interop one
      bool KernelUsesAssert = !(SyclKernel && SyclKernel->isInterop()) &&
                              ProgramManager::getInstance().kernelUsesAssert(
                                  OSModuleHandle, KernelName);
      if (KernelUsesAssert) {
        Event = &MEvent->getHandleRef();
      }
    }

    return enqueueImpKernel(
        MQueue, NDRDesc, Args, ExecKernel->getKernelBundle(), SyclKernel,
        KernelName, OSModuleHandle, RawEvents, Event, getMemAllocationFunc);
  }
  case CG::CGTYPE::CopyUSM: {
    CGCopyUSM *Copy = (CGCopyUSM *)MCommandGroup.get();
    MemoryManager::copy_usm(Copy->getSrc(), MQueue, Copy->getLength(),
                            Copy->getDst(), std::move(RawEvents), Event);

    return CL_SUCCESS;
  }
  case CG::CGTYPE::FillUSM: {
    CGFillUSM *Fill = (CGFillUSM *)MCommandGroup.get();
    MemoryManager::fill_usm(Fill->getDst(), MQueue, Fill->getLength(),
                            Fill->getFill(), std::move(RawEvents), Event);

    return CL_SUCCESS;
  }
  case CG::CGTYPE::PrefetchUSM: {
    CGPrefetchUSM *Prefetch = (CGPrefetchUSM *)MCommandGroup.get();
    MemoryManager::prefetch_usm(Prefetch->getDst(), MQueue,
                                Prefetch->getLength(), std::move(RawEvents),
                                Event);

    return CL_SUCCESS;
  }
  case CG::CGTYPE::AdviseUSM: {
    CGAdviseUSM *Advise = (CGAdviseUSM *)MCommandGroup.get();
    MemoryManager::advise_usm(Advise->getDst(), MQueue, Advise->getLength(),
                              Advise->getAdvice(), std::move(RawEvents), Event);

    return CL_SUCCESS;
  }
  case CG::CGTYPE::CodeplayInteropTask: {
    const detail::plugin &Plugin = MQueue->getPlugin();
    CGInteropTask *ExecInterop = (CGInteropTask *)MCommandGroup.get();
    // Wait for dependencies to complete before dispatching work on the host
    // TODO: Use a callback to dispatch the interop task instead of waiting for
    //  the event
    if (!RawEvents.empty()) {
      Plugin.call<PiApiKind::piEventsWait>(RawEvents.size(), &RawEvents[0]);
    }
    std::vector<interop_handler::ReqToMem> ReqMemObjs;
    // Extract the Mem Objects for all Requirements, to ensure they are available if
    // a user ask for them inside the interop task scope
    const auto& HandlerReq = ExecInterop->MRequirements;
    std::for_each(std::begin(HandlerReq), std::end(HandlerReq), [&](Requirement* Req) {
      AllocaCommandBase *AllocaCmd = getAllocaForReq(Req);
      auto MemArg = reinterpret_cast<pi_mem>(AllocaCmd->getMemAllocation());
      interop_handler::ReqToMem ReqToMem = std::make_pair(Req, MemArg);
      ReqMemObjs.emplace_back(ReqToMem);
    });

    std::sort(std::begin(ReqMemObjs), std::end(ReqMemObjs));
    interop_handler InteropHandler(std::move(ReqMemObjs), MQueue);
    ExecInterop->MInteropTask->call(InteropHandler);
    Plugin.call<PiApiKind::piEnqueueEventsWait>(MQueue->getHandleRef(), 0,
                                                nullptr, Event);

    return CL_SUCCESS;
  }
  case CG::CGTYPE::CodeplayHostTask: {
    CGHostTask *HostTask = static_cast<CGHostTask *>(MCommandGroup.get());

    for (ArgDesc &Arg : HostTask->MArgs) {
      switch (Arg.MType) {
      case kernel_param_kind_t::kind_accessor: {
        Requirement *Req = static_cast<Requirement *>(Arg.MPtr);
        AllocaCommandBase *AllocaCmd = getAllocaForReq(Req);

        Req->MData = AllocaCmd->getMemAllocation();
        break;
      }
      default:
        throw runtime_error("Unsupported arg type", PI_INVALID_VALUE);
      }
    }

    std::vector<interop_handle::ReqToMem> ReqToMem;

    if (HostTask->MHostTask->isInteropTask()) {
      // Extract the Mem Objects for all Requirements, to ensure they are
      // available if a user asks for them inside the interop task scope
      const std::vector<Requirement *> &HandlerReq = HostTask->MRequirements;
      auto ReqToMemConv = [&ReqToMem, HostTask](Requirement *Req) {
        const std::vector<AllocaCommandBase *> &AllocaCmds =
            Req->MSYCLMemObj->MRecord->MAllocaCommands;

        for (AllocaCommandBase *AllocaCmd : AllocaCmds)
          if (HostTask->MQueue->getContextImplPtr() ==
              AllocaCmd->getQueue()->getContextImplPtr()) {
            auto MemArg =
                reinterpret_cast<pi_mem>(AllocaCmd->getMemAllocation());
            ReqToMem.emplace_back(std::make_pair(Req, MemArg));

            return;
          }

        assert(false &&
               "Can't get memory object due to no allocation available");

        throw runtime_error(
            "Can't get memory object due to no allocation available",
            PI_INVALID_MEM_OBJECT);
      };
      std::for_each(std::begin(HandlerReq), std::end(HandlerReq), ReqToMemConv);
      std::sort(std::begin(ReqToMem), std::end(ReqToMem));
    }

    MQueue->getThreadPool().submit<DispatchHostTask>(
        DispatchHostTask(this, std::move(ReqToMem)));

    MShouldCompleteEventIfPossible = false;

    return CL_SUCCESS;
  }
  case CG::CGTYPE::Barrier: {
    if (MQueue->get_device().is_host()) {
      // NOP for host device.
      return PI_SUCCESS;
    }
    const detail::plugin &Plugin = MQueue->getPlugin();
    Plugin.call<PiApiKind::piEnqueueEventsWaitWithBarrier>(
        MQueue->getHandleRef(), 0, nullptr, Event);

    return PI_SUCCESS;
  }
  case CG::CGTYPE::BarrierWaitlist: {
    CGBarrier *Barrier = static_cast<CGBarrier *>(MCommandGroup.get());
    std::vector<detail::EventImplPtr> Events = Barrier->MEventsWaitWithBarrier;
    std::vector<RT::PiEvent> PiEvents = getPiEvents(Events);
    if (MQueue->get_device().is_host() || PiEvents.empty()) {
      // NOP for host device.
      // If Events is empty, then the barrier has no effect.
      return PI_SUCCESS;
    }
    const detail::plugin &Plugin = MQueue->getPlugin();
    Plugin.call<PiApiKind::piEnqueueEventsWaitWithBarrier>(
        MQueue->getHandleRef(), PiEvents.size(), &PiEvents[0], Event);

    return PI_SUCCESS;
  }
  case CG::CGTYPE::None:
    throw runtime_error("CG type not implemented.", PI_INVALID_OPERATION);
  }
  return PI_INVALID_OPERATION;
}

bool ExecCGCommand::producesPiEvent() const {
  return MCommandGroup->getType() != CG::CGTYPE::CodeplayHostTask;
}

bool ExecCGCommand::supportsPostEnqueueCleanup() const {
  // TODO enable cleaning up host task commands and kernels with streams after
  // enqueue
  return Command::supportsPostEnqueueCleanup() &&
         (MCommandGroup->getType() != CG::CGTYPE::CodeplayHostTask) &&
         (MCommandGroup->getType() != CG::CGTYPE::Kernel ||
          !(static_cast<CGExecKernel *>(MCommandGroup.get()))->hasStreams());
}

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
