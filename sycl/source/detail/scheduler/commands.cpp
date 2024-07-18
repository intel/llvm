//===----------- commands.cpp - SYCL commands -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/error_handling/error_handling.hpp>

#include <detail/context_impl.hpp>
#include <detail/event_impl.hpp>
#include <detail/host_pipe_map_entry.hpp>
#include <detail/kernel_bundle_impl.hpp>
#include <detail/kernel_impl.hpp>
#include <detail/kernel_info.hpp>
#include <detail/memory_manager.hpp>
#include <detail/program_manager/program_manager.hpp>
#include <detail/queue_impl.hpp>
#include <detail/sampler_impl.hpp>
#include <detail/scheduler/commands.hpp>
#include <detail/scheduler/scheduler.hpp>
#include <detail/stream_impl.hpp>
#include <detail/xpti_registry.hpp>
#include <sycl/access/access.hpp>
#include <sycl/backend_types.hpp>
#include <sycl/detail/cg_types.hpp>
#include <sycl/detail/kernel_desc.hpp>
#include <sycl/sampler.hpp>

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

namespace sycl {
inline namespace _V1 {
namespace detail {

#ifdef XPTI_ENABLE_INSTRUMENTATION
// Global graph for the application
extern xpti::trace_event_data_t *GSYCLGraphEvent;

static bool CurrentCodeLocationValid() {
  detail::tls_code_loc_t Tls;
  auto CodeLoc = Tls.query();
  auto FileName = CodeLoc.fileName();
  auto FunctionName = CodeLoc.functionName();
  return (FileName && FileName[0] != '\0') ||
         (FunctionName && FunctionName[0] != '\0');
}

void emitInstrumentationGeneral(uint32_t StreamID, uint64_t InstanceID,
                                xpti_td *TraceEvent, uint16_t Type,
                                const void *Addr) {
  if (!(xptiCheckTraceEnabled(StreamID, Type) && TraceEvent))
    return;
  // Trace event notifier that emits a Type event
  xptiNotifySubscribers(StreamID, Type, detail::GSYCLGraphEvent,
                        static_cast<xpti_td *>(TraceEvent), InstanceID, Addr);
}

static size_t deviceToID(const device &Device) {
  return reinterpret_cast<size_t>(getSyclObjImpl(Device)->getHandleRef());
}

static void addDeviceMetadata(xpti_td *TraceEvent, const QueueImplPtr &Queue) {
  xpti::addMetadata(TraceEvent, "sycl_device_type",
                    queueDeviceToString(Queue.get()));
  if (Queue) {
    xpti::addMetadata(TraceEvent, "sycl_device",
                      deviceToID(Queue->get_device()));
    xpti::addMetadata(TraceEvent, "sycl_device_name",
                      getSyclObjImpl(Queue->get_device())->getDeviceName());
  }
}

static unsigned long long getQueueID(const QueueImplPtr &Queue) {
  return Queue ? Queue->getQueueID() : 0;
}
#endif

static ContextImplPtr getContext(const QueueImplPtr &Queue) {
  if (Queue)
    return Queue->getContextImplPtr();
  return nullptr;
}

#ifdef __SYCL_ENABLE_GNU_DEMANGLING
struct DemangleHandle {
  char *p;
  DemangleHandle(char *ptr) : p(ptr) {}

  DemangleHandle(const DemangleHandle &) = delete;
  DemangleHandle &operator=(const DemangleHandle &) = delete;

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

void applyFuncOnFilteredArgs(
    const KernelArgMask *EliminatedArgMask, std::vector<ArgDesc> &Args,
    std::function<void(detail::ArgDesc &Arg, int NextTrueIndex)> Func) {
  if (!EliminatedArgMask) {
    for (ArgDesc &Arg : Args) {
      Func(Arg, Arg.MIndex);
    }
  } else {
    // TODO this is not necessary as long as we can guarantee that the
    // arguments are already sorted (e. g. handle the sorting in handler
    // if necessary due to set_arg(...) usage).
    std::sort(Args.begin(), Args.end(), [](const ArgDesc &A, const ArgDesc &B) {
      return A.MIndex < B.MIndex;
    });
    int LastIndex = -1;
    size_t NextTrueIndex = 0;

    for (ArgDesc &Arg : Args) {
      // Handle potential gaps in set arguments (e. g. if some of them are
      // set on the user side).
      for (int Idx = LastIndex + 1; Idx < Arg.MIndex; ++Idx)
        if (!(*EliminatedArgMask)[Idx])
          ++NextTrueIndex;
      LastIndex = Arg.MIndex;

      if ((*EliminatedArgMask)[Arg.MIndex])
        continue;

      Func(Arg, NextTrueIndex);
      ++NextTrueIndex;
    }
  }
}

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
  case Command::CommandType::FUSION:
    return "kernel_fusion_placeholder_node";
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
  case Command::CommandType::FUSION:
    return "Kernel Fusion Placeholder";
  default:
    return "Unknown Action";
  }
}
#endif

std::vector<sycl::detail::pi::PiEvent>
Command::getPiEvents(const std::vector<EventImplPtr> &EventImpls) const {
  std::vector<sycl::detail::pi::PiEvent> RetPiEvents;
  for (auto &EventImpl : EventImpls) {
    if (EventImpl->getHandleRef() == nullptr)
      continue;

    // Do not add redundant event dependencies for in-order queues.
    // At this stage dependency is definitely pi task and need to check if
    // current one is a host task. In this case we should not skip pi event due
    // to different sync mechanisms for different task types on in-order queue.
    if (MWorkerQueue && EventImpl->getWorkerQueue() == MWorkerQueue &&
        MWorkerQueue->isInOrder() && !isHostTask())
      continue;

    RetPiEvents.push_back(EventImpl->getHandleRef());
  }

  return RetPiEvents;
}

// This function is implemented (duplicating getPiEvents a lot) as short term
// solution for the issue that barrier with wait list could not
// handle empty pi event handles when kernel is enqueued on host task
// completion.
std::vector<sycl::detail::pi::PiEvent> Command::getPiEventsBlocking(
    const std::vector<EventImplPtr> &EventImpls) const {
  std::vector<sycl::detail::pi::PiEvent> RetPiEvents;
  for (auto &EventImpl : EventImpls) {
    // Throwaway events created with empty constructor will not have a context
    // (which is set lazily) calling getContextImpl() would set that
    // context, which we wish to avoid as it is expensive.
    // Skip host task and NOP events also.
    if (EventImpl->isDefaultConstructed() || EventImpl->isHost() ||
        EventImpl->isNOP())
      continue;
    // In this path nullptr native event means that the command has not been
    // enqueued. It may happen if async enqueue in a host task is involved.
    if (!EventImpl->isEnqueued()) {
      if (!EventImpl->getCommand() ||
          !static_cast<Command *>(EventImpl->getCommand())->producesPiEvent())
        continue;
      std::vector<Command *> AuxCmds;
      Scheduler::getInstance().enqueueCommandForCG(EventImpl, AuxCmds,
                                                   BLOCKING);
    }
    // Do not add redundant event dependencies for in-order queues.
    // At this stage dependency is definitely pi task and need to check if
    // current one is a host task. In this case we should not skip pi event due
    // to different sync mechanisms for different task types on in-order queue.
    if (MWorkerQueue && EventImpl->getWorkerQueue() == MWorkerQueue &&
        MWorkerQueue->isInOrder() && !isHostTask())
      continue;

    RetPiEvents.push_back(EventImpl->getHandleRef());
  }

  return RetPiEvents;
}

bool Command::isHostTask() const {
  return (MType == CommandType::RUN_CG) /* host task has this type also */ &&
         ((static_cast<const ExecCGCommand *>(this))->getCG().getType() ==
          CGType::CodeplayHostTask);
}

bool Command::isFusable() const {
  if ((MType != CommandType::RUN_CG)) {
    return false;
  }
  const auto &CG = (static_cast<const ExecCGCommand &>(*this)).getCG();
  return (CG.getType() == CGType::Kernel) &&
         (!static_cast<const CGExecKernel &>(CG).MKernelIsCooperative) &&
         (!static_cast<const CGExecKernel &>(CG).MKernelUsesClusterLaunch);
}

static void flushCrossQueueDeps(const std::vector<EventImplPtr> &EventImpls,
                                const QueueImplPtr &Queue) {
  for (auto &EventImpl : EventImpls) {
    EventImpl->flushIfNeeded(Queue);
  }
}

namespace {

struct EnqueueNativeCommandData {
  sycl::interop_handle ih;
  std::function<void(interop_handle)> func;
};

void InteropFreeFunc(pi_queue, void *InteropData) {
  auto *Data = reinterpret_cast<EnqueueNativeCommandData *>(InteropData);
  return Data->func(Data->ih);
}
} // namespace

class DispatchHostTask {
  ExecCGCommand *MThisCmd;
  std::vector<interop_handle::ReqToMem> MReqToMem;
  std::vector<pi_mem> MReqPiMem;

  bool waitForEvents() const {
    std::map<const PluginPtr, std::vector<EventImplPtr>>
        RequiredEventsPerPlugin;

    for (const EventImplPtr &Event : MThisCmd->MPreparedDepsEvents) {
      const PluginPtr &Plugin = Event->getPlugin();
      RequiredEventsPerPlugin[Plugin].push_back(Event);
    }

    // wait for dependency device events
    // FIXME Current implementation of waiting for events will make the thread
    // 'sleep' until all of dependency events are complete. We need a bit more
    // sophisticated waiting mechanism to allow to utilize this thread for any
    // other available job and resume once all required events are ready.
    for (auto &PluginWithEvents : RequiredEventsPerPlugin) {
      std::vector<sycl::detail::pi::PiEvent> RawEvents =
          MThisCmd->getPiEvents(PluginWithEvents.second);
      if (RawEvents.size() == 0)
        continue;
      try {
        PluginWithEvents.first->call<PiApiKind::piEventsWait>(RawEvents.size(),
                                                              RawEvents.data());
      } catch (const sycl::exception &) {
        MThisCmd->MEvent->getSubmittedQueue()->reportAsyncException(
            std::current_exception());
        return false;
      } catch (...) {
        MThisCmd->MEvent->getSubmittedQueue()->reportAsyncException(
            std::current_exception());
        return false;
      }
    }

    // Wait for dependency host events.
    // Host events can't throw exceptions so don't try to catch it.
    for (const EventImplPtr &Event : MThisCmd->MPreparedHostDepsEvents) {
      Event->waitInternal();
    }

    return true;
  }

public:
  DispatchHostTask(ExecCGCommand *ThisCmd,
                   std::vector<interop_handle::ReqToMem> ReqToMem,
                   std::vector<pi_mem> ReqPiMem)
      : MThisCmd{ThisCmd}, MReqToMem(std::move(ReqToMem)),
        MReqPiMem(std::move(ReqPiMem)) {}

  void operator()() const {
    assert(MThisCmd->getCG().getType() == CGType::CodeplayHostTask);

    CGHostTask &HostTask = static_cast<CGHostTask &>(MThisCmd->getCG());

#ifdef XPTI_ENABLE_INSTRUMENTATION
    // Host task is executed async and in a separate thread that do not allow to
    // use code location data stored in TLS. So we keep submission code location
    // as Command field and put it here to TLS so that thrown exception could
    // query and report it.
    std::unique_ptr<detail::tls_code_loc_t> AsyncCodeLocationPtr;
    if (xptiTraceEnabled() && !CurrentCodeLocationValid()) {
      AsyncCodeLocationPtr.reset(
          new detail::tls_code_loc_t(MThisCmd->MSubmissionCodeLocation));
    }
#endif

    if (!waitForEvents()) {
      std::exception_ptr EPtr = std::make_exception_ptr(sycl::exception(
          make_error_code(errc::runtime),
          std::string("Couldn't wait for host-task's dependencies")));

      MThisCmd->MEvent->getSubmittedQueue()->reportAsyncException(EPtr);
      // reset host-task's lambda and quit
      HostTask.MHostTask.reset();
      Scheduler::getInstance().NotifyHostTaskCompletion(MThisCmd);
      return;
    }

    try {
      // we're ready to call the user-defined lambda now
      if (HostTask.MHostTask->isInteropTask()) {
        assert(HostTask.MQueue &&
               "Host task submissions should have an associated queue");
        interop_handle IH{MReqToMem, HostTask.MQueue,
                          HostTask.MQueue->getDeviceImplPtr(),
                          HostTask.MQueue->getContextImplPtr()};
        // TODO: should all the backends that support this entry point use this
        // for host task?
        auto &Queue = HostTask.MQueue;
        bool NativeCommandSupport = false;
        Queue->getPlugin()->call<PiApiKind::piDeviceGetInfo>(
            detail::getSyclObjImpl(Queue->get_device())->getHandleRef(),
            PI_EXT_ONEAPI_DEVICE_INFO_ENQUEUE_NATIVE_COMMAND_SUPPORT,
            sizeof(NativeCommandSupport), &NativeCommandSupport, nullptr);
        if (NativeCommandSupport) {
          EnqueueNativeCommandData CustomOpData{
              IH, HostTask.MHostTask->MInteropTask};

          // We are assuming that we have already synchronized with the HT's
          // dependent events, and that the user will synchronize before the end
          // of the HT lambda. As such we don't pass in any events, or ask for
          // one back.
          //
          // This entry point is needed in order to migrate memory across
          // devices in the same context for CUDA and HIP backends
          Queue->getPlugin()->call<PiApiKind::piextEnqueueNativeCommand>(
              HostTask.MQueue->getHandleRef(), InteropFreeFunc, &CustomOpData,
              MReqPiMem.size(), MReqPiMem.data(), 0, nullptr, nullptr);
        } else {
          HostTask.MHostTask->call(MThisCmd->MEvent->getHostProfilingInfo(),
                                   IH);
        }
      } else
        HostTask.MHostTask->call(MThisCmd->MEvent->getHostProfilingInfo());
    } catch (...) {
      auto CurrentException = std::current_exception();
#ifdef XPTI_ENABLE_INSTRUMENTATION
      // sycl::exception emit tracing of message with code location if
      // available. For other types of exception we need to explicitly trigger
      // tracing by calling TraceEventXPTI.
      if (xptiTraceEnabled()) {
        try {
          rethrow_exception(CurrentException);
        } catch (const sycl::exception &) {
          // it is already traced, nothing to care about
        } catch (const std::exception &StdException) {
          GlobalHandler::instance().TraceEventXPTI(StdException.what());
        } catch (...) {
          GlobalHandler::instance().TraceEventXPTI(
              "Host task lambda thrown non standard exception");
        }
      }
#endif
      MThisCmd->MEvent->getSubmittedQueue()->reportAsyncException(
          CurrentException);
    }

    HostTask.MHostTask.reset();

#ifdef XPTI_ENABLE_INSTRUMENTATION
    // Host Task is done, clear its submittion location to not interfere with
    // following dependent kernels submission.
    AsyncCodeLocationPtr.reset();
#endif

    try {
      // If we enqueue blocked users - pi level could throw exception that
      // should be treated as async now.
      Scheduler::getInstance().NotifyHostTaskCompletion(MThisCmd);
    } catch (...) {
      auto CurrentException = std::current_exception();
      MThisCmd->MEvent->getSubmittedQueue()->reportAsyncException(
          CurrentException);
    }
  }
};

void Command::waitForPreparedHostEvents() const {
  for (const EventImplPtr &HostEvent : MPreparedHostDepsEvents)
    HostEvent->waitInternal();
}

void Command::waitForEvents(QueueImplPtr Queue,
                            std::vector<EventImplPtr> &EventImpls,
                            sycl::detail::pi::PiEvent &Event) {
#ifndef NDEBUG
  for (const EventImplPtr &Event : EventImpls)
    assert(!Event->isHost() &&
           "Only non-host events are expected to be waited for here");
#endif
  if (!EventImpls.empty()) {
    if (!Queue) {
      // Host queue can wait for events from different contexts, i.e. it may
      // contain events with different contexts in its MPreparedDepsEvents.
      // OpenCL 2.1 spec says that clWaitForEvents will return
      // CL_INVALID_CONTEXT if events specified in the list do not belong to
      // the same context. Thus we split all the events into per-context map.
      // An example. We have two queues for the same CPU device: Q1, Q2. Thus
      // we will have two different contexts for the same CPU device: C1, C2.
      // Also we have default host queue. This queue is accessible via
      // Scheduler. Now, let's assume we have three different events: E1(C1),
      // E2(C1), E3(C2). The command's MPreparedDepsEvents will contain all
      // three events (E1, E2, E3). Now, if piEventsWait is called for all
      // three events we'll experience failure with CL_INVALID_CONTEXT 'cause
      // these events refer to different contexts.
      std::map<context_impl *, std::vector<EventImplPtr>>
          RequiredEventsPerContext;

      for (const EventImplPtr &Event : EventImpls) {
        ContextImplPtr Context = Event->getContextImpl();
        assert(Context.get() &&
               "Only non-host events are expected to be waited for here");
        RequiredEventsPerContext[Context.get()].push_back(Event);
      }

      for (auto &CtxWithEvents : RequiredEventsPerContext) {
        std::vector<sycl::detail::pi::PiEvent> RawEvents =
            getPiEvents(CtxWithEvents.second);
        CtxWithEvents.first->getPlugin()->call<PiApiKind::piEventsWait>(
            RawEvents.size(), RawEvents.data());
      }
    } else {
      std::vector<sycl::detail::pi::PiEvent> RawEvents =
          getPiEvents(EventImpls);
      flushCrossQueueDeps(EventImpls, MWorkerQueue);
      const PluginPtr &Plugin = Queue->getPlugin();

      if (MEvent != nullptr)
        MEvent->setHostEnqueueTime();
      Plugin->call<PiApiKind::piEnqueueEventsWait>(
          Queue->getHandleRef(), RawEvents.size(), &RawEvents[0], &Event);
    }
  }
}

/// It is safe to bind MPreparedDepsEvents and MPreparedHostDepsEvents
/// references to event_impl class members because Command
/// should not outlive the event connected to it.
Command::Command(
    CommandType Type, QueueImplPtr Queue,
    sycl::detail::pi::PiExtCommandBuffer CommandBuffer,
    const std::vector<sycl::detail::pi::PiExtSyncPoint> &SyncPoints)
    : MQueue(std::move(Queue)),
      MEvent(std::make_shared<detail::event_impl>(MQueue)),
      MPreparedDepsEvents(MEvent->getPreparedDepsEvents()),
      MPreparedHostDepsEvents(MEvent->getPreparedHostDepsEvents()), MType(Type),
      MCommandBuffer(CommandBuffer), MSyncPointDeps(SyncPoints) {
  MWorkerQueue = MQueue;
  MEvent->setWorkerQueue(MWorkerQueue);
  MEvent->setSubmittedQueue(MWorkerQueue);
  MEvent->setCommand(this);
  if (MQueue)
    MEvent->setContextImpl(MQueue->getContextImplPtr());
  MEvent->setStateIncomplete();
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
/// @param ObjAddr The address that defines the edge dependency; it is the
/// event address when the edge is for an event and a memory object address if
/// it is due to an accessor
/// @param Prefix Contains "event" if the dependency is an edge and contains
/// the access mode to the buffer if it is due to an accessor
/// @param IsCommand True if the dependency has a command object as the
/// source, false otherwise
void Command::emitEdgeEventForCommandDependence(
    Command *Cmd, void *ObjAddr, bool IsCommand,
    std::optional<access::mode> AccMode) {
#ifdef XPTI_ENABLE_INSTRUMENTATION
  // Bail early if either the source or the target node for the given
  // dependency is undefined or NULL
  constexpr uint16_t NotificationTraceType = xpti::trace_edge_create;
  if (!(xptiCheckTraceEnabled(MStreamID, NotificationTraceType) &&
        MTraceEvent && Cmd && Cmd->MTraceEvent))
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
    xptiNotifySubscribers(MStreamID, NotificationTraceType,
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
void Command::emitEdgeEventForEventDependence(
    Command *Cmd, sycl::detail::pi::PiEvent &PiEventAddr) {
#ifdef XPTI_ENABLE_INSTRUMENTATION
  // If we have failed to create an event to represent the Command, then we
  // cannot emit an edge event. Bail early!
  if (!(xptiCheckTraceEnabled(MStreamID) && MTraceEvent))
    return;

  if (Cmd && Cmd->MTraceEvent) {
    // If the event is associated with a command, we use this command's trace
    // event as the source of edge, hence modeling the control flow
    emitEdgeEventForCommandDependence(Cmd, (void *)PiEventAddr, false);
    return;
  }
  if (PiEventAddr) {
    xpti::utils::StringHelper SH;
    std::string AddressStr =
        SH.addressAsString<sycl::detail::pi::PiEvent>(PiEventAddr);
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
  if (!xptiCheckTraceEnabled(MStreamID))
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
  constexpr uint16_t NotificationTraceType = xpti::trace_node_create;
  if (!(xptiCheckTraceEnabled(MStreamID, NotificationTraceType) && MTraceEvent))
    return;
  assert(MTraceEventPrologComplete);
  xptiNotifySubscribers(MStreamID, NotificationTraceType,
                        detail::GSYCLGraphEvent,
                        static_cast<xpti_td *>(MTraceEvent), MInstanceID,
                        static_cast<const void *>(MCommandNodeType.c_str()));
#endif
}

Command *Command::processDepEvent(EventImplPtr DepEvent, const DepDesc &Dep,
                                  std::vector<Command *> &ToCleanUp) {
  const ContextImplPtr &WorkerContext = getWorkerContext();

  // 1. Non-host events can be ignored if they are not fully initialized.
  // 2. Some types of commands do not produce PI events after they are
  // enqueued (e.g. alloca). Note that we can't check the pi event to make that
  // distinction since the command might still be unenqueued at this point.
  bool PiEventExpected =
      (!DepEvent->isHost() && !DepEvent->isDefaultConstructed());
  if (auto *DepCmd = static_cast<Command *>(DepEvent->getCommand()))
    PiEventExpected &= DepCmd->producesPiEvent();

  if (!PiEventExpected) {
    // call to waitInternal() is in waitForPreparedHostEvents() as it's called
    // from enqueue process functions
    MPreparedHostDepsEvents.push_back(DepEvent);
    return nullptr;
  }

  Command *ConnectionCmd = nullptr;

  ContextImplPtr DepEventContext = DepEvent->getContextImpl();
  // If contexts don't match we'll connect them using host task
  if (DepEventContext != WorkerContext && WorkerContext) {
    Scheduler::GraphBuilder &GB = Scheduler::getInstance().MGraphBuilder;
    ConnectionCmd = GB.connectDepEvent(this, DepEvent, Dep, ToCleanUp);
  } else
    MPreparedDepsEvents.push_back(std::move(DepEvent));

  return ConnectionCmd;
}

ContextImplPtr Command::getWorkerContext() const {
  if (!MQueue)
    return nullptr;
  return MQueue->getContextImplPtr();
}

bool Command::producesPiEvent() const { return true; }

bool Command::supportsPostEnqueueCleanup() const { return true; }

bool Command::readyForCleanup() const {
  return MLeafCounter == 0 &&
         MEnqueueStatus == EnqueueResultT::SyclEnqueueSuccess;
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
  sycl::detail::pi::PiEvent &PiEventAddr = Event->getHandleRef();
  // Now make an edge for the dependent event
  emitEdgeEventForEventDependence(Cmd, PiEventAddr);
#endif

  return processDepEvent(std::move(Event), DepDesc{nullptr, nullptr, nullptr},
                         ToCleanUp);
}

void Command::emitEnqueuedEventSignal(sycl::detail::pi::PiEvent &PiEventAddr) {
#ifdef XPTI_ENABLE_INSTRUMENTATION
  emitInstrumentationGeneral(
      MStreamID, MInstanceID, static_cast<xpti_td *>(MTraceEvent),
      xpti::trace_signal, static_cast<const void *>(PiEventAddr));
#endif
  std::ignore = PiEventAddr;
}

void Command::emitInstrumentation(uint16_t Type, const char *Txt) {
#ifdef XPTI_ENABLE_INSTRUMENTATION
  return emitInstrumentationGeneral(MStreamID, MInstanceID,
                                    static_cast<xpti_td *>(MTraceEvent), Type,
                                    static_cast<const void *>(Txt));
#else
  std::ignore = Type;
  std::ignore = Txt;
#endif
}

bool Command::enqueue(EnqueueResultT &EnqueueResult, BlockingT Blocking,
                      std::vector<Command *> &ToCleanUp) {
#ifdef XPTI_ENABLE_INSTRUMENTATION
  // If command is enqueued from host task thread - it will not have valid
  // submission code location set. So we set it manually to properly trace
  // failures if pi level report any.
  std::unique_ptr<detail::tls_code_loc_t> AsyncCodeLocationPtr;
  if (xptiTraceEnabled() && !CurrentCodeLocationValid()) {
    AsyncCodeLocationPtr.reset(
        new detail::tls_code_loc_t(MSubmissionCodeLocation));
  }
#endif
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
  pi_int32 Res = enqueueImp();

  if (PI_SUCCESS != Res)
    EnqueueResult =
        EnqueueResultT(EnqueueResultT::SyclEnqueueFailed, this, Res);
  else {
    MEvent->setEnqueued();
    if (MShouldCompleteEventIfPossible &&
        (MEvent->isHost() || MEvent->getHandleRef() == nullptr))
      MEvent->setComplete();

    // Consider the command is successfully enqueued if return code is
    // PI_SUCCESS
    MEnqueueStatus = EnqueueResultT::SyclEnqueueSuccess;
    if (MLeafCounter == 0 && supportsPostEnqueueCleanup() &&
        !SYCLConfig<SYCL_DISABLE_EXECUTION_GRAPH_CLEANUP>::get() &&
        !SYCLConfig<SYCL_DISABLE_POST_ENQUEUE_CLEANUP>::get()) {
      assert(!MMarkedForCleanup);
      MMarkedForCleanup = true;
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
  // We have all the Commands that must be completed before the release
  // command can be enqueued; here we'll find the command that is an Alloca
  // with the same SYCLMemObject address and create a dependency line (edge)
  // between them in our sematic modeling
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

void Command::copySubmissionCodeLocation() {
#ifdef XPTI_ENABLE_INSTRUMENTATION
  if (!xptiTraceEnabled())
    return;

  detail::tls_code_loc_t Tls;
  auto TData = Tls.query();
  if (TData.fileName())
    MSubmissionFileName = TData.fileName();
  if (TData.functionName())
    MSubmissionFunctionName = TData.functionName();
  if (MSubmissionFileName.size() || MSubmissionFunctionName.size())
    MSubmissionCodeLocation = {
        MSubmissionFileName.c_str(), MSubmissionFunctionName.c_str(),
        (int)TData.lineNumber(), (int)TData.columnNumber()};
#endif
}

AllocaCommandBase::AllocaCommandBase(CommandType Type, QueueImplPtr Queue,
                                     Requirement Req,
                                     AllocaCommandBase *LinkedAllocaCmd,
                                     bool IsConst)
    : Command(Type, Queue), MLinkedAllocaCmd(LinkedAllocaCmd),
      MIsLeaderAlloca(nullptr == LinkedAllocaCmd), MIsConst(IsConst),
      MRequirement(std::move(Req)), MReleaseCmd(Queue, this) {
  MRequirement.MAccessMode = access::mode::read_write;
  emitInstrumentationDataProxy();
}

void AllocaCommandBase::emitInstrumentationData() {
#ifdef XPTI_ENABLE_INSTRUMENTATION
  if (!xptiCheckTraceEnabled(MStreamID))
    return;
  // Create a payload with the command name and an event using this payload to
  // emit a node_create
  MAddress = MRequirement.MSYCLMemObj;
  makeTraceEventProlog(MAddress);
  // Set the relevant meta data properties for this command
  if (MTraceEvent && MFirstInstance) {
    xpti_td *TE = static_cast<xpti_td *>(MTraceEvent);
    addDeviceMetadata(TE, MQueue);
    xpti::addMetadata(TE, "memory_object", reinterpret_cast<size_t>(MAddress));
    // Since we do NOT add queue_id value to metadata, we are stashing it to TLS
    // as this data is mutable and the metadata is supposed to be invariant
    xpti::framework::stash_tuple(XPTI_QUEUE_INSTANCE_ID_KEY,
                                 getQueueID(MQueue));
  }
#endif
}

bool AllocaCommandBase::producesPiEvent() const { return false; }

bool AllocaCommandBase::supportsPostEnqueueCleanup() const { return false; }

bool AllocaCommandBase::readyForCleanup() const { return false; }

AllocaCommand::AllocaCommand(QueueImplPtr Queue, Requirement Req,
                             bool InitFromUserData,
                             AllocaCommandBase *LinkedAllocaCmd, bool IsConst)
    : AllocaCommandBase(CommandType::ALLOCA, std::move(Queue), std::move(Req),
                        LinkedAllocaCmd, IsConst),
      MInitFromUserData(InitFromUserData) {
  // Node event must be created before the dependent edge is added to this
  // node, so this call must be before the addDep() call.
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
  if (!xptiCheckTraceEnabled(MStreamID))
    return;

  // Only if it is the first event, we emit a node create event
  if (MFirstInstance) {
    makeTraceEventEpilog();
  }
#endif
}

pi_int32 AllocaCommand::enqueueImp() {
  waitForPreparedHostEvents();
  std::vector<EventImplPtr> EventImpls = MPreparedDepsEvents;

  sycl::detail::pi::PiEvent &Event = MEvent->getHandleRef();

  void *HostPtr = nullptr;
  if (!MIsLeaderAlloca) {

    if (!MQueue) {
      // Do not need to make allocation if we have a linked device allocation
      Command::waitForEvents(MQueue, EventImpls, Event);

      return PI_SUCCESS;
    }
    HostPtr = MLinkedAllocaCmd->getMemAllocation();
  }
  // TODO: Check if it is correct to use std::move on stack variable and
  // delete it RawEvents below.
  MMemAllocation = MemoryManager::allocate(getContext(MQueue), getSYCLMemObj(),
                                           MInitFromUserData, HostPtr,
                                           std::move(EventImpls), Event);

  return PI_SUCCESS;
}

void AllocaCommand::printDot(std::ostream &Stream) const {
  Stream << "\"" << this << "\" [style=filled, fillcolor=\"#FFD28A\", label=\"";

  Stream << "ID = " << this << "\\n";
  Stream << "ALLOCA ON " << queueDeviceToString(MQueue.get()) << "\\n";
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
                        /*LinkedAllocaCmd*/ nullptr, /*IsConst*/ false),
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
  if (!xptiCheckTraceEnabled(MStreamID))
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
    xpti::framework::stash_tuple(XPTI_QUEUE_INSTANCE_ID_KEY,
                                 getQueueID(MQueue));
    makeTraceEventEpilog();
  }
#endif
}

void *AllocaSubBufCommand::getMemAllocation() const {
  // In some cases parent`s memory allocation might change (e.g., after
  // map/unmap operations). If parent`s memory allocation changes, sub-buffer
  // memory allocation should be changed as well.
  if (!MQueue) {
    return static_cast<void *>(
        static_cast<char *>(MParentAlloca->getMemAllocation()) +
        MRequirement.MOffsetInBytes);
  }
  return MMemAllocation;
}

pi_int32 AllocaSubBufCommand::enqueueImp() {
  waitForPreparedHostEvents();
  std::vector<EventImplPtr> EventImpls = MPreparedDepsEvents;
  sycl::detail::pi::PiEvent &Event = MEvent->getHandleRef();

  MMemAllocation = MemoryManager::allocateMemSubBuffer(
      getContext(MQueue), MParentAlloca->getMemAllocation(),
      MRequirement.MElemSize, MRequirement.MOffsetInBytes,
      MRequirement.MAccessRange, std::move(EventImpls), Event);

  XPTIRegistry::bufferAssociateNotification(MParentAlloca->getSYCLMemObj(),
                                            MMemAllocation);
  return PI_SUCCESS;
}

void AllocaSubBufCommand::printDot(std::ostream &Stream) const {
  Stream << "\"" << this << "\" [style=filled, fillcolor=\"#FFD28A\", label=\"";

  Stream << "ID = " << this << "\\n";
  Stream << "ALLOCA SUB BUF ON " << queueDeviceToString(MQueue.get()) << "\\n";
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
  if (!xptiCheckTraceEnabled(MStreamID))
    return;
  // Create a payload with the command name and an event using this payload to
  // emit a node_create
  MAddress = MAllocaCmd->getSYCLMemObj();
  makeTraceEventProlog(MAddress);

  if (MFirstInstance) {
    xpti_td *TE = static_cast<xpti_td *>(MTraceEvent);
    addDeviceMetadata(TE, MQueue);
    xpti::addMetadata(TE, "allocation_type",
                      commandToName(MAllocaCmd->getType()));
    // Since we do NOT add queue_id value to metadata, we are stashing it to TLS
    // as this data is mutable and the metadata is supposed to be invariant
    xpti::framework::stash_tuple(XPTI_QUEUE_INSTANCE_ID_KEY,
                                 getQueueID(MQueue));
    makeTraceEventEpilog();
  }
#endif
}

pi_int32 ReleaseCommand::enqueueImp() {
  waitForPreparedHostEvents();
  std::vector<EventImplPtr> EventImpls = MPreparedDepsEvents;
  std::vector<sycl::detail::pi::PiEvent> RawEvents = getPiEvents(EventImpls);
  bool SkipRelease = false;

  // On host side we only allocate memory for full buffers.
  // Thus, deallocating sub buffers leads to double memory freeing.
  SkipRelease |= !MQueue && MAllocaCmd->getType() == ALLOCA_SUB_BUF;

  const bool CurAllocaIsHost = !MAllocaCmd->getQueue();
  bool NeedUnmap = false;
  if (MAllocaCmd->MLinkedAllocaCmd) {

    // When releasing one of the "linked" allocations special rules take
    // place:
    // 1. Device allocation should always be released.
    // 2. Host allocation should be released if host allocation is "leader".
    // 3. Device alloca in the pair should be in active state in order to be
    //    correctly released.

    // There is no actual memory allocation if a host alloca command is
    // created being linked to a device allocation.
    SkipRelease |= CurAllocaIsHost && !MAllocaCmd->MIsLeaderAlloca;

    NeedUnmap |= CurAllocaIsHost == MAllocaCmd->MIsActive;
  }

  if (NeedUnmap) {
    const QueueImplPtr &Queue = CurAllocaIsHost
                                    ? MAllocaCmd->MLinkedAllocaCmd->getQueue()
                                    : MAllocaCmd->getQueue();

    EventImplPtr UnmapEventImpl(new event_impl(Queue));
    UnmapEventImpl->setContextImpl(getContext(Queue));
    UnmapEventImpl->setStateIncomplete();
    sycl::detail::pi::PiEvent &UnmapEvent = UnmapEventImpl->getHandleRef();

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
  sycl::detail::pi::PiEvent &Event = MEvent->getHandleRef();
  if (SkipRelease)
    Command::waitForEvents(MQueue, EventImpls, Event);
  else {
    MemoryManager::release(getContext(MQueue), MAllocaCmd->getSYCLMemObj(),
                           MAllocaCmd->getMemAllocation(),
                           std::move(EventImpls), Event);
  }
  return PI_SUCCESS;
}

void ReleaseCommand::printDot(std::ostream &Stream) const {
  Stream << "\"" << this << "\" [style=filled, fillcolor=\"#FF827A\", label=\"";

  Stream << "ID = " << this << " ; ";
  Stream << "RELEASE ON " << queueDeviceToString(MQueue.get()) << "\\n";
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

bool ReleaseCommand::readyForCleanup() const { return false; }

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
  if (!xptiCheckTraceEnabled(MStreamID))
    return;
  // Create a payload with the command name and an event using this payload to
  // emit a node_create
  MAddress = MSrcAllocaCmd->getSYCLMemObj();
  makeTraceEventProlog(MAddress);

  if (MFirstInstance) {
    xpti_td *TE = static_cast<xpti_td *>(MTraceEvent);
    addDeviceMetadata(TE, MQueue);
    xpti::addMetadata(TE, "memory_object", reinterpret_cast<size_t>(MAddress));
    // Since we do NOT add queue_id value to metadata, we are stashing it to TLS
    // as this data is mutable and the metadata is supposed to be invariant
    xpti::framework::stash_tuple(XPTI_QUEUE_INSTANCE_ID_KEY,
                                 getQueueID(MQueue));
    makeTraceEventEpilog();
  }
#endif
}

pi_int32 MapMemObject::enqueueImp() {
  waitForPreparedHostEvents();
  std::vector<EventImplPtr> EventImpls = MPreparedDepsEvents;
  std::vector<sycl::detail::pi::PiEvent> RawEvents = getPiEvents(EventImpls);
  flushCrossQueueDeps(EventImpls, MWorkerQueue);

  sycl::detail::pi::PiEvent &Event = MEvent->getHandleRef();
  *MDstPtr = MemoryManager::map(
      MSrcAllocaCmd->getSYCLMemObj(), MSrcAllocaCmd->getMemAllocation(), MQueue,
      MMapMode, MSrcReq.MDims, MSrcReq.MMemoryRange, MSrcReq.MAccessRange,
      MSrcReq.MOffset, MSrcReq.MElemSize, std::move(RawEvents), Event);

  return PI_SUCCESS;
}

void MapMemObject::printDot(std::ostream &Stream) const {
  Stream << "\"" << this << "\" [style=filled, fillcolor=\"#77AFFF\", label=\"";

  Stream << "ID = " << this << " ; ";
  Stream << "MAP ON " << queueDeviceToString(MQueue.get()) << "\\n";

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
  if (!xptiCheckTraceEnabled(MStreamID))
    return;
  // Create a payload with the command name and an event using this payload to
  // emit a node_create
  MAddress = MDstAllocaCmd->getSYCLMemObj();
  makeTraceEventProlog(MAddress);

  if (MFirstInstance) {
    xpti_td *TE = static_cast<xpti_td *>(MTraceEvent);
    addDeviceMetadata(TE, MQueue);
    xpti::addMetadata(TE, "memory_object", reinterpret_cast<size_t>(MAddress));
    // Since we do NOT add queue_id value to metadata, we are stashing it to TLS
    // as this data is mutable and the metadata is supposed to be invariant
    xpti::framework::stash_tuple(XPTI_QUEUE_INSTANCE_ID_KEY,
                                 getQueueID(MQueue));
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
  return MQueue && (MQueue->getDeviceImplPtr()->getBackend() !=
                        backend::ext_oneapi_level_zero ||
                    MEvent->getHandleRef() != nullptr);
}

pi_int32 UnMapMemObject::enqueueImp() {
  waitForPreparedHostEvents();
  std::vector<EventImplPtr> EventImpls = MPreparedDepsEvents;
  std::vector<sycl::detail::pi::PiEvent> RawEvents = getPiEvents(EventImpls);
  flushCrossQueueDeps(EventImpls, MWorkerQueue);

  sycl::detail::pi::PiEvent &Event = MEvent->getHandleRef();
  MemoryManager::unmap(MDstAllocaCmd->getSYCLMemObj(),
                       MDstAllocaCmd->getMemAllocation(), MQueue, *MSrcPtr,
                       std::move(RawEvents), Event);

  return PI_SUCCESS;
}

void UnMapMemObject::printDot(std::ostream &Stream) const {
  Stream << "\"" << this << "\" [style=filled, fillcolor=\"#EBC40F\", label=\"";

  Stream << "ID = " << this << " ; ";
  Stream << "UNMAP ON " << queueDeviceToString(MQueue.get()) << "\\n";

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
  if (MSrcQueue) {
    MEvent->setContextImpl(MSrcQueue->getContextImplPtr());
  }

  MWorkerQueue = !MQueue ? MSrcQueue : MQueue;
  MEvent->setWorkerQueue(MWorkerQueue);

  emitInstrumentationDataProxy();
}

void MemCpyCommand::emitInstrumentationData() {
#ifdef XPTI_ENABLE_INSTRUMENTATION
  if (!xptiCheckTraceEnabled(MStreamID))
    return;
  // Create a payload with the command name and an event using this payload to
  // emit a node_create
  MAddress = MSrcAllocaCmd->getSYCLMemObj();
  makeTraceEventProlog(MAddress);

  if (MFirstInstance) {
    xpti_td *CmdTraceEvent = static_cast<xpti_td *>(MTraceEvent);
    addDeviceMetadata(CmdTraceEvent, MQueue);
    xpti::addMetadata(CmdTraceEvent, "memory_object",
                      reinterpret_cast<size_t>(MAddress));
    xpti::addMetadata(CmdTraceEvent, "copy_from",
                      MSrcQueue ? deviceToID(MSrcQueue->get_device()) : 0);
    xpti::addMetadata(CmdTraceEvent, "copy_to",
                      MQueue ? deviceToID(MQueue->get_device()) : 0);
    // Since we do NOT add queue_id value to metadata, we are stashing it to TLS
    // as this data is mutable and the metadata is supposed to be invariant
    xpti::framework::stash_tuple(XPTI_QUEUE_INSTANCE_ID_KEY,
                                 getQueueID(MQueue));
    makeTraceEventEpilog();
  }
#endif
}

ContextImplPtr MemCpyCommand::getWorkerContext() const {
  if (!MWorkerQueue)
    return nullptr;
  return MWorkerQueue->getContextImplPtr();
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
  return !MQueue ||
         MQueue->getDeviceImplPtr()->getBackend() !=
             backend::ext_oneapi_level_zero ||
         MEvent->getHandleRef() != nullptr;
}

pi_int32 MemCpyCommand::enqueueImp() {
  waitForPreparedHostEvents();
  std::vector<EventImplPtr> EventImpls = MPreparedDepsEvents;

  sycl::detail::pi::PiEvent &Event = MEvent->getHandleRef();

  auto RawEvents = getPiEvents(EventImpls);
  flushCrossQueueDeps(EventImpls, MWorkerQueue);

  MemoryManager::copy(
      MSrcAllocaCmd->getSYCLMemObj(), MSrcAllocaCmd->getMemAllocation(),
      MSrcQueue, MSrcReq.MDims, MSrcReq.MMemoryRange, MSrcReq.MAccessRange,
      MSrcReq.MOffset, MSrcReq.MElemSize, MDstAllocaCmd->getMemAllocation(),
      MQueue, MDstReq.MDims, MDstReq.MMemoryRange, MDstReq.MAccessRange,
      MDstReq.MOffset, MDstReq.MElemSize, std::move(RawEvents), Event, MEvent);

  return PI_SUCCESS;
}

void MemCpyCommand::printDot(std::ostream &Stream) const {
  Stream << "\"" << this << "\" [style=filled, fillcolor=\"#C7EB15\" label=\"";

  Stream << "ID = " << this << " ; ";
  Stream << "MEMCPY ON " << queueDeviceToString(MQueue.get()) << "\\n";
  Stream << "From: " << MSrcAllocaCmd << " is host: " << !MSrcQueue << "\\n";
  Stream << "To: " << MDstAllocaCmd << " is host: " << !MQueue << "\\n";

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
  // Default constructed accessors do not add dependencies, but they can be
  // passed to commands. Simply return nullptr, since they are empty and don't
  // really require any memory.
  return nullptr;
}

std::vector<std::shared_ptr<const void>>
ExecCGCommand::getAuxiliaryResources() const {
  if (MCommandGroup->getType() == CGType::Kernel)
    return ((CGExecKernel *)MCommandGroup.get())->getAuxiliaryResources();
  return {};
}

void ExecCGCommand::clearAuxiliaryResources() {
  if (MCommandGroup->getType() == CGType::Kernel)
    ((CGExecKernel *)MCommandGroup.get())->clearAuxiliaryResources();
}

pi_int32 UpdateHostRequirementCommand::enqueueImp() {
  waitForPreparedHostEvents();
  std::vector<EventImplPtr> EventImpls = MPreparedDepsEvents;
  sycl::detail::pi::PiEvent &Event = MEvent->getHandleRef();
  Command::waitForEvents(MQueue, EventImpls, Event);

  assert(MSrcAllocaCmd && "Expected valid alloca command");
  assert(MSrcAllocaCmd->getMemAllocation() && "Expected valid source pointer");
  assert(MDstPtr && "Expected valid target pointer");
  *MDstPtr = MSrcAllocaCmd->getMemAllocation();

  return PI_SUCCESS;
}

void UpdateHostRequirementCommand::printDot(std::ostream &Stream) const {
  Stream << "\"" << this << "\" [style=filled, fillcolor=\"#f1337f\", label=\"";

  Stream << "ID = " << this << "\\n";
  Stream << "UPDATE REQ ON " << queueDeviceToString(MQueue.get()) << "\\n";
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
  if (MSrcQueue) {
    MEvent->setContextImpl(MSrcQueue->getContextImplPtr());
  }

  MWorkerQueue = !MQueue ? MSrcQueue : MQueue;
  MEvent->setWorkerQueue(MWorkerQueue);

  emitInstrumentationDataProxy();
}

void MemCpyCommandHost::emitInstrumentationData() {
#ifdef XPTI_ENABLE_INSTRUMENTATION
  if (!xptiCheckTraceEnabled(MStreamID))
    return;
  // Create a payload with the command name and an event using this payload to
  // emit a node_create
  MAddress = MSrcAllocaCmd->getSYCLMemObj();
  makeTraceEventProlog(MAddress);

  if (MFirstInstance) {
    xpti_td *CmdTraceEvent = static_cast<xpti_td *>(MTraceEvent);
    addDeviceMetadata(CmdTraceEvent, MQueue);
    xpti::addMetadata(CmdTraceEvent, "memory_object",
                      reinterpret_cast<size_t>(MAddress));
    xpti::addMetadata(CmdTraceEvent, "copy_from",
                      MSrcQueue ? deviceToID(MSrcQueue->get_device()) : 0);
    xpti::addMetadata(CmdTraceEvent, "copy_to",
                      MQueue ? deviceToID(MQueue->get_device()) : 0);
    // Since we do NOT add queue_id value to metadata, we are stashing it to TLS
    // as this data is mutable and the metadata is supposed to be invariant
    xpti::framework::stash_tuple(XPTI_QUEUE_INSTANCE_ID_KEY,
                                 getQueueID(MQueue));
    makeTraceEventEpilog();
  }
#endif
}

ContextImplPtr MemCpyCommandHost::getWorkerContext() const {
  if (!MWorkerQueue)
    return nullptr;
  return MWorkerQueue->getContextImplPtr();
}

pi_int32 MemCpyCommandHost::enqueueImp() {
  const QueueImplPtr &Queue = MWorkerQueue;
  waitForPreparedHostEvents();
  std::vector<EventImplPtr> EventImpls = MPreparedDepsEvents;
  std::vector<sycl::detail::pi::PiEvent> RawEvents = getPiEvents(EventImpls);

  sycl::detail::pi::PiEvent &Event = MEvent->getHandleRef();
  // Omit copying if mode is discard one.
  // TODO: Handle this at the graph building time by, for example, creating
  // empty node instead of memcpy.
  if (MDstReq.MAccessMode == access::mode::discard_read_write ||
      MDstReq.MAccessMode == access::mode::discard_write) {
    Command::waitForEvents(Queue, EventImpls, Event);

    return PI_SUCCESS;
  }

  flushCrossQueueDeps(EventImpls, MWorkerQueue);
  MemoryManager::copy(
      MSrcAllocaCmd->getSYCLMemObj(), MSrcAllocaCmd->getMemAllocation(),
      MSrcQueue, MSrcReq.MDims, MSrcReq.MMemoryRange, MSrcReq.MAccessRange,
      MSrcReq.MOffset, MSrcReq.MElemSize, *MDstPtr, MQueue, MDstReq.MDims,
      MDstReq.MMemoryRange, MDstReq.MAccessRange, MDstReq.MOffset,
      MDstReq.MElemSize, std::move(RawEvents), MEvent->getHandleRef(), MEvent);

  return PI_SUCCESS;
}

EmptyCommand::EmptyCommand() : Command(CommandType::EMPTY_TASK, nullptr) {
  emitInstrumentationDataProxy();
}

pi_int32 EmptyCommand::enqueueImp() {
  waitForPreparedHostEvents();
  waitForEvents(MQueue, MPreparedDepsEvents, MEvent->getHandleRef());

  return PI_SUCCESS;
}

void EmptyCommand::addRequirement(Command *DepCmd, AllocaCommandBase *AllocaCmd,
                                  const Requirement *Req) {
  const Requirement &ReqRef = *Req;
  MRequirements.emplace_back(ReqRef);
  const Requirement *const StoredReq = &MRequirements.back();

  // EmptyCommand is always host one, so we believe that result of addDep is
  // nil
  std::vector<Command *> ToCleanUp;
  Command *Cmd = addDep(DepDesc{DepCmd, StoredReq, AllocaCmd}, ToCleanUp);
  assert(Cmd == nullptr && "Conection command should be null for EmptyCommand");
  assert(ToCleanUp.empty() && "addDep should add a command for cleanup only if "
                              "there's a connection command");
  (void)Cmd;
}

void EmptyCommand::emitInstrumentationData() {
#ifdef XPTI_ENABLE_INSTRUMENTATION
  if (!xptiCheckTraceEnabled(MStreamID))
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
    addDeviceMetadata(CmdTraceEvent, MQueue);
    xpti::addMetadata(CmdTraceEvent, "memory_object",
                      reinterpret_cast<size_t>(MAddress));
    // Since we do NOT add queue_id value to metadata, we are stashing it to TLS
    // as this data is mutable and the metadata is supposed to be invariant
    xpti::framework::stash_tuple(XPTI_QUEUE_INSTANCE_ID_KEY,
                                 getQueueID(MQueue));
    makeTraceEventEpilog();
  }
#endif
}

void EmptyCommand::printDot(std::ostream &Stream) const {
  Stream << "\"" << this << "\" [style=filled, fillcolor=\"#8d8f29\", label=\"";

  Stream << "ID = " << this << "\\n";
  Stream << "EMPTY NODE" << "\\n";

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

void MemCpyCommandHost::printDot(std::ostream &Stream) const {
  Stream << "\"" << this << "\" [style=filled, fillcolor=\"#B6A2EB\", label=\"";

  Stream << "ID = " << this << "\\n";
  Stream << "MEMCPY HOST ON " << queueDeviceToString(MQueue.get()) << "\\n";

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
  if (!xptiCheckTraceEnabled(MStreamID))
    return;
  // Create a payload with the command name and an event using this payload to
  // emit a node_create
  MAddress = MSrcAllocaCmd->getSYCLMemObj();
  makeTraceEventProlog(MAddress);

  if (MFirstInstance) {
    xpti_td *CmdTraceEvent = static_cast<xpti_td *>(MTraceEvent);
    addDeviceMetadata(CmdTraceEvent, MQueue);
    xpti::addMetadata(CmdTraceEvent, "memory_object",
                      reinterpret_cast<size_t>(MAddress));
    // Since we do NOT add queue_id value to metadata, we are stashing it to TLS
    // as this data is mutable and the metadata is supposed to be invariant
    xpti::framework::stash_tuple(XPTI_QUEUE_INSTANCE_ID_KEY,
                                 getQueueID(MQueue));
    makeTraceEventEpilog();
  }
#endif
}

static std::string_view cgTypeToString(detail::CGType Type) {
  switch (Type) {
  case detail::CGType::Kernel:
    return "Kernel";
    break;
  case detail::CGType::UpdateHost:
    return "update_host";
    break;
  case detail::CGType::Fill:
    return "fill";
    break;
  case detail::CGType::CopyAccToAcc:
    return "copy acc to acc";
    break;
  case detail::CGType::CopyAccToPtr:
    return "copy acc to ptr";
    break;
  case detail::CGType::CopyPtrToAcc:
    return "copy ptr to acc";
    break;
  case detail::CGType::Barrier:
    return "barrier";
  case detail::CGType::BarrierWaitlist:
    return "barrier waitlist";
  case detail::CGType::CopyUSM:
    return "copy usm";
    break;
  case detail::CGType::FillUSM:
    return "fill usm";
    break;
  case detail::CGType::PrefetchUSM:
    return "prefetch usm";
    break;
  case detail::CGType::CodeplayHostTask:
    return "host task";
    break;
  case detail::CGType::Copy2DUSM:
    return "copy 2d usm";
    break;
  case detail::CGType::Fill2DUSM:
    return "fill 2d usm";
    break;
  case detail::CGType::AdviseUSM:
    return "advise usm";
  case detail::CGType::Memset2DUSM:
    return "memset 2d usm";
    break;
  case detail::CGType::CopyToDeviceGlobal:
    return "copy to device_global";
    break;
  case detail::CGType::CopyFromDeviceGlobal:
    return "copy from device_global";
    break;
  case detail::CGType::ReadWriteHostPipe:
    return "read_write host pipe";
  case detail::CGType::ExecCommandBuffer:
    return "exec command buffer";
  case detail::CGType::CopyImage:
    return "copy image";
  case detail::CGType::SemaphoreWait:
    return "semaphore wait";
  case detail::CGType::SemaphoreSignal:
    return "semaphore signal";
  default:
    return "unknown";
    break;
  }
}

ExecCGCommand::ExecCGCommand(
    std::unique_ptr<detail::CG> CommandGroup, QueueImplPtr Queue,
    bool EventNeeded, sycl::detail::pi::PiExtCommandBuffer CommandBuffer,
    const std::vector<sycl::detail::pi::PiExtSyncPoint> &Dependencies)
    : Command(CommandType::RUN_CG, std::move(Queue), CommandBuffer,
              Dependencies),
      MEventNeeded(EventNeeded), MCommandGroup(std::move(CommandGroup)) {
  if (MCommandGroup->getType() == detail::CGType::CodeplayHostTask) {
    MEvent->setSubmittedQueue(
        static_cast<detail::CGHostTask *>(MCommandGroup.get())->MQueue);
  }
  if (MCommandGroup->getType() == detail::CGType::ProfilingTag)
    MEvent->markAsProfilingTagEvent();

  emitInstrumentationDataProxy();
}

#ifdef XPTI_ENABLE_INSTRUMENTATION
std::string instrumentationGetKernelName(
    const std::shared_ptr<detail::kernel_impl> &SyclKernel,
    const std::string &FunctionName, const std::string &SyclKernelName,
    void *&Address, std::optional<bool> &FromSource) {
  std::string KernelName;
  if (SyclKernel && SyclKernel->isCreatedFromSource()) {
    FromSource = true;
    pi_kernel KernelHandle = SyclKernel->getHandleRef();
    Address = KernelHandle;
    KernelName = FunctionName;
  } else {
    FromSource = false;
    KernelName = demangleKernelName(SyclKernelName);
  }
  return KernelName;
}

void instrumentationAddExtraKernelMetadata(
    xpti_td *&CmdTraceEvent, const NDRDescT &NDRDesc,
    const std::shared_ptr<detail::kernel_bundle_impl> &KernelBundleImplPtr,
    const std::string &KernelName,
    const std::shared_ptr<detail::kernel_impl> &SyclKernel,
    const QueueImplPtr &Queue,
    std::vector<ArgDesc> &CGArgs) // CGArgs are not const since they could be
                                  // sorted in this function
{
  std::vector<ArgDesc> Args;

  auto FilterArgs = [&Args](detail::ArgDesc &Arg, int NextTrueIndex) {
    Args.push_back({Arg.MType, Arg.MPtr, Arg.MSize, NextTrueIndex});
  };
  sycl::detail::pi::PiProgram Program = nullptr;
  sycl::detail::pi::PiKernel Kernel = nullptr;
  std::mutex *KernelMutex = nullptr;
  const KernelArgMask *EliminatedArgMask = nullptr;

  std::shared_ptr<kernel_impl> SyclKernelImpl;
  std::shared_ptr<device_image_impl> DeviceImageImpl;

  // Use kernel_bundle if available unless it is interop.
  // Interop bundles can't be used in the first branch, because the
  // kernels in interop kernel bundles (if any) do not have kernel_id and
  // can therefore not be looked up, but since they are self-contained
  // they can simply be launched directly.
  if (KernelBundleImplPtr && !KernelBundleImplPtr->isInterop()) {
    kernel_id KernelID =
        detail::ProgramManager::getInstance().getSYCLKernelID(KernelName);
    kernel SyclKernel =
        KernelBundleImplPtr->get_kernel(KernelID, KernelBundleImplPtr);
    std::shared_ptr<kernel_impl> KernelImpl =
        detail::getSyclObjImpl(SyclKernel);

    EliminatedArgMask = KernelImpl->getKernelArgMask();
    Program = KernelImpl->getDeviceImage()->get_program_ref();
  } else if (nullptr != SyclKernel) {
    Program = SyclKernel->getProgramRef();
    if (!SyclKernel->isCreatedFromSource())
      EliminatedArgMask = SyclKernel->getKernelArgMask();
  } else {
    assert(Queue && "Kernel submissions should have an associated queue");
    std::tie(Kernel, KernelMutex, EliminatedArgMask, Program) =
        detail::ProgramManager::getInstance().getOrCreateKernel(
            Queue->getContextImplPtr(), Queue->getDeviceImplPtr(), KernelName);
  }

  applyFuncOnFilteredArgs(EliminatedArgMask, CGArgs, FilterArgs);

  xpti::offload_kernel_enqueue_data_t KernelData{
      {NDRDesc.GlobalSize[0], NDRDesc.GlobalSize[1], NDRDesc.GlobalSize[2]},
      {NDRDesc.LocalSize[0], NDRDesc.LocalSize[1], NDRDesc.LocalSize[2]},
      {NDRDesc.GlobalOffset[0], NDRDesc.GlobalOffset[1],
       NDRDesc.GlobalOffset[2]},
      Args.size()};
  xpti::addMetadata(CmdTraceEvent, "enqueue_kernel_data", KernelData);
  for (size_t i = 0; i < Args.size(); i++) {
    std::string Prefix("arg");
    xpti::offload_kernel_arg_data_t arg{(int)Args[i].MType, Args[i].MPtr,
                                        Args[i].MSize, Args[i].MIndex};
    xpti::addMetadata(CmdTraceEvent, Prefix + std::to_string(i), arg);
  }
}

void instrumentationFillCommonData(const std::string &KernelName,
                                   const std::string &FileName, uint64_t Line,
                                   uint64_t Column, const void *const Address,
                                   const QueueImplPtr &Queue,
                                   std::optional<bool> &FromSource,
                                   uint64_t &OutInstanceID,
                                   xpti_td *&OutTraceEvent) {
  //  Get source file, line number information from the CommandGroup object
  //  and create payload using name, address, and source info
  //
  //  On Windows, since the support for builtin functions is not available in
  //  MSVC, the MFileName, MLine will be set to nullptr and "0" respectively.
  //  Handle this condition explicitly here.
  bool HasSourceInfo = false;
  xpti::payload_t Payload;
  if (!FileName.empty()) {
    // File name has a valid string
    Payload = xpti::payload_t(KernelName.c_str(), FileName.c_str(), Line,
                              Column, Address);
    HasSourceInfo = true;
  } else if (Address) {
    // We have a valid function name and an address
    Payload = xpti::payload_t(KernelName.c_str(), Address);
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
    OutInstanceID = CGKernelInstanceNo;
    OutTraceEvent = CmdTraceEvent;
    // If we are seeing this event again, then the instance ID will be greater
    // than 1; in this case, we will skip sending a notification to create a
    // node as this node has already been created.
    if (CGKernelInstanceNo > 1)
      return;

    addDeviceMetadata(CmdTraceEvent, Queue);
    if (!KernelName.empty()) {
      xpti::addMetadata(CmdTraceEvent, "kernel_name", KernelName);
    }
    if (FromSource.has_value()) {
      xpti::addMetadata(CmdTraceEvent, "from_source", FromSource.value());
    }
    if (HasSourceInfo) {
      xpti::addMetadata(CmdTraceEvent, "sym_function_name", KernelName);
      xpti::addMetadata(CmdTraceEvent, "sym_source_file_name", FileName);
      xpti::addMetadata(CmdTraceEvent, "sym_line_no", static_cast<int>(Line));
      xpti::addMetadata(CmdTraceEvent, "sym_column_no",
                        static_cast<int>(Column));
    }
    // We no longer set the 'queue_id' in the metadata structure as it is a
    // mutable value and multiple threads using the same queue created at the
    // same location will overwrite the metadata values creating inconsistencies
  }
}
#endif

#ifdef XPTI_ENABLE_INSTRUMENTATION
std::pair<xpti_td *, uint64_t> emitKernelInstrumentationData(
    int32_t StreamID, const std::shared_ptr<detail::kernel_impl> &SyclKernel,
    const detail::code_location &CodeLoc, const std::string &SyclKernelName,
    const QueueImplPtr &Queue, const NDRDescT &NDRDesc,
    const std::shared_ptr<detail::kernel_bundle_impl> &KernelBundleImplPtr,
    std::vector<ArgDesc> &CGArgs) {

  auto XptiObjects = std::make_pair<xpti_td *, uint64_t>(nullptr, -1);
  constexpr uint16_t NotificationTraceType = xpti::trace_node_create;
  if (!xptiCheckTraceEnabled(StreamID))
    return XptiObjects;

  void *Address = nullptr;
  std::optional<bool> FromSource;
  std::string KernelName = instrumentationGetKernelName(
      SyclKernel, std::string(CodeLoc.functionName()), SyclKernelName, Address,
      FromSource);

  auto &[CmdTraceEvent, InstanceID] = XptiObjects;

  std::string FileName =
      CodeLoc.fileName() ? CodeLoc.fileName() : std::string();
  instrumentationFillCommonData(KernelName, FileName, CodeLoc.lineNumber(),
                                CodeLoc.columnNumber(), Address, Queue,
                                FromSource, InstanceID, CmdTraceEvent);

  if (CmdTraceEvent) {
    // Stash the queue_id mutable metadata in TLS
    xpti::framework::stash_tuple(XPTI_QUEUE_INSTANCE_ID_KEY, getQueueID(Queue));
    instrumentationAddExtraKernelMetadata(CmdTraceEvent, NDRDesc,
                                          KernelBundleImplPtr, SyclKernelName,
                                          SyclKernel, Queue, CGArgs);

    xptiNotifySubscribers(
        StreamID, NotificationTraceType, detail::GSYCLGraphEvent, CmdTraceEvent,
        InstanceID,
        static_cast<const void *>(
            commandToNodeType(Command::CommandType::RUN_CG).c_str()));
  }

  return XptiObjects;
}
#endif

void ExecCGCommand::emitInstrumentationData() {
#ifdef XPTI_ENABLE_INSTRUMENTATION
  constexpr uint16_t NotificationTraceType = xpti::trace_node_create;
  if (!xptiCheckTraceEnabled(MStreamID))
    return;

  std::string KernelName;
  std::optional<bool> FromSource;
  switch (MCommandGroup->getType()) {
  case detail::CGType::Kernel: {
    auto KernelCG =
        reinterpret_cast<detail::CGExecKernel *>(MCommandGroup.get());
    KernelName = instrumentationGetKernelName(
        KernelCG->MSyclKernel, MCommandGroup->MFunctionName,
        KernelCG->getKernelName(), MAddress, FromSource);
  } break;
  default:
    KernelName = getTypeString();
    break;
  }

  xpti_td *CmdTraceEvent = nullptr;
  instrumentationFillCommonData(KernelName, MCommandGroup->MFileName,
                                MCommandGroup->MLine, MCommandGroup->MColumn,
                                MAddress, MQueue, FromSource, MInstanceID,
                                CmdTraceEvent);

  if (CmdTraceEvent) {
    xpti::framework::stash_tuple(XPTI_QUEUE_INSTANCE_ID_KEY,
                                 getQueueID(MQueue));
    MTraceEvent = static_cast<void *>(CmdTraceEvent);
    if (MCommandGroup->getType() == detail::CGType::Kernel) {
      auto KernelCG =
          reinterpret_cast<detail::CGExecKernel *>(MCommandGroup.get());
      instrumentationAddExtraKernelMetadata(
          CmdTraceEvent, KernelCG->MNDRDesc, KernelCG->getKernelBundle(),
          KernelCG->MKernelName, KernelCG->MSyclKernel, MQueue,
          KernelCG->MArgs);
    }

    xptiNotifySubscribers(
        MStreamID, NotificationTraceType, detail::GSYCLGraphEvent,
        CmdTraceEvent, MInstanceID,
        static_cast<const void *>(commandToNodeType(MType).c_str()));
  }
#endif
}

void ExecCGCommand::printDot(std::ostream &Stream) const {
  Stream << "\"" << this << "\" [style=filled, fillcolor=\"#AFFF82\", label=\"";

  Stream << "ID = " << this << "\\n";
  Stream << "EXEC CG ON " << queueDeviceToString(MQueue.get()) << "\\n";

  switch (MCommandGroup->getType()) {
  case detail::CGType::Kernel: {
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
    Stream << "CG type: " << getTypeString() << "\\n";
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

std::string_view ExecCGCommand::getTypeString() const {
  return cgTypeToString(MCommandGroup->getType());
}

// SYCL has a parallel_for_work_group variant where the only NDRange
// characteristics set by a user is the number of work groups. This does not
// map to the OpenCL clEnqueueNDRangeAPI, which requires global work size to
// be set as well. This function determines local work size based on the
// device characteristics and the number of work groups requested by the user,
// then calculates the global work size. SYCL specification (from 4.8.5.3):
// The member function handler::parallel_for_work_group is parameterized by
// the number of work - groups, such that the size of each group is chosen by
// the runtime, or by the number of work - groups and number of work - items
// for users who need more control.
static void adjustNDRangePerKernel(NDRDescT &NDR,
                                   sycl::detail::pi::PiKernel Kernel,
                                   const device_impl &DeviceImpl) {
  if (NDR.GlobalSize[0] != 0)
    return; // GlobalSize is set - no need to adjust
  // check the prerequisites:
  assert(NDR.LocalSize[0] == 0);
  // TODO might be good to cache this info together with the kernel info to
  // avoid get_kernel_work_group_info on every kernel run
  range<3> WGSize = get_kernel_device_specific_info<
      sycl::info::kernel_device_specific::compile_work_group_size>(
      Kernel, DeviceImpl.getHandleRef(), DeviceImpl.getPlugin());

  if (WGSize[0] == 0) {
    WGSize = {1, 1, 1};
  }
  NDR = sycl::detail::NDRDescT{nd_range<3>(NDR.NumWorkGroups * WGSize, WGSize),
                               static_cast<int>(NDR.Dims)};
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
void ReverseRangeDimensionsForKernel(NDRDescT &NDR) {
  if (NDR.Dims > 1) {
    std::swap(NDR.GlobalSize[0], NDR.GlobalSize[NDR.Dims - 1]);
    std::swap(NDR.LocalSize[0], NDR.LocalSize[NDR.Dims - 1]);
    std::swap(NDR.GlobalOffset[0], NDR.GlobalOffset[NDR.Dims - 1]);
  }
}

pi_mem_obj_access AccessModeToPi(access::mode AccessorMode) {
  switch (AccessorMode) {
  case access::mode::read:
    return PI_ACCESS_READ_ONLY;
  case access::mode::write:
  case access::mode::discard_write:
    return PI_ACCESS_WRITE_ONLY;
  default:
    return PI_ACCESS_READ_WRITE;
  }
}

void SetArgBasedOnType(
    const PluginPtr &Plugin, sycl::detail::pi::PiKernel Kernel,
    const std::shared_ptr<device_image_impl> &DeviceImageImpl,
    const std::function<void *(Requirement *Req)> &getMemAllocationFunc,
    const sycl::context &Context, detail::ArgDesc &Arg, size_t NextTrueIndex) {
  switch (Arg.MType) {
  case kernel_param_kind_t::kind_stream:
    break;
  case kernel_param_kind_t::kind_accessor: {
    Requirement *Req = (Requirement *)(Arg.MPtr);

    // getMemAllocationFunc is nullptr when there are no requirements. However,
    // we may pass default constructed accessors to a command, which don't add
    // requirements. In such case, getMemAllocationFunc is nullptr, but it's a
    // valid case, so we need to properly handle it.
    sycl::detail::pi::PiMem MemArg =
        getMemAllocationFunc
            ? (sycl::detail::pi::PiMem)getMemAllocationFunc(Req)
            : nullptr;
    // Only call piKernelSetArg for opencl plugin. Although for now opencl
    // plugin is a thin wrapper for UR plugin, but they still produce different
    // MemArg. For opencl plugin, the MemArg is a straight-forward cl_mem, so it
    // will be fine using piKernelSetArg, which will call urKernelSetArgValue to
    // pass the cl_mem object directly to clSetKernelArg. But when in
    // SYCL_PREFER_UR=1, the MemArg is a cl_mem wrapped by ur_mem_object_t,
    // which will need to unpack by calling piextKernelSetArgMemObj, which calls
    // urKernelSetArgMemObj. If we call piKernelSetArg in such case, the
    // clSetKernelArg will report CL_INVALID_MEM_OBJECT since the arg_value is
    // not a valid cl_mem object but a ur_mem_object_t object.
    if (Context.get_backend() == backend::opencl &&
        !Plugin->hasBackend(backend::all)) {
      // clSetKernelArg (corresponding to piKernelSetArg) returns an error
      // when MemArg is null, which is the case when zero-sized buffers are
      // handled. Below assignment provides later call to clSetKernelArg with
      // acceptable arguments.
      if (!MemArg)
        MemArg = sycl::detail::pi::PiMem();

      Plugin->call<PiApiKind::piKernelSetArg>(
          Kernel, NextTrueIndex, sizeof(sycl::detail::pi::PiMem), &MemArg);
    } else {
      pi_mem_obj_property MemObjData{};
      MemObjData.mem_access = AccessModeToPi(Req->MAccessMode);
      MemObjData.type = PI_KERNEL_ARG_MEM_OBJ_ACCESS;
      Plugin->call<PiApiKind::piextKernelSetArgMemObj>(Kernel, NextTrueIndex,
                                                       &MemObjData, &MemArg);
    }
    break;
  }
  case kernel_param_kind_t::kind_std_layout: {
    Plugin->call<PiApiKind::piKernelSetArg>(Kernel, NextTrueIndex, Arg.MSize,
                                            Arg.MPtr);
    break;
  }
  case kernel_param_kind_t::kind_sampler: {
    sampler *SamplerPtr = (sampler *)Arg.MPtr;
    sycl::detail::pi::PiSampler Sampler =
        detail::getSyclObjImpl(*SamplerPtr)->getOrCreateSampler(Context);
    Plugin->call<PiApiKind::piextKernelSetArgSampler>(Kernel, NextTrueIndex,
                                                      &Sampler);
    break;
  }
  case kernel_param_kind_t::kind_pointer: {
    Plugin->call<PiApiKind::piextKernelSetArgPointer>(Kernel, NextTrueIndex,
                                                      Arg.MSize, Arg.MPtr);
    break;
  }
  case kernel_param_kind_t::kind_specialization_constants_buffer: {
    assert(DeviceImageImpl != nullptr);
    sycl::detail::pi::PiMem SpecConstsBuffer =
        DeviceImageImpl->get_spec_const_buffer_ref();
    // Avoid taking an address of nullptr
    sycl::detail::pi::PiMem *SpecConstsBufferArg =
        SpecConstsBuffer ? &SpecConstsBuffer : nullptr;

    pi_mem_obj_property MemObjData{};
    MemObjData.mem_access = PI_ACCESS_READ_ONLY;
    MemObjData.type = PI_KERNEL_ARG_MEM_OBJ_ACCESS;
    Plugin->call<PiApiKind::piextKernelSetArgMemObj>(
        Kernel, NextTrueIndex, &MemObjData, SpecConstsBufferArg);
    break;
  }
  case kernel_param_kind_t::kind_invalid:
    throw sycl::exception(sycl::make_error_code(sycl::errc::runtime),
                          "Invalid kernel param kind " +
                              codeToString(PI_ERROR_INVALID_VALUE));
    break;
  }
}

static pi_result SetKernelParamsAndLaunch(
    const QueueImplPtr &Queue, std::vector<ArgDesc> &Args,
    const std::shared_ptr<device_image_impl> &DeviceImageImpl,
    sycl::detail::pi::PiKernel Kernel, NDRDescT &NDRDesc,
    std::vector<sycl::detail::pi::PiEvent> &RawEvents,
    const detail::EventImplPtr &OutEventImpl,
    const KernelArgMask *EliminatedArgMask,
    const std::function<void *(Requirement *Req)> &getMemAllocationFunc,
    bool IsCooperative, bool KernelUsesClusterLaunch) {
  assert(Queue && "Kernel submissions should have an associated queue");
  const PluginPtr &Plugin = Queue->getPlugin();

  auto setFunc = [&Plugin, Kernel, &DeviceImageImpl, &getMemAllocationFunc,
                  &Queue](detail::ArgDesc &Arg, size_t NextTrueIndex) {
    SetArgBasedOnType(Plugin, Kernel, DeviceImageImpl, getMemAllocationFunc,
                      Queue->get_context(), Arg, NextTrueIndex);
  };

  applyFuncOnFilteredArgs(EliminatedArgMask, Args, setFunc);

  adjustNDRangePerKernel(NDRDesc, Kernel, *(Queue->getDeviceImplPtr()));

  // Remember this information before the range dimensions are reversed
  const bool HasLocalSize = (NDRDesc.LocalSize[0] != 0);

  ReverseRangeDimensionsForKernel(NDRDesc);

  size_t RequiredWGSize[3] = {0, 0, 0};
  size_t *LocalSize = nullptr;

  if (HasLocalSize)
    LocalSize = &NDRDesc.LocalSize[0];
  else {
    Plugin->call<PiApiKind::piKernelGetGroupInfo>(
        Kernel, Queue->getDeviceImplPtr()->getHandleRef(),
        PI_KERNEL_GROUP_INFO_COMPILE_WORK_GROUP_SIZE, sizeof(RequiredWGSize),
        RequiredWGSize, /* param_value_size_ret = */ nullptr);

    const bool EnforcedLocalSize =
        (RequiredWGSize[0] != 0 || RequiredWGSize[1] != 0 ||
         RequiredWGSize[2] != 0);
    if (EnforcedLocalSize)
      LocalSize = RequiredWGSize;
  }
  if (OutEventImpl != nullptr)
    OutEventImpl->setHostEnqueueTime();
  if (KernelUsesClusterLaunch) {
    std::vector<pi_launch_property> property_list;

    pi_launch_property_value launch_property_value_cluster_range;
    launch_property_value_cluster_range.cluster_dims[0] =
        NDRDesc.ClusterDimensions[0];
    launch_property_value_cluster_range.cluster_dims[1] =
        NDRDesc.ClusterDimensions[1];
    launch_property_value_cluster_range.cluster_dims[2] =
        NDRDesc.ClusterDimensions[2];

    property_list.push_back(
        {pi_launch_property_id::PI_LAUNCH_PROPERTY_CLUSTER_DIMENSION,
         launch_property_value_cluster_range});

    if (IsCooperative) {
      pi_launch_property_value launch_property_value_cooperative;
      launch_property_value_cooperative.cooperative = 1;
      property_list.push_back(
          {pi_launch_property_id::PI_LAUNCH_PROPERTY_COOPERATIVE,
           launch_property_value_cooperative});
    }

    return Plugin->call_nocheck<PiApiKind::piextEnqueueKernelLaunchCustom>(
        Queue->getHandleRef(), Kernel, NDRDesc.Dims, &NDRDesc.GlobalSize[0],
        LocalSize, property_list.size(), property_list.data(), RawEvents.size(),
        RawEvents.empty() ? nullptr : &RawEvents[0],
        OutEventImpl ? &OutEventImpl->getHandleRef() : nullptr);
  }
  pi_result Error =
      [&](auto... Args) {
        if (IsCooperative) {
          return Plugin
              ->call_nocheck<PiApiKind::piextEnqueueCooperativeKernelLaunch>(
                  Args...);
        }
        return Plugin->call_nocheck<PiApiKind::piEnqueueKernelLaunch>(Args...);
      }(Queue->getHandleRef(), Kernel, NDRDesc.Dims, &NDRDesc.GlobalOffset[0],
        &NDRDesc.GlobalSize[0], LocalSize, RawEvents.size(),
        RawEvents.empty() ? nullptr : &RawEvents[0],
        OutEventImpl ? &OutEventImpl->getHandleRef() : nullptr);
  return Error;
}

pi_int32 enqueueImpCommandBufferKernel(
    context Ctx, DeviceImplPtr DeviceImpl,
    sycl::detail::pi::PiExtCommandBuffer CommandBuffer,
    const CGExecKernel &CommandGroup,
    std::vector<sycl::detail::pi::PiExtSyncPoint> &SyncPoints,
    sycl::detail::pi::PiExtSyncPoint *OutSyncPoint,
    sycl::detail::pi::PiExtCommandBufferCommand *OutCommand,
    const std::function<void *(Requirement *Req)> &getMemAllocationFunc) {
  auto ContextImpl = sycl::detail::getSyclObjImpl(Ctx);
  const sycl::detail::PluginPtr &Plugin = ContextImpl->getPlugin();
  pi_kernel PiKernel = nullptr;
  pi_program PiProgram = nullptr;
  std::shared_ptr<kernel_impl> SyclKernelImpl = nullptr;
  std::shared_ptr<device_image_impl> DeviceImageImpl = nullptr;

  auto Kernel = CommandGroup.MSyclKernel;
  auto KernelBundleImplPtr = CommandGroup.MKernelBundle;
  const KernelArgMask *EliminatedArgMask = nullptr;

  // Use kernel_bundle if available unless it is interop.
  // Interop bundles can't be used in the first branch, because the kernels
  // in interop kernel bundles (if any) do not have kernel_id
  // and can therefore not be looked up, but since they are self-contained
  // they can simply be launched directly.
  if (KernelBundleImplPtr && !KernelBundleImplPtr->isInterop()) {
    auto KernelName = CommandGroup.MKernelName;
    kernel_id KernelID =
        detail::ProgramManager::getInstance().getSYCLKernelID(KernelName);
    kernel SyclKernel =
        KernelBundleImplPtr->get_kernel(KernelID, KernelBundleImplPtr);
    SyclKernelImpl = detail::getSyclObjImpl(SyclKernel);
    PiKernel = SyclKernelImpl->getHandleRef();
    DeviceImageImpl = SyclKernelImpl->getDeviceImage();
    PiProgram = DeviceImageImpl->get_program_ref();
    EliminatedArgMask = SyclKernelImpl->getKernelArgMask();
  } else if (Kernel != nullptr) {
    PiKernel = Kernel->getHandleRef();
    PiProgram = Kernel->getProgramRef();
    EliminatedArgMask = Kernel->getKernelArgMask();
  } else {
    std::tie(PiKernel, std::ignore, EliminatedArgMask, PiProgram) =
        sycl::detail::ProgramManager::getInstance().getOrCreateKernel(
            ContextImpl, DeviceImpl, CommandGroup.MKernelName);
  }

  auto SetFunc = [&Plugin, &PiKernel, &DeviceImageImpl, &Ctx,
                  &getMemAllocationFunc](sycl::detail::ArgDesc &Arg,
                                         size_t NextTrueIndex) {
    sycl::detail::SetArgBasedOnType(Plugin, PiKernel, DeviceImageImpl,
                                    getMemAllocationFunc, Ctx, Arg,
                                    NextTrueIndex);
  };
  // Copy args for modification
  auto Args = CommandGroup.MArgs;
  sycl::detail::applyFuncOnFilteredArgs(EliminatedArgMask, Args, SetFunc);

  // Remember this information before the range dimensions are reversed
  const bool HasLocalSize = (CommandGroup.MNDRDesc.LocalSize[0] != 0);

  // Copy NDRDesc for modification
  auto NDRDesc = CommandGroup.MNDRDesc;
  // Reverse kernel dims
  sycl::detail::ReverseRangeDimensionsForKernel(NDRDesc);

  size_t RequiredWGSize[3] = {0, 0, 0};
  size_t *LocalSize = nullptr;

  if (HasLocalSize)
    LocalSize = &NDRDesc.LocalSize[0];
  else {
    Plugin->call<sycl::detail::PiApiKind::piKernelGetGroupInfo>(
        PiKernel, DeviceImpl->getHandleRef(),
        PI_KERNEL_GROUP_INFO_COMPILE_WORK_GROUP_SIZE, sizeof(RequiredWGSize),
        RequiredWGSize,
        /* param_value_size_ret = */ nullptr);

    const bool EnforcedLocalSize =
        (RequiredWGSize[0] != 0 || RequiredWGSize[1] != 0 ||
         RequiredWGSize[2] != 0);
    if (EnforcedLocalSize)
      LocalSize = RequiredWGSize;
  }

  pi_result Res = Plugin->call_nocheck<
      sycl::detail::PiApiKind::piextCommandBufferNDRangeKernel>(
      CommandBuffer, PiKernel, NDRDesc.Dims, &NDRDesc.GlobalOffset[0],
      &NDRDesc.GlobalSize[0], LocalSize, SyncPoints.size(),
      SyncPoints.size() ? SyncPoints.data() : nullptr, OutSyncPoint,
      OutCommand);

  if (!SyclKernelImpl && !Kernel) {
    Plugin->call<PiApiKind::piKernelRelease>(PiKernel);
    Plugin->call<PiApiKind::piProgramRelease>(PiProgram);
  }

  if (Res != pi_result::PI_SUCCESS) {
    const device_impl &DeviceImplem = *(DeviceImpl);
    detail::enqueue_kernel_launch::handleErrorOrWarning(Res, DeviceImplem,
                                                        PiKernel, NDRDesc);
  }

  return Res;
}

void enqueueImpKernel(
    const QueueImplPtr &Queue, NDRDescT &NDRDesc, std::vector<ArgDesc> &Args,
    const std::shared_ptr<detail::kernel_bundle_impl> &KernelBundleImplPtr,
    const std::shared_ptr<detail::kernel_impl> &MSyclKernel,
    const std::string &KernelName,
    std::vector<sycl::detail::pi::PiEvent> &RawEvents,
    const detail::EventImplPtr &OutEventImpl,
    const std::function<void *(Requirement *Req)> &getMemAllocationFunc,
    sycl::detail::pi::PiKernelCacheConfig KernelCacheConfig,
    const bool KernelIsCooperative, const bool KernelUsesClusterLaunch) {
  assert(Queue && "Kernel submissions should have an associated queue");
  // Run OpenCL kernel
  auto ContextImpl = Queue->getContextImplPtr();
  auto DeviceImpl = Queue->getDeviceImplPtr();
  sycl::detail::pi::PiKernel Kernel = nullptr;
  std::mutex *KernelMutex = nullptr;
  sycl::detail::pi::PiProgram Program = nullptr;
  const KernelArgMask *EliminatedArgMask;

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

    EliminatedArgMask = SyclKernelImpl->getKernelArgMask();
    KernelMutex = SyclKernelImpl->getCacheMutex();
  } else if (nullptr != MSyclKernel) {
    assert(MSyclKernel->get_info<info::kernel::context>() ==
           Queue->get_context());
    Kernel = MSyclKernel->getHandleRef();
    Program = MSyclKernel->getProgramRef();

    // Non-cacheable kernels use mutexes from kernel_impls.
    // TODO this can still result in a race condition if multiple SYCL
    // kernels are created with the same native handle. To address this,
    // we need to either store and use a pi_native_handle -> mutex map or
    // reuse and return existing SYCL kernels from make_native to avoid
    // their duplication in such cases.
    KernelMutex = &MSyclKernel->getNoncacheableEnqueueMutex();
    EliminatedArgMask = MSyclKernel->getKernelArgMask();
  } else {
    std::tie(Kernel, KernelMutex, EliminatedArgMask, Program) =
        detail::ProgramManager::getInstance().getOrCreateKernel(
            ContextImpl, DeviceImpl, KernelName, NDRDesc);
  }

  // We may need more events for the launch, so we make another reference.
  std::vector<sycl::detail::pi::PiEvent> &EventsWaitList = RawEvents;

  // Initialize device globals associated with this.
  std::vector<sycl::detail::pi::PiEvent> DeviceGlobalInitEvents =
      ContextImpl->initializeDeviceGlobals(Program, Queue);
  std::vector<sycl::detail::pi::PiEvent> EventsWithDeviceGlobalInits;
  if (!DeviceGlobalInitEvents.empty()) {
    EventsWithDeviceGlobalInits.reserve(RawEvents.size() +
                                        DeviceGlobalInitEvents.size());
    EventsWithDeviceGlobalInits.insert(EventsWithDeviceGlobalInits.end(),
                                       RawEvents.begin(), RawEvents.end());
    EventsWithDeviceGlobalInits.insert(EventsWithDeviceGlobalInits.end(),
                                       DeviceGlobalInitEvents.begin(),
                                       DeviceGlobalInitEvents.end());
    EventsWaitList = EventsWithDeviceGlobalInits;
  }

  pi_result Error = PI_SUCCESS;
  {
    // When KernelMutex is null, this means that in-memory caching is
    // disabled, which means that kernel object is not shared, so no locking
    // is necessary.
    using LockT = std::unique_lock<std::mutex>;
    auto Lock = KernelMutex ? LockT(*KernelMutex) : LockT();

    // Set SLM/Cache configuration for the kernel if non-default value is
    // provided.
    if (KernelCacheConfig == PI_EXT_KERNEL_EXEC_INFO_CACHE_LARGE_SLM ||
        KernelCacheConfig == PI_EXT_KERNEL_EXEC_INFO_CACHE_LARGE_DATA) {
      const PluginPtr &Plugin = Queue->getPlugin();
      Plugin->call<PiApiKind::piKernelSetExecInfo>(
          Kernel, PI_EXT_KERNEL_EXEC_INFO_CACHE_CONFIG,
          sizeof(sycl::detail::pi::PiKernelCacheConfig), &KernelCacheConfig);
    }

    Error = SetKernelParamsAndLaunch(
        Queue, Args, DeviceImageImpl, Kernel, NDRDesc, EventsWaitList,
        OutEventImpl, EliminatedArgMask, getMemAllocationFunc,
        KernelIsCooperative, KernelUsesClusterLaunch);

    const PluginPtr &Plugin = Queue->getPlugin();
    if (!SyclKernelImpl && !MSyclKernel) {
      Plugin->call<PiApiKind::piKernelRelease>(Kernel);
      Plugin->call<PiApiKind::piProgramRelease>(Program);
    }
  }
  if (PI_SUCCESS != Error) {
    // If we have got non-success error code, let's analyze it to emit nice
    // exception explaining what was wrong
    const device_impl &DeviceImpl = *(Queue->getDeviceImplPtr());
    detail::enqueue_kernel_launch::handleErrorOrWarning(Error, DeviceImpl,
                                                        Kernel, NDRDesc);
  }
}

pi_int32
enqueueReadWriteHostPipe(const QueueImplPtr &Queue, const std::string &PipeName,
                         bool blocking, void *ptr, size_t size,
                         std::vector<sycl::detail::pi::PiEvent> &RawEvents,
                         const detail::EventImplPtr &OutEventImpl, bool read) {
  assert(Queue &&
         "ReadWrite host pipe submissions should have an associated queue");
  detail::HostPipeMapEntry *hostPipeEntry =
      ProgramManager::getInstance().getHostPipeEntry(PipeName);

  sycl::detail::pi::PiProgram Program = nullptr;
  device Device = Queue->get_device();
  ContextImplPtr ContextImpl = Queue->getContextImplPtr();
  std::optional<sycl::detail::pi::PiProgram> CachedProgram =
      ContextImpl->getProgramForHostPipe(Device, hostPipeEntry);
  if (CachedProgram)
    Program = *CachedProgram;
  else {
    // If there was no cached program, build one.
    device_image_plain devImgPlain =
        ProgramManager::getInstance().getDeviceImageFromBinaryImage(
            hostPipeEntry->getDevBinImage(), Queue->get_context(),
            Queue->get_device());
    device_image_plain BuiltImage =
        ProgramManager::getInstance().build(devImgPlain, {Device}, {});
    Program = getSyclObjImpl(BuiltImage)->get_program_ref();
  }
  assert(Program && "Program for this hostpipe is not compiled.");

  // Get plugin for calling opencl functions
  const PluginPtr &Plugin = Queue->getPlugin();

  pi_queue pi_q = Queue->getHandleRef();
  pi_result Error;

  auto OutEvent = OutEventImpl ? &OutEventImpl->getHandleRef() : nullptr;
  if (OutEventImpl != nullptr)
    OutEventImpl->setHostEnqueueTime();
  if (read) {
    Error =
        Plugin->call_nocheck<sycl::detail::PiApiKind::piextEnqueueReadHostPipe>(
            pi_q, Program, PipeName.c_str(), blocking, ptr, size,
            RawEvents.size(), RawEvents.empty() ? nullptr : &RawEvents[0],
            OutEvent);
  } else {
    Error =
        Plugin
            ->call_nocheck<sycl::detail::PiApiKind::piextEnqueueWriteHostPipe>(
                pi_q, Program, PipeName.c_str(), blocking, ptr, size,
                RawEvents.size(), RawEvents.empty() ? nullptr : &RawEvents[0],
                OutEvent);
  }
  return Error;
}

pi_int32 ExecCGCommand::enqueueImpCommandBuffer() {
  assert(MQueue && "Command buffer enqueue should have an associated queue");
  // Wait on host command dependencies
  waitForPreparedHostEvents();

  // Any device dependencies need to be waited on here since subsequent
  // submissions of the command buffer itself will not receive dependencies on
  // them, e.g. initial copies from host to device
  std::vector<EventImplPtr> EventImpls = MPreparedDepsEvents;
  flushCrossQueueDeps(EventImpls, MWorkerQueue);
  std::vector<sycl::detail::pi::PiEvent> RawEvents = getPiEvents(EventImpls);
  if (!RawEvents.empty()) {
    const PluginPtr &Plugin = MQueue->getPlugin();
    Plugin->call<PiApiKind::piEventsWait>(RawEvents.size(), &RawEvents[0]);
  }

  // We can omit creating a PI event and create a "discarded" event if either
  // the queue has the discard property or the command has been explicitly
  // marked as not needing an event, e.g. if the user did not ask for one, and
  // if the queue supports discarded PI event and there are no requirements.
  bool DiscardPiEvent = (MQueue->MDiscardEvents || !MEventNeeded) &&
                        MQueue->supportsDiscardingPiEvents() &&
                        MCommandGroup->getRequirements().size() == 0;
  sycl::detail::pi::PiEvent *Event =
      DiscardPiEvent ? nullptr : &MEvent->getHandleRef();
  sycl::detail::pi::PiExtSyncPoint OutSyncPoint;
  sycl::detail::pi::PiExtCommandBufferCommand OutCommand = nullptr;
  switch (MCommandGroup->getType()) {
  case CGType::Kernel: {
    CGExecKernel *ExecKernel = (CGExecKernel *)MCommandGroup.get();

    auto getMemAllocationFunc = [this](Requirement *Req) {
      AllocaCommandBase *AllocaCmd = getAllocaForReq(Req);
      return AllocaCmd->getMemAllocation();
    };

    if (!Event) {
      // Kernel only uses assert if it's non interop one
      bool KernelUsesAssert =
          !(ExecKernel->MSyclKernel && ExecKernel->MSyclKernel->isInterop()) &&
          ProgramManager::getInstance().kernelUsesAssert(
              ExecKernel->MKernelName);
      if (KernelUsesAssert) {
        Event = &MEvent->getHandleRef();
      }
    }
    auto result = enqueueImpCommandBufferKernel(
        MQueue->get_context(), MQueue->getDeviceImplPtr(), MCommandBuffer,
        *ExecKernel, MSyncPointDeps, &OutSyncPoint, &OutCommand,
        getMemAllocationFunc);
    MEvent->setSyncPoint(OutSyncPoint);
    MEvent->setCommandBufferCommand(OutCommand);
    return result;
  }
  case CGType::CopyUSM: {
    CGCopyUSM *Copy = (CGCopyUSM *)MCommandGroup.get();
    MemoryManager::ext_oneapi_copy_usm_cmd_buffer(
        MQueue->getContextImplPtr(), Copy->getSrc(), MCommandBuffer,
        Copy->getLength(), Copy->getDst(), MSyncPointDeps, &OutSyncPoint);
    MEvent->setSyncPoint(OutSyncPoint);
    return PI_SUCCESS;
  }
  case CGType::CopyAccToAcc: {
    CGCopy *Copy = (CGCopy *)MCommandGroup.get();
    Requirement *ReqSrc = (Requirement *)(Copy->getSrc());
    Requirement *ReqDst = (Requirement *)(Copy->getDst());

    AllocaCommandBase *AllocaCmdSrc = getAllocaForReq(ReqSrc);
    AllocaCommandBase *AllocaCmdDst = getAllocaForReq(ReqDst);

    MemoryManager::ext_oneapi_copyD2D_cmd_buffer(
        MQueue->getContextImplPtr(), MCommandBuffer,
        AllocaCmdSrc->getSYCLMemObj(), AllocaCmdSrc->getMemAllocation(),
        ReqSrc->MDims, ReqSrc->MMemoryRange, ReqSrc->MAccessRange,
        ReqSrc->MOffset, ReqSrc->MElemSize, AllocaCmdDst->getMemAllocation(),
        ReqDst->MDims, ReqDst->MMemoryRange, ReqDst->MAccessRange,
        ReqDst->MOffset, ReqDst->MElemSize, std::move(MSyncPointDeps),
        &OutSyncPoint);
    MEvent->setSyncPoint(OutSyncPoint);
    return PI_SUCCESS;
  }
  case CGType::CopyAccToPtr: {
    CGCopy *Copy = (CGCopy *)MCommandGroup.get();
    Requirement *Req = (Requirement *)Copy->getSrc();
    AllocaCommandBase *AllocaCmd = getAllocaForReq(Req);

    MemoryManager::ext_oneapi_copyD2H_cmd_buffer(
        MQueue->getContextImplPtr(), MCommandBuffer, AllocaCmd->getSYCLMemObj(),
        AllocaCmd->getMemAllocation(), Req->MDims, Req->MMemoryRange,
        Req->MAccessRange, Req->MOffset, Req->MElemSize, (char *)Copy->getDst(),
        Req->MDims, Req->MAccessRange,
        /*DstOffset=*/{0, 0, 0}, Req->MElemSize, std::move(MSyncPointDeps),
        &OutSyncPoint);
    MEvent->setSyncPoint(OutSyncPoint);
    return PI_SUCCESS;
  }
  case CGType::CopyPtrToAcc: {
    CGCopy *Copy = (CGCopy *)MCommandGroup.get();
    Requirement *Req = (Requirement *)(Copy->getDst());
    AllocaCommandBase *AllocaCmd = getAllocaForReq(Req);

    MemoryManager::ext_oneapi_copyH2D_cmd_buffer(
        MQueue->getContextImplPtr(), MCommandBuffer, AllocaCmd->getSYCLMemObj(),
        (char *)Copy->getSrc(), Req->MDims, Req->MAccessRange,
        /*SrcOffset*/ {0, 0, 0}, Req->MElemSize, AllocaCmd->getMemAllocation(),
        Req->MDims, Req->MMemoryRange, Req->MAccessRange, Req->MOffset,
        Req->MElemSize, std::move(MSyncPointDeps), &OutSyncPoint);
    MEvent->setSyncPoint(OutSyncPoint);
    return PI_SUCCESS;
  }
  case CGType::Fill: {
    CGFill *Fill = (CGFill *)MCommandGroup.get();
    Requirement *Req = (Requirement *)(Fill->getReqToFill());
    AllocaCommandBase *AllocaCmd = getAllocaForReq(Req);

    MemoryManager::ext_oneapi_fill_cmd_buffer(
        MQueue->getContextImplPtr(), MCommandBuffer, AllocaCmd->getSYCLMemObj(),
        AllocaCmd->getMemAllocation(), Fill->MPattern.size(),
        Fill->MPattern.data(), Req->MDims, Req->MMemoryRange, Req->MAccessRange,
        Req->MOffset, Req->MElemSize, std::move(MSyncPointDeps), &OutSyncPoint);
    MEvent->setSyncPoint(OutSyncPoint);
    return PI_SUCCESS;
  }
  case CGType::FillUSM: {
    CGFillUSM *Fill = (CGFillUSM *)MCommandGroup.get();
    MemoryManager::ext_oneapi_fill_usm_cmd_buffer(
        MQueue->getContextImplPtr(), MCommandBuffer, Fill->getDst(),
        Fill->getLength(), Fill->getPattern(), std::move(MSyncPointDeps),
        &OutSyncPoint);
    MEvent->setSyncPoint(OutSyncPoint);
    return PI_SUCCESS;
  }
  case CGType::PrefetchUSM: {
    CGPrefetchUSM *Prefetch = (CGPrefetchUSM *)MCommandGroup.get();
    MemoryManager::ext_oneapi_prefetch_usm_cmd_buffer(
        MQueue->getContextImplPtr(), MCommandBuffer, Prefetch->getDst(),
        Prefetch->getLength(), std::move(MSyncPointDeps), &OutSyncPoint);
    MEvent->setSyncPoint(OutSyncPoint);
    return PI_SUCCESS;
  }
  case CGType::AdviseUSM: {
    CGAdviseUSM *Advise = (CGAdviseUSM *)MCommandGroup.get();
    MemoryManager::ext_oneapi_advise_usm_cmd_buffer(
        MQueue->getContextImplPtr(), MCommandBuffer, Advise->getDst(),
        Advise->getLength(), Advise->getAdvice(), std::move(MSyncPointDeps),
        &OutSyncPoint);
    MEvent->setSyncPoint(OutSyncPoint);
    return PI_SUCCESS;
  }

  default:
    throw exception(make_error_code(errc::runtime),
                    "CG type not implemented for command buffers.");
  }
}

pi_int32 ExecCGCommand::enqueueImp() {
  if (MCommandBuffer) {
    return enqueueImpCommandBuffer();
  } else {
    return enqueueImpQueue();
  }
}

pi_int32 ExecCGCommand::enqueueImpQueue() {
  if (getCG().getType() != CGType::CodeplayHostTask)
    waitForPreparedHostEvents();
  std::vector<EventImplPtr> EventImpls = MPreparedDepsEvents;
  auto RawEvents = getPiEvents(EventImpls);
  flushCrossQueueDeps(EventImpls, MWorkerQueue);

  // We can omit creating a PI event and create a "discarded" event if either
  // the queue has the discard property or the command has been explicitly
  // marked as not needing an event, e.g. if the user did not ask for one, and
  // if the queue supports discarded PI event and there are no requirements.
  bool DiscardPiEvent = MQueue && (MQueue->MDiscardEvents || !MEventNeeded) &&
                        MQueue->supportsDiscardingPiEvents() &&
                        MCommandGroup->getRequirements().size() == 0;
  sycl::detail::pi::PiEvent *Event =
      DiscardPiEvent ? nullptr : &MEvent->getHandleRef();
  detail::EventImplPtr EventImpl = DiscardPiEvent ? nullptr : MEvent;

  switch (MCommandGroup->getType()) {

  case CGType::UpdateHost: {
    throw sycl::exception(sycl::make_error_code(sycl::errc::runtime),
                          "Update host should be handled by the Scheduler. " +
                              codeToString(PI_ERROR_INVALID_VALUE));
  }
  case CGType::CopyAccToPtr: {
    CGCopy *Copy = (CGCopy *)MCommandGroup.get();
    Requirement *Req = (Requirement *)Copy->getSrc();
    AllocaCommandBase *AllocaCmd = getAllocaForReq(Req);

    MemoryManager::copy(
        AllocaCmd->getSYCLMemObj(), AllocaCmd->getMemAllocation(), MQueue,
        Req->MDims, Req->MMemoryRange, Req->MAccessRange, Req->MOffset,
        Req->MElemSize, Copy->getDst(), nullptr, Req->MDims, Req->MAccessRange,
        Req->MAccessRange, /*DstOffset=*/{0, 0, 0}, Req->MElemSize,
        std::move(RawEvents), MEvent->getHandleRef(), MEvent);

    return PI_SUCCESS;
  }
  case CGType::CopyPtrToAcc: {
    CGCopy *Copy = (CGCopy *)MCommandGroup.get();
    Requirement *Req = (Requirement *)(Copy->getDst());
    AllocaCommandBase *AllocaCmd = getAllocaForReq(Req);

    MemoryManager::copy(
        AllocaCmd->getSYCLMemObj(), Copy->getSrc(), nullptr, Req->MDims,
        Req->MAccessRange, Req->MAccessRange,
        /*SrcOffset*/ {0, 0, 0}, Req->MElemSize, AllocaCmd->getMemAllocation(),
        MQueue, Req->MDims, Req->MMemoryRange, Req->MAccessRange, Req->MOffset,
        Req->MElemSize, std::move(RawEvents), MEvent->getHandleRef(), MEvent);

    return PI_SUCCESS;
  }
  case CGType::CopyAccToAcc: {
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
        MEvent->getHandleRef(), MEvent);

    return PI_SUCCESS;
  }
  case CGType::Fill: {
    CGFill *Fill = (CGFill *)MCommandGroup.get();
    Requirement *Req = (Requirement *)(Fill->getReqToFill());
    AllocaCommandBase *AllocaCmd = getAllocaForReq(Req);

    MemoryManager::fill(
        AllocaCmd->getSYCLMemObj(), AllocaCmd->getMemAllocation(), MQueue,
        Fill->MPattern.size(), Fill->MPattern.data(), Req->MDims,
        Req->MMemoryRange, Req->MAccessRange, Req->MOffset, Req->MElemSize,
        std::move(RawEvents), MEvent->getHandleRef(), MEvent);

    return PI_SUCCESS;
  }
  case CGType::Kernel: {
    assert(MQueue && "Kernel submissions should have an associated queue");
    CGExecKernel *ExecKernel = (CGExecKernel *)MCommandGroup.get();

    NDRDescT &NDRDesc = ExecKernel->MNDRDesc;
    std::vector<ArgDesc> &Args = ExecKernel->MArgs;

    auto getMemAllocationFunc = [this](Requirement *Req) {
      AllocaCommandBase *AllocaCmd = getAllocaForReq(Req);
      // getAllocaForReq may return nullptr if Req is a default constructed
      // accessor. Simply return nullptr in such a case.
      return AllocaCmd ? AllocaCmd->getMemAllocation() : nullptr;
    };

    const std::shared_ptr<detail::kernel_impl> &SyclKernel =
        ExecKernel->MSyclKernel;
    const std::string &KernelName = ExecKernel->MKernelName;

    if (!EventImpl) {
      // Kernel only uses assert if it's non interop one
      bool KernelUsesAssert =
          !(SyclKernel && SyclKernel->isInterop()) &&
          ProgramManager::getInstance().kernelUsesAssert(KernelName);
      if (KernelUsesAssert) {
        EventImpl = MEvent;
      }
    }

    enqueueImpKernel(MQueue, NDRDesc, Args, ExecKernel->getKernelBundle(),
                     SyclKernel, KernelName, RawEvents, EventImpl,
                     getMemAllocationFunc, ExecKernel->MKernelCacheConfig,
                     ExecKernel->MKernelIsCooperative,
                     ExecKernel->MKernelUsesClusterLaunch);

    return PI_SUCCESS;
  }
  case CGType::CopyUSM: {
    CGCopyUSM *Copy = (CGCopyUSM *)MCommandGroup.get();
    MemoryManager::copy_usm(Copy->getSrc(), MQueue, Copy->getLength(),
                            Copy->getDst(), std::move(RawEvents), Event,
                            MEvent);

    return PI_SUCCESS;
  }
  case CGType::FillUSM: {
    CGFillUSM *Fill = (CGFillUSM *)MCommandGroup.get();
    MemoryManager::fill_usm(Fill->getDst(), MQueue, Fill->getLength(),
                            Fill->getPattern(), std::move(RawEvents), Event,
                            MEvent);

    return PI_SUCCESS;
  }
  case CGType::PrefetchUSM: {
    CGPrefetchUSM *Prefetch = (CGPrefetchUSM *)MCommandGroup.get();
    MemoryManager::prefetch_usm(Prefetch->getDst(), MQueue,
                                Prefetch->getLength(), std::move(RawEvents),
                                Event, MEvent);

    return PI_SUCCESS;
  }
  case CGType::AdviseUSM: {
    CGAdviseUSM *Advise = (CGAdviseUSM *)MCommandGroup.get();
    MemoryManager::advise_usm(Advise->getDst(), MQueue, Advise->getLength(),
                              Advise->getAdvice(), std::move(RawEvents), Event,
                              MEvent);

    return PI_SUCCESS;
  }
  case CGType::Copy2DUSM: {
    CGCopy2DUSM *Copy = (CGCopy2DUSM *)MCommandGroup.get();
    MemoryManager::copy_2d_usm(Copy->getSrc(), Copy->getSrcPitch(), MQueue,
                               Copy->getDst(), Copy->getDstPitch(),
                               Copy->getWidth(), Copy->getHeight(),
                               std::move(RawEvents), Event, MEvent);
    return PI_SUCCESS;
  }
  case CGType::Fill2DUSM: {
    CGFill2DUSM *Fill = (CGFill2DUSM *)MCommandGroup.get();
    MemoryManager::fill_2d_usm(Fill->getDst(), MQueue, Fill->getPitch(),
                               Fill->getWidth(), Fill->getHeight(),
                               Fill->getPattern(), std::move(RawEvents), Event,
                               MEvent);
    return PI_SUCCESS;
  }
  case CGType::Memset2DUSM: {
    CGMemset2DUSM *Memset = (CGMemset2DUSM *)MCommandGroup.get();
    MemoryManager::memset_2d_usm(Memset->getDst(), MQueue, Memset->getPitch(),
                                 Memset->getWidth(), Memset->getHeight(),
                                 Memset->getValue(), std::move(RawEvents),
                                 Event, MEvent);
    return PI_SUCCESS;
  }
  case CGType::CodeplayHostTask: {
    CGHostTask *HostTask = static_cast<CGHostTask *>(MCommandGroup.get());

    for (ArgDesc &Arg : HostTask->MArgs) {
      switch (Arg.MType) {
      case kernel_param_kind_t::kind_accessor: {
        Requirement *Req = static_cast<Requirement *>(Arg.MPtr);
        AllocaCommandBase *AllocaCmd = getAllocaForReq(Req);

        if (AllocaCmd)
          Req->MData = AllocaCmd->getMemAllocation();
        break;
      }
      default:
        throw sycl::exception(sycl::make_error_code(sycl::errc::runtime),
                              "Unsupported arg type " +
                                  codeToString(PI_ERROR_INVALID_VALUE));
      }
    }

    std::vector<interop_handle::ReqToMem> ReqToMem;
    std::vector<pi_mem> ReqPiMem;

    if (HostTask->MHostTask->isInteropTask()) {
      // Extract the Mem Objects for all Requirements, to ensure they are
      // available if a user asks for them inside the interop task scope
      const std::vector<Requirement *> &HandlerReq =
          HostTask->getRequirements();
      auto ReqToMemConv = [&ReqToMem, &ReqPiMem, HostTask](Requirement *Req) {
        const std::vector<AllocaCommandBase *> &AllocaCmds =
            Req->MSYCLMemObj->MRecord->MAllocaCommands;

        for (AllocaCommandBase *AllocaCmd : AllocaCmds)
          if (getContext(HostTask->MQueue) ==
              getContext(AllocaCmd->getQueue())) {
            auto MemArg =
                reinterpret_cast<pi_mem>(AllocaCmd->getMemAllocation());
            ReqToMem.emplace_back(std::make_pair(Req, MemArg));
            ReqPiMem.emplace_back(MemArg);

            return;
          }

        assert(false &&
               "Can't get memory object due to no allocation available");

        throw sycl::exception(
            sycl::make_error_code(sycl::errc::runtime),
            "Can't get memory object due to no allocation available " +
                codeToString(PI_ERROR_INVALID_MEM_OBJECT));
      };
      std::for_each(std::begin(HandlerReq), std::end(HandlerReq), ReqToMemConv);
      std::sort(std::begin(ReqToMem), std::end(ReqToMem));
    }

    // Host task is executed asynchronously so we should record where it was
    // submitted to report exception origin properly.
    copySubmissionCodeLocation();

    queue_impl::getThreadPool().submit<DispatchHostTask>(
        DispatchHostTask(this, std::move(ReqToMem), std::move(ReqPiMem)));

    MShouldCompleteEventIfPossible = false;

    return PI_SUCCESS;
  }
  case CGType::Barrier: {
    assert(MQueue && "Barrier submission should have an associated queue");
    const PluginPtr &Plugin = MQueue->getPlugin();
    if (MEvent != nullptr)
      MEvent->setHostEnqueueTime();
    Plugin->call<PiApiKind::piEnqueueEventsWaitWithBarrier>(
        MQueue->getHandleRef(), 0, nullptr, Event);

    return PI_SUCCESS;
  }
  case CGType::BarrierWaitlist: {
    assert(MQueue && "Barrier submission should have an associated queue");
    CGBarrier *Barrier = static_cast<CGBarrier *>(MCommandGroup.get());
    std::vector<detail::EventImplPtr> Events = Barrier->MEventsWaitWithBarrier;
    std::vector<sycl::detail::pi::PiEvent> PiEvents =
        getPiEventsBlocking(Events);
    if (PiEvents.empty()) {
      // If Events is empty, then the barrier has no effect.
      return PI_SUCCESS;
    }
    const PluginPtr &Plugin = MQueue->getPlugin();
    if (MEvent != nullptr)
      MEvent->setHostEnqueueTime();
    Plugin->call<PiApiKind::piEnqueueEventsWaitWithBarrier>(
        MQueue->getHandleRef(), PiEvents.size(), &PiEvents[0], Event);

    return PI_SUCCESS;
  }
  case CGType::ProfilingTag: {
    const PluginPtr &Plugin = MQueue->getPlugin();
    // If the queue is not in-order, we need to insert a barrier. This barrier
    // does not need output events as it will implicitly enforce the following
    // enqueue is blocked until it finishes.
    if (!MQueue->isInOrder())
      Plugin->call<PiApiKind::piEnqueueEventsWaitWithBarrier>(
          MQueue->getHandleRef(), /*num_events_in_wait_list=*/0,
          /*event_wait_list=*/nullptr, /*event=*/nullptr);

    Plugin->call<PiApiKind::piEnqueueTimestampRecordingExp>(
        MQueue->getHandleRef(), /*blocking=*/false,
        /*num_events_in_wait_list=*/0, /*event_wait_list=*/nullptr, Event);

    return PI_SUCCESS;
  }
  case CGType::CopyToDeviceGlobal: {
    CGCopyToDeviceGlobal *Copy = (CGCopyToDeviceGlobal *)MCommandGroup.get();
    MemoryManager::copy_to_device_global(
        Copy->getDeviceGlobalPtr(), Copy->isDeviceImageScoped(), MQueue,
        Copy->getNumBytes(), Copy->getOffset(), Copy->getSrc(),
        std::move(RawEvents), Event, MEvent);

    return CL_SUCCESS;
  }
  case CGType::CopyFromDeviceGlobal: {
    CGCopyFromDeviceGlobal *Copy =
        (CGCopyFromDeviceGlobal *)MCommandGroup.get();
    MemoryManager::copy_from_device_global(
        Copy->getDeviceGlobalPtr(), Copy->isDeviceImageScoped(), MQueue,
        Copy->getNumBytes(), Copy->getOffset(), Copy->getDest(),
        std::move(RawEvents), Event, MEvent);

    return CL_SUCCESS;
  }
  case CGType::ReadWriteHostPipe: {
    CGReadWriteHostPipe *ExecReadWriteHostPipe =
        (CGReadWriteHostPipe *)MCommandGroup.get();
    std::string pipeName = ExecReadWriteHostPipe->getPipeName();
    void *hostPtr = ExecReadWriteHostPipe->getHostPtr();
    size_t typeSize = ExecReadWriteHostPipe->getTypeSize();
    bool blocking = ExecReadWriteHostPipe->isBlocking();
    bool read = ExecReadWriteHostPipe->isReadHostPipe();

    if (!EventImpl) {
      EventImpl = MEvent;
    }
    return enqueueReadWriteHostPipe(MQueue, pipeName, blocking, hostPtr,
                                    typeSize, RawEvents, EventImpl, read);
  }
  case CGType::ExecCommandBuffer: {
    assert(MQueue &&
           "Command buffer submissions should have an associated queue");
    CGExecCommandBuffer *CmdBufferCG =
        static_cast<CGExecCommandBuffer *>(MCommandGroup.get());
    if (MEvent != nullptr)
      MEvent->setHostEnqueueTime();
    return MQueue->getPlugin()
        ->call_nocheck<sycl::detail::PiApiKind::piextEnqueueCommandBuffer>(
            CmdBufferCG->MCommandBuffer, MQueue->getHandleRef(),
            RawEvents.size(), RawEvents.empty() ? nullptr : &RawEvents[0],
            Event);
  }
  case CGType::CopyImage: {
    CGCopyImage *Copy = (CGCopyImage *)MCommandGroup.get();

    sycl::detail::pi::PiMemImageDesc Desc = Copy->getDesc();

    MemoryManager::copy_image_bindless(
        Copy->getSrc(), MQueue, Copy->getDst(), Desc, Copy->getFormat(),
        Copy->getCopyFlags(), Copy->getSrcOffset(), Copy->getDstOffset(),
        Copy->getHostExtent(), Copy->getCopyExtent(), std::move(RawEvents),
        Event);
    return PI_SUCCESS;
  }
  case CGType::SemaphoreWait: {
    assert(MQueue &&
           "Semaphore wait submissions should have an associated queue");
    CGSemaphoreWait *SemWait = (CGSemaphoreWait *)MCommandGroup.get();

    const detail::PluginPtr &Plugin = MQueue->getPlugin();
    auto OptWaitValue = SemWait->getWaitValue();
    uint64_t WaitValue = OptWaitValue.has_value() ? OptWaitValue.value() : 0;
    Plugin->call<PiApiKind::piextWaitExternalSemaphore>(
        MQueue->getHandleRef(), SemWait->getInteropSemaphoreHandle(),
        OptWaitValue.has_value(), WaitValue, 0, nullptr, nullptr);

    return PI_SUCCESS;
  }
  case CGType::SemaphoreSignal: {
    assert(MQueue &&
           "Semaphore signal submissions should have an associated queue");
    CGSemaphoreSignal *SemSignal = (CGSemaphoreSignal *)MCommandGroup.get();

    const detail::PluginPtr &Plugin = MQueue->getPlugin();
    auto OptSignalValue = SemSignal->getSignalValue();
    uint64_t SignalValue =
        OptSignalValue.has_value() ? OptSignalValue.value() : 0;
    Plugin->call<PiApiKind::piextSignalExternalSemaphore>(
        MQueue->getHandleRef(), SemSignal->getInteropSemaphoreHandle(),
        OptSignalValue.has_value(), SignalValue, 0, nullptr, nullptr);

    return PI_SUCCESS;
  }
  case CGType::None:
    throw sycl::exception(sycl::make_error_code(sycl::errc::runtime),
                          "CG type not implemented. " +
                              codeToString(PI_ERROR_INVALID_OPERATION));
  }
  return PI_ERROR_INVALID_OPERATION;
}

bool ExecCGCommand::producesPiEvent() const {
  return !MCommandBuffer &&
         MCommandGroup->getType() != CGType::CodeplayHostTask;
}

bool ExecCGCommand::supportsPostEnqueueCleanup() const {
  // Host tasks are cleaned up upon completion instead.
  return Command::supportsPostEnqueueCleanup() &&
         (MCommandGroup->getType() != CGType::CodeplayHostTask);
}

bool ExecCGCommand::readyForCleanup() const {
  if (MCommandGroup->getType() == CGType::CodeplayHostTask)
    return MLeafCounter == 0 && MEvent->isCompleted();
  return Command::readyForCleanup();
}

KernelFusionCommand::KernelFusionCommand(QueueImplPtr Queue)
    : Command(Command::CommandType::FUSION, Queue),
      MStatus(FusionStatus::ACTIVE) {
  emitInstrumentationDataProxy();
}

std::vector<Command *> &KernelFusionCommand::auxiliaryCommands() {
  return MAuxiliaryCommands;
}

void KernelFusionCommand::addToFusionList(ExecCGCommand *Kernel) {
  MFusionList.push_back(Kernel);
}

std::vector<ExecCGCommand *> &KernelFusionCommand::getFusionList() {
  return MFusionList;
}

bool KernelFusionCommand::producesPiEvent() const { return false; }

pi_int32 KernelFusionCommand::enqueueImp() {
  waitForPreparedHostEvents();
  waitForEvents(MQueue, MPreparedDepsEvents, MEvent->getHandleRef());

  // We need to release the queue here because KernelFusionCommands are
  // held back by the scheduler thus prevent the deallocation of the queue.
  resetQueue();
  return PI_SUCCESS;
}

void KernelFusionCommand::setFusionStatus(FusionStatus Status) {
  MStatus = Status;
}

void KernelFusionCommand::resetQueue() {
  assert(MStatus != FusionStatus::ACTIVE &&
         "Cannot release the queue attached to the KernelFusionCommand if it "
         "is active.");
  MQueue.reset();
  MWorkerQueue.reset();
}

void KernelFusionCommand::emitInstrumentationData() {
#ifdef XPTI_ENABLE_INSTRUMENTATION
  constexpr uint16_t NotificationTraceType = xpti::trace_node_create;
  if (!xptiCheckTraceEnabled(MStreamID)) {
    return;
  }
  // Create a payload with the command name and an event using this payload to
  // emit a node_create
  MCommandNodeType = commandToNodeType(MType);
  MCommandName = commandToName(MType);

  static unsigned FusionNodeCount = 0;
  std::stringstream PayloadStr;
  PayloadStr << "Fusion command #" << FusionNodeCount++;
  xpti::payload_t Payload = xpti::payload_t(PayloadStr.str().c_str());

  uint64_t CommandInstanceNo = 0;
  xpti_td *CmdTraceEvent =
      xptiMakeEvent(MCommandName.c_str(), &Payload, xpti::trace_graph_event,
                    xpti_at::active, &CommandInstanceNo);

  MInstanceID = CommandInstanceNo;
  if (CmdTraceEvent) {
    MTraceEvent = static_cast<void *>(CmdTraceEvent);
    // If we are seeing this event again, then the instance ID
    // will be greater
    // than 1; in this case, we must skip sending a
    // notification to create a node as this node has already
    // been created. We return this value so the epilog method
    // can be called selectively.
    // See makeTraceEventProlog.
    MFirstInstance = (CommandInstanceNo == 1);
  }

  // This function is called in the constructor of the command. At this point
  // the kernel fusion list is still empty, so we don't have a terrible lot of
  // information we could attach to this node here.
  if (MFirstInstance && CmdTraceEvent)
    addDeviceMetadata(CmdTraceEvent, MQueue);

  if (MFirstInstance) {
    // Since we do NOT add queue_id value to metadata, we are stashing it to TLS
    // as this data is mutable and the metadata is supposed to be invariant
    xpti::framework::stash_tuple(XPTI_QUEUE_INSTANCE_ID_KEY,
                                 getQueueID(MQueue));
    xptiNotifySubscribers(MStreamID, NotificationTraceType,
                          detail::GSYCLGraphEvent,
                          static_cast<xpti_td *>(MTraceEvent), MInstanceID,
                          static_cast<const void *>(MCommandNodeType.c_str()));
  }

#endif
}

void KernelFusionCommand::printDot(std::ostream &Stream) const {
  Stream << "\"" << this << "\" [style=filled, fillcolor=\"#AFFF82\", label=\"";

  Stream << "ID = " << this << "\\n";
  Stream << "KERNEL FUSION on " << queueDeviceToString(MQueue.get()) << "\\n"
         << "FUSION LIST: {";
  bool Initial = true;
  for (auto *Cmd : MFusionList) {
    if (!Initial) {
      Stream << ",\\n";
    }
    Initial = false;
    auto *KernelCG = static_cast<detail::CGExecKernel *>(&Cmd->getCG());
    if (KernelCG->MSyclKernel && KernelCG->MSyclKernel->isCreatedFromSource()) {
      Stream << "created from source";
    } else {
      Stream << demangleKernelName(KernelCG->getKernelName());
    }
  }
  Stream << "}\\n";

  Stream << "\"];" << std::endl;

  for (const auto &Dep : MDeps) {
    Stream << "  \"" << this << "\" -> \"" << Dep.MDepCommand << "\""
           << " [ label = \"Access mode: "
           << accessModeToString(Dep.MDepRequirement->MAccessMode) << "\\n"
           << "MemObj: " << Dep.MDepRequirement->MSYCLMemObj << " \" ]"
           << std::endl;
  }
}

UpdateCommandBufferCommand::UpdateCommandBufferCommand(
    QueueImplPtr Queue,
    ext::oneapi::experimental::detail::exec_graph_impl *Graph,
    std::vector<std::shared_ptr<ext::oneapi::experimental::detail::node_impl>>
        Nodes)
    : Command(CommandType::UPDATE_CMD_BUFFER, Queue), MGraph(Graph),
      MNodes(Nodes) {}

pi_int32 UpdateCommandBufferCommand::enqueueImp() {
  waitForPreparedHostEvents();
  std::vector<EventImplPtr> EventImpls = MPreparedDepsEvents;
  sycl::detail::pi::PiEvent &Event = MEvent->getHandleRef();
  Command::waitForEvents(MQueue, EventImpls, Event);

  for (auto &Node : MNodes) {
    auto CG = static_cast<CGExecKernel *>(Node->MCommandGroup.get());
    for (auto &Arg : CG->MArgs) {
      if (Arg.MType != kernel_param_kind_t::kind_accessor) {
        continue;
      }
      // Search through deps to get actual allocation for accessor args.
      for (const DepDesc &Dep : MDeps) {
        Requirement *Req = static_cast<AccessorImplHost *>(Arg.MPtr);
        if (Dep.MDepRequirement == Req) {
          if (Dep.MAllocaCmd) {
            Req->MData = Dep.MAllocaCmd->getMemAllocation();
          } else {
            throw sycl::exception(make_error_code(errc::invalid),
                                  "No allocation available for accessor when "
                                  "updating command buffer!");
          }
        }
      }
    }
    MGraph->updateImpl(Node);
  }

  return PI_SUCCESS;
}

void UpdateCommandBufferCommand::printDot(std::ostream &Stream) const {
  Stream << "\"" << this << "\" [style=filled, fillcolor=\"#8d8f29\", label=\"";

  Stream << "ID = " << this << "\\n";
  Stream << "CommandBuffer Command Update" << "\\n";

  Stream << "\"];" << std::endl;

  for (const auto &Dep : MDeps) {
    Stream << "  \"" << this << "\" -> \"" << Dep.MDepCommand << "\""
           << " [ label = \"Access mode: "
           << accessModeToString(Dep.MDepRequirement->MAccessMode) << "\\n"
           << "MemObj: " << Dep.MDepRequirement->MSYCLMemObj << " \" ]"
           << std::endl;
  }
}

void UpdateCommandBufferCommand::emitInstrumentationData() {}
bool UpdateCommandBufferCommand::producesPiEvent() const { return false; }

} // namespace detail
} // namespace _V1
} // namespace sycl
