//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ur_api.h"
#include <detail/error_handling/error_handling.hpp>

#include <detail/context_impl.hpp>
#include <detail/event_impl.hpp>
#include <detail/helpers.hpp>
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
#include <sycl/detail/helpers.hpp>
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

// MemoryManager:: calls return void and throw exception in case of failure.
// enqueueImp is expected to return status, not exception to correctly handle
// submission error.
template <typename MemOpFuncT, typename... MemOpArgTs>
ur_result_t callMemOpHelper(MemOpFuncT &MemOpFunc, MemOpArgTs &&...MemOpArgs) {
  try {
    MemOpFunc(MemOpArgs...);
  } catch (sycl::exception &e) {
    return static_cast<ur_result_t>(get_ur_error(e));
  }
  return UR_RESULT_SUCCESS;
}

template <typename MemOpRet, typename MemOpFuncT, typename... MemOpArgTs>
ur_result_t callMemOpHelperRet(MemOpRet &MemOpResult, MemOpFuncT &MemOpFunc,
                               MemOpArgTs &&...MemOpArgs) {
  try {
    MemOpResult = MemOpFunc(MemOpArgs...);
  } catch (sycl::exception &e) {
    return static_cast<ur_result_t>(get_ur_error(e));
  }
  return UR_RESULT_SUCCESS;
}

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
  if (!EliminatedArgMask || EliminatedArgMask->size() == 0) {
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

std::vector<ur_event_handle_t>
Command::getUrEvents(const std::vector<EventImplPtr> &EventImpls,
                     const QueueImplPtr &CommandQueue, bool IsHostTaskCommand) {
  std::vector<ur_event_handle_t> RetUrEvents;
  for (auto &EventImpl : EventImpls) {
    auto Handle = EventImpl->getHandle();
    if (Handle == nullptr)
      continue;

    // Do not add redundant event dependencies for in-order queues.
    // At this stage dependency is definitely ur task and need to check if
    // current one is a host task. In this case we should not skip ur event due
    // to different sync mechanisms for different task types on in-order queue.
    if (CommandQueue && EventImpl->getWorkerQueue() == CommandQueue &&
        CommandQueue->isInOrder() && !IsHostTaskCommand)
      continue;

    RetUrEvents.push_back(Handle);
  }

  return RetUrEvents;
}

std::vector<ur_event_handle_t>
Command::getUrEvents(const std::vector<EventImplPtr> &EventImpls) const {
  return getUrEvents(EventImpls, MWorkerQueue, isHostTask());
}

// This function is implemented (duplicating getUrEvents a lot) as short term
// solution for the issue that barrier with wait list could not
// handle empty ur event handles when kernel is enqueued on host task
// completion.
std::vector<ur_event_handle_t> Command::getUrEventsBlocking(
    const std::vector<EventImplPtr> &EventImpls) const {
  std::vector<ur_event_handle_t> RetUrEvents;
  for (auto &EventImpl : EventImpls) {
    // Throwaway events created with empty constructor will not have a context
    // (which is set lazily) calling getContextImpl() would set that
    // context, which we wish to avoid as it is expensive.
    // Skip host task and NOP events also.
    if (EventImpl->isDefaultConstructed() || EventImpl->isHost() ||
        EventImpl->isNOP())
      continue;

    // If command has not been enqueued then we have to enqueue it.
    // It may happen if async enqueue in a host task is involved.
    // Interoperability events are special cases and they are not enqueued, as
    // they don't have an associated queue and command.
    if (!EventImpl->isInterop() && !EventImpl->isEnqueued()) {
      if (!EventImpl->getCommand() ||
          !static_cast<Command *>(EventImpl->getCommand())->producesPiEvent())
        continue;
      std::vector<Command *> AuxCmds;
      Scheduler::getInstance().enqueueCommandForCG(EventImpl, AuxCmds,
                                                   BLOCKING);
    }
    // Do not add redundant event dependencies for in-order queues.
    // At this stage dependency is definitely ur task and need to check if
    // current one is a host task. In this case we should not skip pi event due
    // to different sync mechanisms for different task types on in-order queue.
    if (MWorkerQueue && EventImpl->getWorkerQueue() == MWorkerQueue &&
        MWorkerQueue->isInOrder() && !isHostTask())
      continue;

    RetUrEvents.push_back(EventImpl->getHandle());
  }

  return RetUrEvents;
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

void InteropFreeFunc(ur_queue_handle_t, void *InteropData) {
  auto *Data = reinterpret_cast<EnqueueNativeCommandData *>(InteropData);
  return Data->func(Data->ih);
}
} // namespace

class DispatchHostTask {
  ExecCGCommand *MThisCmd;
  std::vector<interop_handle::ReqToMem> MReqToMem;
  std::vector<ur_mem_handle_t> MReqUrMem;

  bool waitForEvents() const {
    std::map<const AdapterPtr, std::vector<EventImplPtr>>
        RequiredEventsPerAdapter;

    for (const EventImplPtr &Event : MThisCmd->MPreparedDepsEvents) {
      const AdapterPtr &Adapter = Event->getAdapter();
      RequiredEventsPerAdapter[Adapter].push_back(Event);
    }

    // wait for dependency device events
    // FIXME Current implementation of waiting for events will make the thread
    // 'sleep' until all of dependency events are complete. We need a bit more
    // sophisticated waiting mechanism to allow to utilize this thread for any
    // other available job and resume once all required events are ready.
    for (auto &AdapterWithEvents : RequiredEventsPerAdapter) {
      std::vector<ur_event_handle_t> RawEvents =
          MThisCmd->getUrEvents(AdapterWithEvents.second);
      if (RawEvents.size() == 0)
        continue;
      try {
        AdapterWithEvents.first->call<UrApiKind::urEventWait>(RawEvents.size(),
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
                   std::vector<ur_mem_handle_t> ReqUrMem)
      : MThisCmd{ThisCmd}, MReqToMem(std::move(ReqToMem)),
        MReqUrMem(std::move(ReqUrMem)) {}

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
        Queue->getAdapter()->call<UrApiKind::urDeviceGetInfo>(
            detail::getSyclObjImpl(Queue->get_device())->getHandleRef(),
            UR_DEVICE_INFO_ENQUEUE_NATIVE_COMMAND_SUPPORT_EXP,
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
          Queue->getAdapter()->call<UrApiKind::urEnqueueNativeCommandExp>(
              HostTask.MQueue->getHandleRef(), InteropFreeFunc, &CustomOpData,
              MReqUrMem.size(), MReqUrMem.data(), nullptr, 0, nullptr, nullptr);
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
      // If we enqueue blocked users - ur level could throw exception that
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
                            ur_event_handle_t &Event) {
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
      // three events (E1, E2, E3). Now, if urEventWait is called for all
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
        std::vector<ur_event_handle_t> RawEvents =
            getUrEvents(CtxWithEvents.second);
        if (!RawEvents.empty()) {
          CtxWithEvents.first->getAdapter()->call<UrApiKind::urEventWait>(
              RawEvents.size(), RawEvents.data());
        }
      }
    } else {
      std::vector<ur_event_handle_t> RawEvents = getUrEvents(EventImpls);
      flushCrossQueueDeps(EventImpls, MWorkerQueue);
      const AdapterPtr &Adapter = Queue->getAdapter();

      if (MEvent != nullptr)
        MEvent->setHostEnqueueTime();
      Adapter->call<UrApiKind::urEnqueueEventsWait>(
          Queue->getHandleRef(), RawEvents.size(), &RawEvents[0], &Event);
    }
  }
}

/// It is safe to bind MPreparedDepsEvents and MPreparedHostDepsEvents
/// references to event_impl class members because Command
/// should not outlive the event connected to it.
Command::Command(
    CommandType Type, QueueImplPtr Queue,
    ur_exp_command_buffer_handle_t CommandBuffer,
    const std::vector<ur_exp_command_buffer_sync_point_t> &SyncPoints)
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
/// @param UrEventAddr The address that defines the edge dependency, which in
/// this case is an event
void Command::emitEdgeEventForEventDependence(Command *Cmd,
                                              ur_event_handle_t &UrEventAddr) {
#ifdef XPTI_ENABLE_INSTRUMENTATION
  // If we have failed to create an event to represent the Command, then we
  // cannot emit an edge event. Bail early!
  if (!(xptiCheckTraceEnabled(MStreamID) && MTraceEvent))
    return;

  if (Cmd && Cmd->MTraceEvent) {
    // If the event is associated with a command, we use this command's trace
    // event as the source of edge, hence modeling the control flow
    emitEdgeEventForCommandDependence(Cmd, (void *)UrEventAddr, false);
    return;
  }
  if (UrEventAddr) {
    xpti::utils::StringHelper SH;
    std::string AddressStr = SH.addressAsString<ur_event_handle_t>(UrEventAddr);
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
                        reinterpret_cast<size_t>(UrEventAddr));
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
    // than 1; in the previous implementation, we would skip sending a
    // notifications for subsequent instances. With the new implementation, we
    // will send a notification for each instance as this allows for mutable
    // metadata entries for multiple visits to the same code location and
    // maintaining data integrity.
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
  // 2. Some types of commands do not produce UR events after they are
  // enqueued (e.g. alloca). Note that we can't check the ur event to make that
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
  ur_event_handle_t UrEventAddr = Event->getHandle();
  // Now make an edge for the dependent event
  emitEdgeEventForEventDependence(Cmd, UrEventAddr);
#endif

  return processDepEvent(std::move(Event), DepDesc{nullptr, nullptr, nullptr},
                         ToCleanUp);
}

void Command::emitEnqueuedEventSignal(const ur_event_handle_t UrEventAddr) {
#ifdef XPTI_ENABLE_INSTRUMENTATION
  emitInstrumentationGeneral(
      MStreamID, MInstanceID, static_cast<xpti_td *>(MTraceEvent),
      xpti::trace_signal, static_cast<const void *>(UrEventAddr));
#endif
  std::ignore = UrEventAddr;
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
  // failures if ur level report any.
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
  ur_result_t Res = enqueueImp();

  if (UR_RESULT_SUCCESS != Res)
    EnqueueResult =
        EnqueueResultT(EnqueueResultT::SyclEnqueueFailed, this, Res);
  else {
    MEvent->setEnqueued();
    if (MShouldCompleteEventIfPossible && !MEvent->isDiscarded() &&
        (MEvent->isHost() || MEvent->getHandle() == nullptr))
      MEvent->setComplete();

    // Consider the command is successfully enqueued if return code is
    // UR_RESULT_SUCCESS
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
  emitEnqueuedEventSignal(MEvent->getHandle());
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
  default:
    return "Unknown block reason";
  }
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
  // Set the relevant meta data properties for this command; in the 64-bit key
  // based implementation, we would notify the graph events only for the first
  // instance as the trace event structure was invariant across all instances.
  // Due to mutable metadata requirements, we now create and notify them for all
  // instances. In addition to this, we have moved to 128-bit keys in the XPTI
  // internal infrastructure to guarantee collision free universal IDs.
  if (MTraceEvent) {
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

  makeTraceEventEpilog();
#endif
}

ur_result_t AllocaCommand::enqueueImp() {
  waitForPreparedHostEvents();
  std::vector<EventImplPtr> EventImpls = MPreparedDepsEvents;

  ur_event_handle_t UREvent = nullptr;

  void *HostPtr = nullptr;
  if (!MIsLeaderAlloca) {

    if (!MQueue) {
      // Do not need to make allocation if we have a linked device allocation
      Command::waitForEvents(MQueue, EventImpls, UREvent);
      MEvent->setHandle(UREvent);

      return UR_RESULT_SUCCESS;
    }
    HostPtr = MLinkedAllocaCmd->getMemAllocation();
  }
  // TODO: Check if it is correct to use std::move on stack variable and
  // delete it RawEvents below.
  if (auto Result = callMemOpHelperRet(MMemAllocation, MemoryManager::allocate,
                                       getContext(MQueue), getSYCLMemObj(),
                                       MInitFromUserData, HostPtr,
                                       std::move(EventImpls), UREvent);
      Result != UR_RESULT_SUCCESS)
    return Result;

  MEvent->setHandle(UREvent);
  return UR_RESULT_SUCCESS;
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

  xpti_td *TE = static_cast<xpti_td *>(MTraceEvent);
  xpti::addMetadata(TE, "offset", this->MRequirement.MOffsetInBytes);
  xpti::addMetadata(TE, "access_range_start",
                    this->MRequirement.MAccessRange[0]);
  xpti::addMetadata(TE, "access_range_end", this->MRequirement.MAccessRange[1]);
  xpti::framework::stash_tuple(XPTI_QUEUE_INSTANCE_ID_KEY, getQueueID(MQueue));
  makeTraceEventEpilog();
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

ur_result_t AllocaSubBufCommand::enqueueImp() {
  waitForPreparedHostEvents();
  std::vector<EventImplPtr> EventImpls = MPreparedDepsEvents;
  ur_event_handle_t UREvent = nullptr;

  if (auto Result = callMemOpHelperRet(
          MMemAllocation, MemoryManager::allocateMemSubBuffer,
          getContext(MQueue), MParentAlloca->getMemAllocation(),
          MRequirement.MElemSize, MRequirement.MOffsetInBytes,
          MRequirement.MAccessRange, std::move(EventImpls), UREvent);
      Result != UR_RESULT_SUCCESS)
    return Result;

  MEvent->setHandle(UREvent);

  XPTIRegistry::bufferAssociateNotification(MParentAlloca->getSYCLMemObj(),
                                            MMemAllocation);
  return UR_RESULT_SUCCESS;
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

  xpti_td *TE = static_cast<xpti_td *>(MTraceEvent);
  addDeviceMetadata(TE, MQueue);
  xpti::addMetadata(TE, "allocation_type",
                    commandToName(MAllocaCmd->getType()));
  // Since we do NOT add queue_id value to metadata, we are stashing it to TLS
  // as this data is mutable and the metadata is supposed to be invariant
  xpti::framework::stash_tuple(XPTI_QUEUE_INSTANCE_ID_KEY, getQueueID(MQueue));
  makeTraceEventEpilog();
#endif
}

ur_result_t ReleaseCommand::enqueueImp() {
  waitForPreparedHostEvents();
  std::vector<EventImplPtr> EventImpls = MPreparedDepsEvents;
  std::vector<ur_event_handle_t> RawEvents = getUrEvents(EventImpls);
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
    ur_event_handle_t UREvent = nullptr;

    void *Src = CurAllocaIsHost
                    ? MAllocaCmd->getMemAllocation()
                    : MAllocaCmd->MLinkedAllocaCmd->getMemAllocation();

    void *Dst = !CurAllocaIsHost
                    ? MAllocaCmd->getMemAllocation()
                    : MAllocaCmd->MLinkedAllocaCmd->getMemAllocation();

    if (auto Result =
            callMemOpHelper(MemoryManager::unmap, MAllocaCmd->getSYCLMemObj(),
                            Dst, Queue, Src, RawEvents, UREvent);
        Result != UR_RESULT_SUCCESS)
      return Result;

    UnmapEventImpl->setHandle(UREvent);
    std::swap(MAllocaCmd->MIsActive, MAllocaCmd->MLinkedAllocaCmd->MIsActive);
    EventImpls.clear();
    EventImpls.push_back(UnmapEventImpl);
  }
  ur_event_handle_t UREvent = nullptr;
  if (SkipRelease)
    Command::waitForEvents(MQueue, EventImpls, UREvent);
  else {
    if (auto Result = callMemOpHelper(
            MemoryManager::release, getContext(MQueue),
            MAllocaCmd->getSYCLMemObj(), MAllocaCmd->getMemAllocation(),
            std::move(EventImpls), UREvent);
        Result != UR_RESULT_SUCCESS)
      return Result;
  }
  MEvent->setHandle(UREvent);
  return UR_RESULT_SUCCESS;
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

  xpti_td *TE = static_cast<xpti_td *>(MTraceEvent);
  addDeviceMetadata(TE, MQueue);
  xpti::addMetadata(TE, "memory_object", reinterpret_cast<size_t>(MAddress));
  // Since we do NOT add queue_id value to metadata, we are stashing it to TLS
  // as this data is mutable and the metadata is supposed to be invariant
  xpti::framework::stash_tuple(XPTI_QUEUE_INSTANCE_ID_KEY, getQueueID(MQueue));
  makeTraceEventEpilog();
#endif
}

ur_result_t MapMemObject::enqueueImp() {
  waitForPreparedHostEvents();
  std::vector<EventImplPtr> EventImpls = MPreparedDepsEvents;
  std::vector<ur_event_handle_t> RawEvents = getUrEvents(EventImpls);
  flushCrossQueueDeps(EventImpls, MWorkerQueue);

  ur_event_handle_t UREvent = nullptr;
  if (auto Result = callMemOpHelperRet(
          *MDstPtr, MemoryManager::map, MSrcAllocaCmd->getSYCLMemObj(),
          MSrcAllocaCmd->getMemAllocation(), MQueue, MMapMode, MSrcReq.MDims,
          MSrcReq.MMemoryRange, MSrcReq.MAccessRange, MSrcReq.MOffset,
          MSrcReq.MElemSize, std::move(RawEvents), UREvent);
      Result != UR_RESULT_SUCCESS)
    return Result;

  MEvent->setHandle(UREvent);
  return UR_RESULT_SUCCESS;
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

  xpti_td *TE = static_cast<xpti_td *>(MTraceEvent);
  addDeviceMetadata(TE, MQueue);
  xpti::addMetadata(TE, "memory_object", reinterpret_cast<size_t>(MAddress));
  // Since we do NOT add queue_id value to metadata, we are stashing it to TLS
  // as this data is mutable and the metadata is supposed to be invariant
  xpti::framework::stash_tuple(XPTI_QUEUE_INSTANCE_ID_KEY, getQueueID(MQueue));
  makeTraceEventEpilog();
#endif
}

bool UnMapMemObject::producesPiEvent() const {
  // TODO remove this workaround once the batching issue is addressed in Level
  // Zero adapter.
  // Consider the following scenario on Level Zero:
  // 1. Kernel A, which uses buffer A, is submitted to queue A.
  // 2. Kernel B, which uses buffer B, is submitted to queue B.
  // 3. queueA.wait().
  // 4. queueB.wait().
  // DPCPP runtime used to treat unmap/write commands for buffer A/B as host
  // dependencies (i.e. they were waited for prior to enqueueing any command
  // that's dependent on them). This allowed Level Zero adapter to detect that
  // each queue is idle on steps 1/2 and submit the command list right away.
  // This is no longer the case since we started passing these dependencies in
  // an event waitlist and Level Zero adapter attempts to batch these commands,
  // so the execution of kernel B starts only on step 4. This workaround
  // restores the old behavior in this case until this is resolved.
  return MQueue && (MQueue->getDeviceImplPtr()->getBackend() !=
                        backend::ext_oneapi_level_zero ||
                    MEvent->getHandle() != nullptr);
}

ur_result_t UnMapMemObject::enqueueImp() {
  waitForPreparedHostEvents();
  std::vector<EventImplPtr> EventImpls = MPreparedDepsEvents;
  std::vector<ur_event_handle_t> RawEvents = getUrEvents(EventImpls);
  flushCrossQueueDeps(EventImpls, MWorkerQueue);

  ur_event_handle_t UREvent = nullptr;
  if (auto Result =
          callMemOpHelper(MemoryManager::unmap, MDstAllocaCmd->getSYCLMemObj(),
                          MDstAllocaCmd->getMemAllocation(), MQueue, *MSrcPtr,
                          std::move(RawEvents), UREvent);
      Result != UR_RESULT_SUCCESS)
    return Result;

  MEvent->setHandle(UREvent);

  return UR_RESULT_SUCCESS;
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
  xpti::framework::stash_tuple(XPTI_QUEUE_INSTANCE_ID_KEY, getQueueID(MQueue));
  makeTraceEventEpilog();
#endif
}

ContextImplPtr MemCpyCommand::getWorkerContext() const {
  if (!MWorkerQueue)
    return nullptr;
  return MWorkerQueue->getContextImplPtr();
}

bool MemCpyCommand::producesPiEvent() const {
  // TODO remove this workaround once the batching issue is addressed in Level
  // Zero adapter.
  // Consider the following scenario on Level Zero:
  // 1. Kernel A, which uses buffer A, is submitted to queue A.
  // 2. Kernel B, which uses buffer B, is submitted to queue B.
  // 3. queueA.wait().
  // 4. queueB.wait().
  // DPCPP runtime used to treat unmap/write commands for buffer A/B as host
  // dependencies (i.e. they were waited for prior to enqueueing any command
  // that's dependent on them). This allowed Level Zero adapter to detect that
  // each queue is idle on steps 1/2 and submit the command list right away.
  // This is no longer the case since we started passing these dependencies in
  // an event waitlist and Level Zero adapter attempts to batch these commands,
  // so the execution of kernel B starts only on step 4. This workaround
  // restores the old behavior in this case until this is resolved.
  return !MQueue ||
         MQueue->getDeviceImplPtr()->getBackend() !=
             backend::ext_oneapi_level_zero ||
         MEvent->getHandle() != nullptr;
}

ur_result_t MemCpyCommand::enqueueImp() {
  waitForPreparedHostEvents();
  std::vector<EventImplPtr> EventImpls = MPreparedDepsEvents;

  ur_event_handle_t UREvent = nullptr;

  auto RawEvents = getUrEvents(EventImpls);
  flushCrossQueueDeps(EventImpls, MWorkerQueue);

  if (auto Result = callMemOpHelper(
          MemoryManager::copy, MSrcAllocaCmd->getSYCLMemObj(),
          MSrcAllocaCmd->getMemAllocation(), MSrcQueue, MSrcReq.MDims,
          MSrcReq.MMemoryRange, MSrcReq.MAccessRange, MSrcReq.MOffset,
          MSrcReq.MElemSize, MDstAllocaCmd->getMemAllocation(), MQueue,
          MDstReq.MDims, MDstReq.MMemoryRange, MDstReq.MAccessRange,
          MDstReq.MOffset, MDstReq.MElemSize, std::move(RawEvents), UREvent,
          MEvent);
      Result != UR_RESULT_SUCCESS)
    return Result;

  MEvent->setHandle(UREvent);
  return UR_RESULT_SUCCESS;
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

ur_result_t UpdateHostRequirementCommand::enqueueImp() {
  waitForPreparedHostEvents();
  std::vector<EventImplPtr> EventImpls = MPreparedDepsEvents;
  ur_event_handle_t UREvent = nullptr;
  Command::waitForEvents(MQueue, EventImpls, UREvent);
  MEvent->setHandle(UREvent);

  assert(MSrcAllocaCmd && "Expected valid alloca command");
  assert(MSrcAllocaCmd->getMemAllocation() && "Expected valid source pointer");
  assert(MDstPtr && "Expected valid target pointer");
  *MDstPtr = MSrcAllocaCmd->getMemAllocation();

  return UR_RESULT_SUCCESS;
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
  xpti::framework::stash_tuple(XPTI_QUEUE_INSTANCE_ID_KEY, getQueueID(MQueue));
  makeTraceEventEpilog();
#endif
}

ContextImplPtr MemCpyCommandHost::getWorkerContext() const {
  if (!MWorkerQueue)
    return nullptr;
  return MWorkerQueue->getContextImplPtr();
}

ur_result_t MemCpyCommandHost::enqueueImp() {
  const QueueImplPtr &Queue = MWorkerQueue;
  waitForPreparedHostEvents();
  std::vector<EventImplPtr> EventImpls = MPreparedDepsEvents;
  std::vector<ur_event_handle_t> RawEvents = getUrEvents(EventImpls);

  ur_event_handle_t UREvent = nullptr;
  // Omit copying if mode is discard one.
  // TODO: Handle this at the graph building time by, for example, creating
  // empty node instead of memcpy.
  if (MDstReq.MAccessMode == access::mode::discard_read_write ||
      MDstReq.MAccessMode == access::mode::discard_write) {
    Command::waitForEvents(Queue, EventImpls, UREvent);

    return UR_RESULT_SUCCESS;
  }

  flushCrossQueueDeps(EventImpls, MWorkerQueue);

  if (auto Result = callMemOpHelper(
          MemoryManager::copy, MSrcAllocaCmd->getSYCLMemObj(),
          MSrcAllocaCmd->getMemAllocation(), MSrcQueue, MSrcReq.MDims,
          MSrcReq.MMemoryRange, MSrcReq.MAccessRange, MSrcReq.MOffset,
          MSrcReq.MElemSize, *MDstPtr, MQueue, MDstReq.MDims,
          MDstReq.MMemoryRange, MDstReq.MAccessRange, MDstReq.MOffset,
          MDstReq.MElemSize, std::move(RawEvents), UREvent, MEvent);
      Result != UR_RESULT_SUCCESS)
    return Result;

  MEvent->setHandle(UREvent);
  return UR_RESULT_SUCCESS;
}

EmptyCommand::EmptyCommand() : Command(CommandType::EMPTY_TASK, nullptr) {
  emitInstrumentationDataProxy();
}

ur_result_t EmptyCommand::enqueueImp() {
  waitForPreparedHostEvents();
  ur_event_handle_t UREvent = nullptr;
  waitForEvents(MQueue, MPreparedDepsEvents, UREvent);
  MEvent->setHandle(UREvent);
  return UR_RESULT_SUCCESS;
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

  xpti_td *CmdTraceEvent = static_cast<xpti_td *>(MTraceEvent);
  addDeviceMetadata(CmdTraceEvent, MQueue);
  xpti::addMetadata(CmdTraceEvent, "memory_object",
                    reinterpret_cast<size_t>(MAddress));
  // Since we do NOT add queue_id value to metadata, we are stashing it to TLS
  // as this data is mutable and the metadata is supposed to be invariant
  xpti::framework::stash_tuple(XPTI_QUEUE_INSTANCE_ID_KEY, getQueueID(MQueue));
  makeTraceEventEpilog();
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

  xpti_td *CmdTraceEvent = static_cast<xpti_td *>(MTraceEvent);
  addDeviceMetadata(CmdTraceEvent, MQueue);
  xpti::addMetadata(CmdTraceEvent, "memory_object",
                    reinterpret_cast<size_t>(MAddress));
  // Since we do NOT add queue_id value to metadata, we are stashing it to TLS
  // as this data is mutable and the metadata is supposed to be invariant
  xpti::framework::stash_tuple(XPTI_QUEUE_INSTANCE_ID_KEY, getQueueID(MQueue));
  makeTraceEventEpilog();
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
    bool EventNeeded, ur_exp_command_buffer_handle_t CommandBuffer,
    const std::vector<ur_exp_command_buffer_sync_point_t> &Dependencies)
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
    ur_kernel_handle_t KernelHandle = SyclKernel->getHandleRef();
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
  ur_program_handle_t Program = nullptr;
  ur_kernel_handle_t Kernel = nullptr;
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
    Program = KernelImpl->getDeviceImage()->get_ur_program_ref();
  } else if (nullptr != SyclKernel) {
    Program = SyclKernel->getProgramRef();
    if (!SyclKernel->isCreatedFromSource())
      EliminatedArgMask = SyclKernel->getKernelArgMask();
  } else if (Queue) {
    // NOTE: Queue can be null when kernel is directly enqueued to a command
    // buffer
    //       by graph API, when a modifiable graph is finalized.
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
                                   const std::string &FuncName,
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
    Payload = xpti::payload_t(FuncName.empty() ? KernelName.c_str()
                                               : FuncName.c_str(),
                              FileName.c_str(), Line, Column, Address);
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
    const detail::code_location &CodeLoc, bool IsTopCodeLoc,
    const std::string &SyclKernelName, const QueueImplPtr &Queue,
    const NDRDescT &NDRDesc,
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

  // If code location is above sycl layer, use function name from code
  // location instead of kernel name in event payload
  std::string FuncName = (!IsTopCodeLoc && CodeLoc.functionName())
                             ? CodeLoc.functionName()
                             : std::string();

  instrumentationFillCommonData(KernelName, FuncName, FileName,
                                CodeLoc.lineNumber(), CodeLoc.columnNumber(),
                                Address, Queue, FromSource, InstanceID,
                                CmdTraceEvent);

  if (CmdTraceEvent) {
    // Stash the queue_id mutable metadata in TLS
    // NOTE: Queue can be null when kernel is directly enqueued to a command
    // buffer by graph API, when a modifiable graph is finalized.
    if (Queue.get())
      xpti::framework::stash_tuple(XPTI_QUEUE_INSTANCE_ID_KEY,
                                   getQueueID(Queue));
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

  // If code location is above sycl layer, use function name from code
  // location instead of kernel name in event payload
  std::string FuncName;
  if (!MCommandGroup->MIsTopCodeLoc)
    FuncName = MCommandGroup->MFunctionName;

  xpti_td *CmdTraceEvent = nullptr;
  instrumentationFillCommonData(KernelName, FuncName, MCommandGroup->MFileName,
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
static void adjustNDRangePerKernel(NDRDescT &NDR, ur_kernel_handle_t Kernel,
                                   const device_impl &DeviceImpl) {
  if (NDR.GlobalSize[0] != 0)
    return; // GlobalSize is set - no need to adjust
  // check the prerequisites:
  assert(NDR.LocalSize[0] == 0);
  // TODO might be good to cache this info together with the kernel info to
  // avoid get_kernel_work_group_info on every kernel run
  range<3> WGSize = get_kernel_device_specific_info<
      sycl::info::kernel_device_specific::compile_work_group_size>(
      Kernel, DeviceImpl.getHandleRef(), DeviceImpl.getAdapter());

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

ur_mem_flags_t AccessModeToUr(access::mode AccessorMode) {
  switch (AccessorMode) {
  case access::mode::read:
    return UR_MEM_FLAG_READ_ONLY;
  case access::mode::write:
  case access::mode::discard_write:
    return UR_MEM_FLAG_WRITE_ONLY;
  default:
    return UR_MEM_FLAG_READ_WRITE;
  }
}

void SetArgBasedOnType(
    const AdapterPtr &Adapter, ur_kernel_handle_t Kernel,
    const std::shared_ptr<device_image_impl> &DeviceImageImpl,
    const std::function<void *(Requirement *Req)> &getMemAllocationFunc,
    const sycl::context &Context, detail::ArgDesc &Arg, size_t NextTrueIndex) {
  switch (Arg.MType) {
  case kernel_param_kind_t::kind_work_group_memory:
    break;
  case kernel_param_kind_t::kind_stream:
    break;
  case kernel_param_kind_t::kind_accessor: {
    Requirement *Req = (Requirement *)(Arg.MPtr);

    // getMemAllocationFunc is nullptr when there are no requirements. However,
    // we may pass default constructed accessors to a command, which don't add
    // requirements. In such case, getMemAllocationFunc is nullptr, but it's a
    // valid case, so we need to properly handle it.
    ur_mem_handle_t MemArg =
        getMemAllocationFunc
            ? reinterpret_cast<ur_mem_handle_t>(getMemAllocationFunc(Req))
            : nullptr;
    ur_kernel_arg_mem_obj_properties_t MemObjData{};
    MemObjData.stype = UR_STRUCTURE_TYPE_KERNEL_ARG_MEM_OBJ_PROPERTIES;
    MemObjData.memoryAccess = AccessModeToUr(Req->MAccessMode);
    Adapter->call<UrApiKind::urKernelSetArgMemObj>(Kernel, NextTrueIndex,
                                                   &MemObjData, MemArg);
    break;
  }
  case kernel_param_kind_t::kind_std_layout: {
    if (Arg.MPtr) {
      Adapter->call<UrApiKind::urKernelSetArgValue>(
          Kernel, NextTrueIndex, Arg.MSize, nullptr, Arg.MPtr);
    } else {
      Adapter->call<UrApiKind::urKernelSetArgLocal>(Kernel, NextTrueIndex,
                                                    Arg.MSize, nullptr);
    }

    break;
  }
  case kernel_param_kind_t::kind_sampler: {
    sampler *SamplerPtr = (sampler *)Arg.MPtr;
    ur_sampler_handle_t Sampler =
        (ur_sampler_handle_t)detail::getSyclObjImpl(*SamplerPtr)
            ->getOrCreateSampler(Context);
    Adapter->call<UrApiKind::urKernelSetArgSampler>(Kernel, NextTrueIndex,
                                                    nullptr, Sampler);
    break;
  }
  case kernel_param_kind_t::kind_pointer: {
    // We need to de-rerence this to get the actual USM allocation - that's the
    // pointer UR is expecting.
    const void *Ptr = *static_cast<const void *const *>(Arg.MPtr);
    Adapter->call<UrApiKind::urKernelSetArgPointer>(Kernel, NextTrueIndex,
                                                    nullptr, Ptr);
    break;
  }
  case kernel_param_kind_t::kind_specialization_constants_buffer: {
    assert(DeviceImageImpl != nullptr);
    ur_mem_handle_t SpecConstsBuffer =
        DeviceImageImpl->get_spec_const_buffer_ref();

    ur_kernel_arg_mem_obj_properties_t MemObjProps{};
    MemObjProps.pNext = nullptr;
    MemObjProps.stype = UR_STRUCTURE_TYPE_KERNEL_ARG_MEM_OBJ_PROPERTIES;
    MemObjProps.memoryAccess = UR_MEM_FLAG_READ_ONLY;
    Adapter->call<UrApiKind::urKernelSetArgMemObj>(
        Kernel, NextTrueIndex, &MemObjProps, SpecConstsBuffer);
    break;
  }
  case kernel_param_kind_t::kind_invalid:
    throw sycl::exception(sycl::make_error_code(sycl::errc::runtime),
                          "Invalid kernel param kind " +
                              codeToString(UR_RESULT_ERROR_INVALID_VALUE));
    break;
  }
}

static ur_result_t SetKernelParamsAndLaunch(
    const QueueImplPtr &Queue, std::vector<ArgDesc> &Args,
    const std::shared_ptr<device_image_impl> &DeviceImageImpl,
    ur_kernel_handle_t Kernel, NDRDescT &NDRDesc,
    std::vector<ur_event_handle_t> &RawEvents,
    const detail::EventImplPtr &OutEventImpl,
    const KernelArgMask *EliminatedArgMask,
    const std::function<void *(Requirement *Req)> &getMemAllocationFunc,
    bool IsCooperative, bool KernelUsesClusterLaunch,
    const RTDeviceBinaryImage *BinImage, const std::string &KernelName) {
  assert(Queue && "Kernel submissions should have an associated queue");
  const AdapterPtr &Adapter = Queue->getAdapter();

  if (SYCLConfig<SYCL_JIT_AMDGCN_PTX_KERNELS>::get()) {
    std::vector<unsigned char> Empty;
    Kernel = Scheduler::getInstance().completeSpecConstMaterialization(
        Queue, BinImage, KernelName,
        DeviceImageImpl.get() ? DeviceImageImpl->get_spec_const_blob_ref()
                              : Empty);
  }

  auto setFunc = [&Adapter, Kernel, &DeviceImageImpl, &getMemAllocationFunc,
                  &Queue](detail::ArgDesc &Arg, size_t NextTrueIndex) {
    SetArgBasedOnType(Adapter, Kernel, DeviceImageImpl, getMemAllocationFunc,
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
    Adapter->call<UrApiKind::urKernelGetGroupInfo>(
        Kernel, Queue->getDeviceImplPtr()->getHandleRef(),
        UR_KERNEL_GROUP_INFO_COMPILE_WORK_GROUP_SIZE, sizeof(RequiredWGSize),
        RequiredWGSize,
        /* pPropSizeRet = */ nullptr);

    const bool EnforcedLocalSize =
        (RequiredWGSize[0] != 0 || RequiredWGSize[1] != 0 ||
         RequiredWGSize[2] != 0);
    if (EnforcedLocalSize)
      LocalSize = RequiredWGSize;
  }
  if (OutEventImpl != nullptr)
    OutEventImpl->setHostEnqueueTime();
  if (KernelUsesClusterLaunch) {
    std::vector<ur_exp_launch_property_t> property_list;

    ur_exp_launch_property_value_t launch_property_value_cluster_range;
    launch_property_value_cluster_range.clusterDim[0] =
        NDRDesc.ClusterDimensions[0];
    launch_property_value_cluster_range.clusterDim[1] =
        NDRDesc.ClusterDimensions[1];
    launch_property_value_cluster_range.clusterDim[2] =
        NDRDesc.ClusterDimensions[2];

    property_list.push_back({UR_EXP_LAUNCH_PROPERTY_ID_CLUSTER_DIMENSION,
                             launch_property_value_cluster_range});

    if (IsCooperative) {
      ur_exp_launch_property_value_t launch_property_value_cooperative;
      launch_property_value_cooperative.cooperative = 1;
      property_list.push_back({UR_EXP_LAUNCH_PROPERTY_ID_COOPERATIVE,
                               launch_property_value_cooperative});
    }

    ur_event_handle_t UREvent = nullptr;
    ur_result_t Error =
        Adapter->call_nocheck<UrApiKind::urEnqueueKernelLaunchCustomExp>(
            Queue->getHandleRef(), Kernel, NDRDesc.Dims, &NDRDesc.GlobalSize[0],
            LocalSize, property_list.size(), property_list.data(),
            RawEvents.size(), RawEvents.empty() ? nullptr : &RawEvents[0],
            OutEventImpl ? &UREvent : nullptr);
    if (OutEventImpl) {
      OutEventImpl->setHandle(UREvent);
    }
    return Error;
  }
  ur_event_handle_t UREvent = nullptr;
  ur_result_t Error =
      [&](auto... Args) {
        if (IsCooperative) {
          return Adapter
              ->call_nocheck<UrApiKind::urEnqueueCooperativeKernelLaunchExp>(
                  Args...);
        }
        return Adapter->call_nocheck<UrApiKind::urEnqueueKernelLaunch>(Args...);
      }(Queue->getHandleRef(), Kernel, NDRDesc.Dims, &NDRDesc.GlobalOffset[0],
        &NDRDesc.GlobalSize[0], LocalSize, RawEvents.size(),
        RawEvents.empty() ? nullptr : &RawEvents[0],
        OutEventImpl ? &UREvent : nullptr);
  if (Error == UR_RESULT_SUCCESS && OutEventImpl) {
    OutEventImpl->setHandle(UREvent);
  }

  return Error;
}

namespace {
std::tuple<ur_kernel_handle_t, std::shared_ptr<device_image_impl>,
           const KernelArgMask *>
getCGKernelInfo(const CGExecKernel &CommandGroup, ContextImplPtr ContextImpl,
                DeviceImplPtr DeviceImpl,
                std::vector<ur_kernel_handle_t> &UrKernelsToRelease,
                std::vector<ur_program_handle_t> &UrProgramsToRelease) {

  ur_kernel_handle_t UrKernel = nullptr;
  std::shared_ptr<device_image_impl> DeviceImageImpl = nullptr;
  const KernelArgMask *EliminatedArgMask = nullptr;

  // Use kernel_bundle if available unless it is interop.
  // Interop bundles can't be used in the first branch, because the kernels
  // in interop kernel bundles (if any) do not have kernel_id
  // and can therefore not be looked up, but since they are self-contained
  // they can simply be launched directly.
  if (auto KernelBundleImplPtr = CommandGroup.MKernelBundle;
      KernelBundleImplPtr && !KernelBundleImplPtr->isInterop()) {
    auto KernelName = CommandGroup.MKernelName;
    kernel_id KernelID =
        detail::ProgramManager::getInstance().getSYCLKernelID(KernelName);

    kernel SyclKernel =
        KernelBundleImplPtr->get_kernel(KernelID, KernelBundleImplPtr);

    auto SyclKernelImpl = detail::getSyclObjImpl(SyclKernel);
    UrKernel = SyclKernelImpl->getHandleRef();
    DeviceImageImpl = SyclKernelImpl->getDeviceImage();
    EliminatedArgMask = SyclKernelImpl->getKernelArgMask();
  } else if (auto Kernel = CommandGroup.MSyclKernel; Kernel != nullptr) {
    UrKernel = Kernel->getHandleRef();
    EliminatedArgMask = Kernel->getKernelArgMask();
  } else {
    ur_program_handle_t UrProgram = nullptr;
    std::tie(UrKernel, std::ignore, EliminatedArgMask, UrProgram) =
        sycl::detail::ProgramManager::getInstance().getOrCreateKernel(
            ContextImpl, DeviceImpl, CommandGroup.MKernelName);
    UrKernelsToRelease.push_back(UrKernel);
    UrProgramsToRelease.push_back(UrProgram);
  }
  return std::make_tuple(UrKernel, DeviceImageImpl, EliminatedArgMask);
}
} // anonymous namespace

ur_result_t enqueueImpCommandBufferKernel(
    context Ctx, DeviceImplPtr DeviceImpl,
    ur_exp_command_buffer_handle_t CommandBuffer,
    const CGExecKernel &CommandGroup,
    std::vector<ur_exp_command_buffer_sync_point_t> &SyncPoints,
    ur_exp_command_buffer_sync_point_t *OutSyncPoint,
    ur_exp_command_buffer_command_handle_t *OutCommand,
    const std::function<void *(Requirement *Req)> &getMemAllocationFunc) {
  // List of ur objects to be released after UR call. We don't do anything
  // with the ur_program_handle_t objects, but need to update their reference
  // count.
  std::vector<ur_kernel_handle_t> UrKernelsToRelease;
  std::vector<ur_program_handle_t> UrProgramsToRelease;

  ur_kernel_handle_t UrKernel = nullptr;
  std::shared_ptr<device_image_impl> DeviceImageImpl = nullptr;
  const KernelArgMask *EliminatedArgMask = nullptr;

  auto ContextImpl = sycl::detail::getSyclObjImpl(Ctx);
  std::tie(UrKernel, DeviceImageImpl, EliminatedArgMask) =
      getCGKernelInfo(CommandGroup, ContextImpl, DeviceImpl, UrKernelsToRelease,
                      UrProgramsToRelease);

  // Build up the list of UR kernel handles that the UR command could be
  // updated to use.
  std::vector<ur_kernel_handle_t> AltUrKernels;
  const std::vector<std::weak_ptr<sycl::detail::CGExecKernel>>
      &AlternativeKernels = CommandGroup.MAlternativeKernels;
  for (const auto &AltCGKernelWP : AlternativeKernels) {
    auto AltCGKernel = AltCGKernelWP.lock();
    assert(AltCGKernel != nullptr);

    ur_kernel_handle_t AltUrKernel = nullptr;
    std::tie(AltUrKernel, std::ignore, std::ignore) =
        getCGKernelInfo(*AltCGKernel.get(), ContextImpl, DeviceImpl,
                        UrKernelsToRelease, UrProgramsToRelease);
    AltUrKernels.push_back(AltUrKernel);
  }

  const sycl::detail::AdapterPtr &Adapter = ContextImpl->getAdapter();
  auto SetFunc = [&Adapter, &UrKernel, &DeviceImageImpl, &Ctx,
                  &getMemAllocationFunc](sycl::detail::ArgDesc &Arg,
                                         size_t NextTrueIndex) {
    sycl::detail::SetArgBasedOnType(Adapter, UrKernel, DeviceImageImpl,
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
    Adapter->call<UrApiKind::urKernelGetGroupInfo>(
        UrKernel, DeviceImpl->getHandleRef(),
        UR_KERNEL_GROUP_INFO_COMPILE_WORK_GROUP_SIZE, sizeof(RequiredWGSize),
        RequiredWGSize,
        /* pPropSizeRet = */ nullptr);

    const bool EnforcedLocalSize =
        (RequiredWGSize[0] != 0 || RequiredWGSize[1] != 0 ||
         RequiredWGSize[2] != 0);
    if (EnforcedLocalSize)
      LocalSize = RequiredWGSize;
  }

  // Command-buffers which are not updatable cannot return command handles, so
  // we query the descriptor here to check if a handle is required.
  ur_exp_command_buffer_desc_t CommandBufferDesc{};

  Adapter->call<UrApiKind::urCommandBufferGetInfoExp>(
      CommandBuffer,
      ur_exp_command_buffer_info_t::UR_EXP_COMMAND_BUFFER_INFO_DESCRIPTOR,
      sizeof(ur_exp_command_buffer_desc_t), &CommandBufferDesc, nullptr);

  ur_result_t Res =
      Adapter->call_nocheck<UrApiKind::urCommandBufferAppendKernelLaunchExp>(
          CommandBuffer, UrKernel, NDRDesc.Dims, &NDRDesc.GlobalOffset[0],
          &NDRDesc.GlobalSize[0], LocalSize, AltUrKernels.size(),
          AltUrKernels.size() ? AltUrKernels.data() : nullptr,
          SyncPoints.size(), SyncPoints.size() ? SyncPoints.data() : nullptr, 0,
          nullptr, OutSyncPoint, nullptr,
          CommandBufferDesc.isUpdatable ? OutCommand : nullptr);

  for (auto &Kernel : UrKernelsToRelease) {
    Adapter->call<UrApiKind::urKernelRelease>(Kernel);
  }
  for (auto &Program : UrProgramsToRelease) {
    Adapter->call<UrApiKind::urProgramRelease>(Program);
  }

  if (Res != UR_RESULT_SUCCESS) {
    const device_impl &DeviceImplem = *(DeviceImpl);
    detail::enqueue_kernel_launch::handleErrorOrWarning(Res, DeviceImplem,
                                                        UrKernel, NDRDesc);
  }

  return Res;
}

void enqueueImpKernel(
    const QueueImplPtr &Queue, NDRDescT &NDRDesc, std::vector<ArgDesc> &Args,
    const std::shared_ptr<detail::kernel_bundle_impl> &KernelBundleImplPtr,
    const std::shared_ptr<detail::kernel_impl> &MSyclKernel,
    const std::string &KernelName, std::vector<ur_event_handle_t> &RawEvents,
    const detail::EventImplPtr &OutEventImpl,
    const std::function<void *(Requirement *Req)> &getMemAllocationFunc,
    ur_kernel_cache_config_t KernelCacheConfig, const bool KernelIsCooperative,
    const bool KernelUsesClusterLaunch, const RTDeviceBinaryImage *BinImage) {
  assert(Queue && "Kernel submissions should have an associated queue");
  // Run OpenCL kernel
  auto ContextImpl = Queue->getContextImplPtr();
  auto DeviceImpl = Queue->getDeviceImplPtr();
  ur_kernel_handle_t Kernel = nullptr;
  std::mutex *KernelMutex = nullptr;
  ur_program_handle_t Program = nullptr;
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

    Program = DeviceImageImpl->get_ur_program_ref();

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
    // we need to either store and use a ur_native_handle_t -> mutex map or
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
  std::vector<ur_event_handle_t> &EventsWaitList = RawEvents;

  // Initialize device globals associated with this.
  std::vector<ur_event_handle_t> DeviceGlobalInitEvents =
      ContextImpl->initializeDeviceGlobals(Program, Queue);
  std::vector<ur_event_handle_t> EventsWithDeviceGlobalInits;
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

  ur_result_t Error = UR_RESULT_SUCCESS;
  {
    // When KernelMutex is null, this means that in-memory caching is
    // disabled, which means that kernel object is not shared, so no locking
    // is necessary.
    using LockT = std::unique_lock<std::mutex>;
    auto Lock = KernelMutex ? LockT(*KernelMutex) : LockT();

    // Set SLM/Cache configuration for the kernel if non-default value is
    // provided.
    if (KernelCacheConfig == UR_KERNEL_CACHE_CONFIG_LARGE_SLM ||
        KernelCacheConfig == UR_KERNEL_CACHE_CONFIG_LARGE_DATA) {
      const AdapterPtr &Adapter = Queue->getAdapter();
      Adapter->call<UrApiKind::urKernelSetExecInfo>(
          Kernel, UR_KERNEL_EXEC_INFO_CACHE_CONFIG,
          sizeof(ur_kernel_cache_config_t), nullptr, &KernelCacheConfig);
    }

    Error = SetKernelParamsAndLaunch(
        Queue, Args, DeviceImageImpl, Kernel, NDRDesc, EventsWaitList,
        OutEventImpl, EliminatedArgMask, getMemAllocationFunc,
        KernelIsCooperative, KernelUsesClusterLaunch, BinImage, KernelName);

    const AdapterPtr &Adapter = Queue->getAdapter();
    if (!SyclKernelImpl && !MSyclKernel) {
      Adapter->call<UrApiKind::urKernelRelease>(Kernel);
      Adapter->call<UrApiKind::urProgramRelease>(Program);
    }
  }
  if (UR_RESULT_SUCCESS != Error) {
    // If we have got non-success error code, let's analyze it to emit nice
    // exception explaining what was wrong
    const device_impl &DeviceImpl = *(Queue->getDeviceImplPtr());
    detail::enqueue_kernel_launch::handleErrorOrWarning(Error, DeviceImpl,
                                                        Kernel, NDRDesc);
  }
}

ur_result_t enqueueReadWriteHostPipe(const QueueImplPtr &Queue,
                                     const std::string &PipeName, bool blocking,
                                     void *ptr, size_t size,
                                     std::vector<ur_event_handle_t> &RawEvents,
                                     const detail::EventImplPtr &OutEventImpl,
                                     bool read) {
  assert(Queue &&
         "ReadWrite host pipe submissions should have an associated queue");
  detail::HostPipeMapEntry *hostPipeEntry =
      ProgramManager::getInstance().getHostPipeEntry(PipeName);

  ur_program_handle_t Program = nullptr;
  device Device = Queue->get_device();
  ContextImplPtr ContextImpl = Queue->getContextImplPtr();
  std::optional<ur_program_handle_t> CachedProgram =
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
    Program = getSyclObjImpl(BuiltImage)->get_ur_program_ref();
  }
  assert(Program && "Program for this hostpipe is not compiled.");

  const AdapterPtr &Adapter = Queue->getAdapter();

  ur_queue_handle_t ur_q = Queue->getHandleRef();
  ur_result_t Error;

  ur_event_handle_t UREvent = nullptr;
  auto OutEvent = OutEventImpl ? &UREvent : nullptr;
  if (OutEventImpl != nullptr)
    OutEventImpl->setHostEnqueueTime();
  if (read) {
    Error = Adapter->call_nocheck<UrApiKind::urEnqueueReadHostPipe>(
        ur_q, Program, PipeName.c_str(), blocking, ptr, size, RawEvents.size(),
        RawEvents.empty() ? nullptr : &RawEvents[0], OutEvent);
  } else {
    Error = Adapter->call_nocheck<UrApiKind::urEnqueueWriteHostPipe>(
        ur_q, Program, PipeName.c_str(), blocking, ptr, size, RawEvents.size(),
        RawEvents.empty() ? nullptr : &RawEvents[0], OutEvent);
  }
  if (Error == UR_RESULT_SUCCESS && OutEventImpl) {
    OutEventImpl->setHandle(UREvent);
  }
  return Error;
}

ur_result_t ExecCGCommand::enqueueImpCommandBuffer() {
  assert(MQueue && "Command buffer enqueue should have an associated queue");
  // Wait on host command dependencies
  waitForPreparedHostEvents();

  // Any device dependencies need to be waited on here since subsequent
  // submissions of the command buffer itself will not receive dependencies on
  // them, e.g. initial copies from host to device
  std::vector<EventImplPtr> EventImpls = MPreparedDepsEvents;
  flushCrossQueueDeps(EventImpls, MWorkerQueue);
  std::vector<ur_event_handle_t> RawEvents = getUrEvents(EventImpls);
  if (!RawEvents.empty()) {
    MQueue->getAdapter()->call<UrApiKind::urEventWait>(RawEvents.size(),
                                                       &RawEvents[0]);
  }

  ur_exp_command_buffer_sync_point_t OutSyncPoint;
  ur_exp_command_buffer_command_handle_t OutCommand = nullptr;
  switch (MCommandGroup->getType()) {
  case CGType::Kernel: {
    CGExecKernel *ExecKernel = (CGExecKernel *)MCommandGroup.get();

    auto getMemAllocationFunc = [this](Requirement *Req) {
      AllocaCommandBase *AllocaCmd = getAllocaForReq(Req);
      return AllocaCmd->getMemAllocation();
    };

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
    if (auto Result = callMemOpHelper(
            MemoryManager::ext_oneapi_copy_usm_cmd_buffer,
            MQueue->getContextImplPtr(), Copy->getSrc(), MCommandBuffer,
            Copy->getLength(), Copy->getDst(), MSyncPointDeps, &OutSyncPoint);
        Result != UR_RESULT_SUCCESS)
      return Result;

    MEvent->setSyncPoint(OutSyncPoint);
    return UR_RESULT_SUCCESS;
  }
  case CGType::CopyAccToAcc: {
    CGCopy *Copy = (CGCopy *)MCommandGroup.get();
    Requirement *ReqSrc = (Requirement *)(Copy->getSrc());
    Requirement *ReqDst = (Requirement *)(Copy->getDst());

    AllocaCommandBase *AllocaCmdSrc = getAllocaForReq(ReqSrc);
    AllocaCommandBase *AllocaCmdDst = getAllocaForReq(ReqDst);

    if (auto Result = callMemOpHelper(
            MemoryManager::ext_oneapi_copyD2D_cmd_buffer,
            MQueue->getContextImplPtr(), MCommandBuffer,
            AllocaCmdSrc->getSYCLMemObj(), AllocaCmdSrc->getMemAllocation(),
            ReqSrc->MDims, ReqSrc->MMemoryRange, ReqSrc->MAccessRange,
            ReqSrc->MOffset, ReqSrc->MElemSize,
            AllocaCmdDst->getMemAllocation(), ReqDst->MDims,
            ReqDst->MMemoryRange, ReqDst->MAccessRange, ReqDst->MOffset,
            ReqDst->MElemSize, std::move(MSyncPointDeps), &OutSyncPoint);
        Result != UR_RESULT_SUCCESS)
      return Result;

    MEvent->setSyncPoint(OutSyncPoint);
    return UR_RESULT_SUCCESS;
  }
  case CGType::CopyAccToPtr: {
    CGCopy *Copy = (CGCopy *)MCommandGroup.get();
    Requirement *Req = (Requirement *)Copy->getSrc();
    AllocaCommandBase *AllocaCmd = getAllocaForReq(Req);

    if (auto Result = callMemOpHelper(
            MemoryManager::ext_oneapi_copyD2H_cmd_buffer,
            MQueue->getContextImplPtr(), MCommandBuffer,
            AllocaCmd->getSYCLMemObj(), AllocaCmd->getMemAllocation(),
            Req->MDims, Req->MMemoryRange, Req->MAccessRange, Req->MOffset,
            Req->MElemSize, (char *)Copy->getDst(), Req->MDims,
            Req->MAccessRange,
            /*DstOffset=*/sycl::id<3>{0, 0, 0}, Req->MElemSize,
            std::move(MSyncPointDeps), &OutSyncPoint);
        Result != UR_RESULT_SUCCESS)
      return Result;

    MEvent->setSyncPoint(OutSyncPoint);
    return UR_RESULT_SUCCESS;
  }
  case CGType::CopyPtrToAcc: {
    CGCopy *Copy = (CGCopy *)MCommandGroup.get();
    Requirement *Req = (Requirement *)(Copy->getDst());
    AllocaCommandBase *AllocaCmd = getAllocaForReq(Req);

    if (auto Result = callMemOpHelper(
            MemoryManager::ext_oneapi_copyH2D_cmd_buffer,
            MQueue->getContextImplPtr(), MCommandBuffer,
            AllocaCmd->getSYCLMemObj(), (char *)Copy->getSrc(), Req->MDims,
            Req->MAccessRange,
            /*SrcOffset*/ sycl::id<3>{0, 0, 0}, Req->MElemSize,
            AllocaCmd->getMemAllocation(), Req->MDims, Req->MMemoryRange,
            Req->MAccessRange, Req->MOffset, Req->MElemSize,
            std::move(MSyncPointDeps), &OutSyncPoint);
        Result != UR_RESULT_SUCCESS)
      return Result;

    MEvent->setSyncPoint(OutSyncPoint);
    return UR_RESULT_SUCCESS;
  }
  case CGType::Fill: {
    CGFill *Fill = (CGFill *)MCommandGroup.get();
    Requirement *Req = (Requirement *)(Fill->getReqToFill());
    AllocaCommandBase *AllocaCmd = getAllocaForReq(Req);

    if (auto Result = callMemOpHelper(
            MemoryManager::ext_oneapi_fill_cmd_buffer,
            MQueue->getContextImplPtr(), MCommandBuffer,
            AllocaCmd->getSYCLMemObj(), AllocaCmd->getMemAllocation(),
            Fill->MPattern.size(), Fill->MPattern.data(), Req->MDims,
            Req->MMemoryRange, Req->MAccessRange, Req->MOffset, Req->MElemSize,
            std::move(MSyncPointDeps), &OutSyncPoint);
        Result != UR_RESULT_SUCCESS)
      return Result;

    MEvent->setSyncPoint(OutSyncPoint);
    return UR_RESULT_SUCCESS;
  }
  case CGType::FillUSM: {
    CGFillUSM *Fill = (CGFillUSM *)MCommandGroup.get();
    if (auto Result = callMemOpHelper(
            MemoryManager::ext_oneapi_fill_usm_cmd_buffer,
            MQueue->getContextImplPtr(), MCommandBuffer, Fill->getDst(),
            Fill->getLength(), Fill->getPattern(), std::move(MSyncPointDeps),
            &OutSyncPoint);
        Result != UR_RESULT_SUCCESS)
      return Result;

    MEvent->setSyncPoint(OutSyncPoint);
    return UR_RESULT_SUCCESS;
  }
  case CGType::PrefetchUSM: {
    CGPrefetchUSM *Prefetch = (CGPrefetchUSM *)MCommandGroup.get();
    if (auto Result = callMemOpHelper(
            MemoryManager::ext_oneapi_prefetch_usm_cmd_buffer,
            MQueue->getContextImplPtr(), MCommandBuffer, Prefetch->getDst(),
            Prefetch->getLength(), std::move(MSyncPointDeps), &OutSyncPoint);
        Result != UR_RESULT_SUCCESS)
      return Result;

    MEvent->setSyncPoint(OutSyncPoint);
    return UR_RESULT_SUCCESS;
  }
  case CGType::AdviseUSM: {
    CGAdviseUSM *Advise = (CGAdviseUSM *)MCommandGroup.get();
    if (auto Result = callMemOpHelper(
            MemoryManager::ext_oneapi_advise_usm_cmd_buffer,
            MQueue->getContextImplPtr(), MCommandBuffer, Advise->getDst(),
            Advise->getLength(), Advise->getAdvice(), std::move(MSyncPointDeps),
            &OutSyncPoint);
        Result != UR_RESULT_SUCCESS)
      return Result;

    MEvent->setSyncPoint(OutSyncPoint);
    return UR_RESULT_SUCCESS;
  }

  default:
    throw exception(make_error_code(errc::runtime),
                    "CG type not implemented for command buffers.");
  }
}

ur_result_t ExecCGCommand::enqueueImp() {
  if (MCommandBuffer) {
    return enqueueImpCommandBuffer();
  } else {
    return enqueueImpQueue();
  }
}

ur_result_t ExecCGCommand::enqueueImpQueue() {
  if (getCG().getType() != CGType::CodeplayHostTask)
    waitForPreparedHostEvents();
  std::vector<EventImplPtr> EventImpls = MPreparedDepsEvents;
  auto RawEvents = getUrEvents(EventImpls);
  flushCrossQueueDeps(EventImpls, MWorkerQueue);

  // We can omit creating a UR event and create a "discarded" event if either
  // the queue has the discard property or the command has been explicitly
  // marked as not needing an event, e.g. if the user did not ask for one, and
  // if the queue supports discarded UR event and there are no requirements.
  bool DiscardUrEvent = MQueue && (MQueue->MDiscardEvents || !MEventNeeded) &&
                        MQueue->supportsDiscardingPiEvents() &&
                        MCommandGroup->getRequirements().size() == 0;

  ur_event_handle_t UREvent = nullptr;
  ur_event_handle_t *Event = DiscardUrEvent ? nullptr : &UREvent;
  detail::EventImplPtr EventImpl = DiscardUrEvent ? nullptr : MEvent;

  // If we are discarding the UR event, we also need to mark the result event.
  if (DiscardUrEvent)
    MEvent->setStateDiscarded();

  switch (MCommandGroup->getType()) {

  case CGType::UpdateHost: {
    throw sycl::exception(sycl::make_error_code(sycl::errc::runtime),
                          "Update host should be handled by the Scheduler. " +
                              codeToString(UR_RESULT_ERROR_INVALID_VALUE));
  }
  case CGType::CopyAccToPtr: {
    CGCopy *Copy = (CGCopy *)MCommandGroup.get();
    Requirement *Req = (Requirement *)Copy->getSrc();
    AllocaCommandBase *AllocaCmd = getAllocaForReq(Req);

    if (auto Result = callMemOpHelper(
            MemoryManager::copy, AllocaCmd->getSYCLMemObj(),
            AllocaCmd->getMemAllocation(), MQueue, Req->MDims,
            Req->MMemoryRange, Req->MAccessRange, Req->MOffset, Req->MElemSize,
            Copy->getDst(), nullptr, Req->MDims, Req->MAccessRange,
            Req->MAccessRange, /*DstOffset=*/sycl::id<3>{0, 0, 0},
            Req->MElemSize, std::move(RawEvents), UREvent, MEvent);
        Result != UR_RESULT_SUCCESS)
      return Result;

    MEvent->setHandle(UREvent);

    return UR_RESULT_SUCCESS;
  }
  case CGType::CopyPtrToAcc: {
    CGCopy *Copy = (CGCopy *)MCommandGroup.get();
    Requirement *Req = (Requirement *)(Copy->getDst());
    AllocaCommandBase *AllocaCmd = getAllocaForReq(Req);

    if (auto Result = callMemOpHelper(
            MemoryManager::copy, AllocaCmd->getSYCLMemObj(), Copy->getSrc(),
            nullptr, Req->MDims, Req->MAccessRange, Req->MAccessRange,
            /*SrcOffset*/ sycl::id<3>{0, 0, 0}, Req->MElemSize,
            AllocaCmd->getMemAllocation(), MQueue, Req->MDims,
            Req->MMemoryRange, Req->MAccessRange, Req->MOffset, Req->MElemSize,
            std::move(RawEvents), UREvent, MEvent);
        Result != UR_RESULT_SUCCESS)
      return Result;

    MEvent->setHandle(UREvent);
    return UR_RESULT_SUCCESS;
  }
  case CGType::CopyAccToAcc: {
    CGCopy *Copy = (CGCopy *)MCommandGroup.get();
    Requirement *ReqSrc = (Requirement *)(Copy->getSrc());
    Requirement *ReqDst = (Requirement *)(Copy->getDst());

    AllocaCommandBase *AllocaCmdSrc = getAllocaForReq(ReqSrc);
    AllocaCommandBase *AllocaCmdDst = getAllocaForReq(ReqDst);

    if (auto Result = callMemOpHelper(
            MemoryManager::copy, AllocaCmdSrc->getSYCLMemObj(),
            AllocaCmdSrc->getMemAllocation(), MQueue, ReqSrc->MDims,
            ReqSrc->MMemoryRange, ReqSrc->MAccessRange, ReqSrc->MOffset,
            ReqSrc->MElemSize, AllocaCmdDst->getMemAllocation(), MQueue,
            ReqDst->MDims, ReqDst->MMemoryRange, ReqDst->MAccessRange,
            ReqDst->MOffset, ReqDst->MElemSize, std::move(RawEvents), UREvent,
            MEvent);
        Result != UR_RESULT_SUCCESS)
      return Result;

    MEvent->setHandle(UREvent);
    return UR_RESULT_SUCCESS;
  }
  case CGType::Fill: {
    CGFill *Fill = (CGFill *)MCommandGroup.get();
    Requirement *Req = (Requirement *)(Fill->getReqToFill());
    AllocaCommandBase *AllocaCmd = getAllocaForReq(Req);

    if (auto Result = callMemOpHelper(
            MemoryManager::fill, AllocaCmd->getSYCLMemObj(),
            AllocaCmd->getMemAllocation(), MQueue, Fill->MPattern.size(),
            Fill->MPattern.data(), Req->MDims, Req->MMemoryRange,
            Req->MAccessRange, Req->MOffset, Req->MElemSize,
            std::move(RawEvents), UREvent, MEvent);
        Result != UR_RESULT_SUCCESS)
      return Result;

    MEvent->setHandle(UREvent);
    return UR_RESULT_SUCCESS;
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

    const RTDeviceBinaryImage *BinImage = nullptr;
    if (detail::SYCLConfig<detail::SYCL_JIT_AMDGCN_PTX_KERNELS>::get()) {
      std::tie(BinImage, std::ignore) =
          retrieveKernelBinary(MQueue, KernelName.c_str());
      assert(BinImage && "Failed to obtain a binary image.");
    }
    enqueueImpKernel(MQueue, NDRDesc, Args, ExecKernel->getKernelBundle(),
                     SyclKernel, KernelName, RawEvents, EventImpl,
                     getMemAllocationFunc, ExecKernel->MKernelCacheConfig,
                     ExecKernel->MKernelIsCooperative,
                     ExecKernel->MKernelUsesClusterLaunch, BinImage);

    return UR_RESULT_SUCCESS;
  }
  case CGType::CopyUSM: {
    CGCopyUSM *Copy = (CGCopyUSM *)MCommandGroup.get();
    if (auto Result = callMemOpHelper(MemoryManager::copy_usm, Copy->getSrc(),
                                      MQueue, Copy->getLength(), Copy->getDst(),
                                      std::move(RawEvents), Event, MEvent);
        Result != UR_RESULT_SUCCESS)
      return Result;

    if (Event)
      MEvent->setHandle(*Event);
    return UR_RESULT_SUCCESS;
  }
  case CGType::FillUSM: {
    CGFillUSM *Fill = (CGFillUSM *)MCommandGroup.get();
    if (auto Result = callMemOpHelper(
            MemoryManager::fill_usm, Fill->getDst(), MQueue, Fill->getLength(),
            Fill->getPattern(), std::move(RawEvents), Event, MEvent);
        Result != UR_RESULT_SUCCESS)
      return Result;

    if (Event)
      MEvent->setHandle(*Event);
    return UR_RESULT_SUCCESS;
  }
  case CGType::PrefetchUSM: {
    CGPrefetchUSM *Prefetch = (CGPrefetchUSM *)MCommandGroup.get();
    if (auto Result = callMemOpHelper(
            MemoryManager::prefetch_usm, Prefetch->getDst(), MQueue,
            Prefetch->getLength(), std::move(RawEvents), Event, MEvent);
        Result != UR_RESULT_SUCCESS)
      return Result;

    if (Event)
      MEvent->setHandle(*Event);
    return UR_RESULT_SUCCESS;
  }
  case CGType::AdviseUSM: {
    CGAdviseUSM *Advise = (CGAdviseUSM *)MCommandGroup.get();
    if (auto Result =
            callMemOpHelper(MemoryManager::advise_usm, Advise->getDst(), MQueue,
                            Advise->getLength(), Advise->getAdvice(),
                            std::move(RawEvents), Event, MEvent);
        Result != UR_RESULT_SUCCESS)
      return Result;

    if (Event)
      MEvent->setHandle(*Event);
    return UR_RESULT_SUCCESS;
  }
  case CGType::Copy2DUSM: {
    CGCopy2DUSM *Copy = (CGCopy2DUSM *)MCommandGroup.get();
    if (auto Result = callMemOpHelper(
            MemoryManager::copy_2d_usm, Copy->getSrc(), Copy->getSrcPitch(),
            MQueue, Copy->getDst(), Copy->getDstPitch(), Copy->getWidth(),
            Copy->getHeight(), std::move(RawEvents), Event, MEvent);
        Result != UR_RESULT_SUCCESS)
      return Result;

    if (Event)
      MEvent->setHandle(*Event);
    return UR_RESULT_SUCCESS;
  }
  case CGType::Fill2DUSM: {
    CGFill2DUSM *Fill = (CGFill2DUSM *)MCommandGroup.get();
    if (auto Result = callMemOpHelper(
            MemoryManager::fill_2d_usm, Fill->getDst(), MQueue,
            Fill->getPitch(), Fill->getWidth(), Fill->getHeight(),
            Fill->getPattern(), std::move(RawEvents), Event, MEvent);
        Result != UR_RESULT_SUCCESS)
      return Result;

    if (Event)
      MEvent->setHandle(*Event);
    return UR_RESULT_SUCCESS;
  }
  case CGType::Memset2DUSM: {
    CGMemset2DUSM *Memset = (CGMemset2DUSM *)MCommandGroup.get();
    if (auto Result = callMemOpHelper(
            MemoryManager::memset_2d_usm, Memset->getDst(), MQueue,
            Memset->getPitch(), Memset->getWidth(), Memset->getHeight(),
            Memset->getValue(), std::move(RawEvents), Event, MEvent);
        Result != UR_RESULT_SUCCESS)
      return Result;

    if (Event)
      MEvent->setHandle(*Event);
    return UR_RESULT_SUCCESS;
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
                                  codeToString(UR_RESULT_ERROR_INVALID_VALUE));
      }
    }

    std::vector<interop_handle::ReqToMem> ReqToMem;
    std::vector<ur_mem_handle_t> ReqUrMem;

    if (HostTask->MHostTask->isInteropTask()) {
      // Extract the Mem Objects for all Requirements, to ensure they are
      // available if a user asks for them inside the interop task scope
      const std::vector<Requirement *> &HandlerReq =
          HostTask->getRequirements();
      auto ReqToMemConv = [&ReqToMem, &ReqUrMem, HostTask](Requirement *Req) {
        const std::vector<AllocaCommandBase *> &AllocaCmds =
            Req->MSYCLMemObj->MRecord->MAllocaCommands;

        for (AllocaCommandBase *AllocaCmd : AllocaCmds)
          if (getContext(HostTask->MQueue) ==
              getContext(AllocaCmd->getQueue())) {
            auto MemArg = reinterpret_cast<ur_mem_handle_t>(
                AllocaCmd->getMemAllocation());
            ReqToMem.emplace_back(std::make_pair(Req, MemArg));
            ReqUrMem.emplace_back(MemArg);

            return;
          }

        assert(false &&
               "Can't get memory object due to no allocation available");

        throw sycl::exception(
            sycl::make_error_code(sycl::errc::runtime),
            "Can't get memory object due to no allocation available " +
                codeToString(UR_RESULT_ERROR_INVALID_MEM_OBJECT));
      };
      std::for_each(std::begin(HandlerReq), std::end(HandlerReq), ReqToMemConv);
      std::sort(std::begin(ReqToMem), std::end(ReqToMem));
    }

    // Host task is executed asynchronously so we should record where it was
    // submitted to report exception origin properly.
    copySubmissionCodeLocation();

    queue_impl::getThreadPool().submit<DispatchHostTask>(
        DispatchHostTask(this, std::move(ReqToMem), std::move(ReqUrMem)));

    MShouldCompleteEventIfPossible = false;

    return UR_RESULT_SUCCESS;
  }
  case CGType::EnqueueNativeCommand: {
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
                              "Unsupported arg type ");
      }
    }

    std::vector<interop_handle::ReqToMem> ReqToMem;
    std::vector<ur_mem_handle_t> ReqMems;

    if (HostTask->MHostTask->isInteropTask()) {
      // Extract the Mem Objects for all Requirements, to ensure they are
      // available if a user asks for them inside the interop task scope
      const std::vector<Requirement *> &HandlerReq =
          HostTask->getRequirements();
      auto ReqToMemConv = [&ReqToMem, &ReqMems, HostTask](Requirement *Req) {
        const std::vector<AllocaCommandBase *> &AllocaCmds =
            Req->MSYCLMemObj->MRecord->MAllocaCommands;

        for (AllocaCommandBase *AllocaCmd : AllocaCmds)
          if (getContext(HostTask->MQueue) ==
              getContext(AllocaCmd->getQueue())) {
            auto MemArg = reinterpret_cast<ur_mem_handle_t>(
                AllocaCmd->getMemAllocation());
            ReqToMem.emplace_back(std::make_pair(Req, MemArg));
            ReqMems.emplace_back(MemArg);

            return;
          }

        assert(false &&
               "Can't get memory object due to no allocation available");

        throw sycl::exception(
            sycl::make_error_code(sycl::errc::runtime),
            "Can't get memory object due to no allocation available " +
                codeToString(UR_RESULT_ERROR_INVALID_MEM_OBJECT));
      };
      std::for_each(std::begin(HandlerReq), std::end(HandlerReq), ReqToMemConv);
      std::sort(std::begin(ReqToMem), std::end(ReqToMem));
    }

    EnqueueNativeCommandData CustomOpData{
        interop_handle{ReqToMem, HostTask->MQueue,
                       HostTask->MQueue->getDeviceImplPtr(),
                       HostTask->MQueue->getContextImplPtr()},
        HostTask->MHostTask->MInteropTask};

    ur_bool_t NativeCommandSupport = false;
    assert(MQueue && "Native command should have an associated queue");
    MQueue->getAdapter()->call<UrApiKind::urDeviceGetInfo>(
        detail::getSyclObjImpl(MQueue->get_device())->getHandleRef(),
        UR_DEVICE_INFO_ENQUEUE_NATIVE_COMMAND_SUPPORT_EXP,
        sizeof(NativeCommandSupport), &NativeCommandSupport, nullptr);
    assert(NativeCommandSupport && "ext_codeplay_enqueue_native_command is not "
                                   "supported on this device");
    MQueue->getAdapter()->call<UrApiKind::urEnqueueNativeCommandExp>(
        MQueue->getHandleRef(), InteropFreeFunc, &CustomOpData, ReqMems.size(),
        ReqMems.data(), nullptr, RawEvents.size(), RawEvents.data(), Event);
    if (Event)
      MEvent->setHandle(*Event);
    return UR_RESULT_SUCCESS;
  }
  case CGType::Barrier: {
    assert(MQueue && "Barrier submission should have an associated queue");
    const AdapterPtr &Adapter = MQueue->getAdapter();
    if (MEvent != nullptr)
      MEvent->setHostEnqueueTime();
    Adapter->call<UrApiKind::urEnqueueEventsWaitWithBarrier>(
        MQueue->getHandleRef(), 0, nullptr, Event);
    if (Event)
      MEvent->setHandle(*Event);
    return UR_RESULT_SUCCESS;
  }
  case CGType::BarrierWaitlist: {
    assert(MQueue && "Barrier submission should have an associated queue");
    CGBarrier *Barrier = static_cast<CGBarrier *>(MCommandGroup.get());
    std::vector<detail::EventImplPtr> Events = Barrier->MEventsWaitWithBarrier;
    std::vector<ur_event_handle_t> UrEvents = getUrEventsBlocking(Events);
    if (UrEvents.empty()) {
      // If Events is empty, then the barrier has no effect.
      return UR_RESULT_SUCCESS;
    }
    const AdapterPtr &Adapter = MQueue->getAdapter();
    if (MEvent != nullptr)
      MEvent->setHostEnqueueTime();
    Adapter->call<UrApiKind::urEnqueueEventsWaitWithBarrier>(
        MQueue->getHandleRef(), UrEvents.size(), &UrEvents[0], Event);
    if (Event)
      MEvent->setHandle(*Event);
    return UR_RESULT_SUCCESS;
  }
  case CGType::ProfilingTag: {
    assert(MQueue && "Profiling tag requires a valid queue");
    const auto &Adapter = MQueue->getAdapter();

    bool IsInOrderQueue = MQueue->isInOrder();
    ur_event_handle_t *TimestampDeps = nullptr;
    size_t NumTimestampDeps = 0;

    // If the queue is not in-order, the implementation will need to first
    // insert a marker event that the timestamp waits for.
    ur_event_handle_t PreTimestampMarkerEvent{};
    if (!IsInOrderQueue) {
      // FIXME: urEnqueueEventsWait on the L0 adapter requires a double-release.
      //        Use that instead once it has been fixed.
      //        See https://github.com/oneapi-src/unified-runtime/issues/2347.
      Adapter->call<UrApiKind::urEnqueueEventsWaitWithBarrier>(
          MQueue->getHandleRef(),
          /*num_events_in_wait_list=*/0,
          /*event_wait_list=*/nullptr, &PreTimestampMarkerEvent);
      TimestampDeps = &PreTimestampMarkerEvent;
      NumTimestampDeps = 1;
    }

    Adapter->call<UrApiKind::urEnqueueTimestampRecordingExp>(
        MQueue->getHandleRef(),
        /*blocking=*/false, NumTimestampDeps, TimestampDeps, Event);

    // If the queue is not in-order, we need to insert a barrier. This barrier
    // does not need output events as it will implicitly enforce the following
    // enqueue is blocked until it finishes.
    if (!IsInOrderQueue) {
      // We also need to release the timestamp event from the marker.
      Adapter->call<UrApiKind::urEventRelease>(PreTimestampMarkerEvent);
      // FIXME: Due to a bug in the L0 UR adapter, we will leak events if we do
      //        not pass an output event to the UR call. Once that is fixed,
      //        this immediately-deleted event can be removed.
      ur_event_handle_t PostTimestampBarrierEvent{};
      Adapter->call<UrApiKind::urEnqueueEventsWaitWithBarrier>(
          MQueue->getHandleRef(),
          /*num_events_in_wait_list=*/0,
          /*event_wait_list=*/nullptr, &PostTimestampBarrierEvent);
      Adapter->call<UrApiKind::urEventRelease>(PostTimestampBarrierEvent);
    }

    if (Event)
      MEvent->setHandle(*Event);
    return UR_RESULT_SUCCESS;
  }
  case CGType::CopyToDeviceGlobal: {
    CGCopyToDeviceGlobal *Copy = (CGCopyToDeviceGlobal *)MCommandGroup.get();
    if (auto Result = callMemOpHelper(
            MemoryManager::copy_to_device_global, Copy->getDeviceGlobalPtr(),
            Copy->isDeviceImageScoped(), MQueue, Copy->getNumBytes(),
            Copy->getOffset(), Copy->getSrc(), std::move(RawEvents), Event,
            MEvent);
        Result != UR_RESULT_SUCCESS)
      return Result;

    if (Event)
      MEvent->setHandle(*Event);
    return UR_RESULT_SUCCESS;
  }
  case CGType::CopyFromDeviceGlobal: {
    CGCopyFromDeviceGlobal *Copy =
        (CGCopyFromDeviceGlobal *)MCommandGroup.get();
    if (auto Result = callMemOpHelper(
            MemoryManager::copy_from_device_global, Copy->getDeviceGlobalPtr(),
            Copy->isDeviceImageScoped(), MQueue, Copy->getNumBytes(),
            Copy->getOffset(), Copy->getDest(), std::move(RawEvents), Event,
            MEvent);
        Result != UR_RESULT_SUCCESS)
      return Result;

    if (Event)
      MEvent->setHandle(*Event);
    return UR_RESULT_SUCCESS;
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
    ur_result_t Err =
        MQueue->getAdapter()
            ->call_nocheck<UrApiKind::urCommandBufferEnqueueExp>(
                CmdBufferCG->MCommandBuffer, MQueue->getHandleRef(),
                RawEvents.size(), RawEvents.empty() ? nullptr : &RawEvents[0],
                Event);
    if (Event)
      MEvent->setHandle(*Event);

    return Err;
  }
  case CGType::CopyImage: {
    CGCopyImage *Copy = (CGCopyImage *)MCommandGroup.get();

    if (auto Result = callMemOpHelper(
            MemoryManager::copy_image_bindless, MQueue, Copy->getSrc(),
            Copy->getDst(), Copy->getSrcDesc(), Copy->getDstDesc(),
            Copy->getSrcFormat(), Copy->getDstFormat(), Copy->getCopyFlags(),
            Copy->getSrcOffset(), Copy->getDstOffset(), Copy->getCopyExtent(),
            std::move(RawEvents), Event);
        Result != UR_RESULT_SUCCESS)
      return Result;

    if (Event)
      MEvent->setHandle(*Event);

    return UR_RESULT_SUCCESS;
  }
  case CGType::SemaphoreWait: {
    assert(MQueue &&
           "Semaphore wait submissions should have an associated queue");
    CGSemaphoreWait *SemWait = (CGSemaphoreWait *)MCommandGroup.get();
    const detail::AdapterPtr &Adapter = MQueue->getAdapter();
    auto OptWaitValue = SemWait->getWaitValue();
    uint64_t WaitValue = OptWaitValue.has_value() ? OptWaitValue.value() : 0;
    Adapter->call<UrApiKind::urBindlessImagesWaitExternalSemaphoreExp>(
        MQueue->getHandleRef(), SemWait->getExternalSemaphore(),
        OptWaitValue.has_value(), WaitValue, 0, nullptr, nullptr);

    return UR_RESULT_SUCCESS;
  }
  case CGType::SemaphoreSignal: {
    assert(MQueue &&
           "Semaphore signal submissions should have an associated queue");
    CGSemaphoreSignal *SemSignal = (CGSemaphoreSignal *)MCommandGroup.get();
    const detail::AdapterPtr &Adapter = MQueue->getAdapter();
    auto OptSignalValue = SemSignal->getSignalValue();
    uint64_t SignalValue =
        OptSignalValue.has_value() ? OptSignalValue.value() : 0;
    Adapter->call<UrApiKind::urBindlessImagesSignalExternalSemaphoreExp>(
        MQueue->getHandleRef(), SemSignal->getExternalSemaphore(),
        OptSignalValue.has_value(), SignalValue, 0, nullptr, nullptr);

    return UR_RESULT_SUCCESS;
  }
  case CGType::None: {
    if (RawEvents.empty()) {
      // urEnqueueEventsWait with zero events acts like a barrier which is NOT
      // what we want here. On the other hand, there is nothing to wait for, so
      // we don't need to enqueue anything.
      return UR_RESULT_SUCCESS;
    }
    const detail::AdapterPtr &Adapter = MQueue->getAdapter();
    ur_event_handle_t Event;
    ur_result_t Result = Adapter->call_nocheck<UrApiKind::urEnqueueEventsWait>(
        MQueue->getHandleRef(), RawEvents.size(),
        RawEvents.size() ? &RawEvents[0] : nullptr, &Event);
    MEvent->setHandle(Event);
    return Result;
  }
  }
  return UR_RESULT_ERROR_INVALID_OPERATION;
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

UpdateCommandBufferCommand::UpdateCommandBufferCommand(
    QueueImplPtr Queue,
    ext::oneapi::experimental::detail::exec_graph_impl *Graph,
    std::vector<std::shared_ptr<ext::oneapi::experimental::detail::node_impl>>
        Nodes)
    : Command(CommandType::UPDATE_CMD_BUFFER, Queue), MGraph(Graph),
      MNodes(Nodes) {}

ur_result_t UpdateCommandBufferCommand::enqueueImp() {
  waitForPreparedHostEvents();
  std::vector<EventImplPtr> EventImpls = MPreparedDepsEvents;
  ur_event_handle_t UREvent = nullptr;
  Command::waitForEvents(MQueue, EventImpls, UREvent);
  MEvent->setHandle(UREvent);

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

  return UR_RESULT_SUCCESS;
}

void UpdateCommandBufferCommand::printDot(std::ostream &Stream) const {
  Stream << "\"" << this << "\" [style=filled, fillcolor=\"#8d8f29\", label=\"";

  Stream << "ID = " << this << "\\n";
  Stream << "CommandBuffer Command Update"
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

void UpdateCommandBufferCommand::emitInstrumentationData() {}
bool UpdateCommandBufferCommand::producesPiEvent() const { return false; }

} // namespace detail
} // namespace _V1
} // namespace sycl
