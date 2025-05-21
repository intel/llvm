//==------------------ queue_impl.hpp - SYCL queue -------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <detail/adapter.hpp>
#include <detail/config.hpp>
#include <detail/context_impl.hpp>
#include <detail/device_impl.hpp>
#include <detail/event_impl.hpp>
#include <detail/global_handler.hpp>
#include <detail/handler_impl.hpp>
#include <detail/kernel_impl.hpp>
#include <detail/scheduler/scheduler.hpp>
#include <detail/stream_impl.hpp>
#include <detail/thread_pool.hpp>
#include <sycl/context.hpp>
#include <sycl/detail/assert_happened.hpp>
#include <sycl/detail/ur.hpp>
#include <sycl/device.hpp>
#include <sycl/event.hpp>
#include <sycl/exception.hpp>
#include <sycl/exception_list.hpp>
#include <sycl/handler.hpp>
#include <sycl/properties/queue_properties.hpp>
#include <sycl/property_list.hpp>
#include <sycl/queue.hpp>

#include "detail/graph_impl.hpp"

#include <utility>

#ifdef XPTI_ENABLE_INSTRUMENTATION
#include "xpti/xpti_trace_framework.hpp"
#include <detail/xpti_registry.hpp>
#endif

namespace sycl {
inline namespace _V1 {

// forward declaration

namespace ext::oneapi::experimental::detail {
class graph_impl;
}

namespace detail {

using ContextImplPtr = std::shared_ptr<detail::context_impl>;

/// Sets max number of queues supported by FPGA RT.
static constexpr size_t MaxNumQueues = 256;

//// Possible CUDA context types supported by UR CUDA backend
/// TODO: Implement this as a property once there is an extension document
enum class CUDAContextT : char { primary, custom };

/// Default context type created for CUDA backend
constexpr CUDAContextT DefaultContextType = CUDAContextT::custom;

enum QueueOrder { Ordered, OOO };

// Implementation of the submission information storage.
struct SubmissionInfoImpl {
  optional<detail::SubmitPostProcessF> MPostProcessorFunc = std::nullopt;
  std::shared_ptr<detail::queue_impl> MSecondaryQueue = nullptr;
  ext::oneapi::experimental::event_mode_enum MEventMode =
      ext::oneapi::experimental::event_mode_enum::none;
};

class queue_impl {
public:
  // \return a default context for the platform if it includes the device
  // passed and default contexts are enabled, a new context otherwise.
  static ContextImplPtr getDefaultOrNew(device_impl &Device) {
    if (!SYCLConfig<SYCL_ENABLE_DEFAULT_CONTEXTS>::get())
      return detail::getSyclObjImpl(
          context{createSyclObjFromImpl<device>(Device), {}, {}});

    ContextImplPtr DefaultContext =
        detail::getSyclObjImpl(Device.get_platform().khr_get_default_context());
    if (DefaultContext->isDeviceValid(Device))
      return DefaultContext;
    return detail::getSyclObjImpl(
        context{createSyclObjFromImpl<device>(Device), {}, {}});
  }
  /// Constructs a SYCL queue from a device using an async_handler and
  /// property_list provided.
  ///
  /// \param Device is a SYCL device that is used to dispatch tasks submitted
  /// to the queue.
  /// \param AsyncHandler is a SYCL asynchronous exception handler.
  /// \param PropList is a list of properties to use for queue construction.
  queue_impl(device_impl &Device, const async_handler &AsyncHandler,
             const property_list &PropList)
      : queue_impl(Device, getDefaultOrNew(Device), AsyncHandler, PropList) {};

  /// Constructs a SYCL queue with an async_handler and property_list provided
  /// form a device and a context.
  ///
  /// \param Device is a SYCL device that is used to dispatch tasks submitted
  /// to the queue.
  /// \param Context is a SYCL context to associate with the queue being
  /// constructed.
  /// \param AsyncHandler is a SYCL asynchronous exception handler.
  /// \param PropList is a list of properties to use for queue construction.
  queue_impl(device_impl &Device, const ContextImplPtr &Context,
             const async_handler &AsyncHandler, const property_list &PropList)
      : MDevice(Device), MContext(Context), MAsyncHandler(AsyncHandler),
        MPropList(PropList),
        MIsInorder(has_property<property::queue::in_order>()),
        MIsProfilingEnabled(has_property<property::queue::enable_profiling>()),
        MQueueID{
            MNextAvailableQueueID.fetch_add(1, std::memory_order_relaxed)} {
    verifyProps(PropList);
    if (has_property<property::queue::enable_profiling>()) {
      if (!MDevice.has(aspect::queue_profiling)) {
        throw sycl::exception(make_error_code(errc::feature_not_supported),
                              "Cannot enable profiling, the associated device "
                              "does not have the queue_profiling aspect");
      }
    }
    if (has_property<ext::intel::property::queue::compute_index>()) {
      int Idx = get_property<ext::intel::property::queue::compute_index>()
                    .get_index();
      int NumIndices =
          createSyclObjFromImpl<device>(Device)
              .get_info<ext::intel::info::device::max_compute_queue_indices>();
      if (Idx < 0 || Idx >= NumIndices)
        throw sycl::exception(
            make_error_code(errc::invalid),
            "Queue compute index must be a non-negative number less than "
            "device's number of available compute queue indices.");
    }
    if (!Context->isDeviceValid(Device)) {
      if (Context->getBackend() == backend::opencl)
        throw sycl::exception(
            make_error_code(errc::invalid),
            "Queue cannot be constructed with the given context and device "
            "since the device is not a member of the context (descendants of "
            "devices from the context are not supported on OpenCL yet).");
      throw sycl::exception(
          make_error_code(errc::invalid),
          "Queue cannot be constructed with the given context and device "
          "since the device is neither a member of the context nor a "
          "descendant of its member.");
    }
    const QueueOrder QOrder =
        MIsInorder ? QueueOrder::Ordered : QueueOrder::OOO;
    MQueue = createQueue(QOrder);
    // This section is the second part of the instrumentation that uses the
    // tracepoint information and notifies

    // We enable XPTI tracing events using the TLS mechanism; if the code
    // location data is available, then the tracing data will be rich.
#if XPTI_ENABLE_INSTRUMENTATION
    // Emit a trace event for queue creation; we currently do not get code
    // location information, so all queueus will have the same UID with a
    // different instance ID until this gets added.
    constructorNotification();
#endif

    trySwitchingToNoEventsMode();
  }

  sycl::detail::optional<event>
  getLastEvent(const std::shared_ptr<queue_impl> &Self);

public:
  /// Constructs a SYCL queue from adapter interoperability handle.
  ///
  /// \param UrQueue is a raw UR queue handle.
  /// \param Context is a SYCL context to associate with the queue being
  /// constructed.
  /// \param AsyncHandler is a SYCL asynchronous exception handler.
  queue_impl(ur_queue_handle_t UrQueue, const ContextImplPtr &Context,
             const async_handler &AsyncHandler)
      : queue_impl(UrQueue, Context, AsyncHandler, {}) {}

  /// Constructs a SYCL queue from adapter interoperability handle.
  ///
  /// \param UrQueue is a raw UR queue handle.
  /// \param Context is a SYCL context to associate with the queue being
  /// constructed.
  /// \param AsyncHandler is a SYCL asynchronous exception handler.
  /// \param PropList is the queue properties.
  queue_impl(ur_queue_handle_t UrQueue, const ContextImplPtr &Context,
             const async_handler &AsyncHandler, const property_list &PropList)
      : MDevice([&]() -> device_impl & {
          ur_device_handle_t DeviceUr{};
          const AdapterPtr &Adapter = Context->getAdapter();
          // TODO catch an exception and put it to list of asynchronous
          // exceptions
          Adapter->call<UrApiKind::urQueueGetInfo>(
              UrQueue, UR_QUEUE_INFO_DEVICE, sizeof(DeviceUr), &DeviceUr,
              nullptr);
          device_impl *Device = Context->findMatchingDeviceImpl(DeviceUr);
          if (Device == nullptr) {
            throw sycl::exception(
                make_error_code(errc::invalid),
                "Device provided by native Queue not found in Context.");
          }
          return *Device;
        }()),
        MContext(Context), MAsyncHandler(AsyncHandler), MPropList(PropList),
        MQueue(UrQueue), MIsInorder(has_property<property::queue::in_order>()),
        MIsProfilingEnabled(has_property<property::queue::enable_profiling>()),
        MQueueID{
            MNextAvailableQueueID.fetch_add(1, std::memory_order_relaxed)} {
    verifyProps(PropList);
    if (has_property<ext::oneapi::property::queue::discard_events>() &&
        has_property<property::queue::enable_profiling>()) {
      throw sycl::exception(make_error_code(errc::invalid),
                            "Queue cannot be constructed with both of "
                            "discard_events and enable_profiling.");
    }

    // The following commented section provides a guideline on how to use the
    // TLS enabled mechanism to create a tracepoint and notify using XPTI. This
    // is the prolog section and the epilog section will initiate the
    // notification.
#if XPTI_ENABLE_INSTRUMENTATION
    // Emit a trace event for queue creation; we currently do not get code
    // location information, so all queueus will have the same UID with a
    // different instance ID until this gets added.
    constructorNotification();
#endif

    trySwitchingToNoEventsMode();
  }

  ~queue_impl() {
    try {
#if XPTI_ENABLE_INSTRUMENTATION
      // The trace event created in the constructor should be active through the
      // lifetime of the queue object as member variable. We will send a
      // notification and destroy the trace event for this queue.
      destructorNotification();
#endif
      throw_asynchronous();
      auto status =
          getAdapter()->call_nocheck<UrApiKind::urQueueRelease>(MQueue);
      // If loader is already closed, it'll return a not-initialized status
      // which the UR should convert to SUCCESS code. But that isn't always
      // working on Windows. This is a temporary workaround until that is fixed.
      // TODO: Remove this workaround when UR is fixed, and restore
      // ->call<>() instead of ->call_nocheck<>() above.
      if (status != UR_RESULT_SUCCESS &&
          status != UR_RESULT_ERROR_UNINITIALIZED) {
        __SYCL_CHECK_UR_CODE_NO_EXC(status);
      }
    } catch (std::exception &e) {
      __SYCL_REPORT_EXCEPTION_TO_STREAM("exception in ~queue_impl", e);
    }
  }

  /// \return an OpenCL interoperability queue handle.

  cl_command_queue get() {
    ur_native_handle_t nativeHandle = 0;
    getAdapter()->call<UrApiKind::urQueueGetNativeHandle>(MQueue, nullptr,
                                                          &nativeHandle);
    __SYCL_OCL_CALL(clRetainCommandQueue, ur::cast<cl_command_queue>(nativeHandle));
    return ur::cast<cl_command_queue>(nativeHandle);
  }

  /// \return an associated SYCL context.
  context get_context() const {
    return createSyclObjFromImpl<context>(MContext);
  }

  const AdapterPtr &getAdapter() const { return MContext->getAdapter(); }

  const ContextImplPtr &getContextImplPtr() const { return MContext; }

  device_impl &getDeviceImpl() const { return MDevice; }

  /// \return an associated SYCL device.
  device get_device() const { return createSyclObjFromImpl<device>(MDevice); }

  /// \return true if this queue allows for discarded events.
  bool supportsDiscardingPiEvents() const { return MIsInorder; }

  bool isInOrder() const { return MIsInorder; }

  /// Queries SYCL queue for information.
  ///
  /// The return type depends on information being queried.
  template <typename Param> typename Param::return_type get_info() const;

  /// Queries SYCL queue for SYCL backend-specific information.
  ///
  /// The return type depends on information being queried.
  template <typename Param>
  typename Param::return_type get_backend_info() const;

  /// Provides a hint to the backend to execute previously issued commands on
  /// this queue. Overrides normal batching behaviour. Note that this is merely
  /// a hint and not a guarantee.
  void flush() {
    if (MGraph.lock()) {
      throw sycl::exception(make_error_code(errc::invalid),
                            "flush cannot be called for a queue which is "
                            "recording to a command graph.");
    }
    getAdapter()->call<UrApiKind::urQueueFlush>(MQueue);
  }

  /// Submits a command group function object to the queue, in order to be
  /// scheduled for execution on the device.
  ///
  /// On a kernel error, this command group function object is then scheduled
  /// for execution on a secondary queue.
  ///
  /// \param CGF is a function object containing command group.
  /// \param Self is a shared_ptr to this queue.
  /// \param SecondQueue is a shared_ptr to the secondary queue.
  /// \param Loc is the code location of the submit call (default argument)
  /// \param StoreAdditionalInfo makes additional info be stored in event_impl
  /// \return a SYCL event object, which corresponds to the queue the command
  /// group is being enqueued on.
  event submit(const detail::type_erased_cgfo_ty &CGF,
               const std::shared_ptr<queue_impl> &Self,
               const std::shared_ptr<queue_impl> &SecondQueue,
               const detail::code_location &Loc, bool IsTopCodeLoc,
               const SubmitPostProcessF *PostProcess = nullptr) {
    event ResEvent;
    v1::SubmissionInfo SI{};
    SI.SecondaryQueue() = SecondQueue;
    if (PostProcess)
      SI.PostProcessorFunc() = *PostProcess;
    return submit_with_event(CGF, Self, SI, Loc, IsTopCodeLoc);
  }

  /// Submits a command group function object to the queue, in order to be
  /// scheduled for execution on the device.
  ///
  /// \param CGF is a function object containing command group.
  /// \param Self is a shared_ptr to this queue.
  /// \param SubmitInfo is additional optional information for the submission.
  /// \param Loc is the code location of the submit call (default argument)
  /// \param StoreAdditionalInfo makes additional info be stored in event_impl
  /// \return a SYCL event object for the submitted command group.
  event submit_with_event(const detail::type_erased_cgfo_ty &CGF,
                          const std::shared_ptr<queue_impl> &Self,
                          const v1::SubmissionInfo &SubmitInfo,
                          const detail::code_location &Loc, bool IsTopCodeLoc) {

    detail::EventImplPtr ResEvent =
        submit_impl(CGF, Self, SubmitInfo.SecondaryQueue().get(),
                    /*CallerNeedsEvent=*/true, Loc, IsTopCodeLoc, SubmitInfo);
    return createSyclObjFromImpl<event>(ResEvent);
  }

  void submit_without_event(const detail::type_erased_cgfo_ty &CGF,
                            const std::shared_ptr<queue_impl> &Self,
                            const v1::SubmissionInfo &SubmitInfo,
                            const detail::code_location &Loc,
                            bool IsTopCodeLoc) {
    submit_impl(CGF, Self, SubmitInfo.SecondaryQueue().get(),
                /*CallerNeedsEvent=*/false, Loc, IsTopCodeLoc, SubmitInfo);
  }

  /// Performs a blocking wait for the completion of all enqueued tasks in the
  /// queue.
  ///
  /// Synchronous errors will be reported through SYCL exceptions.
  /// @param Loc is the code location of the submit call (default argument)
  void wait(const detail::code_location &Loc = {});

  /// \return list of asynchronous exceptions occurred during execution.
  exception_list getExceptionList() const { return MExceptions; }

  /// @param Loc is the code location of the submit call (default argument)
  void wait_and_throw(const detail::code_location &Loc = {}) {
    wait(Loc);
    throw_asynchronous();
  }

  /// Performs a blocking wait for the completion of all enqueued tasks in the
  /// queue.
  ///
  /// Synchronous errors will be reported through SYCL exceptions.
  /// Asynchronous errors will be passed to the async_handler passed to the
  /// queue on construction. If no async_handler was provided then
  /// asynchronous exceptions will be lost.
  void throw_asynchronous() {
    if (!MAsyncHandler)
      return;

    exception_list Exceptions;
    {
      std::lock_guard<std::mutex> Lock(MMutex);
      std::swap(Exceptions, MExceptions);
    }
    // Unlock the mutex before calling user-provided handler to avoid
    // potential deadlock if the same queue is somehow referenced in the
    // handler.
    if (Exceptions.size())
      MAsyncHandler(std::move(Exceptions));
  }

  /// Creates UR properties array.
  ///
  /// \param PropList SYCL properties.
  /// \param Order specifies whether queue is in-order or out-of-order.
  /// \param Properties UR properties array created from SYCL properties.
  static ur_queue_flags_t createUrQueueFlags(const property_list &PropList,
                                             QueueOrder Order) {
    ur_queue_flags_t CreationFlags = 0;

    if (Order == QueueOrder::OOO) {
      CreationFlags = UR_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE;
    }
    if (PropList.has_property<property::queue::enable_profiling>()) {
      CreationFlags |= UR_QUEUE_FLAG_PROFILING_ENABLE;
    }
    if (PropList.has_property<
            ext::oneapi::cuda::property::queue::use_default_stream>()) {
      CreationFlags |= UR_QUEUE_FLAG_USE_DEFAULT_STREAM;
    }
    // Track that priority settings are not ambiguous.
    bool PrioritySeen = false;
    if (PropList
            .has_property<ext::oneapi::property::queue::priority_normal>()) {
      // Normal is the default priority, don't pass anything.
      PrioritySeen = true;
    }
    if (PropList.has_property<ext::oneapi::property::queue::priority_low>()) {
      if (PrioritySeen) {
        throw sycl::exception(
            make_error_code(errc::invalid),
            "Queue cannot be constructed with different priorities.");
      }
      CreationFlags |= UR_QUEUE_FLAG_PRIORITY_LOW;
      PrioritySeen = true;
    }
    if (PropList.has_property<ext::oneapi::property::queue::priority_high>()) {
      if (PrioritySeen) {
        throw sycl::exception(
            make_error_code(errc::invalid),
            "Queue cannot be constructed with different priorities.");
      }
      CreationFlags |= UR_QUEUE_FLAG_PRIORITY_HIGH;
    }
    {
      // Track that submission modes do not conflict.
      bool no_imm_cmdlist = PropList.has_property<
          ext::intel::property::queue::no_immediate_command_list>();
      bool imm_cmdlist = PropList.has_property<
          ext::intel::property::queue::immediate_command_list>();
      if (no_imm_cmdlist && imm_cmdlist) {
        throw sycl::exception(
            make_error_code(errc::invalid),
            "Queue cannot be constructed with conflicting submission modes.");
      }
      if (no_imm_cmdlist)
        CreationFlags |= UR_QUEUE_FLAG_SUBMISSION_BATCHED;
      if (imm_cmdlist)
        CreationFlags |= UR_QUEUE_FLAG_SUBMISSION_IMMEDIATE;
    }
    return CreationFlags;
  }

  /// Creates UR queue.
  ///
  /// \param Order specifies whether the queue being constructed as in-order
  /// or out-of-order.
  ur_queue_handle_t createQueue(QueueOrder Order) {
    ur_queue_handle_t Queue{};
    ur_context_handle_t Context = MContext->getHandleRef();
    ur_device_handle_t Device = MDevice.getHandleRef();
    const AdapterPtr &Adapter = getAdapter();
    /*
        sycl::detail::pi::PiQueueProperties Properties[] = {
            PI_QUEUE_FLAGS, createPiQueueProperties(MPropList, Order), 0, 0, 0};
        */
    ur_queue_properties_t Properties = {UR_STRUCTURE_TYPE_QUEUE_PROPERTIES,
                                        nullptr, 0};
    Properties.flags = createUrQueueFlags(MPropList, Order);
    ur_queue_index_properties_t IndexProperties = {
        UR_STRUCTURE_TYPE_QUEUE_INDEX_PROPERTIES, nullptr, 0};
    if (has_property<ext::intel::property::queue::compute_index>()) {
      IndexProperties.computeIndex =
          get_property<ext::intel::property::queue::compute_index>()
              .get_index();
      Properties.pNext = &IndexProperties;
    }
    Adapter->call<UrApiKind::urQueueCreate>(Context, Device, &Properties,
                                            &Queue);

    return Queue;
  }

  /// \return a raw UR queue handle. The returned handle is not retained. It
  /// is caller responsibility to make sure queue is still alive.
  ur_queue_handle_t &getHandleRef() { return MQueue; }

  /// \return true if the queue was constructed with property specified by
  /// PropertyT.
  template <typename propertyT> bool has_property() const noexcept {
    return MPropList.has_property<propertyT>();
  }

  /// \return a copy of the property of type PropertyT that the queue was
  /// constructed with. If the queue was not constructed with the PropertyT
  /// property, a SYCL exception with errc::invalid error code will be thrown.
  template <typename propertyT> propertyT get_property() const {
    return MPropList.get_property<propertyT>();
  }

  /// Fills the memory pointed by a USM pointer with the value specified.
  ///
  /// \param Self is a shared_ptr to this queue.
  /// \param Ptr is a USM pointer to the memory to fill.
  /// \param Value is a value to be set. Value is cast as an unsigned char.
  /// \param Count is a number of bytes to fill.
  /// \param DepEvents is a vector of events that specifies the kernel
  /// dependencies.
  /// \param CallerNeedsEvent specifies if the caller expects a usable event.
  /// \return an event representing fill operation.
  event memset(const std::shared_ptr<queue_impl> &Self, void *Ptr, int Value,
               size_t Count, const std::vector<event> &DepEvents,
               bool CallerNeedsEvent);
  /// Copies data from one memory region to another, both pointed by
  /// USM pointers.
  ///
  /// \param Self is a shared_ptr to this queue.
  /// \param Dest is a USM pointer to the destination memory.
  /// \param Src is a USM pointer to the source memory.
  /// \param Count is a number of bytes to copy.
  /// \param DepEvents is a vector of events that specifies the kernel
  /// dependencies.
  /// \param CallerNeedsEvent specifies if the caller expects a usable event.
  /// \return an event representing copy operation.
  event memcpy(const std::shared_ptr<queue_impl> &Self, void *Dest,
               const void *Src, size_t Count,
               const std::vector<event> &DepEvents, bool CallerNeedsEvent,
               const code_location &CodeLoc);
  /// Provides additional information to the underlying runtime about how
  /// different allocations are used.
  ///
  /// \param Self is a shared_ptr to this queue.
  /// \param Ptr is a USM pointer to the allocation.
  /// \param Length is a number of bytes in the allocation.
  /// \param Advice is a device-defined advice for the specified allocation.
  /// \param DepEvents is a vector of events that specifies the kernel
  /// dependencies.
  /// \param CallerNeedsEvent specifies if the caller expects a usable event.
  /// \return an event representing advise operation.
  event mem_advise(const std::shared_ptr<queue_impl> &Self, const void *Ptr,
                   size_t Length, ur_usm_advice_flags_t Advice,
                   const std::vector<event> &DepEvents, bool CallerNeedsEvent);

  /// Puts exception to the list of asynchronous ecxeptions.
  ///
  /// \param ExceptionPtr is a pointer to exception to be put.
  void reportAsyncException(const std::exception_ptr &ExceptionPtr) {
    std::lock_guard<std::mutex> Lock(MMutex);
    MExceptions.PushBack(ExceptionPtr);
  }

  static ThreadPool &getThreadPool() {
    return GlobalHandler::instance().getHostTaskThreadPool();
  }

  /// Gets the native handle of the SYCL queue.
  ///
  /// \return a native handle.
  ur_native_handle_t getNative(int32_t &NativeHandleDesc) const;

  void registerStreamServiceEvent(const EventImplPtr &Event) {
    std::lock_guard<std::mutex> Lock(MStreamsServiceEventsMutex);
    MStreamsServiceEvents.push_back(Event);
  }

  bool queue_empty() const;

  event memcpyToDeviceGlobal(const std::shared_ptr<queue_impl> &Self,
                             void *DeviceGlobalPtr, const void *Src,
                             bool IsDeviceImageScope, size_t NumBytes,
                             size_t Offset, const std::vector<event> &DepEvents,
                             bool CallerNeedsEvent);
  event memcpyFromDeviceGlobal(const std::shared_ptr<queue_impl> &Self,
                               void *Dest, const void *DeviceGlobalPtr,
                               bool IsDeviceImageScope, size_t NumBytes,
                               size_t Offset,
                               const std::vector<event> &DepEvents,
                               bool CallerNeedsEvent);

  void setCommandGraph(
      std::shared_ptr<ext::oneapi::experimental::detail::graph_impl> Graph) {
    std::lock_guard<std::mutex> Lock(MMutex);
    MGraph = Graph;
    MExtGraphDeps.reset();

    if (Graph) {
      MNoEventMode = false;
    } else {
      trySwitchingToNoEventsMode();
    }
  }

  std::shared_ptr<ext::oneapi::experimental::detail::graph_impl>
  getCommandGraph() const {
    return MGraph.lock();
  }

  bool hasCommandGraph() const { return !MGraph.expired(); }

  unsigned long long getQueueID() { return MQueueID; }

  void *getTraceEvent() { return MTraceEvent; }

  void setExternalEvent(const event &Event) {
    MInOrderExternalEvent.set([&](std::optional<event> &InOrderExternalEvent) {
      InOrderExternalEvent = Event;
    });
  }

  std::optional<event> popExternalEvent() {
    std::optional<event> Result = std::nullopt;

    MInOrderExternalEvent.unset(
        [&](std::optional<event> &InOrderExternalEvent) {
          std::swap(Result, InOrderExternalEvent);
        });
    return Result;
  }

  const std::vector<event> &
  getExtendDependencyList(const std::vector<event> &DepEvents,
                          std::vector<event> &MutableVec,
                          std::unique_lock<std::mutex> &QueueLock);

  // Called on host task completion that could block some kernels from enqueue.
  // Approach that tracks almost all tasks to provide barrier sync for both ur
  // tasks and host tasks is applicable for out of order queues only. Not needed
  // for in order ones.
  void revisitUnenqueuedCommandsState(const EventImplPtr &CompletedHostTask);

  static ContextImplPtr getContext(const QueueImplPtr &Queue) {
    return Queue ? Queue->getContextImplPtr() : nullptr;
  }

  // Must be called under MMutex protection
  void doUnenqueuedCommandCleanup(
      const std::shared_ptr<ext::oneapi::experimental::detail::graph_impl>
          &Graph);

  const property_list &getPropList() const { return MPropList; }

  /// Inserts a marker event at the end of the queue. Waiting for this marker
  /// will wait for the completion of all work in the queue at the time of the
  /// insertion, but will not act as a barrier unless the queue is in-order.
  EventImplPtr insertMarkerEvent(const std::shared_ptr<queue_impl> &Self) {
    auto ResEvent = std::make_shared<detail::event_impl>(Self);
    ur_event_handle_t UREvent = nullptr;
    getAdapter()->call<UrApiKind::urEnqueueEventsWait>(getHandleRef(), 0,
                                                       nullptr, &UREvent);
    ResEvent->setHandle(UREvent);
    return ResEvent;
  }

#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
  // CMPLRLLVM-66082
  // These methods are for accessing a member that should live in the
  // sycl::interop_handle class and will be moved on next ABI breaking window.
  ur_exp_command_buffer_handle_t getInteropGraph() const {
    return MInteropGraph;
  }

  void setInteropGraph(ur_exp_command_buffer_handle_t Graph) {
    MInteropGraph = Graph;
  }
#endif

protected:
  template <typename HandlerType = handler>
  EventImplPtr insertHelperBarrier(const HandlerType &Handler) {
    auto ResEvent = std::make_shared<detail::event_impl>(Handler.MQueue);
    ur_event_handle_t UREvent = nullptr;
    getAdapter()->call<UrApiKind::urEnqueueEventsWaitWithBarrier>(
        Handler.MQueue->getHandleRef(), 0, nullptr, &UREvent);
    ResEvent->setHandle(UREvent);
    return ResEvent;
  }

  template <typename HandlerType = handler>
  void synchronizeWithExternalEvent(HandlerType &Handler) {
    // If there is an external event set, add it as a dependency and clear it.
    // We do not need to hold the lock as MLastEventMtx will ensure the last
    // event reflects the corresponding external event dependence as well.
    std::optional<event> ExternalEvent = popExternalEvent();
    if (ExternalEvent)
      Handler.depends_on(*ExternalEvent);
  }

#ifdef __INTEL_PREVIEW_BREAKING_CHANGES
#define parseEvent(arg) (arg)
#else
  inline detail::EventImplPtr parseEvent(const event &Event) {
    const detail::EventImplPtr &EventImpl = getSyclObjImpl(Event);
    return EventImpl->isDiscarded() ? nullptr : EventImpl;
  }
#endif

  bool trySwitchingToNoEventsMode() {
    if (MNoEventMode.load(std::memory_order_relaxed))
      return true;

    if (!MGraph.expired() || !isInOrder())
      return false;

    if (MDefaultGraphDeps.LastEventPtr != nullptr &&
        !Scheduler::CheckEventReadiness(MContext,
                                        MDefaultGraphDeps.LastEventPtr))
      return false;

    MNoEventMode.store(true, std::memory_order_relaxed);
    MDefaultGraphDeps.LastEventPtr = nullptr;
    return true;
  }

  template <typename HandlerType = handler>
  detail::EventImplPtr
  finalizeHandlerInOrderNoEventsUnlocked(HandlerType &Handler) {
    assert(isInOrder());
    assert(MGraph.expired());
    assert(MDefaultGraphDeps.LastEventPtr == nullptr);
    assert(MNoEventMode);

    MEmpty = false;

    synchronizeWithExternalEvent(Handler);

    return parseEvent(Handler.finalize());
  }

  template <typename HandlerType = handler>
  detail::EventImplPtr
  finalizeHandlerInOrderHostTaskUnlocked(HandlerType &Handler) {
    assert(isInOrder());
    assert(Handler.getType() == CGType::CodeplayHostTask);

    auto &EventToBuildDeps = MGraph.expired() ? MDefaultGraphDeps.LastEventPtr
                                              : MExtGraphDeps.LastEventPtr;

    if (EventToBuildDeps && Handler.getType() != CGType::AsyncAlloc) {
      // We are not in no-event mode, so we can use the last event.
      // depends_on after an async alloc is explicitly disallowed. Async alloc
      // handles in order queue dependencies preemptively, so we skip them.
      // Note: This could be improved by moving the handling of dependencies
      // to before calling the CGF.
      Handler.depends_on(EventToBuildDeps);
    } else if (MNoEventMode) {
      // There might be some operations submitted to the queue
      // but the LastEventPtr is not set. If we are to run a host_task,
      // we need to insert a barrier to ensure proper synchronization.
      Handler.depends_on(insertHelperBarrier(Handler));
    }

    MEmpty = false;
    MNoEventMode = false;

    synchronizeWithExternalEvent(Handler);

    EventToBuildDeps = parseEvent(Handler.finalize());
    assert(EventToBuildDeps);
    return EventToBuildDeps;
  }

  template <typename HandlerType = handler>
  detail::EventImplPtr
  finalizeHandlerInOrderWithDepsUnlocked(HandlerType &Handler) {
    // this is handled by finalizeHandlerInOrderHostTask
    assert(Handler.getType() != CGType::CodeplayHostTask);

    if (Handler.getType() == CGType::ExecCommandBuffer && MNoEventMode) {
      // TODO: this shouldn't be needed but without this
      // the legacy adapter doesn't synchronize the operations properly
      // when non-immediate command lists are used.
      Handler.depends_on(insertHelperBarrier(Handler));
    }

    auto &EventToBuildDeps = MGraph.expired() ? MDefaultGraphDeps.LastEventPtr
                                              : MExtGraphDeps.LastEventPtr;

    // depends_on after an async alloc is explicitly disallowed. Async alloc
    // handles in order queue dependencies preemptively, so we skip them.
    // Note: This could be improved by moving the handling of dependencies
    // to before calling the CGF.
    if (EventToBuildDeps && Handler.getType() != CGType::AsyncAlloc) {
      // If we have last event, this means we are no longer in no-event mode.
      assert(!MNoEventMode);
      Handler.depends_on(EventToBuildDeps);
    }

    MEmpty = false;

    synchronizeWithExternalEvent(Handler);

    EventToBuildDeps = parseEvent(Handler.finalize());
    if (EventToBuildDeps)
      MNoEventMode = false;

    // TODO: if the event is NOP we should be able to discard it.
    // However, NOP events are used to describe ordering for graph operations
    // Once https://github.com/intel/llvm/issues/18330 is fixed, we can
    // start relying on command buffer in-order property instead.

    return EventToBuildDeps;
  }

  template <typename HandlerType = handler>
  detail::EventImplPtr finalizeHandlerOutOfOrder(HandlerType &Handler) {
    const CGType Type = getSyclObjImpl(Handler)->MCGType;
    std::lock_guard<std::mutex> Lock{MMutex};

    MEmpty = false;

    // The following code supports barrier synchronization if host task is
    // involved in the scenario. Native barriers cannot handle host task
    // dependency so in the case where some commands were not enqueued
    // (blocked), we track them to prevent barrier from being enqueued
    // earlier.
    MMissedCleanupRequests.unset(
        [&](MissedCleanupRequestsType &MissedCleanupRequests) {
          for (auto &UpdatedGraph : MissedCleanupRequests)
            doUnenqueuedCommandCleanup(UpdatedGraph);
          MissedCleanupRequests.clear();
        });
    auto &Deps = MGraph.expired() ? MDefaultGraphDeps : MExtGraphDeps;
    if (Type == CGType::Barrier && !Deps.UnenqueuedCmdEvents.empty()) {
      Handler.depends_on(Deps.UnenqueuedCmdEvents);
    }
    if (Deps.LastBarrier &&
        (Type == CGType::CodeplayHostTask || (!Deps.LastBarrier->isEnqueued())))
      Handler.depends_on(Deps.LastBarrier);

    EventImplPtr EventRetImpl = parseEvent(Handler.finalize());
    if (Type == CGType::CodeplayHostTask)
      Deps.UnenqueuedCmdEvents.push_back(EventRetImpl);
    else if (Type == CGType::Barrier || Type == CGType::BarrierWaitlist) {
      Deps.LastBarrier = EventRetImpl;
      Deps.UnenqueuedCmdEvents.clear();
    } else if (!EventRetImpl->isEnqueued()) {
      Deps.UnenqueuedCmdEvents.push_back(EventRetImpl);
    }

    return EventRetImpl;
  }

  template <typename HandlerType = handler>
  void handlerPostProcess(HandlerType &Handler,
                          const optional<SubmitPostProcessF> &PostProcessorFunc,
                          event &Event) {
    bool IsKernel = Handler.getType() == CGType::Kernel;
    bool KernelUsesAssert = false;

    if (IsKernel)
      // Kernel only uses assert if it's non interop one
      KernelUsesAssert =
          (!Handler.MKernel || Handler.MKernel->hasSYCLMetadata()) &&
          ProgramManager::getInstance().kernelUsesAssert(
              Handler.MKernelName.data());

    auto &PostProcess = *PostProcessorFunc;
    PostProcess(IsKernel, KernelUsesAssert, Event);
  }

#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
  /// Performs command group submission to the queue.
  ///
  /// \param CGF is a function object containing command group.
  /// \param Self is a pointer to this queue.
  /// \param PrimaryQueue is a pointer to the primary queue. This may be the
  ///        same as Self.
  /// \param SecondaryQueue is a pointer to the secondary queue. This may be the
  ///        same as Self.
  /// \param CallerNeedsEvent is a boolean indicating whether the event is
  ///        required by the user after the call.
  /// \param Loc is the code location of the submit call (default argument)
  /// \param SubmitInfo is additional optional information for the submission.
  /// \return a SYCL event representing submitted command group.
  detail::EventImplPtr
  submit_impl(const detail::type_erased_cgfo_ty &CGF,
              const std::shared_ptr<queue_impl> &Self,
              const std::shared_ptr<queue_impl> &PrimaryQueue,
              const std::shared_ptr<queue_impl> &SecondaryQueue,
              bool CallerNeedsEvent, const detail::code_location &Loc,
              bool IsTopCodeLoc, const SubmissionInfo &SubmitInfo);
#endif

  /// Performs command group submission to the queue.
  ///
  /// \param CGF is a function object containing command group.
  /// \param Self is a pointer to this queue.
  /// \param SecondaryQueue is a pointer to the secondary queue.
  /// \param CallerNeedsEvent is a boolean indicating whether the event is
  ///        required by the user after the call.
  /// \param Loc is the code location of the submit call (default argument)
  /// \param SubmitInfo is additional optional information for the submission.
  /// \return a SYCL event representing submitted command group.
  detail::EventImplPtr submit_impl(const detail::type_erased_cgfo_ty &CGF,
                                   const std::shared_ptr<queue_impl> &Self,
                                   queue_impl *SecondaryQueue,
                                   bool CallerNeedsEvent,
                                   const detail::code_location &Loc,
                                   bool IsTopCodeLoc,
                                   const v1::SubmissionInfo &SubmitInfo);

  /// Helper function for submitting a memory operation with a handler.
  /// \param Self is a shared_ptr to this queue.
  /// \param DepEvents is a vector of dependencies of the operation.
  /// \param HandlerFunc is a function that submits the operation with a
  ///        handler.
  template <typename HandlerFuncT>
  event submitWithHandler(const std::shared_ptr<queue_impl> &Self,
                          const std::vector<event> &DepEvents,
                          bool CallerNeedsEvent, HandlerFuncT HandlerFunc);

  /// Performs submission of a memory operation directly if scheduler can be
  /// bypassed, or with a handler otherwise.
  ///
  /// \param Self is a shared_ptr to this queue.
  /// \param DepEvents is a vector of dependencies of the operation.
  /// \param CallerNeedsEvent specifies if the caller needs an event from this
  ///        memory operation.
  /// \param HandlerFunc is a function that submits the operation with a
  ///        handler.
  /// \param MemMngrFunc is a function that forwards its arguments to the
  ///        appropriate memory manager function.
  /// \param MemMngrArgs are all the arguments that need to be passed to memory
  ///        manager except the last three: dependencies, UR event and
  ///        EventImplPtr are filled out by this helper.
  /// \return an event representing the submitted operation.
  template <typename HandlerFuncT, typename MemMngrFuncT,
            typename... MemMngrArgTs>
  event submitMemOpHelper(const std::shared_ptr<queue_impl> &Self,
                          const std::vector<event> &DepEvents,
                          bool CallerNeedsEvent, HandlerFuncT HandlerFunc,
                          MemMngrFuncT MemMngrFunc, MemMngrArgTs... MemOpArgs);

  // When instrumentation is enabled emits trace event for wait begin and
  // returns the telemetry event generated for the wait
  void *instrumentationProlog(const detail::code_location &CodeLoc,
                              std::string &Name, int32_t StreamID,
                              uint64_t &iid);
  // Uses events generated by the Prolog and emits wait done event
  void instrumentationEpilog(void *TelementryEvent, std::string &Name,
                             int32_t StreamID, uint64_t IId);

  // We need to emit a queue_create notification when a queue object is created
  void constructorNotification();

  // We need to emit a queue_destroy notification when a queue object is
  // destroyed
  void destructorNotification();

  /// queue_impl.addEvent tracks events with weak pointers
  /// but some events have no other owners. addSharedEvent()
  /// follows events with a shared pointer.
  ///
  /// \param Event is the event to be stored
  void addSharedEvent(const event &Event);

  /// Stores an event that should be associated with the queue
  ///
  /// \param EventImpl is the event to be stored
  void addEvent(const detail::EventImplPtr &EventImpl);

  /// Protects all the fields that can be changed by class' methods.
  mutable std::mutex MMutex;

  device_impl &MDevice;
  const ContextImplPtr MContext;

  /// These events are tracked, but not owned, by the queue.
  std::vector<std::weak_ptr<event_impl>> MEventsWeak;

  /// Events without data dependencies (such as USM) need an owner,
  /// additionally, USM operations are not added to the scheduler command graph,
  /// queue is the only owner on the runtime side.
  exception_list MExceptions;
  const async_handler MAsyncHandler;
  const property_list MPropList;

  /// List of queues created for FPGA device from a single SYCL queue.
  ur_queue_handle_t MQueue;

  // Access should be guarded with MMutex
  struct DependencyTrackingItems {
    // This event is employed for enhanced dependency tracking with in-order
    // queue
    EventImplPtr LastEventPtr;
    // The following two items are employed for proper out of order enqueue
    // ordering
    std::vector<EventImplPtr> UnenqueuedCmdEvents;
    EventImplPtr LastBarrier;

    void reset() {
      LastEventPtr = nullptr;
      UnenqueuedCmdEvents.clear();
      LastBarrier = nullptr;
    }
  } MDefaultGraphDeps, MExtGraphDeps;

  // Implement check-lock-check pattern to not lock empty MData as the locks
  // come with runtime overhead.
  template <typename DataType> class CheckLockCheck {
    DataType MData;
    std::atomic_bool MIsSet = false;
    mutable std::mutex MDataMtx;

  public:
    template <typename F> void set(F &&func) {
      std::lock_guard<std::mutex> Lock(MDataMtx);
      MIsSet.store(true, std::memory_order_release);
      std::forward<F>(func)(MData);
    }
    template <typename F> void unset(F &&func) {
      if (MIsSet.load(std::memory_order_acquire)) {
        std::lock_guard<std::mutex> Lock(MDataMtx);
        if (MIsSet.load(std::memory_order_acquire)) {
          std::forward<F>(func)(MData);
          MIsSet.store(false, std::memory_order_release);
        }
      }
    }
    DataType read() {
      if (!MIsSet.load(std::memory_order_acquire))
        return DataType{};
      std::lock_guard<std::mutex> Lock(MDataMtx);
      return MData;
    }
  };

  const bool MIsInorder;

  // Specifies whether this queue records last event. This can only
  // be true if the queue is in-order, the command graph is not
  // associated with the queue and there has never been any host
  // tasks submitted to the queue.
  std::atomic<bool> MNoEventMode = false;

  // Used exclusively in getLastEvent and queue_empty() implementations
  bool MEmpty = true;

  std::vector<EventImplPtr> MStreamsServiceEvents;
  std::mutex MStreamsServiceEventsMutex;

  // All member variable defined here  are needed for the SYCL instrumentation
  // layer. Do not guard these variables below with XPTI_ENABLE_INSTRUMENTATION
  // to ensure we have the same object layout when the macro in the library and
  // SYCL app are not the same.
  void *MTraceEvent = nullptr;
  /// The stream under which the traces are emitted from the queue object
  uint8_t MStreamID = 0;
  /// The instance ID of the trace event for queue object
  uint64_t MInstanceID = 0;

  // This event can be optionally provided by users for in-order queues to add
  // an additional dependency for the subsequent submission in to the queue.
  // Access to the event should be guarded with mutex.
  // NOTE: std::optional must not be exposed in the ABI.
  CheckLockCheck<std::optional<event>> MInOrderExternalEvent;

public:
  const bool MIsProfilingEnabled;

protected:
  // Command graph which is associated with this queue for the purposes of
  // recording commands to it.
  std::weak_ptr<ext::oneapi::experimental::detail::graph_impl> MGraph{};

#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
  // CMPLRLLVM-66082
  // This member should be part of the sycl::interop_handle class, but it
  // is an API breaking change. So member lives here temporarily where it can
  // be accessed through the queue member of the interop_handle
  ur_exp_command_buffer_handle_t MInteropGraph{};
#endif

  unsigned long long MQueueID;
  static std::atomic<unsigned long long> MNextAvailableQueueID;

  using MissedCleanupRequestsType = std::deque<
      std::shared_ptr<ext::oneapi::experimental::detail::graph_impl>>;
  CheckLockCheck<MissedCleanupRequestsType> MMissedCleanupRequests;

  friend class sycl::ext::oneapi::experimental::detail::node_impl;

  void verifyProps(const property_list &Props) const;
};

} // namespace detail
} // namespace _V1
} // namespace sycl
