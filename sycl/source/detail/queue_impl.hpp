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
#include <detail/device_info.hpp>
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
using DeviceImplPtr = std::shared_ptr<detail::device_impl>;

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
  static ContextImplPtr getDefaultOrNew(const DeviceImplPtr &Device) {
    if (!SYCLConfig<SYCL_ENABLE_DEFAULT_CONTEXTS>::get())
      return detail::getSyclObjImpl(
          context{createSyclObjFromImpl<device>(Device), {}, {}});

    ContextImplPtr DefaultContext = detail::getSyclObjImpl(
        Device->get_platform().ext_oneapi_get_default_context());
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
  queue_impl(const DeviceImplPtr &Device, const async_handler &AsyncHandler,
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
  queue_impl(const DeviceImplPtr &Device, const ContextImplPtr &Context,
             const async_handler &AsyncHandler, const property_list &PropList)
      : MDevice(Device), MContext(Context), MAsyncHandler(AsyncHandler),
        MPropList(PropList),
        MIsInorder(has_property<property::queue::in_order>()),
        MDiscardEvents(
            has_property<ext::oneapi::property::queue::discard_events>()),
        MIsProfilingEnabled(has_property<property::queue::enable_profiling>()),
        MQueueID{
            MNextAvailableQueueID.fetch_add(1, std::memory_order_relaxed)} {
    verifyProps(PropList);
    if (has_property<property::queue::enable_profiling>()) {
      if (has_property<ext::oneapi::property::queue::discard_events>())
        throw sycl::exception(make_error_code(errc::invalid),
                              "Queue cannot be constructed with both of "
                              "discard_events and enable_profiling.");
      // fallback profiling support. See MFallbackProfiling
      if (MDevice->has(aspect::queue_profiling)) {
        // When urDeviceGetGlobalTimestamps is not supported, compute the
        // profiling time OpenCL version < 2.1 case
        if (!getDeviceImplPtr()->isGetDeviceAndHostTimerSupported())
          MFallbackProfiling = true;
      } else {
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
    MQueues.push_back(createQueue(QOrder));
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
  }

  event getLastEvent();

private:
  void queue_impl_interop(ur_queue_handle_t UrQueue) {
    if (has_property<ext::oneapi::property::queue::discard_events>() &&
        has_property<property::queue::enable_profiling>()) {
      throw sycl::exception(make_error_code(errc::invalid),
                            "Queue cannot be constructed with both of "
                            "discard_events and enable_profiling.");
    }

    MQueues.push_back(UrQueue);

    ur_device_handle_t DeviceUr{};
    const AdapterPtr &Adapter = getAdapter();
    // TODO catch an exception and put it to list of asynchronous exceptions
    Adapter->call<UrApiKind::urQueueGetInfo>(
        MQueues[0], UR_QUEUE_INFO_DEVICE, sizeof(DeviceUr), &DeviceUr, nullptr);
    MDevice = MContext->findMatchingDeviceImpl(DeviceUr);
    if (MDevice == nullptr) {
      throw sycl::exception(
          make_error_code(errc::invalid),
          "Device provided by native Queue not found in Context.");
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
  }

public:
  /// Constructs a SYCL queue from adapter interoperability handle.
  ///
  /// \param UrQueue is a raw UR queue handle.
  /// \param Context is a SYCL context to associate with the queue being
  /// constructed.
  /// \param AsyncHandler is a SYCL asynchronous exception handler.
  queue_impl(ur_queue_handle_t UrQueue, const ContextImplPtr &Context,
             const async_handler &AsyncHandler)
      : MContext(Context), MAsyncHandler(AsyncHandler),
        MIsInorder(has_property<property::queue::in_order>()),
        MDiscardEvents(
            has_property<ext::oneapi::property::queue::discard_events>()),
        MIsProfilingEnabled(has_property<property::queue::enable_profiling>()),
        MQueueID{
            MNextAvailableQueueID.fetch_add(1, std::memory_order_relaxed)} {
    queue_impl_interop(UrQueue);
  }

  /// Constructs a SYCL queue from adapter interoperability handle.
  ///
  /// \param UrQueue is a raw UR queue handle.
  /// \param Context is a SYCL context to associate with the queue being
  /// constructed.
  /// \param AsyncHandler is a SYCL asynchronous exception handler.
  /// \param PropList is the queue properties.
  queue_impl(ur_queue_handle_t UrQueue, const ContextImplPtr &Context,
             const async_handler &AsyncHandler, const property_list &PropList)
      : MContext(Context), MAsyncHandler(AsyncHandler), MPropList(PropList),
        MIsInorder(has_property<property::queue::in_order>()),
        MDiscardEvents(
            has_property<ext::oneapi::property::queue::discard_events>()),
        MIsProfilingEnabled(has_property<property::queue::enable_profiling>()),
        MQueueID{
            MNextAvailableQueueID.fetch_add(1, std::memory_order_relaxed)} {
    verifyProps(PropList);
    queue_impl_interop(UrQueue);
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
      getAdapter()->call<UrApiKind::urQueueRelease>(MQueues[0]);
    } catch (std::exception &e) {
      __SYCL_REPORT_EXCEPTION_TO_STREAM("exception in ~queue_impl", e);
    }
  }

  /// \return an OpenCL interoperability queue handle.

  cl_command_queue get() {
    getAdapter()->call<UrApiKind::urQueueRetain>(MQueues[0]);
    ur_native_handle_t nativeHandle = 0;
    getAdapter()->call<UrApiKind::urQueueGetNativeHandle>(MQueues[0], nullptr,
                                                          &nativeHandle);
    return ur::cast<cl_command_queue>(nativeHandle);
  }

  /// \return an associated SYCL context.
  context get_context() const {
    return createSyclObjFromImpl<context>(MContext);
  }

  const AdapterPtr &getAdapter() const { return MContext->getAdapter(); }

  const ContextImplPtr &getContextImplPtr() const { return MContext; }

  const DeviceImplPtr &getDeviceImplPtr() const { return MDevice; }

  /// \return an associated SYCL device.
  device get_device() const { return createSyclObjFromImpl<device>(MDevice); }

  /// \return true if the discard event property was set at time of creation.
  bool hasDiscardEventsProperty() const { return MDiscardEvents; }

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
    for (const auto &queue : MQueues) {
      getAdapter()->call<UrApiKind::urQueueFlush>(queue);
    }
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
  event submit(const std::function<void(handler &)> &CGF,
               const std::shared_ptr<queue_impl> &Self,
               const std::shared_ptr<queue_impl> &SecondQueue,
               const detail::code_location &Loc, bool IsTopCodeLoc,
               const SubmitPostProcessF *PostProcess = nullptr) {
    event ResEvent;
    SubmissionInfo SI{};
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
  event submit_with_event(const std::function<void(handler &)> &CGF,
                          const std::shared_ptr<queue_impl> &Self,
                          const SubmissionInfo &SubmitInfo,
                          const detail::code_location &Loc, bool IsTopCodeLoc) {
    if (SubmitInfo.SecondaryQueue()) {
      event ResEvent;
      const std::shared_ptr<queue_impl> SecondQueue =
          SubmitInfo.SecondaryQueue();
      try {
        ResEvent = submit_impl(CGF, Self, Self, SecondQueue,
                               /*CallerNeedsEvent=*/true, Loc, IsTopCodeLoc,
                               SubmitInfo);
      } catch (...) {
        ResEvent = SecondQueue->submit_impl(CGF, SecondQueue, Self, SecondQueue,
                                            /*CallerNeedsEvent=*/true, Loc,
                                            IsTopCodeLoc, SubmitInfo);
      }
      return ResEvent;
    }
    event ResEvent =
        submit_impl(CGF, Self, Self, nullptr,
                    /*CallerNeedsEvent=*/true, Loc, IsTopCodeLoc, SubmitInfo);
    return discard_or_return(ResEvent);
  }

  void submit_without_event(const std::function<void(handler &)> &CGF,
                            const std::shared_ptr<queue_impl> &Self,
                            const SubmissionInfo &SubmitInfo,
                            const detail::code_location &Loc,
                            bool IsTopCodeLoc) {
    if (SubmitInfo.SecondaryQueue()) {
      const std::shared_ptr<queue_impl> SecondQueue =
          SubmitInfo.SecondaryQueue();
      try {
        submit_impl(CGF, Self, Self, SecondQueue,
                    /*CallerNeedsEvent=*/false, Loc, IsTopCodeLoc, SubmitInfo);
      } catch (...) {
        SecondQueue->submit_impl(CGF, SecondQueue, Self, SecondQueue,
                                 /*CallerNeedsEvent=*/false, Loc, IsTopCodeLoc,
                                 SubmitInfo);
      }
    } else {
      submit_impl(CGF, Self, Self, nullptr, /*CallerNeedsEvent=*/false, Loc,
                  IsTopCodeLoc, SubmitInfo);
    }
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
    if (PropList.has_property<ext::oneapi::property::queue::discard_events>()) {
      // Pass this flag to the Level Zero adapter to be able to check it from
      // queue property.
      CreationFlags |= UR_QUEUE_FLAG_DISCARD_EVENTS;
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
    // Track that submission modes do not conflict.
    bool SubmissionSeen = false;
    if (PropList.has_property<
            ext::intel::property::queue::no_immediate_command_list>()) {
      SubmissionSeen = true;
      CreationFlags |= UR_QUEUE_FLAG_SUBMISSION_BATCHED;
    }
    if (PropList.has_property<
            ext::intel::property::queue::immediate_command_list>()) {
      if (SubmissionSeen) {
        throw sycl::exception(
            make_error_code(errc::invalid),
            "Queue cannot be constructed with different submission modes.");
      }
      SubmissionSeen = true;
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
    ur_device_handle_t Device = MDevice->getHandleRef();
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
    ur_result_t Error = Adapter->call_nocheck<UrApiKind::urQueueCreate>(
        Context, Device, &Properties, &Queue);

    // If creating out-of-order queue failed and this property is not
    // supported (for example, on FPGA), it will return
    // UR_RESULT_ERROR_INVALID_QUEUE_PROPERTIES and will try to create in-order
    // queue.
    if (!MEmulateOOO && Error == UR_RESULT_ERROR_INVALID_QUEUE_PROPERTIES) {
      MEmulateOOO = true;
      Queue = createQueue(QueueOrder::Ordered);
    } else {
      Adapter->checkUrResult(Error);
    }

    return Queue;
  }

  /// \return a raw UR handle for a free queue. The returned handle is not
  /// retained. It is caller responsibility to make sure queue is still alive.
  ur_queue_handle_t &getExclusiveUrQueueHandleRef() {
    ur_queue_handle_t *PIQ = nullptr;
    bool ReuseQueue = false;
    {
      std::lock_guard<std::mutex> Lock(MMutex);

      // To achieve parallelism for FPGA with in order execution model with
      // possibility of two kernels to share data with each other we shall
      // create a queue for every kernel enqueued.
      if (MQueues.size() < MaxNumQueues) {
        MQueues.push_back({});
        PIQ = &MQueues.back();
      } else {
        // If the limit of OpenCL queues is going to be exceeded - take the
        // earliest used queue, wait until it finished and then reuse it.
        PIQ = &MQueues[MNextQueueIdx];
        MNextQueueIdx = (MNextQueueIdx + 1) % MaxNumQueues;
        ReuseQueue = true;
      }
    }

    if (!ReuseQueue)
      *PIQ = createQueue(QueueOrder::Ordered);
    else
      getAdapter()->call<UrApiKind::urQueueFinish>(*PIQ);

    return *PIQ;
  }

  /// \return a raw UR queue handle. The returned handle is not retained. It
  /// is caller responsibility to make sure queue is still alive.
  ur_queue_handle_t &getHandleRef() {
    if (!MEmulateOOO)
      return MQueues[0];

    return getExclusiveUrQueueHandleRef();
  }

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

  bool ext_oneapi_empty() const;

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

  bool isProfilingFallback() { return MFallbackProfiling; }

  void setCommandGraph(
      std::shared_ptr<ext::oneapi::experimental::detail::graph_impl> Graph) {
    std::lock_guard<std::mutex> Lock(MMutex);
    MGraph = Graph;
    MExtGraphDeps.reset();
  }

  std::shared_ptr<ext::oneapi::experimental::detail::graph_impl>
  getCommandGraph() const {
    return MGraph.lock();
  }

  unsigned long long getQueueID() { return MQueueID; }

  void *getTraceEvent() { return MTraceEvent; }

  void setExternalEvent(const event &Event) {
    std::lock_guard<std::mutex> Lock(MInOrderExternalEventMtx);
    MInOrderExternalEvent = Event;
  }

  std::optional<event> popExternalEvent() {
    std::lock_guard<std::mutex> Lock(MInOrderExternalEventMtx);
    std::optional<event> Result = std::nullopt;
    std::swap(Result, MInOrderExternalEvent);
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

protected:
  event discard_or_return(const event &Event);

  template <typename HandlerType = handler>
  EventImplPtr insertHelperBarrier(const HandlerType &Handler) {
    auto ResEvent = std::make_shared<detail::event_impl>(Handler.MQueue);
    ur_event_handle_t UREvent = nullptr;
    getAdapter()->call<UrApiKind::urEnqueueEventsWaitWithBarrier>(
        Handler.MQueue->getHandleRef(), 0, nullptr, &UREvent);
    ResEvent->setHandle(UREvent);
    return ResEvent;
  }

  // template is needed for proper unit testing
  template <typename HandlerType = handler>
  void finalizeHandler(HandlerType &Handler, event &EventRet) {
    if (MIsInorder) {
      // Accessing and changing of an event isn't atomic operation.
      // Hence, here is the lock for thread-safety.
      std::lock_guard<std::mutex> Lock{MMutex};

      auto &EventToBuildDeps = MGraph.expired() ? MDefaultGraphDeps.LastEventPtr
                                                : MExtGraphDeps.LastEventPtr;

      // This dependency is needed for the following purposes:
      //    - host tasks are handled by the runtime and cannot be implicitly
      //    synchronized by the backend.
      //    - to prevent the 2nd kernel enqueue when the 1st kernel is blocked
      //    by a host task. This dependency allows to build the enqueue order in
      //    the RT but will not be passed to the backend. See getPIEvents in
      //    Command.
      if (EventToBuildDeps) {
        // In the case where the last event was discarded and we are to run a
        // host_task, we insert a barrier into the queue and use the resulting
        // event as the dependency for the host_task.
        // Note that host_task events can never be discarded, so this will not
        // insert barriers between host_task enqueues.
        if (EventToBuildDeps->isDiscarded() &&
            getSyclObjImpl(Handler)->MCGType == CGType::CodeplayHostTask)
          EventToBuildDeps = insertHelperBarrier(Handler);

        if (!EventToBuildDeps->isDiscarded())
          Handler.depends_on(EventToBuildDeps);
      }

      // If there is an external event set, add it as a dependency and clear it.
      // We do not need to hold the lock as MLastEventMtx will ensure the last
      // event reflects the corresponding external event dependence as well.
      std::optional<event> ExternalEvent = popExternalEvent();
      if (ExternalEvent)
        Handler.depends_on(*ExternalEvent);

      EventRet = Handler.finalize();
      EventToBuildDeps = getSyclObjImpl(EventRet);
    } else {
      const CGType Type = getSyclObjImpl(Handler)->MCGType;
      std::lock_guard<std::mutex> Lock{MMutex};
      // The following code supports barrier synchronization if host task is
      // involved in the scenario. Native barriers cannot handle host task
      // dependency so in the case where some commands were not enqueued
      // (blocked), we track them to prevent barrier from being enqueued
      // earlier.
      {
        std::lock_guard<std::mutex> RequestLock(MMissedCleanupRequestsMtx);
        for (auto &UpdatedGraph : MMissedCleanupRequests)
          doUnenqueuedCommandCleanup(UpdatedGraph);
        MMissedCleanupRequests.clear();
      }
      auto &Deps = MGraph.expired() ? MDefaultGraphDeps : MExtGraphDeps;
      if (Type == CGType::Barrier && !Deps.UnenqueuedCmdEvents.empty()) {
        Handler.depends_on(Deps.UnenqueuedCmdEvents);
      }
      if (Deps.LastBarrier && (Type == CGType::CodeplayHostTask ||
                               (!Deps.LastBarrier->isEnqueued())))
        Handler.depends_on(Deps.LastBarrier);

      EventRet = Handler.finalize();
      EventImplPtr EventRetImpl = getSyclObjImpl(EventRet);
      if (Type == CGType::CodeplayHostTask)
        Deps.UnenqueuedCmdEvents.push_back(EventRetImpl);
      else if (Type == CGType::Barrier || Type == CGType::BarrierWaitlist) {
        Deps.LastBarrier = EventRetImpl;
        Deps.UnenqueuedCmdEvents.clear();
      } else if (!EventRetImpl->isEnqueued()) {
        Deps.UnenqueuedCmdEvents.push_back(EventRetImpl);
      }
    }
  }

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
  event submit_impl(const std::function<void(handler &)> &CGF,
                    const std::shared_ptr<queue_impl> &Self,
                    const std::shared_ptr<queue_impl> &PrimaryQueue,
                    const std::shared_ptr<queue_impl> &SecondaryQueue,
                    bool CallerNeedsEvent, const detail::code_location &Loc,
                    bool IsTopCodeLoc, const SubmissionInfo &SubmitInfo);

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
  /// \param Event is the event to be stored
  void addEvent(const event &Event);

  /// Protects all the fields that can be changed by class' methods.
  mutable std::mutex MMutex;

  DeviceImplPtr MDevice;
  const ContextImplPtr MContext;

  /// These events are tracked, but not owned, by the queue.
  std::vector<std::weak_ptr<event_impl>> MEventsWeak;

  /// Events without data dependencies (such as USM) need an owner,
  /// additionally, USM operations are not added to the scheduler command graph,
  /// queue is the only owner on the runtime side.
  std::vector<event> MEventsShared;
  exception_list MExceptions;
  const async_handler MAsyncHandler;
  const property_list MPropList;

  /// List of queues created for FPGA device from a single SYCL queue.
  std::vector<ur_queue_handle_t> MQueues;
  /// Iterator through MQueues.
  size_t MNextQueueIdx = 0;

  /// Indicates that a native out-of-order queue could not be created and we
  /// need to emulate it with multiple native in-order queues.
  bool MEmulateOOO = false;

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

  const bool MIsInorder;

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

  // the fallback implementation of profiling info
  bool MFallbackProfiling = false;

  // This event can be optionally provided by users for in-order queues to add
  // an additional dependency for the subsequent submission in to the queue.
  // Access to the event should be guarded with MInOrderExternalEventMtx.
  // NOTE: std::optional must not be exposed in the ABI.
  std::optional<event> MInOrderExternalEvent;
  mutable std::mutex MInOrderExternalEventMtx;

public:
  // Queue constructed with the discard_events property
  const bool MDiscardEvents;
  const bool MIsProfilingEnabled;

protected:
  // Command graph which is associated with this queue for the purposes of
  // recording commands to it.
  std::weak_ptr<ext::oneapi::experimental::detail::graph_impl> MGraph{};

  unsigned long long MQueueID;
  static std::atomic<unsigned long long> MNextAvailableQueueID;

  std::deque<std::shared_ptr<ext::oneapi::experimental::detail::graph_impl>>
      MMissedCleanupRequests;
  std::mutex MMissedCleanupRequestsMtx;

  friend class sycl::ext::oneapi::experimental::detail::node_impl;

  void verifyProps(const property_list &Props) const;
};

} // namespace detail
} // namespace _V1
} // namespace sycl
