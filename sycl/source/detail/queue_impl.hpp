//==------------------ queue_impl.hpp - SYCL queue -------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <detail/config.hpp>
#include <detail/context_impl.hpp>
#include <detail/device_impl.hpp>
#include <detail/device_info.hpp>
#include <detail/event_impl.hpp>
#include <detail/global_handler.hpp>
#include <detail/kernel_impl.hpp>
#include <detail/plugin.hpp>
#include <detail/scheduler/scheduler.hpp>
#include <detail/thread_pool.hpp>
#include <sycl/context.hpp>
#include <sycl/detail/assert_happened.hpp>
#include <sycl/detail/cuda_definitions.hpp>
#include <sycl/device.hpp>
#include <sycl/event.hpp>
#include <sycl/exception.hpp>
#include <sycl/exception_list.hpp>
#include <sycl/ext/codeplay/experimental/fusion_properties.hpp>
#include <sycl/handler.hpp>
#include <sycl/properties/context_properties.hpp>
#include <sycl/properties/queue_properties.hpp>
#include <sycl/property_list.hpp>
#include <sycl/queue.hpp>
#include <sycl/stl.hpp>

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

//// Possible CUDA context types supported by PI CUDA backend
/// TODO: Implement this as a property once there is an extension document
enum class CUDAContextT : char { primary, custom };

/// Default context type created for CUDA backend
constexpr CUDAContextT DefaultContextType = CUDAContextT::custom;

enum QueueOrder { Ordered, OOO };

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
      : queue_impl(Device, getDefaultOrNew(Device), AsyncHandler, PropList){};

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
        MPropList(PropList), MHostQueue(MDevice->is_host()),
        MAssertHappenedBuffer(range<1>{1}),
        MIsInorder(has_property<property::queue::in_order>()),
        MDiscardEvents(
            has_property<ext::oneapi::property::queue::discard_events>()),
        MIsProfilingEnabled(has_property<property::queue::enable_profiling>()),
        MHasDiscardEventsSupport(MDiscardEvents &&
                                 (MHostQueue ? true : MIsInorder)),
        MQueueID{
            MNextAvailableQueueID.fetch_add(1, std::memory_order_relaxed)} {
    if (has_property<property::queue::enable_profiling>()) {
      if (has_property<ext::oneapi::property::queue::discard_events>())
        throw sycl::exception(make_error_code(errc::invalid),
                              "Queue cannot be constructed with both of "
                              "discard_events and enable_profiling.");
      // fallback profiling support. See MFallbackProfiling
      if (MDevice->has(aspect::queue_profiling)) {
        // When piGetDeviceAndHostTimer is not supported, compute the
        // profiling time OpenCL version < 2.1 case
        if (!getDeviceImplPtr()->is_host() &&
            !getDeviceImplPtr()->isGetDeviceAndHostTimerSupported())
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
    if (has_property<
            ext::codeplay::experimental::property::queue::enable_fusion>() &&
        !MDevice->get_info<
            ext::codeplay::experimental::info::device::supports_fusion>()) {
      throw sycl::exception(
          make_error_code(errc::invalid),
          "Cannot enable fusion if device does not support fusion");
    }
    if (!Context->isDeviceValid(Device)) {
      if (!Context->is_host() && Context->getBackend() == backend::opencl)
        throw sycl::invalid_object_error(
            "Queue cannot be constructed with the given context and device "
            "since the device is not a member of the context (descendants of "
            "devices from the context are not supported on OpenCL yet).",
            PI_ERROR_INVALID_DEVICE);
      throw sycl::invalid_object_error(
          "Queue cannot be constructed with the given context and device "
          "since the device is neither a member of the context nor a "
          "descendant of its member.",
          PI_ERROR_INVALID_DEVICE);
    }
    if (!MHostQueue) {
      const QueueOrder QOrder =
          MIsInorder ? QueueOrder::Ordered : QueueOrder::OOO;
      MQueues.push_back(createQueue(QOrder));
      // This section is the second part of the instrumentation that uses the
      // tracepoint information and notifies
    }
    // We enable XPTI tracing events using the TLS mechanism; if the code
    // location data is available, then the tracing data will be rich.
#if XPTI_ENABLE_INSTRUMENTATION
    constexpr uint16_t NotificationTraceType =
        static_cast<uint16_t>(xpti::trace_point_type_t::queue_create);
    XPTIScope PrepareNotify((void *)this, NotificationTraceType,
                            SYCL_STREAM_NAME, "queue_create");
    // Cache the trace event, stream id and instance IDs for the destructor
    if (xptiCheckTraceEnabled(PrepareNotify.streamID(),
                              NotificationTraceType)) {
      MTraceEvent = (void *)PrepareNotify.traceEvent();
      MStreamID = PrepareNotify.streamID();
      MInstanceID = PrepareNotify.instanceID();
      // Add the function to capture meta data for the XPTI trace event
      PrepareNotify.addMetadata([&](auto TEvent) {
        xpti::addMetadata(TEvent, "sycl_context",
                          reinterpret_cast<size_t>(MContext->getHandleRef()));
        if (MDevice) {
          xpti::addMetadata(TEvent, "sycl_device_name",
                            MDevice->getDeviceName());
          xpti::addMetadata(
              TEvent, "sycl_device",
              reinterpret_cast<size_t>(
                  MDevice->is_host() ? 0 : MDevice->getHandleRef()));
        }
        xpti::addMetadata(TEvent, "is_inorder", MIsInorder);
        xpti::addMetadata(TEvent, "queue_id", MQueueID);
        if (!MHostQueue)
          xpti::addMetadata(TEvent, "queue_handle",
                            reinterpret_cast<size_t>(getHandleRef()));
      });
      PrepareNotify.notify();
    }
#endif
  }

  event getLastEvent() const;

private:
  void queue_impl_interop(sycl::detail::pi::PiQueue PiQueue) {
    if (has_property<ext::oneapi::property::queue::discard_events>() &&
        has_property<property::queue::enable_profiling>()) {
      throw sycl::exception(make_error_code(errc::invalid),
                            "Queue cannot be constructed with both of "
                            "discard_events and enable_profiling.");
    }

    MQueues.push_back(pi::cast<sycl::detail::pi::PiQueue>(PiQueue));

    sycl::detail::pi::PiDevice DevicePI{};
    const PluginPtr &Plugin = getPlugin();
    // TODO catch an exception and put it to list of asynchronous exceptions
    Plugin->call<PiApiKind::piQueueGetInfo>(
        MQueues[0], PI_QUEUE_INFO_DEVICE, sizeof(DevicePI), &DevicePI, nullptr);
    MDevice = MContext->findMatchingDeviceImpl(DevicePI);
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
    constexpr uint16_t NotificationTraceType =
        static_cast<uint16_t>(xpti::trace_point_type_t::queue_create);
    XPTIScope PrepareNotify((void *)this, NotificationTraceType,
                            SYCL_STREAM_NAME, "queue_create");
    if (xptiCheckTraceEnabled(PrepareNotify.streamID(),
                              NotificationTraceType)) {
      // Cache the trace event, stream id and instance IDs for the destructor
      MTraceEvent = (void *)PrepareNotify.traceEvent();
      MStreamID = PrepareNotify.streamID();
      MInstanceID = PrepareNotify.instanceID();

      // Add the function to capture meta data for the XPTI trace event
      PrepareNotify.addMetadata([&](auto TEvent) {
        xpti::addMetadata(TEvent, "sycl_context",
                          reinterpret_cast<size_t>(MContext->getHandleRef()));
        if (MDevice) {
          xpti::addMetadata(TEvent, "sycl_device_name",
                            MDevice->getDeviceName());
          xpti::addMetadata(
              TEvent, "sycl_device",
              reinterpret_cast<size_t>(
                  MDevice->is_host() ? 0 : MDevice->getHandleRef()));
        }
        xpti::addMetadata(TEvent, "is_inorder", MIsInorder);
        xpti::addMetadata(TEvent, "queue_id", MQueueID);
        if (!MHostQueue)
          xpti::addMetadata(TEvent, "queue_handle", getHandleRef());
      });
      PrepareNotify.notify();
    }
#endif
  }

public:
  /// Constructs a SYCL queue from plugin interoperability handle.
  ///
  /// \param PiQueue is a raw PI queue handle.
  /// \param Context is a SYCL context to associate with the queue being
  /// constructed.
  /// \param AsyncHandler is a SYCL asynchronous exception handler.
  queue_impl(sycl::detail::pi::PiQueue PiQueue, const ContextImplPtr &Context,
             const async_handler &AsyncHandler)
      : MContext(Context), MAsyncHandler(AsyncHandler), MHostQueue(false),
        MAssertHappenedBuffer(range<1>{1}),
        MIsInorder(has_property<property::queue::in_order>()),
        MDiscardEvents(
            has_property<ext::oneapi::property::queue::discard_events>()),
        MIsProfilingEnabled(has_property<property::queue::enable_profiling>()),
        MHasDiscardEventsSupport(MDiscardEvents &&
                                 (MHostQueue ? true : MIsInorder)),
        MQueueID{
            MNextAvailableQueueID.fetch_add(1, std::memory_order_relaxed)} {
    queue_impl_interop(PiQueue);
  }

  /// Constructs a SYCL queue from plugin interoperability handle.
  ///
  /// \param PiQueue is a raw PI queue handle.
  /// \param Context is a SYCL context to associate with the queue being
  /// constructed.
  /// \param AsyncHandler is a SYCL asynchronous exception handler.
  /// \param PropList is the queue properties.
  queue_impl(sycl::detail::pi::PiQueue PiQueue, const ContextImplPtr &Context,
             const async_handler &AsyncHandler, const property_list &PropList)
      : MContext(Context), MAsyncHandler(AsyncHandler), MPropList(PropList),
        MHostQueue(false), MAssertHappenedBuffer(range<1>{1}),
        MIsInorder(has_property<property::queue::in_order>()),
        MDiscardEvents(
            has_property<ext::oneapi::property::queue::discard_events>()),
        MIsProfilingEnabled(has_property<property::queue::enable_profiling>()),
        MHasDiscardEventsSupport(MDiscardEvents &&
                                 (MHostQueue ? true : MIsInorder)) {
    queue_impl_interop(PiQueue);
  }

  ~queue_impl() {
    // The trace event created in the constructor should be active through the
    // lifetime of the queue object as member variables when ABI breakage is
    // allowed. This example shows MTraceEvent as a member variable.
#if XPTI_ENABLE_INSTRUMENTATION
    constexpr uint16_t NotificationTraceType =
        static_cast<uint16_t>(xpti::trace_point_type_t::queue_destroy);
    if (xptiCheckTraceEnabled(MStreamID, NotificationTraceType)) {
      // Used cached information in member variables
      xptiNotifySubscribers(MStreamID, NotificationTraceType, nullptr,
                            (xpti::trace_event_data_t *)MTraceEvent,
                            MInstanceID,
                            static_cast<const void *>("queue_destroy"));
      xptiReleaseEvent((xpti::trace_event_data_t *)MTraceEvent);
    }
#endif
    throw_asynchronous();
    if (!MHostQueue) {
      cleanup_fusion_cmd();
      getPlugin()->call<PiApiKind::piQueueRelease>(MQueues[0]);
    }
  }

  /// \return an OpenCL interoperability queue handle.
  cl_command_queue get() {
    if (MHostQueue) {
      throw invalid_object_error(
          "This instance of queue doesn't support OpenCL interoperability",
          PI_ERROR_INVALID_QUEUE);
    }
    getPlugin()->call<PiApiKind::piQueueRetain>(MQueues[0]);
    return pi::cast<cl_command_queue>(MQueues[0]);
  }

  /// \return an associated SYCL context.
  context get_context() const {
    return createSyclObjFromImpl<context>(MContext);
  }

  const PluginPtr &getPlugin() const { return MContext->getPlugin(); }

  const ContextImplPtr &getContextImplPtr() const { return MContext; }

  const DeviceImplPtr &getDeviceImplPtr() const { return MDevice; }

  /// \return an associated SYCL device.
  device get_device() const { return createSyclObjFromImpl<device>(MDevice); }

  /// \return true if this queue is a SYCL host queue.
  bool is_host() const { return MHostQueue; }

  /// \return true if this queue has discard_events support.
  bool has_discard_events_support() const { return MHasDiscardEventsSupport; }

  bool isInOrder() const { return MIsInorder; }

  /// Queries SYCL queue for information.
  ///
  /// The return type depends on information being queried.
  template <typename Param> typename Param::return_type get_info() const;

  using SubmitPostProcessF = std::function<void(bool, bool, event &)>;

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
               const detail::code_location &Loc,
               const SubmitPostProcessF *PostProcess = nullptr) {
    try {
      return submit_impl(CGF, Self, Self, SecondQueue, Loc, PostProcess);
    } catch (...) {
      return SecondQueue->submit_impl(CGF, SecondQueue, Self, SecondQueue, Loc,
                                      PostProcess);
    }
  }

  /// Submits a command group function object to the queue, in order to be
  /// scheduled for execution on the device.
  ///
  /// \param CGF is a function object containing command group.
  /// \param Self is a shared_ptr to this queue.
  /// \param Loc is the code location of the submit call (default argument)
  /// \param StoreAdditionalInfo makes additional info be stored in event_impl
  /// \return a SYCL event object for the submitted command group.
  event submit(const std::function<void(handler &)> &CGF,
               const std::shared_ptr<queue_impl> &Self,
               const detail::code_location &Loc,
               const SubmitPostProcessF *PostProcess = nullptr) {
    return submit_impl(CGF, Self, Self, nullptr, Loc, PostProcess);
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

  /// Creates PI properties array.
  ///
  /// \param PropList SYCL properties.
  /// \param Order specifies whether queue is in-order or out-of-order.
  /// \param Properties PI properties array created from SYCL properties.
  static sycl::detail::pi::PiQueueProperties
  createPiQueueProperties(const property_list &PropList, QueueOrder Order) {
    sycl::detail::pi::PiQueueProperties CreationFlags = 0;

    if (Order == QueueOrder::OOO) {
      CreationFlags = PI_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE;
    }
    if (PropList.has_property<property::queue::enable_profiling>()) {
      CreationFlags |= PI_QUEUE_FLAG_PROFILING_ENABLE;
    }
    if (PropList.has_property<
            ext::oneapi::cuda::property::queue::use_default_stream>()) {
      CreationFlags |= __SYCL_PI_CUDA_USE_DEFAULT_STREAM;
    }
    if (PropList.has_property<ext::oneapi::property::queue::discard_events>()) {
      // Pass this flag to the Level Zero plugin to be able to check it from
      // queue property.
      CreationFlags |= PI_EXT_ONEAPI_QUEUE_FLAG_DISCARD_EVENTS;
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
      CreationFlags |= PI_EXT_ONEAPI_QUEUE_FLAG_PRIORITY_LOW;
      PrioritySeen = true;
    }
    if (PropList.has_property<ext::oneapi::property::queue::priority_high>()) {
      if (PrioritySeen) {
        throw sycl::exception(
            make_error_code(errc::invalid),
            "Queue cannot be constructed with different priorities.");
      }
      CreationFlags |= PI_EXT_ONEAPI_QUEUE_FLAG_PRIORITY_HIGH;
    }
    // Track that submission modes do not conflict.
    bool SubmissionSeen = false;
    if (PropList.has_property<
            ext::intel::property::queue::no_immediate_command_list>()) {
      SubmissionSeen = true;
      CreationFlags |= PI_EXT_QUEUE_FLAG_SUBMISSION_NO_IMMEDIATE;
    }
    if (PropList.has_property<
            ext::intel::property::queue::immediate_command_list>()) {
      if (SubmissionSeen) {
        throw sycl::exception(
            make_error_code(errc::invalid),
            "Queue cannot be constructed with different submission modes.");
      }
      SubmissionSeen = true;
      CreationFlags |= PI_EXT_QUEUE_FLAG_SUBMISSION_IMMEDIATE;
    }
    return CreationFlags;
  }

  /// Creates PI queue.
  ///
  /// \param Order specifies whether the queue being constructed as in-order
  /// or out-of-order.
  sycl::detail::pi::PiQueue createQueue(QueueOrder Order) {
    sycl::detail::pi::PiQueue Queue{};
    sycl::detail::pi::PiContext Context = MContext->getHandleRef();
    sycl::detail::pi::PiDevice Device = MDevice->getHandleRef();
    const PluginPtr &Plugin = getPlugin();

    sycl::detail::pi::PiQueueProperties Properties[] = {
        PI_QUEUE_FLAGS, createPiQueueProperties(MPropList, Order), 0, 0, 0};
    if (has_property<ext::intel::property::queue::compute_index>()) {
      int Idx = get_property<ext::intel::property::queue::compute_index>()
                    .get_index();
      Properties[2] = PI_QUEUE_COMPUTE_INDEX;
      Properties[3] = static_cast<sycl::detail::pi::PiQueueProperties>(Idx);
    }
    sycl::detail::pi::PiResult Error =
        Plugin->call_nocheck<PiApiKind::piextQueueCreate>(Context, Device,
                                                          Properties, &Queue);

    // If creating out-of-order queue failed and this property is not
    // supported (for example, on FPGA), it will return
    // PI_ERROR_INVALID_QUEUE_PROPERTIES and will try to create in-order queue.
    if (!MEmulateOOO && Error == PI_ERROR_INVALID_QUEUE_PROPERTIES) {
      MEmulateOOO = true;
      Queue = createQueue(QueueOrder::Ordered);
    } else {
      Plugin->checkPiResult(Error);
    }

    return Queue;
  }

  /// \return a raw PI handle for a free queue. The returned handle is not
  /// retained. It is caller responsibility to make sure queue is still alive.
  sycl::detail::pi::PiQueue &getExclusiveQueueHandleRef() {
    sycl::detail::pi::PiQueue *PIQ = nullptr;
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
      getPlugin()->call<PiApiKind::piQueueFinish>(*PIQ);

    return *PIQ;
  }

  /// \return a raw PI queue handle. The returned handle is not retained. It
  /// is caller responsibility to make sure queue is still alive.
  sycl::detail::pi::PiQueue &getHandleRef() {
    if (!MEmulateOOO)
      return MQueues[0];

    return getExclusiveQueueHandleRef();
  }

  /// \return true if the queue was constructed with property specified by
  /// PropertyT.
  template <typename propertyT> bool has_property() const noexcept {
    return MPropList.has_property<propertyT>();
  }

  /// \return a copy of the property of type PropertyT that the queue was
  /// constructed with. If the queue was not constructed with the PropertyT
  /// property, an invalid_object_error SYCL exception.
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
  /// \return an event representing fill operation.
  event memset(const std::shared_ptr<queue_impl> &Self, void *Ptr, int Value,
               size_t Count, const std::vector<event> &DepEvents);
  /// Copies data from one memory region to another, both pointed by
  /// USM pointers.
  ///
  /// \param Self is a shared_ptr to this queue.
  /// \param Dest is a USM pointer to the destination memory.
  /// \param Src is a USM pointer to the source memory.
  /// \param Count is a number of bytes to copy.
  /// \param DepEvents is a vector of events that specifies the kernel
  /// dependencies.
  /// \return an event representing copy operation.
  event memcpy(const std::shared_ptr<queue_impl> &Self, void *Dest,
               const void *Src, size_t Count,
               const std::vector<event> &DepEvents,
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
  /// \return an event representing advise operation.
  event mem_advise(const std::shared_ptr<queue_impl> &Self, const void *Ptr,
                   size_t Length, pi_mem_advice Advice,
                   const std::vector<event> &DepEvents);

  /// Puts exception to the list of asynchronous ecxeptions.
  ///
  /// \param ExceptionPtr is a pointer to exception to be put.
  void reportAsyncException(const std::exception_ptr &ExceptionPtr) {
    std::lock_guard<std::mutex> Lock(MMutex);
    MExceptions.PushBack(ExceptionPtr);
  }

  ThreadPool &getThreadPool() {
    return GlobalHandler::instance().getHostTaskThreadPool();
  }

  /// Gets the native handle of the SYCL queue.
  ///
  /// \return a native handle.
  pi_native_handle getNative(int32_t &NativeHandleDesc) const;

  buffer<AssertHappened, 1> &getAssertHappenedBuffer() {
    return MAssertHappenedBuffer;
  }

  void registerStreamServiceEvent(const EventImplPtr &Event) {
    std::lock_guard<std::mutex> Lock(MMutex);
    MStreamsServiceEvents.push_back(Event);
  }

  bool ext_oneapi_empty() const;

  /// Check whether the queue is in fusion mode.
  ///
  /// \return true if the queue is in fusion mode, false otherwise.
  bool is_in_fusion_mode() {
    return detail::Scheduler::getInstance().isInFusionMode(
        std::hash<typename std::shared_ptr<queue_impl>::element_type *>()(
            this));
  }

  event memcpyToDeviceGlobal(const std::shared_ptr<queue_impl> &Self,
                             void *DeviceGlobalPtr, const void *Src,
                             bool IsDeviceImageScope, size_t NumBytes,
                             size_t Offset,
                             const std::vector<event> &DepEvents);
  event memcpyFromDeviceGlobal(const std::shared_ptr<queue_impl> &Self,
                               void *Dest, const void *DeviceGlobalPtr,
                               bool IsDeviceImageScope, size_t NumBytes,
                               size_t Offset,
                               const std::vector<event> &DepEvents);

  bool isProfilingFallback() { return MFallbackProfiling; }

  void setCommandGraph(
      std::shared_ptr<ext::oneapi::experimental::detail::graph_impl> Graph) {
    std::lock_guard<std::mutex> Lock(MMutex);
    MGraph = Graph;
  }

  std::shared_ptr<ext::oneapi::experimental::detail::graph_impl>
  getCommandGraph() const {
    return MGraph.lock();
  }

  unsigned long long getQueueID() { return MQueueID; }

protected:
  // Hook to the scheduler to clean up any fusion command held on destruction.
  void cleanup_fusion_cmd();

  // template is needed for proper unit testing
  template <typename HandlerType = handler>
  void finalizeHandler(HandlerType &Handler, const CG::CGTYPE &Type,
                       event &EventRet) {
    if (MIsInorder) {

      auto IsExpDepManaged = [](const CG::CGTYPE &Type) {
        return Type == CG::CGTYPE::CodeplayHostTask;
      };

      // Accessing and changing of an event isn't atomic operation.
      // Hence, here is the lock for thread-safety.
      std::lock_guard<std::mutex> Lock{MLastEventMtx};

      if (MLastCGType == CG::CGTYPE::None)
        MLastCGType = Type;
      // Also handles case when sync model changes. E.g. Last is host, new is
      // kernel.
      bool NeedSeparateDependencyMgmt =
          IsExpDepManaged(Type) || IsExpDepManaged(MLastCGType);

      if (NeedSeparateDependencyMgmt)
        Handler.depends_on(MLastEvent);

      EventRet = Handler.finalize();

      MLastEvent = EventRet;
      MLastCGType = Type;
    } else
      EventRet = Handler.finalize();
  }

protected:
  /// Performs command group submission to the queue.
  ///
  /// \param CGF is a function object containing command group.
  /// \param Self is a pointer to this queue.
  /// \param PrimaryQueue is a pointer to the primary queue. This may be the
  ///        same as Self.
  /// \param SecondaryQueue is a pointer to the secondary queue. This may be the
  ///        same as Self.
  /// \param Loc is the code location of the submit call (default argument)
  /// \return a SYCL event representing submitted command group.
  event submit_impl(const std::function<void(handler &)> &CGF,
                    const std::shared_ptr<queue_impl> &Self,
                    const std::shared_ptr<queue_impl> &PrimaryQueue,
                    const std::shared_ptr<queue_impl> &SecondaryQueue,
                    const detail::code_location &Loc,
                    const SubmitPostProcessF *PostProcess) {
    // Flag used to detect nested calls to submit and report an error.
    thread_local static bool PreventSubmit = false;

    if (PreventSubmit) {
      throw sycl::exception(
          make_error_code(errc::invalid),
          "Calls to sycl::queue::submit cannot be nested. Command group "
          "function objects should use the sycl::handler API instead.");
    }

    handler Handler(Self, PrimaryQueue, SecondaryQueue, MHostQueue);
    Handler.saveCodeLoc(Loc);
    PreventSubmit = true;
    try {
      CGF(Handler);
    } catch (...) {
      PreventSubmit = false;
      throw;
    }
    PreventSubmit = false;

    // Scheduler will later omit events, that are not required to execute tasks.
    // Host and interop tasks, however, are not submitted to low-level runtimes
    // and require separate dependency management.
    const CG::CGTYPE Type = Handler.getType();
    event Event = detail::createSyclObjFromImpl<event>(
        std::make_shared<detail::event_impl>());

    if (PostProcess) {
      bool IsKernel = Type == CG::Kernel;
      bool KernelUsesAssert = false;

      if (IsKernel)
      // Kernel only uses assert if it's non interop one
#ifdef __INTEL_PREVIEW_BREAKING_CHANGES
        KernelUsesAssert = !(Handler.MKernel && Handler.MKernel->isInterop()) &&
                           ProgramManager::getInstance().kernelUsesAssert(
                               Handler.MKernelName.c_str());
#else
        KernelUsesAssert =
            !(Handler.MKernel && Handler.MKernel->isInterop()) &&
            ProgramManager::getInstance().kernelUsesAssert(Handler.MKernelName);
#endif
      finalizeHandler(Handler, Type, Event);

      (*PostProcess)(IsKernel, KernelUsesAssert, Event);
    } else
      finalizeHandler(Handler, Type, Event);

    addEvent(Event);
    return Event;
  }

  // When instrumentation is enabled emits trace event for wait begin and
  // returns the telemetry event generated for the wait
  void *instrumentationProlog(const detail::code_location &CodeLoc,
                              std::string &Name, int32_t StreamID,
                              uint64_t &iid);
  // Uses events generated by the Prolog and emits wait done event
  void instrumentationEpilog(void *TelementryEvent, std::string &Name,
                             int32_t StreamID, uint64_t IId);

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
  std::vector<sycl::detail::pi::PiQueue> MQueues;
  /// Iterator through MQueues.
  size_t MNextQueueIdx = 0;

  const bool MHostQueue = false;
  /// Indicates that a native out-of-order queue could not be created and we
  /// need to emulate it with multiple native in-order queues.
  bool MEmulateOOO = false;

  // Buffer to store assert failure descriptor
  buffer<AssertHappened, 1> MAssertHappenedBuffer;

  // This event is employed for enhanced dependency tracking with in-order queue
  // Access to the event should be guarded with MLastEventMtx
  event MLastEvent;
  mutable std::mutex MLastEventMtx;
  // Used for in-order queues in pair with MLastEvent
  // Host tasks are explicitly synchronized in RT, pi tasks - implicitly by
  // backend. Using type to setup explicit sync between host and pi tasks.
  CG::CGTYPE MLastCGType = CG::CGTYPE::None;

  const bool MIsInorder;

  std::vector<EventImplPtr> MStreamsServiceEvents;

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

public:
  // Queue constructed with the discard_events property
  const bool MDiscardEvents;
  const bool MIsProfilingEnabled;

protected:
  // This flag says if we can discard events based on a queue "setup" which will
  // be common for all operations submitted to the queue. This is a must
  // condition for discarding, but even if it's true, in some cases, we won't be
  // able to discard events, because the final decision is made right before the
  // operation itself.
  const bool MHasDiscardEventsSupport;

  // Command graph which is associated with this queue for the purposes of
  // recording commands to it.
  std::weak_ptr<ext::oneapi::experimental::detail::graph_impl> MGraph{};

  unsigned long long MQueueID;
  static std::atomic<unsigned long long> MNextAvailableQueueID;

  friend class sycl::ext::oneapi::experimental::detail::node_impl;
};

} // namespace detail
} // namespace _V1
} // namespace sycl
