//==------------------ queue_impl.hpp - SYCL queue -------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/context.hpp>
#include <CL/sycl/detail/assert_happened.hpp>
#include <CL/sycl/detail/cuda_definitions.hpp>
#include <CL/sycl/device.hpp>
#include <CL/sycl/event.hpp>
#include <CL/sycl/exception.hpp>
#include <CL/sycl/exception_list.hpp>
#include <CL/sycl/handler.hpp>
#include <CL/sycl/properties/context_properties.hpp>
#include <CL/sycl/properties/queue_properties.hpp>
#include <CL/sycl/property_list.hpp>
#include <CL/sycl/stl.hpp>
#include <detail/config.hpp>
#include <detail/context_impl.hpp>
#include <detail/device_impl.hpp>
#include <detail/event_impl.hpp>
#include <detail/global_handler.hpp>
#include <detail/kernel_impl.hpp>
#include <detail/plugin.hpp>
#include <detail/scheduler/scheduler.hpp>
#include <detail/thread_pool.hpp>

#include <utility>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
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

    if (DefaultContext->hasDevice(Device))
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
        MHasDiscardEventsSupport(
            MDiscardEvents &&
            (MHostQueue ? true
                        : (MIsInorder && getPlugin().getBackend() !=
                                             backend::ext_oneapi_level_zero))) {
    if (has_property<ext::oneapi::property::queue::discard_events>() &&
        has_property<property::queue::enable_profiling>()) {
      throw sycl::exception(make_error_code(errc::invalid),
                            "Queue cannot be constructed with both of "
                            "discard_events and enable_profiling.");
    }
    if (!Context->hasDevice(Device))
      throw cl::sycl::invalid_parameter_error(
          "Queue cannot be constructed with the given context and device "
          "as the context does not contain the given device.",
          PI_INVALID_DEVICE);
    if (!MHostQueue) {
      const QueueOrder QOrder =
          MPropList.has_property<property::queue::in_order>()
              ? QueueOrder::Ordered
              : QueueOrder::OOO;
      MQueues.push_back(createQueue(QOrder));
    }
  }

  /// Constructs a SYCL queue from plugin interoperability handle.
  ///
  /// \param PiQueue is a raw PI queue handle.
  /// \param Context is a SYCL context to associate with the queue being
  /// constructed.
  /// \param AsyncHandler is a SYCL asynchronous exception handler.
  queue_impl(RT::PiQueue PiQueue, const ContextImplPtr &Context,
             const async_handler &AsyncHandler)
      : MContext(Context), MAsyncHandler(AsyncHandler), MPropList(),
        MHostQueue(false), MAssertHappenedBuffer(range<1>{1}),
        MIsInorder(has_property<property::queue::in_order>()),
        MDiscardEvents(
            has_property<ext::oneapi::property::queue::discard_events>()),
        MHasDiscardEventsSupport(
            MDiscardEvents &&
            (MHostQueue ? true
                        : (MIsInorder && getPlugin().getBackend() !=
                                             backend::ext_oneapi_level_zero))) {
    if (has_property<ext::oneapi::property::queue::discard_events>() &&
        has_property<property::queue::enable_profiling>()) {
      throw sycl::exception(make_error_code(errc::invalid),
                            "Queue cannot be constructed with both of "
                            "discard_events and enable_profiling.");
    }

    MQueues.push_back(pi::cast<RT::PiQueue>(PiQueue));

    RT::PiDevice Device{};
    const detail::plugin &Plugin = getPlugin();
    // TODO catch an exception and put it to list of asynchronous exceptions
    Plugin.call<PiApiKind::piQueueGetInfo>(MQueues[0], PI_QUEUE_INFO_DEVICE,
                                           sizeof(Device), &Device, nullptr);
    MDevice =
        DeviceImplPtr(new device_impl(Device, Context->getPlatformImpl()));
  }

  ~queue_impl() {
    throw_asynchronous();
    if (!MHostQueue) {
      getPlugin().call<PiApiKind::piQueueRelease>(MQueues[0]);
    }
  }

  /// \return an OpenCL interoperability queue handle.
  cl_command_queue get() {
    if (MHostQueue) {
      throw invalid_object_error(
          "This instance of queue doesn't support OpenCL interoperability",
          PI_INVALID_QUEUE);
    }
    getPlugin().call<PiApiKind::piQueueRetain>(MQueues[0]);
    return pi::cast<cl_command_queue>(MQueues[0]);
  }

  /// \return an associated SYCL context.
  context get_context() const {
    return createSyclObjFromImpl<context>(MContext);
  }

  const plugin &getPlugin() const { return MContext->getPlugin(); }

  const ContextImplPtr &getContextImplPtr() const { return MContext; }

  const DeviceImplPtr &getDeviceImplPtr() const { return MDevice; }

  /// \return an associated SYCL device.
  device get_device() const { return createSyclObjFromImpl<device>(MDevice); }

  /// \return true if this queue is a SYCL host queue.
  bool is_host() const { return MHostQueue; }

  /// \return true if this queue has discard_events support.
  bool has_discard_events_support() const { return MHasDiscardEventsSupport; }

  /// Queries SYCL queue for information.
  ///
  /// The return type depends on information being queried.
  template <info::queue Param>
  typename info::param_traits<info::queue, Param>::return_type get_info() const;

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
      {
        std::lock_guard<std::mutex> Lock(MMutex);
        MExceptions.PushBack(std::current_exception());
      }
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

  /// Creates PI queue.
  ///
  /// \param Order specifies whether the queue being constructed as in-order
  /// or out-of-order.
  RT::PiQueue createQueue(QueueOrder Order) {
    RT::PiQueueProperties CreationFlags = 0;

    if (Order == QueueOrder::OOO) {
      CreationFlags = PI_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
    }
    if (MPropList.has_property<property::queue::enable_profiling>()) {
      CreationFlags |= PI_QUEUE_PROFILING_ENABLE;
    }
    if (MPropList.has_property<
            ext::oneapi::cuda::property::queue::use_default_stream>()) {
      CreationFlags |= __SYCL_PI_CUDA_USE_DEFAULT_STREAM;
    }
    RT::PiQueue Queue{};
    RT::PiContext Context = MContext->getHandleRef();
    RT::PiDevice Device = MDevice->getHandleRef();
    const detail::plugin &Plugin = getPlugin();

    assert(Plugin.getBackend() == MDevice->getPlugin().getBackend());
    RT::PiResult Error = Plugin.call_nocheck<PiApiKind::piQueueCreate>(
        Context, Device, CreationFlags, &Queue);

    // If creating out-of-order queue failed and this property is not
    // supported (for example, on FPGA), it will return
    // PI_INVALID_QUEUE_PROPERTIES and will try to create in-order queue.
    if (MSupportOOO && Error == PI_INVALID_QUEUE_PROPERTIES) {
      MSupportOOO = false;
      Queue = createQueue(QueueOrder::Ordered);
    } else {
      Plugin.checkPiResult(Error);
    }

    return Queue;
  }

  /// \return a raw PI handle for a free queue. The returned handle is not
  /// retained. It is caller responsibility to make sure queue is still alive.
  RT::PiQueue &getExclusiveQueueHandleRef() {
    RT::PiQueue *PIQ = nullptr;
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
      getPlugin().call<PiApiKind::piQueueFinish>(*PIQ);

    return *PIQ;
  }

  /// \return a raw PI queue handle. The returned handle is not retained. It
  /// is caller responsibility to make sure queue is still alive.
  RT::PiQueue &getHandleRef() {
    if (MSupportOOO)
      return MQueues[0];

    return getExclusiveQueueHandleRef();
  }

  /// \return true if the queue was constructed with property specified by
  /// PropertyT.
  template <typename propertyT> bool has_property() const {
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
               const std::vector<event> &DepEvents);
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
  pi_native_handle getNative() const;

  buffer<AssertHappened, 1> &getAssertHappenedBuffer() {
    return MAssertHappenedBuffer;
  }

private:
  void finalizeHandler(handler &Handler, const CG::CGTYPE &Type,
                       event &EventRet) {
    if (MIsInorder) {

      auto IsExpDepManaged = [](const CG::CGTYPE &Type) {
        return (Type == CG::CGTYPE::CodeplayHostTask ||
                Type == CG::CGTYPE::CodeplayInteropTask);
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
    handler Handler(Self, PrimaryQueue, SecondaryQueue, MHostQueue);
    Handler.saveCodeLoc(Loc);
    CGF(Handler);

    // Scheduler will later omit events, that are not required to execute tasks.
    // Host and interop tasks, however, are not submitted to low-level runtimes
    // and require separate dependency management.
    const CG::CGTYPE Type = Handler.getType();
    event Event;

    if (PostProcess) {
      bool IsKernel = Type == CG::Kernel;
      bool KernelUsesAssert = false;

      if (IsKernel)
        // Kernel only uses assert if it's non interop one
        KernelUsesAssert = !(Handler.MKernel && Handler.MKernel->isInterop()) &&
                           ProgramManager::getInstance().kernelUsesAssert(
                               Handler.MOSModuleHandle, Handler.MKernelName);

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
  std::mutex MMutex;

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
  std::vector<RT::PiQueue> MQueues;
  /// Iterator through MQueues.
  size_t MNextQueueIdx = 0;

  const bool MHostQueue = false;
  // Assume OOO support by default.
  bool MSupportOOO = true;

  // Buffer to store assert failure descriptor
  buffer<AssertHappened, 1> MAssertHappenedBuffer;

  // This event is employed for enhanced dependency tracking with in-order queue
  // Access to the event should be guarded with MLastEventMtx
  event MLastEvent;
  std::mutex MLastEventMtx;
  // Used for in-order queues in pair with MLastEvent
  // Host tasks is explicitly synchronized in RT, pi tasks - implicitly by
  // backend Using type to setup explicit sync between host and pi tasks.
  CG::CGTYPE MLastCGType = CG::CGTYPE::None;

  const bool MIsInorder;

public:
  // Queue constructed with the discard_events property
  const bool MDiscardEvents;

private:
  // This flag says if we can discard events based on a queue "setup" which will
  // be common for all operations submitted to the queue. This is a must
  // condition for discarding, but even if it's true, in some cases, we won't be
  // able to discard events, because the final decision is made right before the
  // operation itself.
  const bool MHasDiscardEventsSupport;
};

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
