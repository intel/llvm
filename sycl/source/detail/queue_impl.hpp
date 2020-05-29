//==------------------ queue_impl.hpp - SYCL queue -------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/context.hpp>
#include <CL/sycl/device.hpp>
#include <CL/sycl/event.hpp>
#include <CL/sycl/exception.hpp>
#include <CL/sycl/exception_list.hpp>
#include <CL/sycl/handler.hpp>
#include <CL/sycl/property_list.hpp>
#include <CL/sycl/stl.hpp>
#include <detail/context_impl.hpp>
#include <detail/device_impl.hpp>
#include <detail/event_impl.hpp>
#include <detail/plugin.hpp>
#include <detail/scheduler/scheduler.hpp>
#include <detail/thread_pool.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

using ContextImplPtr = std::shared_ptr<detail::context_impl>;
using DeviceImplPtr = shared_ptr_class<detail::device_impl>;

/// Sets max number of queues supported by FPGA RT.
const size_t MaxNumQueues = 256;

//// Possible CUDA context types supported by PI CUDA backend
/// TODO: Implement this as a property once there is an extension document
enum class cuda_context_type : char { primary, custom };

/// Default context type created for CUDA backend
constexpr cuda_context_type DefaultContextType = cuda_context_type::custom;

enum QueueOrder { Ordered, OOO };

class queue_impl {
public:
  /// Constructs a SYCL queue from a device using an async_handler and
  /// property_list provided.
  ///
  /// \param Device is a SYCL device that is used to dispatch tasks submitted
  /// to the queue.
  /// \param AsyncHandler is a SYCL asynchronous exception handler.
  /// \param Order specifies whether the queue being constructed as in-order
  /// or out-of-order.
  /// \param PropList is a list of properties to use for queue construction.
  queue_impl(DeviceImplPtr Device, async_handler AsyncHandler, QueueOrder Order,
             const property_list &PropList)
      : queue_impl(Device,
                   detail::getSyclObjImpl(context(
                       createSyclObjFromImpl<device>(Device), {},
                       (DefaultContextType == cuda_context_type::primary))),
                   AsyncHandler, Order, PropList){};

  /// Constructs a SYCL queue with an async_handler and property_list provided
  /// form a device and a context.
  ///
  /// \param Device is a SYCL device that is used to dispatch tasks submitted
  /// to the queue.
  /// \param Context is a SYCL context to associate with the queue being
  /// constructed.
  /// \param AsyncHandler is a SYCL asynchronous exception handler.
  /// \param Order specifies whether the queue being constructed as in-order
  /// or out-of-order.
  /// \param PropList is a list of properties to use for queue construction.
  queue_impl(DeviceImplPtr Device, ContextImplPtr Context,
             async_handler AsyncHandler, QueueOrder Order,
             const property_list &PropList)
      : MDevice(Device), MContext(Context), MAsyncHandler(AsyncHandler),
        MPropList(PropList), MHostQueue(MDevice->is_host()),
        MOpenCLInterop(!MHostQueue) {
    if (!Context->hasDevice(Device))
      throw cl::sycl::invalid_parameter_error(
          "Queue cannot be constructed with the given context and device "
          "as the context does not contain the given device.",
          PI_INVALID_DEVICE);
    if (!MHostQueue) {
      MCommandQueue = createQueue(Order);
    }
  }

  /// Constructs a SYCL queue from plugin interoperability handle.
  ///
  /// \param PiQueue is a raw PI queue handle.
  /// \param Context is a SYCL context to associate with the queue being
  /// constructed.
  /// \param AsyncHandler is a SYCL asynchronous exception handler.
  queue_impl(RT::PiQueue PiQueue, ContextImplPtr Context,
             const async_handler &AsyncHandler)
      : MContext(Context), MAsyncHandler(AsyncHandler), MHostQueue(false),
        MOpenCLInterop(true) {

    MCommandQueue = pi::cast<RT::PiQueue>(PiQueue);

    RT::PiDevice Device = nullptr;
    const detail::plugin &Plugin = getPlugin();
    // TODO catch an exception and put it to list of asynchronous exceptions
    Plugin.call<PiApiKind::piQueueGetInfo>(MCommandQueue, PI_QUEUE_INFO_DEVICE,
                                           sizeof(Device), &Device, nullptr);
    MDevice =
        DeviceImplPtr(new device_impl(Device, Context->getPlatformImpl()));

    // TODO catch an exception and put it to list of asynchronous exceptions
    Plugin.call<PiApiKind::piQueueRetain>(MCommandQueue);
  }

  ~queue_impl() {
    throw_asynchronous();
    if (MOpenCLInterop) {
      getPlugin().call<PiApiKind::piQueueRelease>(MCommandQueue);
    }
  }

  /// \return an OpenCL interoperability queue handle.
  cl_command_queue get() {
    if (MOpenCLInterop) {
      getPlugin().call<PiApiKind::piQueueRetain>(MCommandQueue);
      return pi::cast<cl_command_queue>(MCommandQueue);
    }
    throw invalid_object_error(
        "This instance of queue doesn't support OpenCL interoperability",
        PI_INVALID_QUEUE);
  }

  /// \return an associated SYCL context.
  context get_context() const {
    return createSyclObjFromImpl<context>(MContext);
  }

  const plugin &getPlugin() const { return MContext->getPlugin(); }

  ContextImplPtr getContextImplPtr() const { return MContext; }

  /// \return an associated SYCL device.
  device get_device() const { return createSyclObjFromImpl<device>(MDevice); }

  /// \return true if this queue is a SYCL host queue.
  bool is_host() const { return MHostQueue; }

  /// Queries SYCL queue for information.
  ///
  /// The return type depends on information being queried.
  template <info::queue param>
  typename info::param_traits<info::queue, param>::return_type get_info() const;

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
  /// \return a SYCL event object, which corresponds to the queue the command
  /// group is being enqueued on.
  event submit(const function_class<void(handler &)> &CGF,
               shared_ptr_class<queue_impl> Self,
               shared_ptr_class<queue_impl> SecondQueue,
               const detail::code_location &Loc) {
    try {
      return submit_impl(CGF, Self, Loc);
    } catch (...) {
      {
        std::lock_guard<mutex_class> Guard(MMutex);
        MExceptions.PushBack(std::current_exception());
      }
      return SecondQueue->submit(CGF, SecondQueue, Loc);
    }
  }

  /// Submits a command group function object to the queue, in order to be
  /// scheduled for execution on the device.
  ///
  /// \param CGF is a function object containing command group.
  /// \param Self is a shared_ptr to this queue.
  /// \param Loc is the code location of the submit call (default argument)
  /// \return a SYCL event object for the submitted command group.
  event submit(const function_class<void(handler &)> &CGF,
               shared_ptr_class<queue_impl> Self,
               const detail::code_location &Loc) {
    return submit_impl(CGF, std::move(Self), Loc);
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
    std::unique_lock<mutex_class> lock(MMutex);

    if (MAsyncHandler && MExceptions.size()) {
      exception_list Exceptions;

      std::swap(MExceptions, Exceptions);

      // Unlock the mutex before calling user-provided handler to avoid
      // potential deadlock if the same queue is somehow referenced in the
      // handler.
      lock.unlock();

      MAsyncHandler(std::move(Exceptions));
    }
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
    RT::PiQueue Queue;
    RT::PiContext Context = MContext->getHandleRef();
    RT::PiDevice Device = MDevice->getHandleRef();
    const detail::plugin &Plugin = getPlugin();

    assert(Plugin.getBackend() == MDevice->getPlugin().getBackend());
    RT::PiResult Error = Plugin.call_nocheck<PiApiKind::piQueueCreate>(
        Context, Device, CreationFlags, &Queue);

    // If creating out-of-order queue failed and this property is not
    // supported (for example, on FPGA), it will return
    // CL_INVALID_QUEUE_PROPERTIES and will try to create in-order queue.
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
    std::lock_guard<mutex_class> Guard(MMutex);

    // To achieve parallelism for FPGA with in order execution model with
    // possibility of two kernels to share data with each other we shall
    // create a queue for every kernel enqueued.
    if (MQueues.size() < MaxNumQueues) {
      MQueues.push_back(createQueue(QueueOrder::Ordered));
      return MQueues.back();
    }

    // If the limit of OpenCL queues is going to be exceeded - take the
    // earliest used queue, wait until it finished and then reuse it.
    MQueueNumber %= MaxNumQueues;
    size_t FreeQueueNum = MQueueNumber++;

    getPlugin().call<PiApiKind::piQueueFinish>(MQueues[FreeQueueNum]);
    return MQueues[FreeQueueNum];
  }

  /// \return a raw PI queue handle. The returned handle is not retained. It
  /// is caller responsibility to make sure queue is still alive.
  RT::PiQueue &getHandleRef() {
    if (MSupportOOO) {
      return MCommandQueue;
    }

    {
      // Reduce the scope since this mutex is also
      // locked inside of getExclusiveQueueHandleRef()
      std::lock_guard<mutex_class> Guard(MMutex);

      if (MQueues.empty()) {
        MQueues.push_back(MCommandQueue);
        return MCommandQueue;
      }
    }

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
  /// \param Impl is a shared_ptr to this queue.
  /// \param Ptr is a USM pointer to the memory to fill.
  /// \param Value is a value to be set. Value is cast as an unsigned char.
  /// \param Count is a number of bytes to fill.
  /// \return an event representing fill operation.
  event memset(shared_ptr_class<queue_impl> Impl, void *Ptr, int Value,
               size_t Count);
  /// Copies data from one memory region to another, both pointed by
  /// USM pointers.
  ///
  /// \param Impl is a shared_ptr to this queue.
  /// \param Dest is a USM pointer to the destination memory.
  /// \param Src is a USM pointer to the source memory.
  /// \param Count is a number of bytes to copy.
  event memcpy(shared_ptr_class<queue_impl> Impl, void *Dest, const void *Src,
               size_t Count);
  /// Provides additional information to the underlying runtime about how
  /// different allocations are used.
  ///
  /// \param Ptr is a USM pointer to the allocation.
  /// \param Length is a number of bytes in the allocation.
  /// \param Advice is a device-defined advice for the specified allocation.
  event mem_advise(const void *Ptr, size_t Length, pi_mem_advice Advice);

  /// Puts exception to the list of asynchronous ecxeptions.
  ///
  /// \param ExceptionPtr is a pointer to exception to be put.
  void reportAsyncException(std::exception_ptr ExceptionPtr) {
    std::lock_guard<mutex_class> Guard(MMutex);
    MExceptions.PushBack(ExceptionPtr);
  }

  ThreadPool &getThreadPool() {
    if (!MHostTaskThreadPool)
      initHostTaskAndEventCallbackThreadPool();

    return *MHostTaskThreadPool;
  }

  /// Gets the native handle of the SYCL queue.
  ///
  /// \return a native handle.
  pi_native_handle getNative() const;

private:
  /// Performs command group submission to the queue.
  ///
  /// \param CGF is a function object containing command group.
  /// \param Self is a pointer to this queue.
  /// \param Loc is the code location of the submit call (default argument)
  /// \return a SYCL event representing submitted command group.
  event submit_impl(const function_class<void(handler &)> &CGF,
                    shared_ptr_class<queue_impl> Self,
                    const detail::code_location &Loc) {
    handler Handler(std::move(Self), MHostQueue);
    Handler.saveCodeLoc(Loc);
    CGF(Handler);
    event Event = Handler.finalize();
    addEvent(Event);
    return Event;
  }

  // When instrumentation is enabled emits trace event for wait begin and
  // returns the telemetry event generated for the wait
  void *instrumentationProlog(const detail::code_location &CodeLoc,
                              string_class &Name, int32_t StreamID,
                              uint64_t &iid);
  // Uses events generated by the Prolog and emits wait done event
  void instrumentationEpilog(void *TelementryEvent, string_class &Name,
                             int32_t StreamID, uint64_t IId);

  void initHostTaskAndEventCallbackThreadPool();

  /// Stores a USM operation event that should be associated with the queue
  ///
  /// \param Event is the event to be stored
  void addUSMEvent(event Event);

  /// Stores an event that should be associated with the queue
  ///
  /// \param Event is the event to be stored
  void addEvent(event Event);

  /// Protects all the fields that can be changed by class' methods.
  mutex_class MMutex;

  DeviceImplPtr MDevice;
  const ContextImplPtr MContext;
  vector_class<std::weak_ptr<event_impl>> MEvents;
  // USM operations are not added to the scheduler command graph,
  // queue is the only owner on the runtime side.
  vector_class<event> MUSMEvents;
  exception_list MExceptions;
  const async_handler MAsyncHandler;
  const property_list MPropList;

  RT::PiQueue MCommandQueue = nullptr;

  /// List of queues created for FPGA device from a single SYCL queue.
  vector_class<RT::PiQueue> MQueues;
  /// Iterator through MQueues.
  size_t MQueueNumber = 0;

  const bool MHostQueue = false;
  const bool MOpenCLInterop = false;
  // Assume OOO support by default.
  bool MSupportOOO = true;

  // Thread pool for host task and event callbacks execution.
  // The thread pool is instantiated upon the very first call to getThreadPool()
  std::unique_ptr<ThreadPool> MHostTaskThreadPool;
};

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
