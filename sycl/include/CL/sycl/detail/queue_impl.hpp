//==------------------ queue_impl.hpp - SYCL queue -------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/context.hpp>
#include <CL/sycl/detail/context_impl.hpp>
#include <CL/sycl/detail/device_impl.hpp>
#include <CL/sycl/detail/event_impl.hpp>
#include <CL/sycl/detail/scheduler/scheduler.hpp>
#include <CL/sycl/detail/function_class.hpp>
#include <CL/sycl/device.hpp>
#include <CL/sycl/event.hpp>
#include <CL/sycl/exception.hpp>
#include <CL/sycl/exception_list.hpp>
#include <CL/sycl/handler.hpp>
#include <CL/sycl/property_list.hpp>
#include <functional>

__SYCL_INLINE namespace cl {
namespace sycl {
namespace detail {

using ContextImplPtr = shared_ptr_class<detail::context_impl>;
using DeviceImplPtr = shared_ptr_class<detail::device_impl>;

// Set max number of queues supported by FPGA RT.
const size_t MaxNumQueues = 256;

enum QueueOrder { Ordered, OOO };

class queue_impl {
public:
  queue_impl(DeviceImplPtr Device, async_handler AsyncHandler,
             QueueOrder Order, const property_list &PropList)
      : queue_impl(Device,
                   detail::getSyclObjImpl(
                       context(createSyclObjFromImpl<device>(Device))),
                   AsyncHandler, Order, PropList){};

  queue_impl(DeviceImplPtr Device, ContextImplPtr Context,
             async_handler AsyncHandler, QueueOrder Order,
             const property_list &PropList)
      : MDevice(Device), MContext(Context), MAsyncHandler(AsyncHandler),
        MPropList(PropList), MHostQueue(MDevice->is_host()),
        MOpenCLInterop(!MHostQueue) {
    if (!MHostQueue) {
      MCommandQueue = createQueue(Order);
    }
  }

  queue_impl(cl_command_queue CLQueue, ContextImplPtr Context,
             const async_handler &AsyncHandler)
      : MContext(Context), MAsyncHandler(AsyncHandler),
        MHostQueue(false), MOpenCLInterop(true) {

    MCommandQueue = pi::cast<RT::PiQueue>(CLQueue);

    RT::PiDevice Device = nullptr;
    // TODO catch an exception and put it to list of asynchronous exceptions
    PI_CALL(piQueueGetInfo)(MCommandQueue, PI_QUEUE_INFO_DEVICE,
                            sizeof(Device), &Device, nullptr);
    MDevice = std::make_shared<device_impl>(Device);

    // TODO catch an exception and put it to list of asynchronous exceptions
    PI_CALL(piQueueRetain)(MCommandQueue);
  }

  ~queue_impl() {
    throw_asynchronous();
    if (MOpenCLInterop) {
      PI_CALL(piQueueRelease)(MCommandQueue);
    }
  }

  cl_command_queue get() {
    if (MOpenCLInterop) {
      PI_CALL(piQueueRetain)(MCommandQueue);
      return pi::cast<cl_command_queue>(MCommandQueue);
    }
    throw invalid_object_error(
        "This instance of queue doesn't support OpenCL interoperability");
  }

  context get_context() const { return createSyclObjFromImpl<context>(MContext); }

  ContextImplPtr get_context_impl() const {
    return MContext;
  }

  device get_device() const { return createSyclObjFromImpl<device>(MDevice); }

  bool is_host() const { return MHostQueue; }

  template <info::queue param>
  typename info::param_traits<info::queue, param>::return_type get_info() const;

  event submit(const function_class<void(handler&)> &CGF, shared_ptr_class<queue_impl> Self,
               shared_ptr_class<queue_impl> SecondQueue) {
    try {
      return submit_impl(CGF, Self);
    } catch (...) {
      {
        std::lock_guard<mutex_class> guard(MMutex);
        MExceptions.PushBack(std::current_exception());
      }
      return SecondQueue->submit(CGF, SecondQueue);
    }
  }

  event submit(const function_class<void(handler&)> &CGF, shared_ptr_class<queue_impl> Self) {
    return submit_impl(CGF, std::move(Self));
  }

  void wait() {
    std::lock_guard<mutex_class> guard(MMutex);
    for (auto &Event : MEvents)
      Event.wait();
    MEvents.clear();
  }

  exception_list getExceptionList() const { return MExceptions; }

  void wait_and_throw() {
    wait();
    throw_asynchronous();
  }

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
    RT::PiResult Error =
        PI_CALL_NOCHECK(piQueueCreate)(Context, Device, CreationFlags, &Queue);

    // If creating out-of-order queue failed and this property is not
    // supported (for example, on FPGA), it will return
    // CL_INVALID_QUEUE_PROPERTIES and will try to create in-order queue.
    if (MSupportOOO && Error == PI_INVALID_QUEUE_PROPERTIES) {
      MSupportOOO = false;
      Queue = createQueue(QueueOrder::Ordered);
    } else {
      RT::checkPiResult(Error);
    }

    return Queue;
  }

  // Warning. Returned reference will be invalid if queue_impl was destroyed.
  RT::PiQueue &getExclusiveQueueHandleRef() {
    std::lock_guard<mutex_class> guard(MMutex);

    // To achive parallelism for FPGA with in order execution model with
    // possibility of two kernels to share data with each other we shall
    // create a queue for every kernel enqueued.
    if (MQueues.size() < MaxNumQueues) {
      MQueues.push_back(createQueue(QueueOrder::Ordered));
      return MQueues.back();
    }

    // If the limit of OpenCL queues is going to be exceeded - take the earliest
    // used queue, wait until it finished and then reuse it.
    MQueueNumber %= MaxNumQueues;
    size_t FreeQueueNum = MQueueNumber++;

    PI_CALL(piQueueFinish)(MQueues[FreeQueueNum]);
    return MQueues[FreeQueueNum];
  }

  RT::PiQueue &getHandleRef() {
    if (MSupportOOO) {
      return MCommandQueue;
    }

    {
      // Reduce the scope since this mutex is also
      // locked inside of getExclusiveQueueHandleRef()
      std::lock_guard<mutex_class> guard(MMutex);

      if (MQueues.empty()) {
        MQueues.push_back(MCommandQueue);
        return MCommandQueue;
      }
    }

    return getExclusiveQueueHandleRef();
  }

  template <typename propertyT> bool has_property() const {
    return MPropList.has_property<propertyT>();
  }

  template <typename propertyT> propertyT get_property() const {
    return MPropList.get_property<propertyT>();
  }

  event memset(shared_ptr_class<queue_impl> Impl, void *Ptr, int Value,
               size_t Count);
  event memcpy(shared_ptr_class<queue_impl> Impl, void *Dest, const void *Src,
               size_t Count);
  event mem_advise(const void *Ptr, size_t Length, int Advice);

  void reportAsyncException(std::exception_ptr ExceptionPtr) {
    std::lock_guard<mutex_class> guard(MMutex);
    MExceptions.PushBack(ExceptionPtr);
  }

private:
  event submit_impl(const function_class<void(handler&)> &CGF, shared_ptr_class<queue_impl> Self) {
    handler Handler(std::move(Self), MHostQueue);
    CGF(Handler);
    event Event = Handler.finalize();
    {
      std::lock_guard<mutex_class> guard(MMutex);
      MEvents.push_back(Event);
    }
    return Event;
  }

  // Protects all the fields that can be changed by class' methods
  mutex_class MMutex;

  DeviceImplPtr MDevice;
  const ContextImplPtr MContext;
  vector_class<event> MEvents;
  exception_list MExceptions;
  const async_handler MAsyncHandler;
  const property_list MPropList;

  RT::PiQueue MCommandQueue = nullptr;

  // List of queues created for FPGA device from a single SYCL queue.
  vector_class<RT::PiQueue> MQueues;
  // Iterator through m_Queues.
  size_t MQueueNumber = 0;

  const bool MHostQueue = false;
  const bool MOpenCLInterop = false;
  // Assume OOO support by default.
  bool MSupportOOO = true;
};

} // namespace detail
} // namespace sycl
} // namespace cl
