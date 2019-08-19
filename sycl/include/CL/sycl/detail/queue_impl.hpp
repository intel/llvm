//==------------------ queue_impl.hpp - SYCL queue -------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/context.hpp>
#include <CL/sycl/detail/scheduler/scheduler.hpp>
#include <CL/sycl/device.hpp>
#include <CL/sycl/event.hpp>
#include <CL/sycl/exception.hpp>
#include <CL/sycl/exception_list.hpp>
#include <CL/sycl/handler.hpp>
#include <CL/sycl/property_list.hpp>

namespace cl {
namespace sycl {
namespace detail {

// Set max number of queues supported by FPGA RT.
const size_t MaxNumQueues = 256;

class queue_impl {
public:
  queue_impl(const device &SyclDevice, async_handler AsyncHandler,
             const property_list &PropList)
      : queue_impl(SyclDevice, context(SyclDevice), AsyncHandler, PropList) {};

  queue_impl(const device &SyclDevice, const context &Context,
             async_handler AsyncHandler, const property_list &PropList)
      : m_Device(SyclDevice), m_Context(Context), m_AsyncHandler(AsyncHandler),
        m_PropList(PropList), m_HostQueue(m_Device.is_host()) {
    m_OpenCLInterop = !m_HostQueue;
    if (!m_HostQueue) {
      m_CommandQueue = createQueue();
    }
  }

  queue_impl(cl_command_queue CLQueue, const context &SyclContext,
             const async_handler &AsyncHandler)
      : m_Context(SyclContext), m_AsyncHandler(AsyncHandler),
        m_OpenCLInterop(true), m_HostQueue(false) {

    m_CommandQueue = pi::cast<RT::PiQueue>(CLQueue);

    RT::PiDevice Device = nullptr;
    // TODO catch an exception and put it to list of asynchronous exceptions
    PI_CALL(RT::piQueueGetInfo(m_CommandQueue, PI_QUEUE_INFO_DEVICE,
                               sizeof(Device), &Device, nullptr));
    m_Device = createSyclObjFromImpl<device>(
        std::make_shared<device_impl_pi>(Device));

    // TODO catch an exception and put it to list of asynchronous exceptions
    PI_CALL(RT::piQueueRetain(m_CommandQueue));
  }

  ~queue_impl() {
    try {
      throw_asynchronous();
    } catch (...) {
      assert(!"The asynchronous error handler should not throw exceptions from "
              "the queue destructor.");
    }
    if (m_OpenCLInterop) {
      PI_CALL(RT::piQueueRelease(m_CommandQueue));
    }
  }

  cl_command_queue get() {
    if (m_OpenCLInterop) {
      PI_CALL(RT::piQueueRetain(m_CommandQueue));
      return pi::cast<cl_command_queue>(m_CommandQueue);
    }
    throw invalid_object_error(
        "This instance of queue doesn't support OpenCL interoperability");
  }

  context get_context() const { return m_Context; }

  device get_device() const { return m_Device; }

  bool is_host() const { return m_HostQueue; }

  template <info::queue param>
  typename info::param_traits<info::queue, param>::return_type get_info() const;

  template <typename T>
  event submit(T cgf, std::shared_ptr<queue_impl> self,
               std::shared_ptr<queue_impl> second_queue) {
    event Event;
    try {
      Event = submit_impl(cgf, self);
    } catch (...) {
      m_Exceptions.PushBack(std::current_exception());
      Event = second_queue->submit(cgf, second_queue);
    }
    return Event;
  }

  template <typename T> event submit(T cgf, std::shared_ptr<queue_impl> self) {
    event Event;
    try {
      Event = submit_impl(cgf, self);
    } catch (...) {
      m_Exceptions.PushBack(std::current_exception());
    }
    return Event;
  }

  void wait() {
    // TODO: Make thread safe.
    for (auto &evnt : m_Events)
      evnt.wait();
    m_Events.clear();
  }

  exception_list getExceptionList() const { return m_Exceptions; }

  void wait_and_throw() {
    wait();
    throw_asynchronous();
  }

  void throw_asynchronous() {
    if (m_AsyncHandler && m_Exceptions.size()) {
      m_AsyncHandler(m_Exceptions);
    }
    m_Exceptions.Clear();
  }

  RT::PiQueue createQueue() {
    RT::PiQueueProperties CreationFlags = 0;

    if (m_SupportOOO) {
      CreationFlags = PI_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
    }

    if (m_PropList.has_property<property::queue::enable_profiling>()) {
      CreationFlags |= PI_QUEUE_PROFILING_ENABLE;
    }

    RT::PiQueue Queue;
    RT::PiContext Context = detail::getSyclObjImpl(m_Context)->getHandleRef();
    RT::PiDevice Device = detail::getSyclObjImpl(m_Device)->getHandleRef();
    RT::PiResult Error = PI_CALL_RESULT(RT::piQueueCreate(Context, Device,
        CreationFlags, &Queue));

    // If creating out-of-order queue failed and this property is not
    // supported (for example, on FPGA), it will return
    // CL_INVALID_QUEUE_PROPERTIES and will try to create in-order queue.
    if (m_SupportOOO && Error == PI_INVALID_QUEUE_PROPERTIES) {
      m_SupportOOO = false;
      Queue = createQueue();
    } else {
      PI_CHECK(Error);
    }
    return Queue;
  }

  // Warning. Returned reference will be invalid if queue_impl was destroyed.
  RT::PiQueue &getExclusiveQueueHandleRef() {
    // To achive parallelism for FPGA with in order execution model with
    // possibility of two kernels to share data with each other we shall
    // create a queue for every kernel enqueued.
    if (m_Queues.size() < MaxNumQueues) {
      m_Queues.push_back(createQueue());
      return m_Queues.back();
    }

    // If the limit of OpenCL queues is going to be exceeded - take the earliest
    // used queue, wait until it finished and then reuse it.
    m_QueueNumber %= MaxNumQueues;
    size_t FreeQueueNum = m_QueueNumber++;

    PI_CALL(RT::piQueueFinish(m_Queues[FreeQueueNum]));
    return m_Queues[FreeQueueNum];
  }

  RT::PiQueue &getHandleRef() {
    if (m_SupportOOO) {
      return m_CommandQueue;
    }

    if (m_Queues.empty()) {
      m_Queues.push_back(m_CommandQueue);
      return m_CommandQueue;
    }

    return getExclusiveQueueHandleRef();
  }

  template <typename propertyT> bool has_property() const {
    return m_PropList.has_property<propertyT>();
  }

  template <typename propertyT> propertyT get_property() const {
    return m_PropList.get_property<propertyT>();
  }

  event memset(void* ptr, int value, size_t count);
  event memcpy(void* dest, const void* src, size_t count);

private:
  template <typename T>
  event submit_impl(T cgf, std::shared_ptr<queue_impl> self) {
    handler Handler(std::move(self), m_HostQueue);
    cgf(Handler);
    event Event = Handler.finalize();
    // TODO: Make thread safe.
    m_Events.push_back(Event);
    return Event;
  }

  device m_Device;
  context m_Context;
  vector_class<event> m_Events;
  exception_list m_Exceptions;
  async_handler m_AsyncHandler;
  property_list m_PropList;

  RT::PiQueue m_CommandQueue = nullptr;

  // List of queues created for FPGA device from a single SYCL queue.
  vector_class<RT::PiQueue> m_Queues;
  // Iterator through m_Queues.
  size_t m_QueueNumber = 0;

  bool m_OpenCLInterop = false;
  bool m_HostQueue = false;
  // Assume OOO support by default.
  bool m_SupportOOO = true;
};

} // namespace detail
} // namespace sycl
} // namespace cl
