//==-------------------- queue.hpp - SYCL queue ----------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/device.hpp>
#include <CL/sycl/device_selector.hpp>
#include <CL/sycl/event.hpp>
#include <CL/sycl/exception_list.hpp>
#include <CL/sycl/handler.hpp>
#include <CL/sycl/info/info_desc.hpp>
#include <CL/sycl/property_list.hpp>
#include <CL/sycl/stl.hpp>

#include <utility>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {

// Forward declaration
class context;
class device;
namespace detail {
class queue_impl;
}

class queue {
public:
  /// Constructs a SYCL queue instance using the device returned by an instance
  /// of default_selector.
  ///
  /// \param PropList is a list of properties for queue construction.
  explicit queue(const property_list &PropList = {})
      : queue(default_selector(), async_handler{}, PropList) {}

  /// Constructs a SYCL queue instance with an async_handler using the device
  /// returned by an instance of default_selector.
  ///
  /// \param AsyncHandler is a SYCL asynchronous exception handler.
  /// \param PropList is a list of properties for queue construction.
  queue(const async_handler &AsyncHandler, const property_list &PropList = {})
      : queue(default_selector(), AsyncHandler, PropList) {}

  /// Constructs a SYCL queue instance using the device returned by the
  /// DeviceSelector provided.
  ///
  /// \param DeviceSelector is an instance of SYCL device selector.
  /// \param PropList is a list of properties for queue construction.
  queue(const device_selector &DeviceSelector,
        const property_list &PropList = {})
      : queue(DeviceSelector.select_device(), async_handler{}, PropList) {}

  /// Constructs a SYCL queue instance with an async_handler using the device
  /// returned by the DeviceSelector provided.
  ///
  /// \param DeviceSelector is an instance of SYCL device selector.
  /// \param AsyncHandler is a SYCL asynchronous exception handler.
  /// \param PropList is a list of properties for queue construction.
  queue(const device_selector &DeviceSelector,
        const async_handler &AsyncHandler, const property_list &PropList = {})
      : queue(DeviceSelector.select_device(), AsyncHandler, PropList) {}

  /// Constructs a SYCL queue instance using the device provided.
  ///
  /// \param SyclDevice is an instance of SYCL device.
  /// \param PropList is a list of properties for queue construction.
  explicit queue(const device &SyclDevice, const property_list &PropList = {})
      : queue(SyclDevice, async_handler{}, PropList) {}

  /// Constructs a SYCL queue instance with an async_handler using the device
  /// provided.
  ///
  /// \param SyclDevice is an instance of SYCL device.
  /// \param AsyncHandler is a SYCL asynchronous exception handler.
  /// \param PropList is a list of properties for queue construction.
  explicit queue(const device &SyclDevice, const async_handler &AsyncHandler,
                 const property_list &PropList = {});

  /// Constructs a SYCL queue instance that is associated with the context
  /// provided, using the device returned by the device selector.
  ///
  /// \param SyclContext is an instance of SYCL context.
  /// \param DeviceSelector is an instance of SYCL device selector.
  /// \param PropList is a list of properties for queue construction.
  queue(const context &SyclContext, const device_selector &DeviceSelector,
        const property_list &PropList = {});

  /// Constructs a SYCL queue instance with an async_handler that is associated
  /// with the context provided, using the device returned by the device
  /// selector.
  ///
  /// \param SyclContext is an instance of SYCL context.
  /// \param DeviceSelector is an instance of SYCL device selector.
  /// \param AsyncHandler is a SYCL asynchronous exception handler.
  /// \param PropList is a list of properties for queue construction.
  queue(const context &SyclContext, const device_selector &DeviceSelector,
        const async_handler &AsyncHandler, const property_list &PropList = {});

  /// Constructs a SYCL queue associated with the given context, device
  /// and optional properties list.
  ///
  /// \param SyclContext is an instance of SYCL context.
  /// \param SyclDevice is an instance of SYCL device.
  /// \param PropList is a list of properties for queue construction.
  queue(const context &SyclContext, const device &SyclDevice,
        const property_list &PropList = {});

  /// Constructs a SYCL queue associated with the given context, device,
  /// asynchronous exception handler and optional properties list.
  ///
  /// \param SyclContext is an instance of SYCL context.
  /// \param SyclDevice is an instance of SYCL device.
  /// \param AsyncHandler is a SYCL asynchronous exception handler.
  /// \param PropList is a list of properties for queue construction.
  queue(const context &SyclContext, const device &SyclDevice,
        const async_handler &AsyncHandler, const property_list &PropList = {});

  /// Constructs a SYCL queue with an optional async_handler from an OpenCL
  /// cl_command_queue.
  ///
  /// The instance of cl_command_queue is retained on construction.
  ///
  /// \param ClQueue is a valid instance of OpenCL queue.
  /// \param SyclContext is a valid SYCL context.
  /// \param AsyncHandler is a SYCL asynchronous exception handler.
  queue(cl_command_queue ClQueue, const context &SyclContext,
        const async_handler &AsyncHandler = {});

  queue(const queue &rhs) = default;

  queue(queue &&rhs) = default;

  queue &operator=(const queue &rhs) = default;

  queue &operator=(queue &&rhs) = default;

  bool operator==(const queue &rhs) const { return impl == rhs.impl; }

  bool operator!=(const queue &rhs) const { return !(*this == rhs); }

  /// \return a valid instance of OpenCL queue, which is retained before being
  /// returned.
  cl_command_queue get() const;

  /// \return an associated SYCL context.
  context get_context() const;

  /// \return SYCL device this queue was constructed with.
  device get_device() const;

  /// \return true if this queue is a SYCL host queue.
  bool is_host() const;

  /// Queries SYCL queue for information.
  ///
  /// The return type depends on information being queried.
  template <info::queue param>
  typename info::param_traits<info::queue, param>::return_type get_info() const;

  /// Submits a command group function object to the queue, in order to be
  /// scheduled for execution on the device.
  ///
  /// \param CGF is a function object containing command group.
  /// \param CodeLoc is the code location of the submit call (default argument)
  /// \return a SYCL event object for the submitted command group.
  template <typename T>
  event
  submit(T CGF
#ifndef DISABLE_SYCL_INSTRUMENTATION_METADATA
         ,
         const detail::code_location &CodeLoc = detail::code_location::current()
#endif
  ) {
#ifdef DISABLE_SYCL_INSTRUMENTATION_METADATA
    const detail::code_location &CodeLoc = {};
#endif
    return submit_impl(CGF, CodeLoc);
  }

  /// Submits a command group function object to the queue, in order to be
  /// scheduled for execution on the device.
  ///
  /// On a kernel error, this command group function object is then scheduled
  /// for execution on a secondary queue.
  ///
  /// \param CGF is a function object containing command group.
  /// \param SecondaryQueue is a fallback SYCL queue.
  /// \param CodeLoc is the code location of the submit call (default argument)
  /// \return a SYCL event object, which corresponds to the queue the command
  /// group is being enqueued on.
  template <typename T>
  event
  submit(T CGF, queue &SecondaryQueue
#ifndef DISABLE_SYCL_INSTRUMENTATION_METADATA
         ,
         const detail::code_location &CodeLoc = detail::code_location::current()
#endif
  ) {
#ifdef DISABLE_SYCL_INSTRUMENTATION_METADATA
    const detail::code_location &CodeLoc = {};
#endif
    return submit_impl(CGF, SecondaryQueue, CodeLoc);
  }

  /// Performs a blocking wait for the completion of all enqueued tasks in the
  /// queue.
  ///
  /// Synchronous errors will be reported through SYCL exceptions.
  /// @param CodeLoc is the code location of the submit call (default argument)
  void wait(
#ifndef DISABLE_SYCL_INSTRUMENTATION_METADATA
      const detail::code_location &CodeLoc = detail::code_location::current()
#endif
  ) {
#ifdef DISABLE_SYCL_INSTRUMENTATION_METADATA
    const detail::code_location &CodeLoc = {};
#endif
    wait_proxy(CodeLoc);
  }

  /// Performs a blocking wait for the completion of all enqueued tasks in the
  /// queue.
  ///
  /// Synchronous errors will be reported through SYCL exceptions. Asynchronous
  /// errors will be passed to the async_handler passed to the queue on
  /// construction. If no async_handler was provided then asynchronous
  /// exceptions will be lost.
  /// @param CodeLoc is the code location of the submit call (default argument)
  void wait_and_throw(
#ifndef DISABLE_SYCL_INSTRUMENTATION_METADATA
      const detail::code_location &CodeLoc = detail::code_location::current()
#endif
  ) {
#ifdef DISABLE_SYCL_INSTRUMENTATION_METADATA
    const detail::code_location &CodeLoc = {};
#endif
    wait_and_throw_proxy(CodeLoc);
  }

  /// Proxy method for wait to forward the code location information to the
  /// implementation
  void wait_proxy(const detail::code_location &CodeLoc);
  /// Proxy method for wait_and_throw to forward the code location information
  /// to the implementation
  void wait_and_throw_proxy(const detail::code_location &CodeLoc);

  /// Checks if any asynchronous errors have been produced by the queue and if
  /// so reports them to the async_handler passed on the queue construction.
  ///
  /// If no async_handler was provided then asynchronous exceptions will be
  /// lost.
  void throw_asynchronous();

  /// \return true if the queue was constructed with property specified by
  /// PropertyT.
  template <typename PropertyT> bool has_property() const;

  /// \return a copy of the property of type PropertyT that the queue was
  /// constructed with. If the queue was not constructed with the PropertyT
  /// property, an invalid_object_error SYCL exception.
  template <typename propertyT> propertyT get_property() const;

  /// Fills the memory pointed by a USM pointer with the value specified.
  ///
  /// \param Ptr is a USM pointer to the memory to fill.
  /// \param Value is a value to be set. Value is cast as an unsigned char.
  /// \param Count is a number of bytes to fill.
  /// \return an event representing fill operation.
  event memset(void *Ptr, int Value, size_t Count);

  /// Copies data from one memory region to another, both pointed by
  /// USM pointers.
  ///
  /// \param Dest is a USM pointer to the destination memory.
  /// \param Src is a USM pointer to the source memory.
  /// \param Count is a number of bytes to copy.
  /// \return an event representing copy operation.
  event memcpy(void *Dest, const void *Src, size_t Count);

  /// Provides additional information to the underlying runtime about how
  /// different allocations are used.
  ///
  /// \param Ptr is a USM pointer to the allocation.
  /// \param Length is a number of bytes in the allocation.
  /// \param Advice is a device-defined advice for the specified allocation.
  /// \return an event representing advice operation.
  event mem_advise(const void *Ptr, size_t Length, pi_mem_advice Advice);

  /// Provides hints to the runtime library that data should be made available
  /// on a device earlier than Unified Shared Memory would normally require it
  /// to be available.
  ///
  /// \param Ptr is a USM pointer to the memory to be prefetched to the device.
  /// \param Count is a number of bytes to be prefetched.
  event prefetch(const void* Ptr, size_t Count) {
    return submit([=](handler &CGH) { CGH.prefetch(Ptr, Count); });
  }

  /// single_task version with a kernel represented as a lambda.
  ///
  /// \param KernelFunc is the Kernel functor or lambda
  /// \param CodeLoc contains the code location of user code
  template <typename KernelName = detail::auto_name, typename KernelType>
  event single_task(
      KernelType KernelFunc
#ifndef DISABLE_SYCL_INSTRUMENTATION_METADATA
      ,
      const detail::code_location &CodeLoc = detail::code_location::current()
#endif
  ) {
#ifdef DISABLE_SYCL_INSTRUMENTATION_METADATA
    const detail::code_location &CodeLoc = {};
#endif
    return submit(
        [&](handler &CGH) {
          CGH.template single_task<KernelName, KernelType>(KernelFunc);
        },
        CodeLoc);
  }

  /// single_task version with a kernel represented as a lambda.
  ///
  /// \param DepEvent is an event that specifies the kernel dependencies
  /// \param KernelFunc is the Kernel functor or lambda
  /// \param CodeLoc contains the code location of user code
  template <typename KernelName = detail::auto_name, typename KernelType>
  event single_task(
      event DepEvent, KernelType KernelFunc
#ifndef DISABLE_SYCL_INSTRUMENTATION_METADATA
      ,
      const detail::code_location &CodeLoc = detail::code_location::current()
#endif
  ) {
#ifdef DISABLE_SYCL_INSTRUMENTATION_METADATA
    const detail::code_location &CodeLoc = {};
#endif
    return submit(
        [&](handler &CGH) {
          CGH.depends_on(DepEvent);
          CGH.template single_task<KernelName, KernelType>(KernelFunc);
        },
        CodeLoc);
  }

  /// single_task version with a kernel represented as a lambda.
  ///
  /// \param DepEvents is a vector of events that specifies the kernel
  /// dependencies
  /// \param KernelFunc is the Kernel functor or lambda
  /// \param CodeLoc contains the code location of user code
  template <typename KernelName = detail::auto_name, typename KernelType>
  event single_task(
      const vector_class<event> &DepEvents, KernelType KernelFunc
#ifndef DISABLE_SYCL_INSTRUMENTATION_METADATA
      ,
      const detail::code_location &CodeLoc = detail::code_location::current()
#endif
  ) {
#ifdef DISABLE_SYCL_INSTRUMENTATION_METADATA
    const detail::code_location &CodeLoc = {};
#endif
    return submit(
        [&](handler &CGH) {
          CGH.depends_on(DepEvents);
          CGH.template single_task<KernelName, KernelType>(KernelFunc);
        },
        CodeLoc);
  }

  /// parallel_for version with a kernel represented as a lambda + range that
  /// specifies global size only.
  ///
  /// \param NumWorkItems is a range that specifies the work space of the kernel
  /// \param KernelFunc is the Kernel functor or lambda
  /// \param CodeLoc contains the code location of user code
  template <typename KernelName = detail::auto_name, typename KernelType,
            int Dims>
  event parallel_for(
      range<Dims> NumWorkItems, KernelType KernelFunc
#ifndef DISABLE_SYCL_INSTRUMENTATION_METADATA
      ,
      const detail::code_location &CodeLoc = detail::code_location::current()
#endif
  ) {
#ifdef DISABLE_SYCL_INSTRUMENTATION_METADATA
    const detail::code_location &CodeLoc = {};
#endif
    return submit(
        [&](handler &CGH) {
          CGH.template parallel_for<KernelName, KernelType, Dims>(NumWorkItems,
                                                                  KernelFunc);
        },
        CodeLoc);
  }

  /// parallel_for version with a kernel represented as a lambda + range that
  /// specifies global size only.
  ///
  /// \param NumWorkItems is a range that specifies the work space of the kernel
  /// \param DepEvent is an event that specifies the kernel dependencies
  /// \param KernelFunc is the Kernel functor or lambda
  /// \param CodeLoc contains the code location of user code
  template <typename KernelName = detail::auto_name, typename KernelType,
            int Dims>
  event parallel_for(
      range<Dims> NumWorkItems, event DepEvent, KernelType KernelFunc
#ifndef DISABLE_SYCL_INSTRUMENTATION_METADATA
      ,
      const detail::code_location &CodeLoc = detail::code_location::current()
#endif
  ) {
#ifdef DISABLE_SYCL_INSTRUMENTATION_METADATA
    const detail::code_location &CodeLoc = {};
#endif
    return submit(
        [&](handler &CGH) {
          CGH.depends_on(DepEvent);
          CGH.template parallel_for<KernelName, KernelType, Dims>(NumWorkItems,
                                                                  KernelFunc);
        },
        CodeLoc);
  }

  /// parallel_for version with a kernel represented as a lambda + range that
  /// specifies global size only.
  ///
  /// \param NumWorkItems is a range that specifies the work space of the kernel
  /// \param DepEvents is a vector of events that specifies the kernel
  /// dependencies
  /// \param KernelFunc is the Kernel functor or lambda
  /// \param CodeLoc contains the code location of user code
  template <typename KernelName = detail::auto_name, typename KernelType,
            int Dims>
  event parallel_for(
      range<Dims> NumWorkItems, const vector_class<event> &DepEvents,
      KernelType KernelFunc
#ifndef DISABLE_SYCL_INSTRUMENTATION_METADATA
      ,
      const detail::code_location &CodeLoc = detail::code_location::current()
#endif
  ) {
#ifdef DISABLE_SYCL_INSTRUMENTATION_METADATA
    const detail::code_location &CodeLoc = {};
#endif
    return submit(
        [&](handler &CGH) {
          CGH.depends_on(DepEvents);
          CGH.template parallel_for<KernelName, KernelType, Dims>(NumWorkItems,
                                                                  KernelFunc);
        },
        CodeLoc);
  }

  /// parallel_for version with a kernel represented as a lambda + range and
  /// offset that specify global size and global offset correspondingly.
  ///
  /// \param NumWorkItems is a range that specifies the work space of the kernel
  /// \param WorkItemOffset specifies the offset for each work item id
  /// \param KernelFunc is the Kernel functor or lambda
  /// \param CodeLoc contains the code location of user code
  template <typename KernelName = detail::auto_name, typename KernelType,
            int Dims>
  event parallel_for(
      range<Dims> NumWorkItems, id<Dims> WorkItemOffset, KernelType KernelFunc
#ifndef DISABLE_SYCL_INSTRUMENTATION_METADATA
      ,
      const detail::code_location &CodeLoc = detail::code_location::current()
#endif
  ) {
#ifdef DISABLE_SYCL_INSTRUMENTATION_METADATA
    const detail::code_location &CodeLoc = {};
#endif
    return submit(
        [&](handler &CGH) {
          CGH.template parallel_for<KernelName, KernelType, Dims>(
              NumWorkItems, WorkItemOffset, KernelFunc);
        },
        CodeLoc);
  }

  /// parallel_for version with a kernel represented as a lambda + range and
  /// offset that specify global size and global offset correspondingly.
  ///
  /// \param NumWorkItems is a range that specifies the work space of the kernel
  /// \param WorkItemOffset specifies the offset for each work item id
  /// \param DepEvent is an event that specifies the kernel dependencies
  /// \param KernelFunc is the Kernel functor or lambda
  /// \param CodeLoc contains the code location of user code
  template <typename KernelName = detail::auto_name, typename KernelType,
            int Dims>
  event parallel_for(
      range<Dims> NumWorkItems, id<Dims> WorkItemOffset, event DepEvent,
      KernelType KernelFunc
#ifndef DISABLE_SYCL_INSTRUMENTATION_METADATA
      ,
      const detail::code_location &CodeLoc = detail::code_location::current()
#endif
  ) {
#ifdef DISABLE_SYCL_INSTRUMENTATION_METADATA
    const detail::code_location &CodeLoc = {};
#endif
    return submit(
        [&](handler &CGH) {
          CGH.depends_on(DepEvent);
          CGH.template parallel_for<KernelName, KernelType, Dims>(
              NumWorkItems, WorkItemOffset, KernelFunc);
        },
        CodeLoc);
  }

  /// parallel_for version with a kernel represented as a lambda + range and
  /// offset that specify global size and global offset correspondingly.
  ///
  /// \param NumWorkItems is a range that specifies the work space of the kernel
  /// \param WorkItemOffset specifies the offset for each work item id
  /// \param DepEvents is a vector of events that specifies the kernel
  /// dependencies
  /// \param KernelFunc is the Kernel functor or lambda
  /// \param CodeLoc contains the code location of user code
  template <typename KernelName = detail::auto_name, typename KernelType,
            int Dims>
  event parallel_for(
      range<Dims> NumWorkItems, id<Dims> WorkItemOffset,
      const vector_class<event> &DepEvents, KernelType KernelFunc
#ifndef DISABLE_SYCL_INSTRUMENTATION_METADATA
      ,
      const detail::code_location &CodeLoc = detail::code_location::current()
#endif
  ) {
#ifdef DISABLE_SYCL_INSTRUMENTATION_METADATA
    const detail::code_location &CodeLoc = {};
#endif
    return submit(
        [&](handler &CGH) {
          CGH.depends_on(DepEvents);
          CGH.template parallel_for<KernelName, KernelType, Dims>(
              NumWorkItems, WorkItemOffset, KernelFunc);
        },
        CodeLoc);
  }

  /// parallel_for version with a kernel represented as a lambda + nd_range that
  /// specifies global, local sizes and offset.
  ///
  /// \param ExecutionRange is a range that specifies the work space of the
  /// kernel
  /// \param KernelFunc is the Kernel functor or lambda
  /// \param CodeLoc contains the code location of user code
  template <typename KernelName = detail::auto_name, typename KernelType,
            int Dims>
  event parallel_for(
      nd_range<Dims> ExecutionRange, KernelType KernelFunc
#ifndef DISABLE_SYCL_INSTRUMENTATION_METADATA
      ,
      const detail::code_location &CodeLoc = detail::code_location::current()
#endif
  ) {
#ifdef DISABLE_SYCL_INSTRUMENTATION_METADATA
    const detail::code_location &CodeLoc = {};
#endif
    return submit(
        [&](handler &CGH) {
          CGH.template parallel_for<KernelName, KernelType, Dims>(
              ExecutionRange, KernelFunc);
        },
        CodeLoc);
  }

  /// parallel_for version with a kernel represented as a lambda + nd_range that
  /// specifies global, local sizes and offset.
  ///
  /// \param ExecutionRange is a range that specifies the work space of the
  /// kernel
  /// \param DepEvent is an event that specifies the kernel dependencies
  /// \param KernelFunc is the Kernel functor or lambda
  /// \param CodeLoc contains the code location of user code
  template <typename KernelName = detail::auto_name, typename KernelType,
            int Dims>
  event parallel_for(
      nd_range<Dims> ExecutionRange, event DepEvent, KernelType KernelFunc
#ifndef DISABLE_SYCL_INSTRUMENTATION_METADATA
      ,
      const detail::code_location &CodeLoc = detail::code_location::current()
#endif
  ) {
#ifdef DISABLE_SYCL_INSTRUMENTATION_METADATA
    const detail::code_location &CodeLoc = {};
#endif
    return submit(
        [&](handler &CGH) {
          CGH.depends_on(DepEvent);
          CGH.template parallel_for<KernelName, KernelType, Dims>(
              ExecutionRange, KernelFunc);
        },
        CodeLoc);
  }

  /// parallel_for version with a kernel represented as a lambda + nd_range that
  /// specifies global, local sizes and offset.
  ///
  /// \param ExecutionRange is a range that specifies the work space of the
  /// kernel
  /// \param DepEvents is a vector of events that specifies the kernel
  /// dependencies
  /// \param KernelFunc is the Kernel functor or lambda
  /// \param CodeLoc contains the code location of user code
  template <typename KernelName = detail::auto_name, typename KernelType,
            int Dims>
  event parallel_for(
      nd_range<Dims> ExecutionRange, const vector_class<event> &DepEvents,
      KernelType KernelFunc
#ifndef DISABLE_SYCL_INSTRUMENTATION_METADATA
      ,
      const detail::code_location &CodeLoc = detail::code_location::current()
#endif
  ) {
#ifdef DISABLE_SYCL_INSTRUMENTATION_METADATA
    const detail::code_location &CodeLoc = {};
#endif
    return submit(
        [&](handler &CGH) {
          CGH.depends_on(DepEvents);
          CGH.template parallel_for<KernelName, KernelType, Dims>(
              ExecutionRange, KernelFunc);
        },
        CodeLoc);
  }

  /// Returns whether the queue is in order or OoO
  ///
  /// Equivalent to has_property<property::queue::in_order>()
  bool is_in_order() const;

private:
  shared_ptr_class<detail::queue_impl> impl;
  template <class Obj>
  friend decltype(Obj::impl) detail::getSyclObjImpl(const Obj &SyclObject);

  /// A template-free version of submit.
  event submit_impl(function_class<void(handler &)> CGH,
                    const detail::code_location &CodeLoc);
  /// A template-free version of submit.
  event submit_impl(function_class<void(handler &)> CGH, queue secondQueue,
                    const detail::code_location &CodeLoc);
};

} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)

namespace std {
template <> struct hash<cl::sycl::queue> {
  size_t operator()(const cl::sycl::queue &q) const {
    return std::hash<
        cl::sycl::shared_ptr_class<cl::sycl::detail::queue_impl>>()(
        cl::sycl::detail::getSyclObjImpl(q));
  }
};
} // namespace std
