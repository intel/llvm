//==-------------------- queue.hpp - SYCL queue ----------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/assert_happened.hpp>
#include <CL/sycl/detail/backend_traits.hpp>
#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/export.hpp>
#include <CL/sycl/detail/service_kernel_names.hpp>
#include <CL/sycl/device.hpp>
#include <CL/sycl/device_selector.hpp>
#include <CL/sycl/event.hpp>
#include <CL/sycl/exception_list.hpp>
#include <CL/sycl/handler.hpp>
#include <CL/sycl/info/info_desc.hpp>
#include <CL/sycl/property_list.hpp>
#include <CL/sycl/stl.hpp>

// Explicitly request format macros
#ifndef __STDC_FORMAT_MACROS
#define __STDC_FORMAT_MACROS 1
#endif
#include <cinttypes>
#include <utility>

// having _TWO_ mid-param #ifdefs makes the functions very difficult to read.
// Here we simplify the &CodeLoc declaration to be _CODELOCPARAM(&CodeLoc) and
// _CODELOCARG(&CodeLoc) Similarly, the KernelFunc param is simplified to be
// _KERNELFUNCPARAM(KernelFunc) Once the queue kernel functions are defined,
// these macros are #undef immediately.

// replace _CODELOCPARAM(&CodeLoc) with nothing
// or :   , const detail::code_location &CodeLoc =
// detail::code_location::current()
// replace _CODELOCARG(&CodeLoc) with nothing
// or :  const detail::code_location &CodeLoc = {}

#ifndef DISABLE_SYCL_INSTRUMENTATION_METADATA
#define _CODELOCONLYPARAM(a)                                                   \
  const detail::code_location a = detail::code_location::current()
#define _CODELOCPARAM(a)                                                       \
  , const detail::code_location a = detail::code_location::current()

#define _CODELOCARG(a)
#define _CODELOCFW(a) , a
#else
#define _CODELOCONLYPARAM(a)
#define _CODELOCPARAM(a)

#define _CODELOCARG(a) const detail::code_location a = {}
#define _CODELOCFW(a)
#endif

// replace _KERNELFUNCPARAM(KernelFunc) with   KernelType KernelFunc
//                                     or     const KernelType &KernelFunc
#ifdef __SYCL_NONCONST_FUNCTOR__
#define _KERNELFUNCPARAM(a) KernelType a
#else
#define _KERNELFUNCPARAM(a) const KernelType &a
#endif


__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {

// Forward declaration
class context;
class device;
class queue;

namespace detail {
class queue_impl;
#if __SYCL_USE_FALLBACK_ASSERT
static event submitAssertCapture(queue &, event &, queue *,
                                 const detail::code_location &);
#endif
} // namespace detail

/// Encapsulates a single SYCL queue which schedules kernels on a SYCL device.
///
/// A SYCL queue can be used to submit command groups to be executed by the SYCL
/// runtime.
///
/// \sa device
/// \sa handler
/// \sa event
/// \sa kernel
///
/// \ingroup sycl_api
class __SYCL_EXPORT queue {
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
#ifdef __SYCL_INTERNAL_API
  queue(cl_command_queue ClQueue, const context &SyclContext,
        const async_handler &AsyncHandler = {});
#endif

  queue(const queue &RHS) = default;

  queue(queue &&RHS) = default;

  queue &operator=(const queue &RHS) = default;

  queue &operator=(queue &&RHS) = default;

  bool operator==(const queue &RHS) const { return impl == RHS.impl; }

  bool operator!=(const queue &RHS) const { return !(*this == RHS); }

  /// \return a valid instance of OpenCL queue, which is retained before being
  /// returned.
#ifdef __SYCL_INTERNAL_API
  cl_command_queue get() const;
#endif

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

private:
  // A shorthand for `get_device().has()' which is expected to be a bit quicker
  // than the long version
  bool device_has(aspect Aspect) const;

public:
  /// Submits a command group function object to the queue, in order to be
  /// scheduled for execution on the device.
  ///
  /// \param CGF is a function object containing command group.
  /// \param CodeLoc is the code location of the submit call (default argument)
  /// \return a SYCL event object for the submitted command group.
  template <typename T> event submit(T CGF _CODELOCPARAM(&CodeLoc)) {
    _CODELOCARG(&CodeLoc);

#if __SYCL_USE_FALLBACK_ASSERT
    if (!is_host()) {
      auto PostProcess = [this, &CodeLoc](bool IsKernel, bool KernelUsesAssert,
                                          event &E) {
        if (IsKernel && !device_has(aspect::ext_oneapi_native_assert) &&
            KernelUsesAssert && !device_has(aspect::accelerator)) {
          // __devicelib_assert_fail isn't supported by Device-side Runtime
          // Linking against fallback impl of __devicelib_assert_fail is
          // performed by program manager class
          // Fallback assert isn't supported for FPGA
          submitAssertCapture(*this, E, /* SecondaryQueue = */ nullptr,
                              CodeLoc);
        }
      };

      auto Event = submit_impl_and_postprocess(CGF, CodeLoc, PostProcess);
      return discard_or_return(Event);
    } else
#endif // __SYCL_USE_FALLBACK_ASSERT
    {
      auto Event = submit_impl(CGF, CodeLoc);
      return discard_or_return(Event);
    }
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
  event submit(T CGF, queue &SecondaryQueue _CODELOCPARAM(&CodeLoc)) {
    _CODELOCARG(&CodeLoc);

#if __SYCL_USE_FALLBACK_ASSERT
    if (!is_host()) {
      auto PostProcess = [this, &SecondaryQueue, &CodeLoc](
                             bool IsKernel, bool KernelUsesAssert, event &E) {
        if (IsKernel && !device_has(aspect::ext_oneapi_native_assert) &&
            KernelUsesAssert && !device_has(aspect::accelerator)) {
          // Only secondary queues on devices need to be added to the assert
          // capture.
          // TODO: Handle case where primary queue is host but the secondary
          // queue is not.
          queue *DeviceSecondaryQueue =
              SecondaryQueue.is_host() ? nullptr : &SecondaryQueue;
          // __devicelib_assert_fail isn't supported by Device-side Runtime
          // Linking against fallback impl of __devicelib_assert_fail is
          // performed by program manager class
          // Fallback assert isn't supported for FPGA
          submitAssertCapture(*this, E, DeviceSecondaryQueue, CodeLoc);
        }
      };

      auto Event = submit_impl_and_postprocess(CGF, SecondaryQueue, CodeLoc,
                                               PostProcess);
      return discard_or_return(Event);
    } else
#endif // __SYCL_USE_FALLBACK_ASSERT
    {
      auto Event = submit_impl(CGF, SecondaryQueue, CodeLoc);
      return discard_or_return(Event);
    }
  }

  /// Prevents any commands submitted afterward to this queue from executing
  /// until all commands previously submitted to this queue have entered the
  /// complete state.
  ///
  /// \param CodeLoc is the code location of the submit call (default argument)
  /// \return a SYCL event object, which corresponds to the queue the command
  /// group is being enqueued on.
  event ext_oneapi_submit_barrier(_CODELOCONLYPARAM(&CodeLoc)) {
    return submit(
        [=](handler &CGH) { CGH.ext_oneapi_barrier(); } _CODELOCFW(CodeLoc));
  }

  /// Prevents any commands submitted afterward to this queue from executing
  /// until all commands previously submitted to this queue have entered the
  /// complete state.
  ///
  /// \param CodeLoc is the code location of the submit call (default argument)
  /// \return a SYCL event object, which corresponds to the queue the command
  /// group is being enqueued on.
  __SYCL2020_DEPRECATED("use 'ext_oneapi_submit_barrier' instead")
  event submit_barrier(_CODELOCONLYPARAM(&CodeLoc)) {
    _CODELOCARG(&CodeLoc);
    return ext_oneapi_submit_barrier(CodeLoc);
  }

  /// Prevents any commands submitted afterward to this queue from executing
  /// until all events in WaitList have entered the complete state. If WaitList
  /// is empty, then ext_oneapi_submit_barrier has no effect.
  ///
  /// \param WaitList is a vector of valid SYCL events that need to complete
  /// before barrier command can be executed.
  /// \param CodeLoc is the code location of the submit call (default argument)
  /// \return a SYCL event object, which corresponds to the queue the command
  /// group is being enqueued on.
  event ext_oneapi_submit_barrier(
      const std::vector<event> &WaitList _CODELOCPARAM(&CodeLoc)) {
    return submit([=](handler &CGH) {
      CGH.ext_oneapi_barrier(WaitList);
    } _CODELOCFW(CodeLoc));
  }

  /// Prevents any commands submitted afterward to this queue from executing
  /// until all events in WaitList have entered the complete state. If WaitList
  /// is empty, then submit_barrier has no effect.
  ///
  /// \param WaitList is a vector of valid SYCL events that need to complete
  /// before barrier command can be executed.
  /// \param CodeLoc is the code location of the submit call (default argument)
  /// \return a SYCL event object, which corresponds to the queue the command
  /// group is being enqueued on.
  __SYCL2020_DEPRECATED("use 'ext_oneapi_submit_barrier' instead")
  event
  submit_barrier(const std::vector<event> &WaitList _CODELOCPARAM(&CodeLoc)) {
    _CODELOCARG(&CodeLoc);
    return ext_oneapi_submit_barrier(WaitList, CodeLoc);
  }

  /// Performs a blocking wait for the completion of all enqueued tasks in the
  /// queue.
  ///
  /// Synchronous errors will be reported through SYCL exceptions.
  /// @param CodeLoc is the code location of the submit call (default argument)
  void wait(_CODELOCONLYPARAM(&CodeLoc)) {
    _CODELOCARG(&CodeLoc);

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
  void wait_and_throw(_CODELOCONLYPARAM(&CodeLoc)) {
    _CODELOCARG(&CodeLoc);

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
  template <typename PropertyT> PropertyT get_property() const;

  /// Fills the specified memory with the specified pattern.
  ///
  /// \param Ptr is the pointer to the memory to fill.
  /// \param Pattern is the pattern to fill into the memory.  T should be
  /// trivially copyable.
  /// \param Count is the number of times to fill Pattern into Ptr.
  /// \return an event representing fill operation.
  template <typename T> event fill(void *Ptr, const T &Pattern, size_t Count) {
    return submit([&](handler &CGH) { CGH.fill<T>(Ptr, Pattern, Count); });
  }

  /// Fills the specified memory with the specified pattern.
  ///
  /// \param Ptr is the pointer to the memory to fill.
  /// \param Pattern is the pattern to fill into the memory.  T should be
  /// trivially copyable.
  /// \param Count is the number of times to fill Pattern into Ptr.
  /// \param DepEvent is an event that specifies the kernel dependencies.
  /// \return an event representing fill operation.
  template <typename T>
  event fill(void *Ptr, const T &Pattern, size_t Count, event DepEvent) {
    return submit([&](handler &CGH) {
      CGH.depends_on(DepEvent);
      CGH.fill<T>(Ptr, Pattern, Count);
    });
  }

  /// Fills the specified memory with the specified pattern.
  ///
  /// \param Ptr is the pointer to the memory to fill.
  /// \param Pattern is the pattern to fill into the memory.  T should be
  /// trivially copyable.
  /// \param Count is the number of times to fill Pattern into Ptr.
  /// \param DepEvents is a vector of events that specifies the kernel
  /// dependencies.
  /// \return an event representing fill operation.
  template <typename T>
  event fill(void *Ptr, const T &Pattern, size_t Count,
             const std::vector<event> &DepEvents) {
    return submit([&](handler &CGH) {
      CGH.depends_on(DepEvents);
      CGH.fill<T>(Ptr, Pattern, Count);
    });
  }

  /// Fills the memory pointed by a USM pointer with the value specified.
  /// No operations is done if \param Count is zero. An exception is thrown
  /// if \param Ptr is nullptr. The behavior is undefined if \param Ptr
  /// is invalid.
  ///
  /// \param Ptr is a USM pointer to the memory to fill.
  /// \param Value is a value to be set. Value is cast as an unsigned char.
  /// \param Count is a number of bytes to fill.
  /// \return an event representing fill operation.
  event memset(void *Ptr, int Value, size_t Count);

  /// Fills the memory pointed by a USM pointer with the value specified.
  /// No operations is done if \param Count is zero. An exception is thrown
  /// if \param Ptr is nullptr. The behavior is undefined if \param Ptr
  /// is invalid.
  ///
  /// \param Ptr is a USM pointer to the memory to fill.
  /// \param Value is a value to be set. Value is cast as an unsigned char.
  /// \param Count is a number of bytes to fill.
  /// \param DepEvent is an event that specifies the kernel dependencies.
  /// \return an event representing fill operation.
  event memset(void *Ptr, int Value, size_t Count, event DepEvent);

  /// Fills the memory pointed by a USM pointer with the value specified.
  /// No operations is done if \param Count is zero. An exception is thrown
  /// if \param Ptr is nullptr. The behavior is undefined if \param Ptr
  /// is invalid.
  ///
  /// \param Ptr is a USM pointer to the memory to fill.
  /// \param Value is a value to be set. Value is cast as an unsigned char.
  /// \param Count is a number of bytes to fill.
  /// \param DepEvents is a vector of events that specifies the kernel
  /// dependencies.
  /// \return an event representing fill operation.
  event memset(void *Ptr, int Value, size_t Count,
               const std::vector<event> &DepEvents);

  /// Copies data from one memory region to another, both pointed by
  /// USM pointers.
  /// No operations is done if \param Count is zero. An exception is thrown
  /// if either \param Dest or \param Src is nullptr. The behavior is undefined
  /// if any of the pointer parameters is invalid.
  ///
  /// \param Dest is a USM pointer to the destination memory.
  /// \param Src is a USM pointer to the source memory.
  /// \param Count is a number of bytes to copy.
  /// \return an event representing copy operation.
  event memcpy(void *Dest, const void *Src, size_t Count);

  /// Copies data from one memory region to another, both pointed by
  /// USM pointers.
  /// No operations is done if \param Count is zero. An exception is thrown
  /// if either \param Dest or \param Src is nullptr. The behavior is undefined
  /// if any of the pointer parameters is invalid.
  ///
  /// \param Dest is a USM pointer to the destination memory.
  /// \param Src is a USM pointer to the source memory.
  /// \param Count is a number of bytes to copy.
  /// \param DepEvent is an event that specifies the kernel dependencies.
  /// \return an event representing copy operation.
  event memcpy(void *Dest, const void *Src, size_t Count, event DepEvent);

  /// Copies data from one memory region to another, both pointed by
  /// USM pointers.
  /// No operations is done if \param Count is zero. An exception is thrown
  /// if either \param Dest or \param Src is nullptr. The behavior is undefined
  /// if any of the pointer parameters is invalid.
  ///
  /// \param Dest is a USM pointer to the destination memory.
  /// \param Src is a USM pointer to the source memory.
  /// \param Count is a number of bytes to copy.
  /// \param DepEvents is a vector of events that specifies the kernel
  /// dependencies.
  /// \return an event representing copy operation.
  event memcpy(void *Dest, const void *Src, size_t Count,
               const std::vector<event> &DepEvents);

  /// Copies data from one memory region to another, both pointed by
  /// USM pointers.
  /// No operations is done if \param Count is zero. An exception is thrown
  /// if either \param Dest or \param Src is nullptr. The behavior is undefined
  /// if any of the pointer parameters is invalid.
  ///
  /// \param Src is a USM pointer to the source memory.
  /// \param Dest is a USM pointer to the destination memory.
  /// \param Count is a number of elements of type T to copy.
  /// \return an event representing copy operation.
  template <typename T> event copy(const T *Src, T *Dest, size_t Count) {
    return this->memcpy(Dest, Src, Count * sizeof(T));
  }

  /// Copies data from one memory region to another, both pointed by
  /// USM pointers.
  /// No operations is done if \param Count is zero. An exception is thrown
  /// if either \param Dest or \param Src is nullptr. The behavior is undefined
  /// if any of the pointer parameters is invalid.
  ///
  /// \param Src is a USM pointer to the source memory.
  /// \param Dest is a USM pointer to the destination memory.
  /// \param Count is a number of elements of type T to copy.
  /// \param DepEvent is an event that specifies the kernel dependencies.
  /// \return an event representing copy operation.
  template <typename T>
  event copy(const T *Src, T *Dest, size_t Count, event DepEvent) {
    return this->memcpy(Dest, Src, Count * sizeof(T), DepEvent);
  }

  /// Copies data from one memory region to another, both pointed by
  /// USM pointers.
  /// No operations is done if \param Count is zero. An exception is thrown
  /// if either \param Dest or \param Src is nullptr. The behavior is undefined
  /// if any of the pointer parameters is invalid.
  ///
  /// \param Src is a USM pointer to the source memory.
  /// \param Dest is a USM pointer to the destination memory.
  /// \param Count is a number of elements of type T to copy.
  /// \param DepEvents is a vector of events that specifies the kernel
  /// \return an event representing copy operation.
  template <typename T>
  event copy(const T *Src, T *Dest, size_t Count,
             const std::vector<event> &DepEvents) {
    return this->memcpy(Dest, Src, Count * sizeof(T), DepEvents);
  }

  /// Provides additional information to the underlying runtime about how
  /// different allocations are used.
  ///
  /// \param Ptr is a USM pointer to the allocation.
  /// \param Length is a number of bytes in the allocation.
  /// \param Advice is a device-defined advice for the specified allocation.
  /// \return an event representing advice operation.
  __SYCL2020_DEPRECATED("use the overload with int Advice instead")
  event mem_advise(const void *Ptr, size_t Length, pi_mem_advice Advice);

  /// Provides additional information to the underlying runtime about how
  /// different allocations are used.
  ///
  /// \param Ptr is a USM pointer to the allocation.
  /// \param Length is a number of bytes in the allocation.
  /// \param Advice is a device-defined advice for the specified allocation.
  /// \return an event representing advice operation.
  event mem_advise(const void *Ptr, size_t Length, int Advice);

  /// Provides additional information to the underlying runtime about how
  /// different allocations are used.
  ///
  /// \param Ptr is a USM pointer to the allocation.
  /// \param Length is a number of bytes in the allocation.
  /// \param Advice is a device-defined advice for the specified allocation.
  /// \param DepEvent is an event that specifies the kernel dependencies.
  /// \return an event representing advice operation.
  event mem_advise(const void *Ptr, size_t Length, int Advice, event DepEvent);

  /// Provides additional information to the underlying runtime about how
  /// different allocations are used.
  ///
  /// \param Ptr is a USM pointer to the allocation.
  /// \param Length is a number of bytes in the allocation.
  /// \param Advice is a device-defined advice for the specified allocation.
  /// \param DepEvents is a vector of events that specifies the kernel
  /// dependencies.
  /// \return an event representing advice operation.
  event mem_advise(const void *Ptr, size_t Length, int Advice,
                   const std::vector<event> &DepEvents);

  /// Provides hints to the runtime library that data should be made available
  /// on a device earlier than Unified Shared Memory would normally require it
  /// to be available.
  ///
  /// \param Ptr is a USM pointer to the memory to be prefetched to the device.
  /// \param Count is a number of bytes to be prefetched.
  /// \return an event representing prefetch operation.
  event prefetch(const void *Ptr, size_t Count) {
    return submit([=](handler &CGH) { CGH.prefetch(Ptr, Count); });
  }

  /// Provides hints to the runtime library that data should be made available
  /// on a device earlier than Unified Shared Memory would normally require it
  /// to be available.
  ///
  /// \param Ptr is a USM pointer to the memory to be prefetched to the device.
  /// \param Count is a number of bytes to be prefetched.
  /// \param DepEvent is an event that specifies the kernel dependencies.
  /// \return an event representing prefetch operation.
  event prefetch(const void *Ptr, size_t Count, event DepEvent) {
    return submit([=](handler &CGH) {
      CGH.depends_on(DepEvent);
      CGH.prefetch(Ptr, Count);
    });
  }

  /// Provides hints to the runtime library that data should be made available
  /// on a device earlier than Unified Shared Memory would normally require it
  /// to be available.
  ///
  /// \param Ptr is a USM pointer to the memory to be prefetched to the device.
  /// \param Count is a number of bytes to be prefetched.
  /// \param DepEvents is a vector of events that specifies the kernel
  /// dependencies.
  /// \return an event representing prefetch operation.
  event prefetch(const void *Ptr, size_t Count,
                 const std::vector<event> &DepEvents) {
    return submit([=](handler &CGH) {
      CGH.depends_on(DepEvents);
      CGH.prefetch(Ptr, Count);
    });
  }

  /// single_task version with a kernel represented as a lambda.
  ///
  /// \param KernelFunc is the Kernel functor or lambda
  /// \param CodeLoc contains the code location of user code
  template <typename KernelName = detail::auto_name, typename KernelType>
  event single_task(_KERNELFUNCPARAM(KernelFunc) _CODELOCPARAM(&CodeLoc)) {
    static_assert(
        (detail::check_fn_signature<detail::remove_reference_t<KernelType>,
                                    void()>::value ||
         detail::check_fn_signature<detail::remove_reference_t<KernelType>,
                                    void(kernel_handler)>::value),
        "sycl::queue.single_task() requires a kernel instead of command group. "
        "Use queue.submit() instead");
    _CODELOCARG(&CodeLoc);
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
  event single_task(event DepEvent,
                    _KERNELFUNCPARAM(KernelFunc) _CODELOCPARAM(&CodeLoc)) {
    static_assert(
        (detail::check_fn_signature<detail::remove_reference_t<KernelType>,
                                    void()>::value ||
         detail::check_fn_signature<detail::remove_reference_t<KernelType>,
                                    void(kernel_handler)>::value),
        "sycl::queue.single_task() requires a kernel instead of command group. "
        "Use queue.submit() instead");
    _CODELOCARG(&CodeLoc);
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
  event single_task(const std::vector<event> &DepEvents,
                    _KERNELFUNCPARAM(KernelFunc) _CODELOCPARAM(&CodeLoc)) {
    static_assert(
        (detail::check_fn_signature<detail::remove_reference_t<KernelType>,
                                    void()>::value ||
         detail::check_fn_signature<detail::remove_reference_t<KernelType>,
                                    void(kernel_handler)>::value),
        "sycl::queue.single_task() requires a kernel instead of command group. "
        "Use queue.submit() instead");
    _CODELOCARG(&CodeLoc);
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
  template <typename KernelName = detail::auto_name, typename KernelType>
  event parallel_for(range<1> NumWorkItems,
                     _KERNELFUNCPARAM(KernelFunc) _CODELOCPARAM(&CodeLoc)) {
    _CODELOCARG(&CodeLoc);
    return parallel_for_impl<KernelName>(NumWorkItems, KernelFunc, CodeLoc);
  }

  /// parallel_for version with a kernel represented as a lambda + range that
  /// specifies global size only.
  ///
  /// \param NumWorkItems is a range that specifies the work space of the kernel
  /// \param KernelFunc is the Kernel functor or lambda
  /// \param CodeLoc contains the code location of user code
  template <typename KernelName = detail::auto_name, typename KernelType>
  event parallel_for(range<2> NumWorkItems,
                     _KERNELFUNCPARAM(KernelFunc) _CODELOCPARAM(&CodeLoc)) {
    _CODELOCARG(&CodeLoc);
    return parallel_for_impl<KernelName>(NumWorkItems, KernelFunc, CodeLoc);
  }

  /// parallel_for version with a kernel represented as a lambda + range that
  /// specifies global size only.
  ///
  /// \param NumWorkItems is a range that specifies the work space of the kernel
  /// \param KernelFunc is the Kernel functor or lambda
  /// \param CodeLoc contains the code location of user code
  template <typename KernelName = detail::auto_name, typename KernelType>
  event parallel_for(range<3> NumWorkItems,
                     _KERNELFUNCPARAM(KernelFunc) _CODELOCPARAM(&CodeLoc)) {
    _CODELOCARG(&CodeLoc);
    return parallel_for_impl<KernelName>(NumWorkItems, KernelFunc, CodeLoc);
  }

  /// parallel_for version with a kernel represented as a lambda + range that
  /// specifies global size only.
  ///
  /// \param NumWorkItems is a range that specifies the work space of the kernel
  /// \param DepEvent is an event that specifies the kernel dependencies
  /// \param KernelFunc is the Kernel functor or lambda
  /// \param CodeLoc contains the code location of user code
  template <typename KernelName = detail::auto_name, typename KernelType>
  event parallel_for(range<1> NumWorkItems, event DepEvent,
                     _KERNELFUNCPARAM(KernelFunc) _CODELOCPARAM(&CodeLoc)) {
    _CODELOCARG(&CodeLoc);
    return parallel_for_impl<KernelName>(NumWorkItems, DepEvent, KernelFunc,
                                         CodeLoc);
  }

  /// parallel_for version with a kernel represented as a lambda + range that
  /// specifies global size only.
  ///
  /// \param NumWorkItems is a range that specifies the work space of the kernel
  /// \param DepEvent is an event that specifies the kernel dependencies
  /// \param KernelFunc is the Kernel functor or lambda
  /// \param CodeLoc contains the code location of user code
  template <typename KernelName = detail::auto_name, typename KernelType>
  event parallel_for(range<2> NumWorkItems, event DepEvent,
                     _KERNELFUNCPARAM(KernelFunc) _CODELOCPARAM(&CodeLoc)) {
    _CODELOCARG(&CodeLoc);
    return parallel_for_impl<KernelName>(NumWorkItems, DepEvent, KernelFunc,
                                         CodeLoc);
  }

  /// parallel_for version with a kernel represented as a lambda + range that
  /// specifies global size only.
  ///
  /// \param NumWorkItems is a range that specifies the work space of the kernel
  /// \param DepEvent is an event that specifies the kernel dependencies
  /// \param KernelFunc is the Kernel functor or lambda
  /// \param CodeLoc contains the code location of user code
  template <typename KernelName = detail::auto_name, typename KernelType>
  event parallel_for(range<3> NumWorkItems, event DepEvent,
                     _KERNELFUNCPARAM(KernelFunc) _CODELOCPARAM(&CodeLoc)) {
    _CODELOCARG(&CodeLoc);
    return parallel_for_impl<KernelName>(NumWorkItems, DepEvent, KernelFunc,
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
  template <typename KernelName = detail::auto_name, typename KernelType>
  event parallel_for(range<1> NumWorkItems, const std::vector<event> &DepEvents,
                     _KERNELFUNCPARAM(KernelFunc) _CODELOCPARAM(&CodeLoc)) {
    _CODELOCARG(&CodeLoc);
    return parallel_for_impl<KernelName>(NumWorkItems, DepEvents, KernelFunc,
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
  template <typename KernelName = detail::auto_name, typename KernelType>
  event parallel_for(range<2> NumWorkItems, const std::vector<event> &DepEvents,
                     _KERNELFUNCPARAM(KernelFunc) _CODELOCPARAM(&CodeLoc)) {
    _CODELOCARG(&CodeLoc);
    return parallel_for_impl<KernelName>(NumWorkItems, DepEvents, KernelFunc,
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
  template <typename KernelName = detail::auto_name, typename KernelType>
  event parallel_for(range<3> NumWorkItems, const std::vector<event> &DepEvents,
                     _KERNELFUNCPARAM(KernelFunc) _CODELOCPARAM(&CodeLoc)) {
    _CODELOCARG(&CodeLoc);
    return parallel_for_impl<KernelName>(NumWorkItems, DepEvents, KernelFunc,
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
  event parallel_for(range<Dims> NumWorkItems, id<Dims> WorkItemOffset,
                     _KERNELFUNCPARAM(KernelFunc) _CODELOCPARAM(&CodeLoc)) {
    _CODELOCARG(&CodeLoc);
    return submit(
        [&](handler &CGH) {
          CGH.template parallel_for<KernelName, KernelType>(
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
  event parallel_for(range<Dims> NumWorkItems, id<Dims> WorkItemOffset,
                     event DepEvent,
                     _KERNELFUNCPARAM(KernelFunc) _CODELOCPARAM(&CodeLoc)) {
    _CODELOCARG(&CodeLoc);
    return submit(
        [&](handler &CGH) {
          CGH.depends_on(DepEvent);
          CGH.template parallel_for<KernelName, KernelType>(
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
  event parallel_for(range<Dims> NumWorkItems, id<Dims> WorkItemOffset,
                     const std::vector<event> &DepEvents,
                     _KERNELFUNCPARAM(KernelFunc) _CODELOCPARAM(&CodeLoc)) {
    _CODELOCARG(&CodeLoc);
    return submit(
        [&](handler &CGH) {
          CGH.depends_on(DepEvents);
          CGH.template parallel_for<KernelName, KernelType>(
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
  event parallel_for(nd_range<Dims> ExecutionRange,
                     _KERNELFUNCPARAM(KernelFunc) _CODELOCPARAM(&CodeLoc)) {
    _CODELOCARG(&CodeLoc);
    return submit(
        [&](handler &CGH) {
          CGH.template parallel_for<KernelName, KernelType>(ExecutionRange,
                                                            KernelFunc);
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
  event parallel_for(nd_range<Dims> ExecutionRange, event DepEvent,
                     _KERNELFUNCPARAM(KernelFunc) _CODELOCPARAM(&CodeLoc)) {
    _CODELOCARG(&CodeLoc);
    return submit(
        [&](handler &CGH) {
          CGH.depends_on(DepEvent);
          CGH.template parallel_for<KernelName, KernelType>(ExecutionRange,
                                                            KernelFunc);
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
  event parallel_for(nd_range<Dims> ExecutionRange,
                     const std::vector<event> &DepEvents,
                     _KERNELFUNCPARAM(KernelFunc) _CODELOCPARAM(&CodeLoc)) {
    _CODELOCARG(&CodeLoc);
    return submit(
        [&](handler &CGH) {
          CGH.depends_on(DepEvents);
          CGH.template parallel_for<KernelName, KernelType>(ExecutionRange,
                                                            KernelFunc);
        },
        CodeLoc);
  }

  /// parallel_for version with a kernel represented as a lambda + nd_range that
  /// specifies global, local sizes and offset.
  ///
  /// \param ExecutionRange is a range that specifies the work space of the
  /// kernel
  /// \param Redu is a reduction operation
  /// \param KernelFunc is the Kernel functor or lambda
  /// \param CodeLoc contains the code location of user code
  template <typename KernelName = detail::auto_name, typename KernelType,
            int Dims, typename Reduction>
  event parallel_for(nd_range<Dims> ExecutionRange, Reduction Redu,
                     _KERNELFUNCPARAM(KernelFunc) _CODELOCPARAM(&CodeLoc)) {
    _CODELOCARG(&CodeLoc);
    return submit(
        [&](handler &CGH) {
          CGH.template parallel_for<KernelName, KernelType, Dims, Reduction>(
              ExecutionRange, Redu, KernelFunc);
        },
        CodeLoc);
  }

// Clean up CODELOC and KERNELFUNC macros.
#undef _CODELOCPARAM
#undef _CODELOCONLYPARAM
#undef _CODELOCARG
#undef _CODELOCFW
#undef _KERNELFUNCPARAM

  /// Returns whether the queue is in order or OoO
  ///
  /// Equivalent to has_property<property::queue::in_order>()
  bool is_in_order() const;

  /// Returns the backend associated with this queue.
  ///
  /// \return the backend associated with this queue.
  backend get_backend() const noexcept;

  /// Gets the native handle of the SYCL queue.
  ///
  /// \return a native handle, the type of which defined by the backend.
  template <backend Backend>
  __SYCL_DEPRECATED("Use SYCL 2020 sycl::get_native free function")
  backend_return_t<Backend, queue> get_native() const {
    return reinterpret_cast<backend_return_t<Backend, queue>>(getNative());
  }

private:
  pi_native_handle getNative() const;

  std::shared_ptr<detail::queue_impl> impl;
  queue(std::shared_ptr<detail::queue_impl> impl) : impl(impl) {}

  template <class Obj>
  friend decltype(Obj::impl) detail::getSyclObjImpl(const Obj &SyclObject);
  template <class T>
  friend T detail::createSyclObjFromImpl(decltype(T::impl) ImplObj);

#if __SYCL_USE_FALLBACK_ASSERT
  friend event detail::submitAssertCapture(queue &, event &, queue *,
                                           const detail::code_location &);
#endif

  /// A template-free version of submit.
  event submit_impl(std::function<void(handler &)> CGH,
                    const detail::code_location &CodeLoc);
  /// A template-free version of submit.
  event submit_impl(std::function<void(handler &)> CGH, queue secondQueue,
                    const detail::code_location &CodeLoc);

  /// Checks if the event needs to be discarded and if so, discards it and
  /// returns a discarded event. Otherwise, it returns input event.
  event discard_or_return(const event &Event);

  // Function to postprocess submitted command
  // Arguments:
  // bool IsKernel - true if the submitted command was kernel, false otherwise
  // bool KernelUsesAssert - true if submitted kernel uses assert, only
  //                         meaningful when IsKernel is true
  // event &Event - event after which post processing should be executed
  using SubmitPostProcessF = std::function<void(bool, bool, event &)>;

  /// A template-free version of submit.
  /// \param CGH command group function/handler
  /// \param CodeLoc code location
  ///
  /// This method stores additional information within event_impl class instance
  event submit_impl_and_postprocess(std::function<void(handler &)> CGH,
                                    const detail::code_location &CodeLoc,
                                    const SubmitPostProcessF &PostProcess);
  /// A template-free version of submit.
  /// \param CGH command group function/handler
  /// \param secondQueue fallback queue
  /// \param CodeLoc code location
  ///
  /// This method stores additional information within event_impl class instance
  event submit_impl_and_postprocess(std::function<void(handler &)> CGH,
                                    queue secondQueue,
                                    const detail::code_location &CodeLoc,
                                    const SubmitPostProcessF &PostProcess);

  /// parallel_for_impl with a kernel represented as a lambda + range that
  /// specifies global size only.
  ///
  /// \param NumWorkItems is a range that specifies the work space of the kernel
  /// \param KernelFunc is the Kernel functor or lambda
  /// \param CodeLoc contains the code location of user code
  template <typename KernelName = detail::auto_name, typename KernelType,
            int Dims>
  event parallel_for_impl(
      range<Dims> NumWorkItems, KernelType KernelFunc,
      const detail::code_location &CodeLoc = detail::code_location::current()) {
    return submit(
        [&](handler &CGH) {
          CGH.template parallel_for<KernelName, KernelType>(NumWorkItems,
                                                            KernelFunc);
        },
        CodeLoc);
  }

  /// parallel_for_impl with a kernel represented as a lambda + range that
  /// specifies global size only.
  ///
  /// \param NumWorkItems is a range that specifies the work space of the kernel
  /// \param DepEvent is an event that specifies the kernel dependencies
  /// \param KernelFunc is the Kernel functor or lambda
  /// \param CodeLoc contains the code location of user code
  template <typename KernelName = detail::auto_name, typename KernelType,
            int Dims>
  event parallel_for_impl(range<Dims> NumWorkItems, event DepEvent,
                          KernelType KernelFunc,
                          const detail::code_location &CodeLoc) {
    return submit(
        [&](handler &CGH) {
          CGH.depends_on(DepEvent);
          CGH.template parallel_for<KernelName, KernelType>(NumWorkItems,
                                                            KernelFunc);
        },
        CodeLoc);
  }

  /// parallel_for_impl version with a kernel represented as a lambda + range
  /// that specifies global size only.
  ///
  /// \param NumWorkItems is a range that specifies the work space of the kernel
  /// \param DepEvents is a vector of events that specifies the kernel
  /// dependencies
  /// \param KernelFunc is the Kernel functor or lambda
  /// \param CodeLoc contains the code location of user code
  template <typename KernelName = detail::auto_name, typename KernelType,
            int Dims>
  event parallel_for_impl(range<Dims> NumWorkItems,
                          const std::vector<event> &DepEvents,
                          KernelType KernelFunc,
                          const detail::code_location &CodeLoc) {
    return submit(
        [&](handler &CGH) {
          CGH.depends_on(DepEvents);
          CGH.template parallel_for<KernelName, KernelType>(NumWorkItems,
                                                            KernelFunc);
        },
        CodeLoc);
  }

  buffer<detail::AssertHappened, 1> &getAssertHappenedBuffer();
};

namespace detail {
#if __SYCL_USE_FALLBACK_ASSERT
#define __SYCL_ASSERT_START 1
/**
 * Submit copy task for assert failure flag and host-task to check the flag
 * \param Event kernel's event to depend on i.e. the event represents the
 *              kernel to check for assertion failure
 * \param SecondaryQueue secondary queue for submit process, null if not used
 * \returns host tasks event
 *
 * This method doesn't belong to queue class to overcome msvc behaviour due to
 * which it gets compiled and exported without any integration header and, thus,
 * with no proper KernelInfo instance.
 */
event submitAssertCapture(queue &Self, event &Event, queue *SecondaryQueue,
                          const detail::code_location &CodeLoc) {
  using AHBufT = buffer<detail::AssertHappened, 1>;

  AHBufT &Buffer = Self.getAssertHappenedBuffer();

  event CopierEv, CheckerEv, PostCheckerEv;
  auto CopierCGF = [&](handler &CGH) {
    CGH.depends_on(Event);

    auto Acc = Buffer.get_access<access::mode::write>(CGH);

    CGH.single_task<__sycl_service_kernel__::AssertInfoCopier>([Acc] {
#if defined(__SYCL_DEVICE_ONLY__) && !defined(__NVPTX__)
      __devicelib_assert_read(&Acc[0]);
#else
      (void)Acc;
#endif // defined(__SYCL_DEVICE_ONLY__) && !defined(__NVPTX__)
    });
  };
  auto CheckerCGF = [&CopierEv, &Buffer](handler &CGH) {
    CGH.depends_on(CopierEv);
    using mode = access::mode;
    using target = access::target;

    auto Acc = Buffer.get_access<mode::read, target::host_buffer>(CGH);

    CGH.host_task([=] {
      const detail::AssertHappened *AH = &Acc[0];

      // Don't use assert here as msvc will insert reference to __imp__wassert
      // which won't be properly resolved in separate compile use-case
#ifndef NDEBUG
      if (AH->Flag == __SYCL_ASSERT_START)
        throw sycl::runtime_error(
            "Internal Error. Invalid value in assert description.",
            PI_INVALID_VALUE);
#endif

      if (AH->Flag) {
        const char *Expr = AH->Expr[0] ? AH->Expr : "<unknown expr>";
        const char *File = AH->File[0] ? AH->File : "<unknown file>";
        const char *Func = AH->Func[0] ? AH->Func : "<unknown func>";

        fprintf(stderr,
                "%s:%d: %s: global id: [%" PRIu64 ",%" PRIu64 ",%" PRIu64
                "], local id: [%" PRIu64 ",%" PRIu64 ",%" PRIu64 "] "
                "Assertion `%s` failed.\n",
                File, AH->Line, Func, AH->GID0, AH->GID1, AH->GID2, AH->LID0,
                AH->LID1, AH->LID2, Expr);
        abort(); // no need to release memory as it's abort anyway
      }
    });
  };

  if (SecondaryQueue) {
    CopierEv = Self.submit_impl(CopierCGF, *SecondaryQueue, CodeLoc);
    CheckerEv = Self.submit_impl(CheckerCGF, *SecondaryQueue, CodeLoc);
  } else {
    CopierEv = Self.submit_impl(CopierCGF, CodeLoc);
    CheckerEv = Self.submit_impl(CheckerCGF, CodeLoc);
  }

  return CheckerEv;
}
#undef __SYCL_ASSERT_START
#endif // __SYCL_USE_FALLBACK_ASSERT
} // namespace detail

} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)

namespace std {
template <> struct hash<cl::sycl::queue> {
  size_t operator()(const cl::sycl::queue &Q) const {
    return std::hash<std::shared_ptr<cl::sycl::detail::queue_impl>>()(
        cl::sycl::detail::getSyclObjImpl(Q));
  }
};
} // namespace std

#undef __SYCL_USE_FALLBACK_ASSERT
