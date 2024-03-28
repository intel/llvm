//==-------------------- queue.hpp - SYCL queue ----------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/access/access.hpp>             // for target, access...
#include <sycl/accessor.hpp>                  // for accessor
#include <sycl/aspects.hpp>                   // for aspect
#include <sycl/async_handler.hpp>             // for async_handler
#include <sycl/backend_types.hpp>             // for backend, backe...
#include <sycl/buffer.hpp>                    // for buffer
#include <sycl/context.hpp>                   // for context
#include <sycl/detail/assert_happened.hpp>    // for AssertHappened
#include <sycl/detail/cg_types.hpp>           // for check_fn_signa...
#include <sycl/detail/common.hpp>             // for code_location
#include <sycl/detail/defines_elementary.hpp> // for __SYCL2020_DEP...
#include <sycl/detail/export.hpp>             // for __SYCL_EXPORT
#include <sycl/detail/info_desc_helpers.hpp>  // for is_queue_info_...
#include <sycl/detail/kernel_desc.hpp>        // for KernelInfo
#include <sycl/detail/owner_less_base.hpp>    // for OwnerLessBase
#include <sycl/detail/pi.h>                   // for pi_mem_advice
#include <sycl/device.hpp>                    // for device
#include <sycl/device_selector.hpp>           // for device_selector
#include <sycl/event.hpp>                     // for event
#include <sycl/exception.hpp>                 // for make_error_code
#include <sycl/exception_list.hpp>            // for defaultAsyncHa...
#include <sycl/ext/oneapi/bindless_images_descriptor.hpp> // for image_descriptor
#include <sycl/ext/oneapi/bindless_images_interop.hpp> // for interop_semaph...
#include <sycl/ext/oneapi/bindless_images_memory.hpp>  // for image_mem_handle
#include <sycl/ext/oneapi/device_global/device_global.hpp> // for device_global
#include <sycl/ext/oneapi/device_global/properties.hpp> // for device_image_s...
#include <sycl/ext/oneapi/experimental/graph.hpp>       // for command_graph...
#include <sycl/ext/oneapi/properties/properties.hpp>    // for empty_properti...
#include <sycl/handler.hpp>                             // for handler, isDev...
#include <sycl/id.hpp>                                  // for id
#include <sycl/kernel.hpp>                              // for auto_name
#include <sycl/kernel_handler.hpp>                      // for kernel_handler
#include <sycl/nd_range.hpp>                            // for nd_range
#include <sycl/property_list.hpp>                       // for property_list
#include <sycl/range.hpp>                               // for range

#include <cstddef>     // for size_t
#include <functional>  // for function
#include <memory>      // for shared_ptr, hash
#include <stdint.h>    // for int32_t
#include <tuple>       // for tuple
#include <type_traits> // for remove_all_ext...
#include <variant>     // for hash
#include <vector>      // for vector

// having _TWO_ mid-param #ifdefs makes the functions very difficult to read.
// Here we simplify the KernelFunc param is simplified to be
// _KERNELFUNCPARAM(KernelFunc) Once the queue kernel functions are defined,
// these macros are #undef immediately.
// replace _KERNELFUNCPARAM(KernelFunc) with   KernelType KernelFunc
//                                     or     const KernelType &KernelFunc
#ifdef __SYCL_NONCONST_FUNCTOR__
#define _KERNELFUNCPARAM(a) KernelType a
#else
#define _KERNELFUNCPARAM(a) const KernelType &a
#endif

// Helper macro to identify if fallback assert is needed
// FIXME remove __NVPTX__ condition once devicelib supports CUDA
#if defined(SYCL_FALLBACK_ASSERT)
#define __SYCL_USE_FALLBACK_ASSERT SYCL_FALLBACK_ASSERT
#else
#define __SYCL_USE_FALLBACK_ASSERT 0
#endif

namespace sycl {
inline namespace _V1 {

// Forward declaration
class context;
class device;
class event;
class queue;

template <backend BackendName, class SyclObjectT>
auto get_native(const SyclObjectT &Obj)
    -> backend_return_t<BackendName, SyclObjectT>;

namespace detail {
class queue_impl;

#if __SYCL_USE_FALLBACK_ASSERT
inline event submitAssertCapture(queue &, event &, queue *,
                                 const detail::code_location &);
#endif
} // namespace detail

namespace ext {
namespace oneapi {
namespace experimental {
// State of a queue with regards to graph recording,
// returned by info::queue::state
enum class queue_state { executing, recording };
} // namespace experimental
} // namespace oneapi
} // namespace ext

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
class __SYCL_EXPORT queue : public detail::OwnerLessBase<queue> {
public:
  /// Constructs a SYCL queue instance using the device returned by an instance
  /// of default_selector.
  ///
  /// \param PropList is a list of properties for queue construction.
  explicit queue(const property_list &PropList = {})
      : queue(default_selector_v, detail::defaultAsyncHandler, PropList) {}

  /// Constructs a SYCL queue instance with an async_handler using the device
  /// returned by an instance of default_selector.
  ///
  /// \param AsyncHandler is a SYCL asynchronous exception handler.
  /// \param PropList is a list of properties for queue construction.
  queue(const async_handler &AsyncHandler, const property_list &PropList = {})
      : queue(default_selector_v, AsyncHandler, PropList) {}

  /// Constructs a SYCL queue instance using the device identified by the
  /// device selector provided.
  /// \param DeviceSelector is SYCL 2020 Device Selector, a simple callable that
  /// takes a device and returns an int
  /// \param AsyncHandler is a SYCL asynchronous exception handler.
  /// \param PropList is a list of properties for queue construction.
  template <typename DeviceSelector,
            typename =
                detail::EnableIfSYCL2020DeviceSelectorInvocable<DeviceSelector>>
  explicit queue(const DeviceSelector &deviceSelector,
                 const async_handler &AsyncHandler,
                 const property_list &PropList = {})
      : queue(detail::select_device(deviceSelector), AsyncHandler, PropList) {}

  /// Constructs a SYCL queue instance using the device identified by the
  /// device selector provided.
  /// \param DeviceSelector is SYCL 2020 Device Selector, a simple callable that
  /// takes a device and returns an int
  /// \param PropList is a list of properties for queue construction.
  template <typename DeviceSelector,
            typename =
                detail::EnableIfSYCL2020DeviceSelectorInvocable<DeviceSelector>>
  explicit queue(const DeviceSelector &deviceSelector,
                 const property_list &PropList = {})
      : queue(detail::select_device(deviceSelector),
              detail::defaultAsyncHandler, PropList) {}

  /// Constructs a SYCL queue instance using the device identified by the
  /// device selector provided.
  /// \param SyclContext is an instance of SYCL context.
  /// \param DeviceSelector is SYCL 2020 Device Selector, a simple callable that
  /// takes a device and returns an int
  /// \param PropList is a list of properties for queue construction.
  template <typename DeviceSelector,
            typename =
                detail::EnableIfSYCL2020DeviceSelectorInvocable<DeviceSelector>>
  explicit queue(const context &syclContext,
                 const DeviceSelector &deviceSelector,
                 const property_list &propList = {})
      : queue(syclContext, detail::select_device(deviceSelector, syclContext),
              propList) {}

  /// Constructs a SYCL queue instance using the device identified by the
  /// device selector provided.
  /// \param SyclContext is an instance of SYCL context.
  /// \param DeviceSelector is SYCL 2020 Device Selector, a simple callable that
  /// takes a device and returns an int
  /// \param AsyncHandler is a SYCL asynchronous exception handler.
  /// \param PropList is a list of properties for queue construction.
  template <typename DeviceSelector,
            typename =
                detail::EnableIfSYCL2020DeviceSelectorInvocable<DeviceSelector>>
  explicit queue(const context &syclContext,
                 const DeviceSelector &deviceSelector,
                 const async_handler &AsyncHandler,
                 const property_list &propList = {})
      : queue(syclContext, detail::select_device(deviceSelector, syclContext),
              AsyncHandler, propList) {}

  /// Constructs a SYCL queue instance using the device returned by the
  /// DeviceSelector provided.
  ///
  /// \param DeviceSelector is an instance of a SYCL 1.2.1 device_selector.
  /// \param PropList is a list of properties for queue construction.
  __SYCL2020_DEPRECATED("SYCL 1.2.1 device selectors are deprecated. Please "
                        "use SYCL 2020 device selectors instead.")
  queue(const device_selector &DeviceSelector,
        const property_list &PropList = {})
      : queue(DeviceSelector.select_device(), detail::defaultAsyncHandler,
              PropList) {}

  /// Constructs a SYCL queue instance with an async_handler using the device
  /// returned by the DeviceSelector provided.
  ///
  /// \param DeviceSelector is an instance of SYCL 1.2.1 device_selector.
  /// \param AsyncHandler is a SYCL asynchronous exception handler.
  /// \param PropList is a list of properties for queue construction.
  __SYCL2020_DEPRECATED("SYCL 1.2.1 device selectors are deprecated. Please "
                        "use SYCL 2020 device selectors instead.")
  queue(const device_selector &DeviceSelector,
        const async_handler &AsyncHandler, const property_list &PropList = {})
      : queue(DeviceSelector.select_device(), AsyncHandler, PropList) {}

  /// Constructs a SYCL queue instance using the device provided.
  ///
  /// \param SyclDevice is an instance of SYCL device.
  /// \param PropList is a list of properties for queue construction.
  explicit queue(const device &SyclDevice, const property_list &PropList = {})
      : queue(SyclDevice, detail::defaultAsyncHandler, PropList) {}

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
  __SYCL2020_DEPRECATED("SYCL 1.2.1 device selectors are deprecated. Please "
                        "use SYCL 2020 device selectors instead.")
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
  __SYCL2020_DEPRECATED("SYCL 1.2.1 device selectors are deprecated. Please "
                        "use SYCL 2020 device selectors instead.")
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

  /// \return State the queue is currently in.
  ext::oneapi::experimental::queue_state ext_oneapi_get_state() const;

  /// \return Graph when the queue is recording.
  ext::oneapi::experimental::command_graph<
      ext::oneapi::experimental::graph_state::modifiable>
  ext_oneapi_get_graph() const;

  /// \return true if this queue is a SYCL host queue.
  __SYCL2020_DEPRECATED(
      "is_host() is deprecated as the host device is no longer supported.")
  bool is_host() const;

  /// Queries SYCL queue for information.
  ///
  /// The return type depends on information being queried.
  template <typename Param>
  typename detail::is_queue_info_desc<Param>::return_type get_info() const;

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
  template <typename T>
  std::enable_if_t<std::is_invocable_r_v<void, T, handler &>, event> submit(
      T CGF,
      const detail::code_location &CodeLoc = detail::code_location::current()) {
    detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
#if __SYCL_USE_FALLBACK_ASSERT
    auto PostProcess = [this, &CodeLoc](bool IsKernel, bool KernelUsesAssert,
                                        event &E) {
      if (IsKernel && !device_has(aspect::ext_oneapi_native_assert) &&
          KernelUsesAssert && !device_has(aspect::accelerator)) {
        // __devicelib_assert_fail isn't supported by Device-side Runtime
        // Linking against fallback impl of __devicelib_assert_fail is
        // performed by program manager class
        // Fallback assert isn't supported for FPGA
        submitAssertCapture(*this, E, /* SecondaryQueue = */ nullptr, CodeLoc);
      }
    };

    return submit_impl_and_postprocess(CGF, CodeLoc, PostProcess);
#else
    return submit_impl(CGF, CodeLoc);
#endif // __SYCL_USE_FALLBACK_ASSERT
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
  std::enable_if_t<std::is_invocable_r_v<void, T, handler &>, event> submit(
      T CGF, queue &SecondaryQueue,
      const detail::code_location &CodeLoc = detail::code_location::current()) {
    detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
#if __SYCL_USE_FALLBACK_ASSERT
    auto PostProcess = [this, &SecondaryQueue, &CodeLoc](
                           bool IsKernel, bool KernelUsesAssert, event &E) {
      if (IsKernel && !device_has(aspect::ext_oneapi_native_assert) &&
          KernelUsesAssert && !device_has(aspect::accelerator)) {
        // Only secondary queues on devices need to be added to the assert
        // capture.
        // __devicelib_assert_fail isn't supported by Device-side Runtime
        // Linking against fallback impl of __devicelib_assert_fail is
        // performed by program manager class
        // Fallback assert isn't supported for FPGA
        submitAssertCapture(*this, E, &SecondaryQueue, CodeLoc);
      }
    };

    return submit_impl_and_postprocess(CGF, SecondaryQueue, CodeLoc,
                                       PostProcess);
#else
    return submit_impl(CGF, SecondaryQueue, CodeLoc);
#endif // __SYCL_USE_FALLBACK_ASSERT
  }

  /// Prevents any commands submitted afterward to this queue from executing
  /// until all commands previously submitted to this queue have entered the
  /// complete state.
  ///
  /// \param CodeLoc is the code location of the submit call (default argument)
  /// \return a SYCL event object, which corresponds to the queue the command
  /// group is being enqueued on.
  event ext_oneapi_submit_barrier(
      const detail::code_location &CodeLoc = detail::code_location::current());

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
      const std::vector<event> &WaitList,
      const detail::code_location &CodeLoc = detail::code_location::current());

  /// Performs a blocking wait for the completion of all enqueued tasks in the
  /// queue.
  ///
  /// Synchronous errors will be reported through SYCL exceptions.
  /// @param CodeLoc is the code location of the submit call (default argument)
  void wait(
      const detail::code_location &CodeLoc = detail::code_location::current()) {
    detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
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
      const detail::code_location &CodeLoc = detail::code_location::current()) {
    detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
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
  template <typename PropertyT> bool has_property() const noexcept;

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
  template <typename T>
  event fill(
      void *Ptr, const T &Pattern, size_t Count,
      const detail::code_location &CodeLoc = detail::code_location::current()) {
    detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
    return submit([&](handler &CGH) { CGH.fill<T>(Ptr, Pattern, Count); },
                  CodeLoc);
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
  event fill(
      void *Ptr, const T &Pattern, size_t Count, event DepEvent,
      const detail::code_location &CodeLoc = detail::code_location::current()) {
    detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
    return submit(
        [&](handler &CGH) {
          CGH.depends_on(DepEvent);
          CGH.fill<T>(Ptr, Pattern, Count);
        },
        CodeLoc);
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
  event fill(
      void *Ptr, const T &Pattern, size_t Count,
      const std::vector<event> &DepEvents,
      const detail::code_location &CodeLoc = detail::code_location::current()) {
    detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
    return submit(
        [&](handler &CGH) {
          CGH.depends_on(DepEvents);
          CGH.fill<T>(Ptr, Pattern, Count);
        },
        CodeLoc);
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
  event memset(
      void *Ptr, int Value, size_t Count,
      const detail::code_location &CodeLoc = detail::code_location::current());

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
  event memset(
      void *Ptr, int Value, size_t Count, event DepEvent,
      const detail::code_location &CodeLoc = detail::code_location::current());

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
  event memset(
      void *Ptr, int Value, size_t Count, const std::vector<event> &DepEvents,
      const detail::code_location &CodeLoc = detail::code_location::current());

  /// Copies data from one memory region to another, each is either a host
  /// pointer or a pointer within USM allocation accessible on the device
  /// associated with this queue.
  /// No operations is done if \param Count is zero. An exception is thrown
  /// if either \param Dest or \param Src is nullptr. The behavior is undefined
  /// if any of the pointer parameters is invalid.
  ///
  /// \param Dest is a USM pointer to the destination memory.
  /// \param Src is a USM pointer to the source memory.
  /// \param Count is a number of bytes to copy.
  /// \return an event representing copy operation.
  event memcpy(
      void *Dest, const void *Src, size_t Count,
      const detail::code_location &CodeLoc = detail::code_location::current());

  /// Copies data from one memory region to another, each is either a host
  /// pointer or a pointer within USM allocation accessible on the device
  /// associated with this queue.
  /// No operations is done if \param Count is zero. An exception is thrown
  /// if either \param Dest or \param Src is nullptr. The behavior is undefined
  /// if any of the pointer parameters is invalid.
  ///
  /// \param Dest is a USM pointer to the destination memory.
  /// \param Src is a USM pointer to the source memory.
  /// \param Count is a number of bytes to copy.
  /// \param DepEvent is an event that specifies the kernel dependencies.
  /// \return an event representing copy operation.
  event memcpy(
      void *Dest, const void *Src, size_t Count, event DepEvent,
      const detail::code_location &CodeLoc = detail::code_location::current());

  /// Copies data from one memory region to another, each is either a host
  /// pointer or a pointer within USM allocation accessible on the device
  /// associated with this queue.
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
  event memcpy(
      void *Dest, const void *Src, size_t Count,
      const std::vector<event> &DepEvents,
      const detail::code_location &CodeLoc = detail::code_location::current());

  /// Copies data from one memory region to another, each is either a host
  /// pointer or a pointer within USM allocation accessible on the device
  /// associated with this queue.
  /// No operations is done if \param Count is zero. An exception is thrown
  /// if either \param Dest or \param Src is nullptr. The behavior is undefined
  /// if any of the pointer parameters is invalid.
  ///
  /// \param Src is a USM pointer to the source memory.
  /// \param Dest is a USM pointer to the destination memory.
  /// \param Count is a number of elements of type T to copy.
  /// \param CodeLoc contains the code location of user code
  /// \return an event representing copy operation.
  template <typename T>
  event copy(
      const T *Src, T *Dest, size_t Count,
      const detail::code_location &CodeLoc = detail::code_location::current()) {
    detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
    return this->memcpy(Dest, Src, Count * sizeof(T));
  }

  /// Copies data from one memory region to another, each is either a host
  /// pointer or a pointer within USM allocation accessible on the device
  /// associated with this queue.
  /// No operations is done if \param Count is zero. An exception is thrown
  /// if either \param Dest or \param Src is nullptr. The behavior is undefined
  /// if any of the pointer parameters is invalid.
  ///
  /// \param Src is a USM pointer to the source memory.
  /// \param Dest is a USM pointer to the destination memory.
  /// \param Count is a number of elements of type T to copy.
  /// \param DepEvent is an event that specifies the kernel dependencies.
  /// \param CodeLoc contains the code location of user code
  /// \return an event representing copy operation.
  template <typename T>
  event copy(
      const T *Src, T *Dest, size_t Count, event DepEvent,
      const detail::code_location &CodeLoc = detail::code_location::current()) {
    detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
    return this->memcpy(Dest, Src, Count * sizeof(T), DepEvent);
  }

  /// Copies data from one memory region to another, each is either a host
  /// pointer or a pointer within USM allocation accessible on the device
  /// associated with this queue.
  /// No operations is done if \param Count is zero. An exception is thrown
  /// if either \param Dest or \param Src is nullptr. The behavior is undefined
  /// if any of the pointer parameters is invalid.
  ///
  /// \param Src is a USM pointer to the source memory.
  /// \param Dest is a USM pointer to the destination memory.
  /// \param Count is a number of elements of type T to copy.
  /// \param DepEvents is a vector of events that specifies the kernel
  /// \param CodeLoc contains the code location of user code
  /// \return an event representing copy operation.
  template <typename T>
  event copy(
      const T *Src, T *Dest, size_t Count, const std::vector<event> &DepEvents,
      const detail::code_location &CodeLoc = detail::code_location::current()) {
    detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
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
  event mem_advise(
      const void *Ptr, size_t Length, pi_mem_advice Advice,
      const detail::code_location &CodeLoc = detail::code_location::current());

  /// Provides additional information to the underlying runtime about how
  /// different allocations are used.
  ///
  /// \param Ptr is a USM pointer to the allocation.
  /// \param Length is a number of bytes in the allocation.
  /// \param Advice is a device-defined advice for the specified allocation.
  /// \return an event representing advice operation.
  event mem_advise(
      const void *Ptr, size_t Length, int Advice,
      const detail::code_location &CodeLoc = detail::code_location::current());

  /// Provides additional information to the underlying runtime about how
  /// different allocations are used.
  ///
  /// \param Ptr is a USM pointer to the allocation.
  /// \param Length is a number of bytes in the allocation.
  /// \param Advice is a device-defined advice for the specified allocation.
  /// \param DepEvent is an event that specifies the kernel dependencies.
  /// \return an event representing advice operation.
  event mem_advise(
      const void *Ptr, size_t Length, int Advice, event DepEvent,
      const detail::code_location &CodeLoc = detail::code_location::current());

  /// Provides additional information to the underlying runtime about how
  /// different allocations are used.
  ///
  /// \param Ptr is a USM pointer to the allocation.
  /// \param Length is a number of bytes in the allocation.
  /// \param Advice is a device-defined advice for the specified allocation.
  /// \param DepEvents is a vector of events that specifies the kernel
  /// dependencies.
  /// \return an event representing advice operation.
  event mem_advise(
      const void *Ptr, size_t Length, int Advice,
      const std::vector<event> &DepEvents,
      const detail::code_location &CodeLoc = detail::code_location::current());

  /// Provides hints to the runtime library that data should be made available
  /// on a device earlier than Unified Shared Memory would normally require it
  /// to be available.
  ///
  /// \param Ptr is a USM pointer to the memory to be prefetched to the device.
  /// \param Count is a number of bytes to be prefetched.
  /// \return an event representing prefetch operation.
  event prefetch(
      const void *Ptr, size_t Count,
      const detail::code_location &CodeLoc = detail::code_location::current()) {
    detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
    return submit([=](handler &CGH) { CGH.prefetch(Ptr, Count); }, CodeLoc);
  }

  /// Provides hints to the runtime library that data should be made available
  /// on a device earlier than Unified Shared Memory would normally require it
  /// to be available.
  ///
  /// \param Ptr is a USM pointer to the memory to be prefetched to the device.
  /// \param Count is a number of bytes to be prefetched.
  /// \param DepEvent is an event that specifies the kernel dependencies.
  /// \return an event representing prefetch operation.
  event prefetch(
      const void *Ptr, size_t Count, event DepEvent,
      const detail::code_location &CodeLoc = detail::code_location::current()) {
    detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
    return submit(
        [=](handler &CGH) {
          CGH.depends_on(DepEvent);
          CGH.prefetch(Ptr, Count);
        },
        CodeLoc);
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
  event prefetch(
      const void *Ptr, size_t Count, const std::vector<event> &DepEvents,
      const detail::code_location &CodeLoc = detail::code_location::current()) {
    detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
    return submit(
        [=](handler &CGH) {
          CGH.depends_on(DepEvents);
          CGH.prefetch(Ptr, Count);
        },
        CodeLoc);
  }

  /// Copies data from one 2D memory region to another, both pointed by
  /// USM pointers.
  /// No operations is done if \param Width or \param Height is zero. An
  /// exception is thrown if either \param Dest or \param Src is nullptr or if
  /// \param Width is strictly greater than either \param DestPitch or
  /// \param SrcPitch. The behavior is undefined if any of the pointer
  /// parameters is invalid.
  ///
  /// NOTE: Function is dependent to prevent the fallback kernels from
  /// materializing without the use of the function.
  ///
  /// \param Dest is a USM pointer to the destination memory.
  /// \param DestPitch is the pitch of the rows in \param Dest.
  /// \param Src is a USM pointer to the source memory.
  /// \param SrcPitch is the pitch of the rows in \param Src.
  /// \param Width is the width in bytes of the 2D region to copy.
  /// \param Height is the height in number of rows of the 2D region to copy.
  /// \return an event representing the copy operation.
  template <typename T = unsigned char,
            typename = std::enable_if_t<std::is_same_v<T, unsigned char>>>
  event ext_oneapi_memcpy2d(
      void *Dest, size_t DestPitch, const void *Src, size_t SrcPitch,
      size_t Width, size_t Height,
      const detail::code_location &CodeLoc = detail::code_location::current()) {
    return submit(
        [=](handler &CGH) {
          CGH.ext_oneapi_memcpy2d<T>(Dest, DestPitch, Src, SrcPitch, Width,
                                     Height);
        },
        CodeLoc);
  }

  /// Copies data from one 2D memory region to another, both pointed by
  /// USM pointers.
  /// No operations is done if \param Width or \param Height is zero. An
  /// exception is thrown if either \param Dest or \param Src is nullptr or if
  /// \param Width is strictly greater than either \param DestPitch or
  /// \param SrcPitch. The behavior is undefined if any of the pointer
  /// parameters is invalid.
  ///
  /// NOTE: Function is dependent to prevent the fallback kernels from
  /// materializing without the use of the function.
  ///
  /// \param Dest is a USM pointer to the destination memory.
  /// \param DestPitch is the pitch of the rows in \param Dest.
  /// \param Src is a USM pointer to the source memory.
  /// \param SrcPitch is the pitch of the rows in \param Src.
  /// \param Width is the width in bytes of the 2D region to copy.
  /// \param Height is the height in number of rows of the 2D region to copy.
  /// \param DepEvent is an event that specifies the kernel dependencies.
  /// \return an event representing the copy operation.
  template <typename T = unsigned char,
            typename = std::enable_if_t<std::is_same_v<T, unsigned char>>>
  event ext_oneapi_memcpy2d(
      void *Dest, size_t DestPitch, const void *Src, size_t SrcPitch,
      size_t Width, size_t Height, event DepEvent,
      const detail::code_location &CodeLoc = detail::code_location::current());

  /// Copies data from one 2D memory region to another, both pointed by
  /// USM pointers.
  /// No operations is done if \param Width or \param Height is zero. An
  /// exception is thrown if either \param Dest or \param Src is nullptr or if
  /// \param Width is strictly greater than either \param DestPitch or
  /// \param SrcPitch. The behavior is undefined if any of the pointer
  /// parameters is invalid.
  ///
  /// NOTE: Function is dependent to prevent the fallback kernels from
  /// materializing without the use of the function.
  ///
  /// \param Dest is a USM pointer to the destination memory.
  /// \param DestPitch is the pitch of the rows in \param Dest.
  /// \param Src is a USM pointer to the source memory.
  /// \param SrcPitch is the pitch of the rows in \param Src.
  /// \param Width is the width in bytes of the 2D region to copy.
  /// \param Height is the height in number of rows of the 2D region to copy.
  /// \param DepEvents is a vector of events that specifies the kernel
  /// dependencies.
  /// \return an event representing the copy operation.
  template <typename T = unsigned char,
            typename = std::enable_if_t<std::is_same_v<T, unsigned char>>>
  event ext_oneapi_memcpy2d(
      void *Dest, size_t DestPitch, const void *Src, size_t SrcPitch,
      size_t Width, size_t Height, const std::vector<event> &DepEvents,
      const detail::code_location &CodeLoc = detail::code_location::current());

  /// Copies data from one 2D memory region to another, both pointed by
  /// USM pointers.
  /// No operations is done if \param Width or \param Height is zero. An
  /// exception is thrown if either \param Dest or \param Src is nullptr or if
  /// \param Width is strictly greater than either \param DestPitch or
  /// \param SrcPitch. The behavior is undefined if any of the pointer
  /// parameters is invalid.
  ///
  /// \param Src is a USM pointer to the source memory.
  /// \param SrcPitch is the pitch of the rows in \param Src.
  /// \param Dest is a USM pointer to the destination memory.
  /// \param DestPitch is the pitch of the rows in \param Dest.
  /// \param Width is the width in number of elements of the 2D region to copy.
  /// \param Height is the height in number of rows of the 2D region to copy.
  /// \return an event representing the copy operation.
  template <typename T>
  event ext_oneapi_copy2d(
      const T *Src, size_t SrcPitch, T *Dest, size_t DestPitch, size_t Width,
      size_t Height,
      const detail::code_location &CodeLoc = detail::code_location::current());

  /// Copies data from one 2D memory region to another, both pointed by
  /// USM pointers.
  /// No operations is done if \param Width or \param Height is zero. An
  /// exception is thrown if either \param Dest or \param Src is nullptr or if
  /// \param Width is strictly greater than either \param DestPitch or
  /// \param SrcPitch. The behavior is undefined if any of the pointer
  /// parameters is invalid.
  ///
  /// \param Src is a USM pointer to the source memory.
  /// \param SrcPitch is the pitch of the rows in \param Src.
  /// \param Dest is a USM pointer to the destination memory.
  /// \param DestPitch is the pitch of the rows in \param Dest.
  /// \param Width is the width in number of elements of the 2D region to copy.
  /// \param Height is the height in number of rows of the 2D region to copy.
  /// \param DepEvent is an event that specifies the kernel dependencies.
  /// \return an event representing the copy operation.
  template <typename T>
  event ext_oneapi_copy2d(
      const T *Src, size_t SrcPitch, T *Dest, size_t DestPitch, size_t Width,
      size_t Height, event DepEvent,
      const detail::code_location &CodeLoc = detail::code_location::current());

  /// Copies data from one 2D memory region to another, both pointed by
  /// USM pointers.
  /// No operations is done if \param Width or \param Height is zero. An
  /// exception is thrown if either \param Dest or \param Src is nullptr or if
  /// \param Width is strictly greater than either \param DestPitch or
  /// \param SrcPitch. The behavior is undefined if any of the pointer
  /// parameters is invalid.
  ///
  /// \param Src is a USM pointer to the source memory.
  /// \param SrcPitch is the pitch of the rows in \param Src.
  /// \param Dest is a USM pointer to the destination memory.
  /// \param DestPitch is the pitch of the rows in \param Dest.
  /// \param Width is the width in number of elements of the 2D region to copy.
  /// \param Height is the height in number of rows of the 2D region to copy.
  /// \param DepEvents is a vector of events that specifies the kernel
  /// dependencies.
  /// \return an event representing the copy operation.
  template <typename T>
  event ext_oneapi_copy2d(
      const T *Src, size_t SrcPitch, T *Dest, size_t DestPitch, size_t Width,
      size_t Height, const std::vector<event> &DepEvents,
      const detail::code_location &CodeLoc = detail::code_location::current());

  /// Fills the memory pointed by a USM pointer with the value specified.
  /// No operations is done if \param Width or \param Height is zero. An
  /// exception is thrown if either \param Dest or \param Src is nullptr or if
  /// \param Width is strictly greater than \param DestPitch. The behavior is
  /// undefined if any of the pointer parameters is invalid.
  ///
  /// NOTE: Function is dependent to prevent the fallback kernels from
  /// materializing without the use of the function.
  ///
  /// \param Dest is a USM pointer to the destination memory.
  /// \param DestPitch is the pitch of the rows in \param Dest.
  /// \param Value is the value to fill into the region in \param Dest. Value is
  /// cast as an unsigned char.
  /// \param Width is the width in number of elements of the 2D region to fill.
  /// \param Height is the height in number of rows of the 2D region to fill.
  /// \return an event representing the fill operation.
  template <typename T = unsigned char,
            typename = std::enable_if_t<std::is_same_v<T, unsigned char>>>
  event ext_oneapi_memset2d(
      void *Dest, size_t DestPitch, int Value, size_t Width, size_t Height,
      const detail::code_location &CodeLoc = detail::code_location::current());

  /// Fills the memory pointed by a USM pointer with the value specified.
  /// No operations is done if \param Width or \param Height is zero. An
  /// exception is thrown if either \param Dest or \param Src is nullptr or if
  /// \param Width is strictly greater than \param DestPitch. The behavior is
  /// undefined if any of the pointer parameters is invalid.
  ///
  /// NOTE: Function is dependent to prevent the fallback kernels from
  /// materializing without the use of the function.
  ///
  /// \param Dest is a USM pointer to the destination memory.
  /// \param DestPitch is the pitch of the rows in \param Dest.
  /// \param Value is the value to fill into the region in \param Dest. Value is
  /// cast as an unsigned char.
  /// \param Width is the width in number of elements of the 2D region to fill.
  /// \param Height is the height in number of rows of the 2D region to fill.
  /// \param DepEvent is an event that specifies the kernel dependencies.
  /// \return an event representing the fill operation.
  template <typename T = unsigned char,
            typename = std::enable_if_t<std::is_same_v<T, unsigned char>>>
  event ext_oneapi_memset2d(
      void *Dest, size_t DestPitch, int Value, size_t Width, size_t Height,
      event DepEvent,
      const detail::code_location &CodeLoc = detail::code_location::current());

  /// Fills the memory pointed by a USM pointer with the value specified.
  /// No operations is done if \param Width or \param Height is zero. An
  /// exception is thrown if either \param Dest or \param Src is nullptr or if
  /// \param Width is strictly greater than \param DestPitch. The behavior is
  /// undefined if any of the pointer parameters is invalid.
  ///
  /// NOTE: Function is dependent to prevent the fallback kernels from
  /// materializing without the use of the function.
  ///
  /// \param Dest is a USM pointer to the destination memory.
  /// \param DestPitch is the pitch of the rows in \param Dest.
  /// \param Value is the value to fill into the region in \param Dest. Value is
  /// cast as an unsigned char.
  /// \param Width is the width in number of elements of the 2D region to fill.
  /// \param Height is the height in number of rows of the 2D region to fill.
  /// \param DepEvents is a vector of events that specifies the kernel
  /// dependencies.
  /// \return an event representing the fill operation.
  template <typename T = unsigned char,
            typename = std::enable_if_t<std::is_same_v<T, unsigned char>>>
  event ext_oneapi_memset2d(
      void *Dest, size_t DestPitch, int Value, size_t Width, size_t Height,
      const std::vector<event> &DepEvents,
      const detail::code_location &CodeLoc = detail::code_location::current());

  /// Fills the memory pointed by a USM pointer with the value specified.
  /// No operations is done if \param Width or \param Height is zero. An
  /// exception is thrown if either \param Dest or \param Src is nullptr or if
  /// \param Width is strictly greater than \param DestPitch. The behavior is
  /// undefined if any of the pointer parameters is invalid.
  ///
  /// \param Dest is a USM pointer to the destination memory.
  /// \param DestPitch is the pitch of the rows in \param Dest.
  /// \param Pattern is the pattern to fill into the memory.  T should be
  /// trivially copyable.
  /// \param Width is the width in number of elements of the 2D region to fill.
  /// \param Height is the height in number of rows of the 2D region to fill.
  /// \return an event representing the fill operation.
  template <typename T>
  event ext_oneapi_fill2d(
      void *Dest, size_t DestPitch, const T &Pattern, size_t Width,
      size_t Height,
      const detail::code_location &CodeLoc = detail::code_location::current());

  /// Fills the memory pointed by a USM pointer with the value specified.
  /// No operations is done if \param Width or \param Height is zero. An
  /// exception is thrown if either \param Dest or \param Src is nullptr or if
  /// \param Width is strictly greater than \param DestPitch. The behavior is
  /// undefined if any of the pointer parameters is invalid.
  ///
  /// \param Dest is a USM pointer to the destination memory.
  /// \param DestPitch is the pitch of the rows in \param Dest.
  /// \param Pattern is the pattern to fill into the memory.  T should be
  /// trivially copyable.
  /// \param Width is the width in number of elements of the 2D region to fill.
  /// \param Height is the height in number of rows of the 2D region to fill.
  /// \param DepEvent is an event that specifies the kernel dependencies.
  /// \return an event representing the fill operation.
  template <typename T>
  event ext_oneapi_fill2d(
      void *Dest, size_t DestPitch, const T &Pattern, size_t Width,
      size_t Height, event DepEvent,
      const detail::code_location &CodeLoc = detail::code_location::current());

  /// Fills the memory pointed by a USM pointer with the value specified.
  /// No operations is done if \param Width or \param Height is zero. An
  /// exception is thrown if either \param Dest or \param Src is nullptr or if
  /// \param Width is strictly greater than \param DestPitch. The behavior is
  /// undefined if any of the pointer parameters is invalid.
  ///
  /// \param Dest is a USM pointer to the destination memory.
  /// \param DestPitch is the pitch of the rows in \param Dest.
  /// \param Pattern is the pattern to fill into the memory.  T should be
  /// trivially copyable.
  /// \param Width is the width in number of elements of the 2D region to fill.
  /// \param Height is the height in number of rows of the 2D region to fill.
  /// \param DepEvents is a vector of events that specifies the kernel
  /// dependencies.
  /// \return an event representing the fill operation.
  template <typename T>
  event ext_oneapi_fill2d(
      void *Dest, size_t DestPitch, const T &Pattern, size_t Width,
      size_t Height, const std::vector<event> &DepEvents,
      const detail::code_location &CodeLoc = detail::code_location::current());

  /// Copies data from a USM memory region to a device_global.
  /// Throws an exception if the copy operation intends to write outside the
  /// memory range \param Dest, as specified through \param NumBytes and
  /// \param Offset.
  ///
  /// \param Dest is the destination device_glboal.
  /// \param Src is a USM pointer to the source memory.
  /// \param NumBytes is a number of bytes to copy.
  /// \param Offset is the offset into \param Dest to copy to.
  /// \param DepEvents is a vector of events that specifies the operation
  /// dependencies.
  /// \return an event representing copy operation.
  template <typename T, typename PropertyListT>
  event memcpy(
      ext::oneapi::experimental::device_global<T, PropertyListT> &Dest,
      const void *Src, size_t NumBytes, size_t Offset,
      const std::vector<event> &DepEvents,
      const detail::code_location &CodeLoc = detail::code_location::current()) {
    detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
    if (sizeof(T) < Offset + NumBytes)
      throw sycl::exception(make_error_code(errc::invalid),
                            "Copy to device_global is out of bounds.");

    if (!detail::isDeviceGlobalUsedInKernel(&Dest)) {
      // device_global is unregistered so we need a fallback. We let the handler
      // implement this fallback.
      return submit(
          [&](handler &CGH) {
            CGH.depends_on(DepEvents);
            return CGH.memcpy(Dest, Src, NumBytes, Offset);
          },
          CodeLoc);
    }

    constexpr bool IsDeviceImageScoped = PropertyListT::template has_property<
        ext::oneapi::experimental::device_image_scope_key>();
    return memcpyToDeviceGlobal(&Dest, Src, IsDeviceImageScoped, NumBytes,
                                Offset, DepEvents);
  }

  /// Copies data from a USM memory region to a device_global.
  /// Throws an exception if the copy operation intends to write outside the
  /// memory range \param Dest, as specified through \param NumBytes and
  /// \param Offset.
  ///
  /// \param Dest is the destination device_glboal.
  /// \param Src is a USM pointer to the source memory.
  /// \param NumBytes is a number of bytes to copy.
  /// \param Offset is the offset into \param Dest to copy to.
  /// \param DepEvent is a vector of event that specifies the operation
  /// dependency.
  /// \return an event representing copy operation.
  template <typename T, typename PropertyListT>
  event memcpy(
      ext::oneapi::experimental::device_global<T, PropertyListT> &Dest,
      const void *Src, size_t NumBytes, size_t Offset, event DepEvent,
      const detail::code_location &CodeLoc = detail::code_location::current()) {
    detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
    return this->memcpy(Dest, Src, NumBytes, Offset,
                        std::vector<event>{DepEvent});
  }

  /// Copies data from a USM memory region to a device_global.
  /// Throws an exception if the copy operation intends to write outside the
  /// memory range \param Dest, as specified through \param NumBytes and
  /// \param Offset.
  ///
  /// \param Dest is the destination device_glboal.
  /// \param Src is a USM pointer to the source memory.
  /// \param NumBytes is a number of bytes to copy.
  /// \param Offset is the offset into \param Dest to copy to.
  /// \return an event representing copy operation.
  template <typename T, typename PropertyListT>
  event memcpy(
      ext::oneapi::experimental::device_global<T, PropertyListT> &Dest,
      const void *Src, size_t NumBytes = sizeof(T), size_t Offset = 0,
      const detail::code_location &CodeLoc = detail::code_location::current()) {
    detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
    return this->memcpy(Dest, Src, NumBytes, Offset, std::vector<event>{});
  }

  /// Copies data from a device_global to USM memory.
  /// Throws an exception if the copy operation intends to read outside the
  /// memory range \param Src, as specified through \param NumBytes and
  /// \param Offset.
  ///
  /// \param Dest is a USM pointer to copy to.
  /// \param Src is the source device_global.
  /// \param NumBytes is a number of bytes to copy.
  /// \param Offset is the offset into \param Src to copy from.
  /// \param DepEvents is a vector of events that specifies the operation
  /// dependencies.
  /// \return an event representing copy operation.
  template <typename T, typename PropertyListT>
  event memcpy(
      void *Dest,
      const ext::oneapi::experimental::device_global<T, PropertyListT> &Src,
      size_t NumBytes, size_t Offset, const std::vector<event> &DepEvents,
      const detail::code_location &CodeLoc = detail::code_location::current()) {
    detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
    if (sizeof(T) < Offset + NumBytes)
      throw sycl::exception(make_error_code(errc::invalid),
                            "Copy from device_global is out of bounds.");

    if (!detail::isDeviceGlobalUsedInKernel(&Src)) {
      // device_global is unregistered so we need a fallback. We let the handler
      // implement this fallback.
      return submit([&](handler &CGH) {
        CGH.depends_on(DepEvents);
        return CGH.memcpy(Dest, Src, NumBytes, Offset);
      });
    }

    constexpr bool IsDeviceImageScoped = PropertyListT::template has_property<
        ext::oneapi::experimental::device_image_scope_key>();
    return memcpyFromDeviceGlobal(Dest, &Src, IsDeviceImageScoped, NumBytes,
                                  Offset, DepEvents);
  }

  /// Copies data from a device_global to USM memory.
  /// Throws an exception if the copy operation intends to read outside the
  /// memory range \param Src, as specified through \param NumBytes and
  /// \param Offset.
  ///
  /// \param Dest is a USM pointer to copy to.
  /// \param Src is the source device_global.
  /// \param NumBytes is a number of bytes to copy.
  /// \param Offset is the offset into \param Src to copy from.
  /// \param DepEvent is a vector of event that specifies the operation
  /// dependency.
  /// \return an event representing copy operation.
  template <typename T, typename PropertyListT>
  event memcpy(
      void *Dest,
      const ext::oneapi::experimental::device_global<T, PropertyListT> &Src,
      size_t NumBytes, size_t Offset, event DepEvent,
      const detail::code_location &CodeLoc = detail::code_location::current()) {
    detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
    return this->memcpy(Dest, Src, NumBytes, Offset,
                        std::vector<event>{DepEvent});
  }

  /// Copies data from a device_global to USM memory.
  /// Throws an exception if the copy operation intends to read outside the
  /// memory range \param Src, as specified through \param NumBytes and
  /// \param Offset.
  ///
  /// \param Dest is a USM pointer to copy to.
  /// \param Src is the source device_global.
  /// \param NumBytes is a number of bytes to copy.
  /// \param Offset is the offset into \param Src to copy from.
  /// \return an event representing copy operation.
  template <typename T, typename PropertyListT>
  event memcpy(
      void *Dest,
      const ext::oneapi::experimental::device_global<T, PropertyListT> &Src,
      size_t NumBytes = sizeof(T), size_t Offset = 0,
      const detail::code_location &CodeLoc = detail::code_location::current()) {
    detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
    return this->memcpy(Dest, Src, NumBytes, Offset, std::vector<event>{});
  }

  /// Copies elements of type `std::remove_all_extents_t<T>` from a USM memory
  /// region to a device_global.
  /// Throws an exception if the copy operation intends to write outside the
  /// memory range \param Dest, as specified through \param Count and
  /// \param StartIndex.
  ///
  /// \param Src is a USM pointer to the source memory.
  /// \param Dest is the destination device_glboal.
  /// \param Count is a number of elements to copy.
  /// \param StartIndex is the index of the first element in Dest to copy to.
  /// \param DepEvents is a vector of events that specifies the operation
  /// dependencies.
  /// \return an event representing copy operation.
  template <typename T, typename PropertyListT>
  event copy(
      const std::remove_all_extents_t<T> *Src,
      ext::oneapi::experimental::device_global<T, PropertyListT> &Dest,
      size_t Count, size_t StartIndex, const std::vector<event> &DepEvents,
      const detail::code_location &CodeLoc = detail::code_location::current()) {
    detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
    return this->memcpy(Dest, Src, Count * sizeof(std::remove_all_extents_t<T>),
                        StartIndex * sizeof(std::remove_all_extents_t<T>),
                        DepEvents);
  }

  /// Copies elements of type `std::remove_all_extents_t<T>` from a USM memory
  /// region to a device_global.
  /// Throws an exception if the copy operation intends to write outside the
  /// memory range \param Dest, as specified through \param Count and
  /// \param StartIndex.
  ///
  /// \param Src is a USM pointer to the source memory.
  /// \param Dest is the destination device_glboal.
  /// \param Count is a number of elements to copy.
  /// \param StartIndex is the index of the first element in Dest to copy to.
  /// \param DepEvent is a vector of event that specifies the operation
  /// dependency.
  /// \return an event representing copy operation.
  template <typename T, typename PropertyListT>
  event copy(
      const std::remove_all_extents_t<T> *Src,
      ext::oneapi::experimental::device_global<T, PropertyListT> &Dest,
      size_t Count, size_t StartIndex, event DepEvent,
      const detail::code_location &CodeLoc = detail::code_location::current()) {
    detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
    return this->memcpy(Dest, Src, Count * sizeof(std::remove_all_extents_t<T>),
                        StartIndex * sizeof(std::remove_all_extents_t<T>),
                        DepEvent);
  }

  /// Copies elements of type `std::remove_all_extents_t<T>` from a USM memory
  /// region to a device_global.
  /// Throws an exception if the copy operation intends to write outside the
  /// memory range \param Dest, as specified through \param Count and
  /// \param StartIndex.
  ///
  /// \param Src is a USM pointer to the source memory.
  /// \param Dest is the destination device_glboal.
  /// \param Count is a number of elements to copy.
  /// \param StartIndex is the index of the first element in Dest to copy to.
  /// \return an event representing copy operation.
  template <typename T, typename PropertyListT>
  event copy(
      const std::remove_all_extents_t<T> *Src,
      ext::oneapi::experimental::device_global<T, PropertyListT> &Dest,
      size_t Count = sizeof(T) / sizeof(std::remove_all_extents_t<T>),
      size_t StartIndex = 0,
      const detail::code_location &CodeLoc = detail::code_location::current()) {
    detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
    return this->memcpy(Dest, Src, Count * sizeof(std::remove_all_extents_t<T>),
                        StartIndex * sizeof(std::remove_all_extents_t<T>));
  }

  /// Copies elements of type `std::remove_all_extents_t<T>` from a
  /// device_global to a USM memory region.
  /// Throws an exception if the copy operation intends to write outside the
  /// memory range \param Src, as specified through \param Count and
  /// \param StartIndex.
  ///
  /// \param Src is the source device_global.
  /// \param Dest is a USM pointer to copy to.
  /// \param Count is a number of elements to copy.
  /// \param StartIndex is the index of the first element in Src to copy from.
  /// \param DepEvents is a vector of events that specifies the operation
  /// dependencies.
  /// \return an event representing copy operation.
  template <typename T, typename PropertyListT>
  event copy(
      const ext::oneapi::experimental::device_global<T, PropertyListT> &Src,
      std::remove_all_extents_t<T> *Dest, size_t Count, size_t StartIndex,
      const std::vector<event> &DepEvents,
      const detail::code_location &CodeLoc = detail::code_location::current()) {
    detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
    return this->memcpy(Dest, Src, Count * sizeof(std::remove_all_extents_t<T>),
                        StartIndex * sizeof(std::remove_all_extents_t<T>),
                        DepEvents);
  }

  /// Copies elements of type `std::remove_all_extents_t<T>` from a
  /// device_global to a USM memory region.
  /// Throws an exception if the copy operation intends to write outside the
  /// memory range \param Src, as specified through \param Count and
  /// \param StartIndex.
  ///
  /// \param Src is the source device_global.
  /// \param Dest is a USM pointer to copy to.
  /// \param Count is a number of elements to copy.
  /// \param StartIndex is the index of the first element in Src to copy from.
  /// \param DepEvent is a vector of event that specifies the operation
  /// dependency.
  /// \return an event representing copy operation.
  template <typename T, typename PropertyListT>
  event copy(
      const ext::oneapi::experimental::device_global<T, PropertyListT> &Src,
      std::remove_all_extents_t<T> *Dest, size_t Count, size_t StartIndex,
      event DepEvent,
      const detail::code_location &CodeLoc = detail::code_location::current()) {
    detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
    return this->memcpy(Dest, Src, Count * sizeof(std::remove_all_extents_t<T>),
                        StartIndex * sizeof(std::remove_all_extents_t<T>),
                        DepEvent);
  }

  /// Copies elements of type `std::remove_all_extents_t<T>` from a
  /// device_global to a USM memory region.
  /// Throws an exception if the copy operation intends to write outside the
  /// memory range \param Src, as specified through \param Count and
  /// \param StartIndex.
  ///
  /// \param Src is the source device_global.
  /// \param Dest is a USM pointer to copy to.
  /// \param Count is a number of elements to copy.
  /// \param StartIndex is the index of the first element in Src to copy from.
  /// \return an event representing copy operation.
  template <typename T, typename PropertyListT>
  event copy(
      const ext::oneapi::experimental::device_global<T, PropertyListT> &Src,
      std::remove_all_extents_t<T> *Dest,
      size_t Count = sizeof(T) / sizeof(std::remove_all_extents_t<T>),
      size_t StartIndex = 0,
      const detail::code_location &CodeLoc = detail::code_location::current()) {
    detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
    return this->memcpy(Dest, Src, Count * sizeof(std::remove_all_extents_t<T>),
                        StartIndex * sizeof(std::remove_all_extents_t<T>));
  }

  /// Copies data from one memory region to another, where \p Src is a USM
  /// pointer and \p Dest is an opaque image memory handle wrapper. An exception
  /// is thrown if either \p Src is nullptr or \p Dest is incomplete. The
  /// behavior is undefined if \p DestImgDesc is inconsistent with the allocated
  /// memory region.
  ///
  /// \param Src is a USM pointer to the source memory.
  /// \param Dest is a wrapper for an opaque image memory handle to the
  /// destination memory.
  /// \param DestImgDesc is the image descriptor (format, order, dimensions).
  /// \return an event representing the copy operation.
  event ext_oneapi_copy(
      void *Src, ext::oneapi::experimental::image_mem_handle Dest,
      const ext::oneapi::experimental::image_descriptor &DestImgDesc,
      const detail::code_location &CodeLoc = detail::code_location::current()) {
    detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
    return submit(
        [&](handler &CGH) { CGH.ext_oneapi_copy(Src, Dest, DestImgDesc); },
        CodeLoc);
  }

  /// Copies data from one memory region to another, where \p Src is a USM
  /// pointer and \p Dest is an opaque image memory handle. Allows for a
  /// sub-region copy, where \p SrcOffset , \p DestOffset , and \p CopyExtent
  /// are used to determine the sub-region. An exception is thrown if either \p
  /// Src is nullptr or \p CopyExtent is incomplete.
  ///
  /// \param Src is a USM pointer to the source memory.
  /// \param SrcOffset is an offset from the origin where the x, y, and z
  ///                  components are measured in bytes, rows, and slices
  ///                  respectively
  /// \param SrcExtent is the extent of the source memory to copy, measured in
  ///                  pixels (pixel size determined by \p DestImgDesc )
  /// \param Dest is an opaque image memory handle to the destination memory.
  /// \param DestOffset is an offset from the destination origin measured in
  ///                   pixels (pixel size determined by \p DestImgDesc )
  /// \param DestImgDesc is the destination image descriptor
  /// \param CopyExtent is the width, height, and depth of the region to copy
  ///                   measured in pixels as determined by \p DestImgDesc
  /// \return an event representing the copy operation.
  event ext_oneapi_copy(
      void *Src, sycl::range<3> SrcOffset, sycl::range<3> SrcExtent,
      ext::oneapi::experimental::image_mem_handle Dest,
      sycl::range<3> DestOffset,
      const ext::oneapi::experimental::image_descriptor &DestImgDesc,
      sycl::range<3> CopyExtent,
      const detail::code_location &CodeLoc = detail::code_location::current()) {
    detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
    return submit(
        [&](handler &CGH) {
          CGH.ext_oneapi_copy(Src, SrcOffset, SrcExtent, Dest, DestOffset,
                              DestImgDesc, CopyExtent);
        },
        CodeLoc);
  }

  /// Copies data from one memory region to another, where \p Src is a USM
  /// pointer and \p Dest is an opaque image memory handle wrapper. An exception
  /// is thrown if either \p Src is nullptr or \p Dest is incomplete. The
  /// behavior is undefined if \p DestImgDesc is inconsistent with the allocated
  /// memory region.
  ///
  /// \param Src is a USM pointer to the source memory.
  /// \param Dest is a wrapper for an opaque image memory handle to the
  /// destination memory.
  /// \param DestImgDesc is the destination image descriptor
  /// \param DepEvent is an event that specifies the kernel dependencies.
  /// \return an event representing the copy operation.
  event ext_oneapi_copy(
      void *Src, ext::oneapi::experimental::image_mem_handle Dest,
      const ext::oneapi::experimental::image_descriptor &DestImgDesc,
      event DepEvent,
      const detail::code_location &CodeLoc = detail::code_location::current()) {
    detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
    return submit(
        [&](handler &CGH) {
          CGH.depends_on(DepEvent);
          CGH.ext_oneapi_copy(Src, Dest, DestImgDesc);
        },
        CodeLoc);
  }

  /// Copies data from one memory region to another, where \p Src is a USM
  /// pointer and \p Dest is an opaque image memory handle. Allows for a
  /// sub-region copy, where \p SrcOffset , \p DestOffset , and \p CopyExtent
  /// are used to determine the sub-region. An exception is thrown if either \p
  /// Src is nullptr or \p Dest is incomplete.
  ///
  /// \param Src is a USM pointer to the source memory.
  /// \param SrcOffset is an offset from the origin where the x, y, and z
  ///                  components are measured in bytes, rows, and slices
  ///                  respectively
  /// \param SrcExtent is the extent of the source memory to copy, measured in
  ///                  pixels (pixel size determined by \p DestImgDesc )
  /// \param Dest is an opaque image memory handle to the destination memory.
  /// \param DestOffset is an offset from the destination origin measured in
  ///                   pixels (pixel size determined by \p DestImgDesc )
  /// \param DestImgDesc is the destination image descriptor
  /// \param CopyExtent is the width, height, and depth of the region to copy
  ///               measured in pixels as determined by \p DestImgDesc
  /// \param DepEvent is an event that specifies the kernel dependencies.
  /// \return an event representing the copy operation.
  event ext_oneapi_copy(
      void *Src, sycl::range<3> SrcOffset, sycl::range<3> SrcExtent,
      ext::oneapi::experimental::image_mem_handle Dest,
      sycl::range<3> DestOffset,
      const ext::oneapi::experimental::image_descriptor &DestImgDesc,
      sycl::range<3> CopyExtent, event DepEvent,
      const detail::code_location &CodeLoc = detail::code_location::current()) {
    detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
    return submit(
        [&](handler &CGH) {
          CGH.depends_on(DepEvent);
          CGH.ext_oneapi_copy(Src, SrcOffset, SrcExtent, Dest, DestOffset,
                              DestImgDesc, CopyExtent);
        },
        CodeLoc);
  }

  /// Copies data from one memory region to another, where \p Src is a USM
  /// pointer and \p Dest is an opaque image memory handle wrapper. An exception
  /// is thrown if either \p Src is nullptr or \p Dest is incomplete. The
  /// behavior is undefined if \p DestImgDesc is inconsistent with the allocated
  /// memory region.
  ///
  /// \param Src is a USM pointer to the source memory.
  /// \param Dest is a wrapper for an opaque image memory handle to the
  /// destination memory.
  /// \param DestImgDesc is the destination image descriptor
  /// \param DepEvents is a vector of events that specifies the kernel
  /// dependencies.
  /// \return an event representing the copy operation.
  event ext_oneapi_copy(
      void *Src, ext::oneapi::experimental::image_mem_handle Dest,
      const ext::oneapi::experimental::image_descriptor &DestImgDesc,
      const std::vector<event> &DepEvents,
      const detail::code_location &CodeLoc = detail::code_location::current()) {
    detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
    return submit(
        [&](handler &CGH) {
          CGH.depends_on(DepEvents);
          CGH.ext_oneapi_copy(Src, Dest, DestImgDesc);
        },
        CodeLoc);
  }

  /// Copies data from one memory region to another, where \p Src is a USM
  /// pointer and \p Dest is an opaque image memory handle. Allows for a
  /// sub-region copy, where \p SrcOffset , \p DestOffset , and \p CopyExtent
  /// are used to determine the sub-region. An exception is thrown if either \p
  /// Src is nullptr or \p Dest is incomplete.
  ///
  /// \param Src is a USM pointer to the source memory.
  /// \param SrcOffset is an offset from the origin where the x, y, and z
  ///                  components are measured in bytes, rows, and slices
  ///                  respectively
  /// \param SrcExtent is the extent of the source memory to copy, measured in
  ///                  pixels (pixel size determined by \p DestImgDesc )
  /// \param Dest is an opaque image memory handle to the destination memory.
  /// \param DestOffset is an offset from the destination origin measured in
  ///                   pixels (pixel size determined by \p DestImgDesc )
  /// \param DestImgDesc is the destination image descriptor
  /// \param CopyExtent is the width, height, and depth of the region to copy
  ///               measured in pixels as determined by \p DestImgDesc
  /// \param DepEvents is a vector of events that specifies the kernel
  ///                  dependencies.
  /// \return an event representing the copy operation.
  event ext_oneapi_copy(
      void *Src, sycl::range<3> SrcOffset, sycl::range<3> SrcExtent,
      ext::oneapi::experimental::image_mem_handle Dest,
      sycl::range<3> DestOffset,
      const ext::oneapi::experimental::image_descriptor &DestImgDesc,
      sycl::range<3> CopyExtent, const std::vector<event> &DepEvents,
      const detail::code_location &CodeLoc = detail::code_location::current()) {
    detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
    return submit(
        [&](handler &CGH) {
          CGH.depends_on(DepEvents);
          CGH.ext_oneapi_copy(Src, SrcOffset, SrcExtent, Dest, DestOffset,
                              DestImgDesc, CopyExtent);
        },
        CodeLoc);
  }

  /// Copies data from one memory region to another, where \p Src is an opaque
  /// image memory handle and \p Dest is a USM pointer.
  /// An exception is thrown if either \p Src is incomplete or \p Dest is
  /// nullptr. The behavior is undefined if \p SrcImgDesc is inconsistent with
  /// the allocated memory region.
  ///
  /// \param Src is an opaque image memory handle to the source memory.
  /// \param Dest is a USM pointer to the destination memory.
  /// \param SrcImgDesc is the source image descriptor.
  /// \return an event representing the copy operation.
  event ext_oneapi_copy(
      ext::oneapi::experimental::image_mem_handle Src, void *Dest,
      const ext::oneapi::experimental::image_descriptor &SrcImgDesc,
      const detail::code_location &CodeLoc = detail::code_location::current()) {
    detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
    return submit(
        [&](handler &CGH) { CGH.ext_oneapi_copy(Src, Dest, SrcImgDesc); },
        CodeLoc);
  }

  /// Copies data from one memory region to another, where \p Src is an opaque
  /// image memory handle and \p Dest is a USM pointer. Allows for a
  /// sub-region copy, where \p SrcOffset , \p DestOffset , and \p CopyExtent
  /// are used to determine the sub-region.  Pixel size is determined by \p
  /// SrcImgDesc An exception is thrown if either \p Src is nullptr or \p Dest
  /// is incomplete.
  ///
  /// \param Src is an opaque image memory handle to the source memory.
  /// \param SrcOffset is an offset from the origin of source measured in pixels
  ///                   (pixel size determined by \p SrcImgDesc )
  /// \param SrcImgDesc is the source image descriptor
  /// \param Dest is a USM pointer to the destination memory.
  /// \param DestOffset is an offset from the destination origin where the
  ///                  x, y, and z components are measured in bytes, rows,
  ///                  and slices respectively
  /// \param DestExtent is the extent of the dest memory to copy, measured in
  ///                   pixels (pixel size determined by \p DestImgDesc )
  /// \param CopyExtent is the width, height, and depth of the region to copy
  ///               measured in pixels (pixel size determined by
  ///               \p SrcImgDesc )
  /// \return an event representing the copy operation.
  event ext_oneapi_copy(
      ext::oneapi::experimental::image_mem_handle Src, sycl::range<3> SrcOffset,
      const ext::oneapi::experimental::image_descriptor &SrcImgDesc, void *Dest,
      sycl::range<3> DestOffset, sycl::range<3> DestExtent,
      sycl::range<3> CopyExtent,
      const detail::code_location &CodeLoc = detail::code_location::current()) {
    detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
    return submit(
        [&](handler &CGH) {
          CGH.ext_oneapi_copy(Src, SrcOffset, SrcImgDesc, Dest, DestOffset,
                              DestExtent, CopyExtent);
        },
        CodeLoc);
  }

  /// Copies data from one memory region to another, where \p Src is an opaque
  /// image memory handle and \p Dest is a USM pointer.
  /// An exception is thrown if either \p Src is incomplete or \p Dest is
  /// nullptr. The behavior is undefined if \p SrcImgDesc is inconsistent with
  /// the allocated memory region.
  ///
  /// \param Src is an opaque image memory handle to the source memory.
  /// \param Dest is a USM pointer to the destination memory.
  /// \param SrcImgDesc is the image descriptor (format, order, dimensions).
  /// \param DepEvent is an event that specifies the kernel dependencies.
  /// \return an event representing the copy operation.
  event ext_oneapi_copy(
      ext::oneapi::experimental::image_mem_handle Src, void *Dest,
      const ext::oneapi::experimental::image_descriptor &SrcImgDesc,
      event DepEvent,
      const detail::code_location &CodeLoc = detail::code_location::current()) {
    detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
    return submit(
        [&](handler &CGH) {
          CGH.depends_on(DepEvent);
          CGH.ext_oneapi_copy(Src, Dest, SrcImgDesc);
        },
        CodeLoc);
  }

  /// Copies data from one memory region to another, where \p Src is an opaque
  /// image memory handle and \p Dest is a USM pointer. Allows for a
  /// sub-region copy, where \p SrcOffset , \p DestOffset , and \p CopyExtent
  /// are used to determine the sub-region.  Pixel size is determined by \p
  /// SrcImgDesc An exception is thrown if either \p Src is nullptr or \p Dest
  /// is incomplete.
  ///
  /// \param Src is an opaque image memory handle to the source memory.
  /// \param SrcOffset is an offset from the origin of source measured in pixels
  ///                   (pixel size determined by \p SrcImgDesc )
  /// \param SrcImgDesc is the source image descriptor
  /// \param Dest is a USM pointer to the destination memory.
  /// \param DestOffset is an offset from the destination origin where the
  ///                  x, y, and z components are measured in bytes, rows,
  ///                  and slices respectively
  /// \param DestExtent is the extent of the dest memory to copy, measured in
  ///                   pixels (pixel size determined by \p DestImgDesc )
  /// \param CopyExtent is the width, height, and depth of the region to copy
  ///               measured in pixels (pixel size determined by
  ///               \p SrcImgDesc )
  /// \param DepEvent is an event that specifies the kernel dependencies.
  /// \return an event representing the copy operation.
  event ext_oneapi_copy(
      ext::oneapi::experimental::image_mem_handle Src, sycl::range<3> SrcOffset,
      const ext::oneapi::experimental::image_descriptor &SrcImgDesc, void *Dest,
      sycl::range<3> DestOffset, sycl::range<3> DestExtent,
      sycl::range<3> CopyExtent, event DepEvent,
      const detail::code_location &CodeLoc = detail::code_location::current()) {
    detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
    return submit(
        [&](handler &CGH) {
          CGH.depends_on(DepEvent);
          CGH.ext_oneapi_copy(Src, SrcOffset, SrcImgDesc, Dest, DestOffset,
                              DestExtent, CopyExtent);
        },
        CodeLoc);
  }

  /// Copies data from one memory region to another, where \p Src is an opaque
  /// image memory handle and \p Dest is a USM pointer.
  /// An exception is thrown if either \p Src is incomplete or \p Dest is
  /// nullptr. The behavior is undefined if \p SrcImgDesc is inconsistent with
  /// the allocated memory region.
  ///
  /// \param Src is an opaque image memory handle to the source memory.
  /// \param Dest is a USM pointer to the destination memory.
  /// \param SrcImgDesc is the image descriptor (format, order, dimensions).
  /// \param DepEvents is a vector of events that specifies the kernel
  /// dependencies.
  /// \return an event representing the copy operation.
  event ext_oneapi_copy(
      ext::oneapi::experimental::image_mem_handle Src, void *Dest,
      const ext::oneapi::experimental::image_descriptor &SrcImgDesc,
      const std::vector<event> &DepEvents,
      const detail::code_location &CodeLoc = detail::code_location::current()) {
    detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
    return submit(
        [&](handler &CGH) {
          CGH.depends_on(DepEvents);
          CGH.ext_oneapi_copy(Src, Dest, SrcImgDesc);
        },
        CodeLoc);
  }

  /// Copies data from one memory region to another, where \p Src is an opaque
  /// image memory handle and \p Dest is a USM pointer. Allows for a
  /// sub-region copy, where \p SrcOffset , \p DestOffset , and \p CopyExtent
  /// are used to determine the sub-region.  Pixel size is determined by \p
  /// SrcImgDesc An exception is thrown if either \p Src is nullptr or \p Dest
  /// is incomplete.
  ///
  /// \param Src is an opaque image memory handle to the source memory.
  /// \param SrcOffset is an offset from the origin of source measured in pixels
  ///                   (pixel size determined by \p SrcImgDesc )
  /// \param SrcImgDesc is the source image descriptor
  /// \param Dest is a USM pointer to the destination memory.
  /// \param DestOffset is an offset from the destination origin where the
  ///                  x, y, and z components are measured in bytes, rows,
  ///                  and slices respectively
  /// \param DestExtent is the extent of the dest memory to copy, measured in
  ///                   pixels (pixel size determined by \p DestImgDesc )
  /// \param CopyExtent is the width, height, and depth of the region to copy
  ///               measured in pixels (pixel size determined by
  ///               \p SrcImgDesc )
  /// \param DepEvents is a vector of events that specifies the kernel
  ///                  dependencies.
  /// \return an event representing the copy operation.
  event ext_oneapi_copy(
      ext::oneapi::experimental::image_mem_handle Src, sycl::range<3> SrcOffset,
      const ext::oneapi::experimental::image_descriptor &SrcImgDesc, void *Dest,
      sycl::range<3> DestOffset, sycl::range<3> DestExtent,
      sycl::range<3> CopyExtent, const std::vector<event> &DepEvents,
      const detail::code_location &CodeLoc = detail::code_location::current()) {
    detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
    return submit(
        [&](handler &CGH) {
          CGH.depends_on(DepEvents);
          CGH.ext_oneapi_copy(Src, SrcOffset, SrcImgDesc, Dest, DestOffset,
                              DestExtent, CopyExtent);
        },
        CodeLoc);
  }

  /// Copies data from one memory region to another, where \p Src and \p Dest
  /// are USM pointers. An exception is thrown if either \p Src is nullptr, \p
  /// Dest is nullptr, or \p Pitch is inconsistent with hardware requirements.
  /// The behavior is undefined if \p DeviceImgDesc is inconsistent with the
  /// allocated memory region.
  ///
  /// \param Src is a USM pointer to the source memory.
  /// \param Dest is a USM pointer to the destination memory.
  /// \param DeviceImgDesc is the image descriptor
  /// \param DeviceRowPitch is the DeviceRowPitch of the rows on the device.
  /// \return an event representing the copy operation.
  event ext_oneapi_copy(
      void *Src, void *Dest,
      const ext::oneapi::experimental::image_descriptor &DeviceImgDesc,
      size_t DeviceRowPitch,
      const detail::code_location &CodeLoc = detail::code_location::current()) {
    detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
    return submit(
        [&](handler &CGH) {
          CGH.ext_oneapi_copy(Src, Dest, DeviceImgDesc, DeviceRowPitch);
        },
        CodeLoc);
  }

  /// Copies data from one memory region to another, where \p Src and \p Dest
  /// are USM pointers. Allows for a sub-region copy, where \p SrcOffset ,
  /// \p DestOffset , and \p Extent are used to determine the sub-region.
  /// An exception is thrown if either \p Src is nullptr or \p Dest is
  /// incomplete.
  ///
  /// \param Src is a USM pointer to the source memory.
  /// \param SrcOffset is an destination offset from the origin where the
  ///                  x, y, and z components are measured in bytes, rows,
  ///                  and slices respectively
  /// \param Dest is a USM pointer to the destination memory.
  /// \param DestOffset is an destination offset from the origin where the
  ///                  x, y, and z components are measured in bytes, rows,
  ///                  and slices respectively
  /// \param DeviceImgDesc is the device image descriptor
  /// \param DeviceRowPitch is the row pitch on the device
  /// \param HostExtent is the extent of the host memory to copy, measured in
  ///                   pixels (pixel size determined by \p DeviceImgDesc )
  /// \param CopyExtent is the width, height, and depth of the region to copy
  ///               measured in pixels (pixel size determined by
  ///               \p DeviceImgDesc )
  /// \return an event representing the copy operation.
  event ext_oneapi_copy(
      void *Src, sycl::range<3> SrcOffset, void *Dest,
      sycl::range<3> DestOffset,
      const ext::oneapi::experimental::image_descriptor &DeviceImgDesc,
      size_t DeviceRowPitch, sycl::range<3> HostExtent,
      sycl::range<3> CopyExtent,
      const detail::code_location &CodeLoc = detail::code_location::current()) {
    detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
    return submit(
        [&](handler &CGH) {
          CGH.ext_oneapi_copy(Src, SrcOffset, Dest, DestOffset, DeviceImgDesc,
                              DeviceRowPitch, HostExtent, CopyExtent);
        },
        CodeLoc);
  }

  /// Copies data from one memory region to another, where \p Src and \p Dest
  /// are USM pointers. An exception is thrown if either \p Src is nullptr, \p
  /// Dest is nullptr, or \p Pitch is inconsistent with hardware requirements.
  /// The behavior is undefined if \p DeviceImgDesc is inconsistent with the
  /// allocated memory region.
  ///
  /// \param Src is a USM pointer to the source memory.
  /// \param Dest is a USM pointer to the destination memory.
  /// \param DeviceImgDesc is the image descriptor
  /// \param DeviceRowPitch is the pitch of the rows on the device.
  /// \param DepEvent is an event that specifies the kernel dependencies.
  /// \return an event representing the copy operation.
  event ext_oneapi_copy(
      void *Src, void *Dest,
      const ext::oneapi::experimental::image_descriptor &DeviceImgDesc,
      size_t DeviceRowPitch, event DepEvent,
      const detail::code_location &CodeLoc = detail::code_location::current()) {
    detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
    return submit(
        [&](handler &CGH) {
          CGH.depends_on(DepEvent);
          CGH.ext_oneapi_copy(Src, Dest, DeviceImgDesc, DeviceRowPitch);
        },
        CodeLoc);
  }

  /// Copies data from one memory region to another, where \p Src and \p Dest
  /// are USM pointers. Allows for a sub-region copy, where \p SrcOffset ,
  /// \p DestOffset , and \p Extent are used to determine the sub-region.
  /// An exception is thrown if either \p Src is nullptr or \p Dest is
  /// incomplete.
  ///
  /// \param Src is a USM pointer to the source memory.
  /// \param SrcOffset is an destination offset from the origin where the
  ///                  x, y, and z components are measured in bytes, rows,
  ///                  and slices respectively
  /// \param Dest is a USM pointer to the destination memory.
  /// \param DestOffset is an destination offset from the origin where the
  ///                  x, y, and z components are measured in bytes, rows,
  ///                  and slices respectively
  /// \param DeviceImgDesc is the destination image descriptor
  /// \param DeviceRowPitch is the row pitch on the device
  /// \param HostExtent is the extent of the host memory to copy, measured in
  ///                   pixels (pixel size determined by \p DeviceImgDesc )
  /// \param CopyExtent is the width, height, and depth of the region to copy
  ///               measured in pixels (pixel size determined by
  ///               \p DeviceImgDesc )
  /// \param DepEvent is an event that specifies the kernel dependencies.
  /// \return an event representing the copy operation.
  event ext_oneapi_copy(
      void *Src, sycl::range<3> SrcOffset, void *Dest,
      sycl::range<3> DestOffset,
      const ext::oneapi::experimental::image_descriptor &DeviceImgDesc,
      size_t DeviceRowPitch, sycl::range<3> HostExtent,
      sycl::range<3> CopyExtent, event DepEvent,
      const detail::code_location &CodeLoc = detail::code_location::current()) {
    detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
    return submit(
        [&](handler &CGH) {
          CGH.depends_on(DepEvent);
          CGH.ext_oneapi_copy(Src, SrcOffset, Dest, DestOffset, DeviceImgDesc,
                              DeviceRowPitch, HostExtent, CopyExtent);
        },
        CodeLoc);
  }

  /// Copies data from one memory region to another, where \p Src and \p Dest
  /// are USM pointers. An exception is thrown if either \p Src is nullptr, \p
  /// Dest is nullptr, or \p Pitch is inconsistent with hardware requirements.
  /// The behavior is undefined if \p DeviceImgDesc is inconsistent with the
  /// allocated memory region.
  ///
  /// \param Src is a USM pointer to the source memory.
  /// \param Dest is a USM pointer to the destination memory.
  /// \param DeviceImgDesc is the image descriptor
  /// \param DeviceRowPitch is the pitch of the rows on the device.
  /// \param DepEvents is a vector of events that specifies the kernel
  /// dependencies.
  /// \return an event representing the copy operation.
  event ext_oneapi_copy(
      void *Src, void *Dest,
      const ext::oneapi::experimental::image_descriptor &DeviceImgDesc,
      size_t DeviceRowPitch, const std::vector<event> &DepEvents,
      const detail::code_location &CodeLoc = detail::code_location::current()) {
    detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
    return submit(
        [&](handler &CGH) {
          CGH.depends_on(DepEvents);
          CGH.ext_oneapi_copy(Src, Dest, DeviceImgDesc, DeviceRowPitch);
        },
        CodeLoc);
  }

  /// Copies data from one memory region to another, where \p Src and \p Dest
  /// are USM pointers. Allows for a sub-region copy, where \p SrcOffset ,
  /// \p DestOffset , and \p Extent are used to determine the sub-region.
  /// An exception is thrown if either \p Src is nullptr or \p Dest is
  /// incomplete.
  ///
  /// \param Src is a USM pointer to the source memory.
  /// \param SrcOffset is an destination offset from the origin where the
  ///                  x, y, and z components are measured in bytes, rows,
  ///                  and slices respectively
  /// \param Dest is a USM pointer to the destination memory.
  /// \param DestOffset is an destination offset from the origin where the
  ///                  x, y, and z components are measured in bytes, rows,
  ///                  and slices respectively
  /// \param DeviceImgDesc is the destination image descriptor
  /// \param DeviceRowPitch is the row pitch on the device
  /// \param HostExtent is the extent of the host memory to copy, measured in
  ///                   pixels (pixel size determined by \p DeviceImgDesc )
  /// \param CopyExtent is the width, height, and depth of the region to copy
  ///               measured in pixels (pixel size determined by
  ///               \p DeviceImgDesc )
  /// \param DepEvents is a vector of events that specifies the kernel
  ///                  dependencies.
  /// \return an event representing the copy operation.
  event ext_oneapi_copy(
      void *Src, sycl::range<3> SrcOffset, void *Dest,
      sycl::range<3> DestOffset,
      const ext::oneapi::experimental::image_descriptor &DeviceImgDesc,
      size_t DeviceRowPitch, sycl::range<3> HostExtent,
      sycl::range<3> CopyExtent, const std::vector<event> &DepEvents,
      const detail::code_location &CodeLoc = detail::code_location::current()) {
    detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
    return submit(
        [&](handler &CGH) {
          CGH.depends_on(DepEvents);
          CGH.ext_oneapi_copy(Src, SrcOffset, Dest, DestOffset, DeviceImgDesc,
                              DeviceRowPitch, HostExtent, CopyExtent);
        },
        CodeLoc);
  }

  /// Instruct the queue with a non-blocking wait on an external semaphore.
  /// An exception is thrown if \p SemaphoreHandle is incomplete.
  ///
  /// \param SemaphoreHandle is an opaque external interop semaphore handle
  /// \return an event representing the wait operation.
  event ext_oneapi_wait_external_semaphore(
      sycl::ext::oneapi::experimental::interop_semaphore_handle SemaphoreHandle,
      const detail::code_location &CodeLoc = detail::code_location::current()) {
    detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
    return submit(
        [&](handler &CGH) {
          CGH.ext_oneapi_wait_external_semaphore(SemaphoreHandle);
        },
        CodeLoc);
  }

  /// Instruct the queue with a non-blocking wait on an external semaphore.
  /// An exception is thrown if \p SemaphoreHandle is incomplete.
  ///
  /// \param SemaphoreHandle is an opaque external interop semaphore handle
  /// \param DepEvent is an event that specifies the kernel dependencies.
  /// \return an event representing the wait operation.
  event ext_oneapi_wait_external_semaphore(
      sycl::ext::oneapi::experimental::interop_semaphore_handle SemaphoreHandle,
      event DepEvent,
      const detail::code_location &CodeLoc = detail::code_location::current()) {
    detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
    return submit(
        [&](handler &CGH) {
          CGH.depends_on(DepEvent);
          CGH.ext_oneapi_wait_external_semaphore(SemaphoreHandle);
        },
        CodeLoc);
  }

  /// Instruct the queue with a non-blocking wait on an external semaphore.
  /// An exception is thrown if \p SemaphoreHandle is incomplete.
  ///
  /// \param SemaphoreHandle is an opaque external interop semaphore handle
  /// \param DepEvents is a vector of events that specifies the kernel
  /// dependencies.
  /// \return an event representing the wait operation.
  event ext_oneapi_wait_external_semaphore(
      sycl::ext::oneapi::experimental::interop_semaphore_handle SemaphoreHandle,
      const std::vector<event> &DepEvents,
      const detail::code_location &CodeLoc = detail::code_location::current()) {
    detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
    return submit(
        [&](handler &CGH) {
          CGH.depends_on(DepEvents);
          CGH.ext_oneapi_wait_external_semaphore(SemaphoreHandle);
        },
        CodeLoc);
  }

  /// Instruct the queue to signal the external semaphore once all previous
  /// commands have completed execution.
  /// An exception is thrown if \p SemaphoreHandle is incomplete.
  ///
  /// \param SemaphoreHandle is an opaque external interop semaphore handle
  /// \return an event representing the signal operation.
  event ext_oneapi_signal_external_semaphore(
      sycl::ext::oneapi::experimental::interop_semaphore_handle SemaphoreHandle,
      const detail::code_location &CodeLoc = detail::code_location::current()) {
    detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
    return submit(
        [&](handler &CGH) {
          CGH.ext_oneapi_signal_external_semaphore(SemaphoreHandle);
        },
        CodeLoc);
  }

  /// Instruct the queue to signal the external semaphore once all previous
  /// commands have completed execution.
  /// An exception is thrown if \p SemaphoreHandle is incomplete.
  ///
  /// \param SemaphoreHandle is an opaque external interop semaphore handle
  /// \param DepEvent is an event that specifies the kernel dependencies.
  /// \return an event representing the signal operation.
  event ext_oneapi_signal_external_semaphore(
      sycl::ext::oneapi::experimental::interop_semaphore_handle SemaphoreHandle,
      event DepEvent,
      const detail::code_location &CodeLoc = detail::code_location::current()) {
    detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
    return submit(
        [&](handler &CGH) {
          CGH.depends_on(DepEvent);
          CGH.ext_oneapi_signal_external_semaphore(SemaphoreHandle);
        },
        CodeLoc);
  }

  /// Instruct the queue to signal the external semaphore once all previous
  /// commands have completed execution.
  /// An exception is thrown if \p SemaphoreHandle is incomplete.
  ///
  /// \param SemaphoreHandle is an opaque external interop semaphore handle
  /// \param DepEvents is a vector of events that specifies the kernel
  /// dependencies.
  /// \return an event representing the signal operation.
  event ext_oneapi_signal_external_semaphore(
      sycl::ext::oneapi::experimental::interop_semaphore_handle SemaphoreHandle,
      const std::vector<event> &DepEvents,
      const detail::code_location &CodeLoc = detail::code_location::current()) {
    detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
    return submit(
        [&](handler &CGH) {
          CGH.depends_on(DepEvents);
          CGH.ext_oneapi_signal_external_semaphore(SemaphoreHandle);
        },
        CodeLoc);
  }

  /// single_task version with a kernel represented as a lambda.
  ///
  /// \param Properties is the kernel properties.
  /// \param KernelFunc is the Kernel functor or lambda
  /// \param CodeLoc contains the code location of user code
  template <typename KernelName = detail::auto_name, typename KernelType,
            typename PropertiesT>
  std::enable_if_t<
      ext::oneapi::experimental::is_property_list<PropertiesT>::value, event>
  single_task(
      PropertiesT Properties, _KERNELFUNCPARAM(KernelFunc),
      const detail::code_location &CodeLoc = detail::code_location::current()) {
    static_assert(
        (detail::check_fn_signature<std::remove_reference_t<KernelType>,
                                    void()>::value ||
         detail::check_fn_signature<std::remove_reference_t<KernelType>,
                                    void(kernel_handler)>::value),
        "sycl::queue.single_task() requires a kernel instead of command group. "
        "Use queue.submit() instead");

    detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
    return submit(
        [&](handler &CGH) {
          CGH.template single_task<KernelName, KernelType, PropertiesT>(
              Properties, KernelFunc);
        },
        CodeLoc);
  }

  /// single_task version with a kernel represented as a lambda.
  ///
  /// \param KernelFunc is the Kernel functor or lambda
  /// \param CodeLoc contains the code location of user code
  template <typename KernelName = detail::auto_name, typename KernelType>
  event single_task(
      _KERNELFUNCPARAM(KernelFunc),
      const detail::code_location &CodeLoc = detail::code_location::current()) {
    return single_task<KernelName, KernelType>(
        ext::oneapi::experimental::empty_properties_t{}, KernelFunc, CodeLoc);
  }

  /// single_task version with a kernel represented as a lambda.
  ///
  /// \param DepEvent is an event that specifies the kernel dependencies
  /// \param Properties is the kernel properties.
  /// \param KernelFunc is the Kernel functor or lambda
  /// \param CodeLoc contains the code location of user code
  template <typename KernelName = detail::auto_name, typename KernelType,
            typename PropertiesT>
  std::enable_if_t<
      ext::oneapi::experimental::is_property_list<PropertiesT>::value, event>
  single_task(
      event DepEvent, PropertiesT Properties, _KERNELFUNCPARAM(KernelFunc),
      const detail::code_location &CodeLoc = detail::code_location::current()) {
    static_assert(
        (detail::check_fn_signature<std::remove_reference_t<KernelType>,
                                    void()>::value ||
         detail::check_fn_signature<std::remove_reference_t<KernelType>,
                                    void(kernel_handler)>::value),
        "sycl::queue.single_task() requires a kernel instead of command group. "
        "Use queue.submit() instead");

    detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
    return submit(
        [&](handler &CGH) {
          CGH.depends_on(DepEvent);
          CGH.template single_task<KernelName, KernelType, PropertiesT>(
              Properties, KernelFunc);
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
      event DepEvent, _KERNELFUNCPARAM(KernelFunc),
      const detail::code_location &CodeLoc = detail::code_location::current()) {
    return single_task<KernelName, KernelType>(
        DepEvent, ext::oneapi::experimental::empty_properties_t{}, KernelFunc,
        CodeLoc);
  }

  /// single_task version with a kernel represented as a lambda.
  ///
  /// \param DepEvents is a vector of events that specifies the kernel
  /// dependencies
  /// \param Properties is the kernel properties.
  /// \param KernelFunc is the Kernel functor or lambda
  /// \param CodeLoc contains the code location of user code
  template <typename KernelName = detail::auto_name, typename KernelType,
            typename PropertiesT>
  std::enable_if_t<
      ext::oneapi::experimental::is_property_list<PropertiesT>::value, event>
  single_task(
      const std::vector<event> &DepEvents, PropertiesT Properties,
      _KERNELFUNCPARAM(KernelFunc),
      const detail::code_location &CodeLoc = detail::code_location::current()) {
    static_assert(
        (detail::check_fn_signature<std::remove_reference_t<KernelType>,
                                    void()>::value ||
         detail::check_fn_signature<std::remove_reference_t<KernelType>,
                                    void(kernel_handler)>::value),
        "sycl::queue.single_task() requires a kernel instead of command group. "
        "Use queue.submit() instead");

    detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
    return submit(
        [&](handler &CGH) {
          CGH.depends_on(DepEvents);
          CGH.template single_task<KernelName, KernelType, PropertiesT>(
              Properties, KernelFunc);
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
      const std::vector<event> &DepEvents, _KERNELFUNCPARAM(KernelFunc),
      const detail::code_location &CodeLoc = detail::code_location::current()) {
    return single_task<KernelName, KernelType>(
        DepEvents, ext::oneapi::experimental::empty_properties_t{}, KernelFunc,
        CodeLoc);
  }

  /// parallel_for version with a kernel represented as a lambda + range that
  /// specifies global size only.
  ///
  /// \param Range specifies the global work space of the kernel
  /// \param Rest acts as-if: "ReductionTypes&&... Reductions,
  /// const KernelType &KernelFunc".
  template <typename KernelName = detail::auto_name, typename... RestT>
  event parallel_for(range<1> Range, RestT &&...Rest) {
    return parallel_for_impl<KernelName>(Range, Rest...);
  }

  /// parallel_for version with a kernel represented as a lambda + range that
  /// specifies global size only.
  ///
  /// \param Range specifies the global work space of the kernel
  /// \param Rest acts as-if: "ReductionTypes&&... Reductions,
  /// const KernelType &KernelFunc".
  template <typename KernelName = detail::auto_name, typename... RestT>
  event parallel_for(range<2> Range, RestT &&...Rest) {
    return parallel_for_impl<KernelName>(Range, Rest...);
  }

  /// parallel_for version with a kernel represented as a lambda + range that
  /// specifies global size only.
  ///
  /// \param Range specifies the global work space of the kernel
  /// \param Rest acts as-if: "ReductionTypes&&... Reductions,
  /// const KernelType &KernelFunc".
  template <typename KernelName = detail::auto_name, typename... RestT>
  event parallel_for(range<3> Range, RestT &&...Rest) {
    return parallel_for_impl<KernelName>(Range, Rest...);
  }

  /// parallel_for version with a kernel represented as a lambda + range that
  /// specifies global size only.
  ///
  /// \param Range specifies the global work space of the kernel
  /// \param DepEvent is an event that specifies the kernel dependencies
  /// \param Rest acts as-if: "ReductionTypes&&... Reductions,
  /// const KernelType &KernelFunc".
  template <typename KernelName = detail::auto_name, typename... RestT>
  event parallel_for(range<1> Range, event DepEvent, RestT &&...Rest) {
    return parallel_for_impl<KernelName>(Range, DepEvent, Rest...);
  }

  /// parallel_for version with a kernel represented as a lambda + range that
  /// specifies global size only.
  ///
  /// \param Range specifies the global work space of the kernel
  /// \param DepEvent is an event that specifies the kernel dependencies
  /// \param Rest acts as-if: "ReductionTypes&&... Reductions,
  /// const KernelType &KernelFunc".
  template <typename KernelName = detail::auto_name, typename... RestT>
  event parallel_for(range<2> Range, event DepEvent, RestT &&...Rest) {
    return parallel_for_impl<KernelName>(Range, DepEvent, Rest...);
  }

  /// parallel_for version with a kernel represented as a lambda + range that
  /// specifies global size only.
  ///
  /// \param Range specifies the global work space of the kernel
  /// \param DepEvent is an event that specifies the kernel dependencies
  /// \param Rest acts as-if: "ReductionTypes&&... Reductions,
  /// const KernelType &KernelFunc".
  template <typename KernelName = detail::auto_name, typename... RestT>
  event parallel_for(range<3> Range, event DepEvent, RestT &&...Rest) {
    return parallel_for_impl<KernelName>(Range, DepEvent, Rest...);
  }

  /// parallel_for version with a kernel represented as a lambda + range that
  /// specifies global size only.
  ///
  /// \param Range specifies the global work space of the kernel
  /// \param DepEvents is a vector of events that specifies the kernel
  /// dependencies
  /// \param Rest acts as-if: "ReductionTypes&&... Reductions,
  /// const KernelType &KernelFunc".
  template <typename KernelName = detail::auto_name, typename... RestT>
  event parallel_for(range<1> Range, const std::vector<event> &DepEvents,
                     RestT &&...Rest) {
    return parallel_for_impl<KernelName>(Range, DepEvents, Rest...);
  }

  /// parallel_for version with a kernel represented as a lambda + range that
  /// specifies global size only.
  ///
  /// \param Range specifies the global work space of the kernel
  /// \param DepEvents is a vector of events that specifies the kernel
  /// dependencies
  /// \param Rest acts as-if: "ReductionTypes&&... Reductions,
  /// const KernelType &KernelFunc".
  template <typename KernelName = detail::auto_name, typename... RestT>
  event parallel_for(range<2> Range, const std::vector<event> &DepEvents,
                     RestT &&...Rest) {
    return parallel_for_impl<KernelName>(Range, DepEvents, Rest...);
  }

  /// parallel_for version with a kernel represented as a lambda + range that
  /// specifies global size only.
  ///
  /// \param Range specifies the global work space of the kernel
  /// \param DepEvents is a vector of events that specifies the kernel
  /// dependencies
  /// \param Rest acts as-if: "ReductionTypes&&... Reductions,
  /// const KernelType &KernelFunc".
  template <typename KernelName = detail::auto_name, typename... RestT>
  event parallel_for(range<3> Range, const std::vector<event> &DepEvents,
                     RestT &&...Rest) {
    return parallel_for_impl<KernelName>(Range, DepEvents, Rest...);
  }

  // While other shortcuts with offsets are able to go through parallel_for(...,
  // RestT &&...Rest), those that accept dependency events vector have to be
  // overloaded to allow implicit construction from an init-list.
  /// parallel_for version with a kernel represented as a lambda + range and
  /// offset that specify global size and global offset correspondingly.
  ///
  /// \param Range specifies the global work space of the kernel
  /// \param WorkItemOffset specifies the offset for each work item id
  /// \param KernelFunc is the Kernel functor or lambda
  /// \param CodeLoc contains the code location of user code
  template <typename KernelName = detail::auto_name, typename KernelType,
            int Dim>
  event parallel_for(range<Dim> Range, id<Dim> WorkItemOffset,
                     const std::vector<event> &DepEvents,
                     _KERNELFUNCPARAM(KernelFunc)) {
    static_assert(1 <= Dim && Dim <= 3, "Invalid number of dimensions");
    return parallel_for_impl<KernelName>(Range, WorkItemOffset, DepEvents,
                                         KernelFunc);
  }

  /// parallel_for version with a kernel represented as a lambda + range and
  /// offset that specify global size and global offset correspondingly.
  ///
  /// \param Range specifies the global work space of the kernel
  /// \param WorkItemOffset specifies the offset for each work item id
  /// \param KernelFunc is the Kernel functor or lambda
  /// \param CodeLoc contains the code location of user code
  template <typename KernelName = detail::auto_name, typename KernelType,
            int Dims>
  __SYCL2020_DEPRECATED("offsets are deprecated in SYCL 2020")
  event parallel_for_impl(range<Dims> Range, id<Dims> WorkItemOffset,
                          _KERNELFUNCPARAM(KernelFunc)) {
    // Actual code location needs to be captured from KernelInfo object.
    const detail::code_location CodeLoc = {};
    return submit(
        [&](handler &CGH) {
          CGH.template parallel_for<KernelName>(Range, WorkItemOffset,
                                                KernelFunc);
        },
        CodeLoc);
  }

  /// parallel_for version with a kernel represented as a lambda + range and
  /// offset that specify global size and global offset correspondingly.
  ///
  /// \param Range specifies the global work space of the kernel
  /// \param WorkItemOffset specifies the offset for each work item id
  /// \param DepEvent is an event that specifies the kernel dependencies
  /// \param KernelFunc is the Kernel functor or lambda
  /// \param CodeLoc contains the code location of user code
  template <typename KernelName = detail::auto_name, typename KernelType,
            int Dims>
  __SYCL2020_DEPRECATED("offsets are deprecated in SYCL 2020")
  event parallel_for_impl(range<Dims> Range, id<Dims> WorkItemOffset,
                          event DepEvent, _KERNELFUNCPARAM(KernelFunc)) {
    // Actual code location needs to be captured from KernelInfo object.
    const detail::code_location CodeLoc = {};
    return submit(
        [&](handler &CGH) {
          CGH.depends_on(DepEvent);
          CGH.template parallel_for<KernelName>(Range, WorkItemOffset,
                                                KernelFunc);
        },
        CodeLoc);
  }

  /// parallel_for version with a kernel represented as a lambda + range and
  /// offset that specify global size and global offset correspondingly.
  ///
  /// \param Range specifies the global work space of the kernel
  /// \param WorkItemOffset specifies the offset for each work item id
  /// \param DepEvents is a vector of events that specifies the kernel
  /// dependencies
  /// \param KernelFunc is the Kernel functor or lambda
  /// \param CodeLoc contains the code location of user code
  template <typename KernelName = detail::auto_name, typename KernelType,
            int Dims>
  __SYCL2020_DEPRECATED("offsets are deprecated in SYCL 2020")
  event parallel_for_impl(range<Dims> Range, id<Dims> WorkItemOffset,
                          const std::vector<event> &DepEvents,
                          _KERNELFUNCPARAM(KernelFunc)) {
    // Actual code location needs to be captured from KernelInfo object.
    const detail::code_location CodeLoc = {};
    return submit(
        [&](handler &CGH) {
          CGH.depends_on(DepEvents);
          CGH.template parallel_for<KernelName>(Range, WorkItemOffset,
                                                KernelFunc);
        },
        CodeLoc);
  }

  /// parallel_for version with a kernel represented as a lambda + nd_range that
  /// specifies global, local sizes and offset.
  ///
  /// \param Range specifies the global and local work spaces of the kernel
  /// \param Properties is the kernel properties.
  /// \param Rest acts as-if: "ReductionTypes&&... Reductions,
  /// const KernelType &KernelFunc".
  template <typename KernelName = detail::auto_name, int Dims,
            typename PropertiesT, typename... RestT>
  std::enable_if_t<
      detail::AreAllButLastReductions<RestT...>::value &&
          ext::oneapi::experimental::is_property_list<PropertiesT>::value,
      event>
  parallel_for(nd_range<Dims> Range, PropertiesT Properties, RestT &&...Rest) {
    using KI = sycl::detail::KernelInfo<KernelName>;
    constexpr detail::code_location CodeLoc(
        KI::getFileName(), KI::getFunctionName(), KI::getLineNumber(),
        KI::getColumnNumber());
    detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
    return submit(
        [&](handler &CGH) {
          CGH.template parallel_for<KernelName>(Range, Properties, Rest...);
        },
        CodeLoc);
  }

  /// parallel_for version with a kernel represented as a lambda + nd_range that
  /// specifies global, local sizes and offset.
  ///
  /// \param Range specifies the global and local work spaces of the kernel
  /// \param Rest acts as-if: "ReductionTypes&&... Reductions,
  /// const KernelType &KernelFunc".
  template <typename KernelName = detail::auto_name, int Dims,
            typename... RestT>
  std::enable_if_t<detail::AreAllButLastReductions<RestT...>::value, event>
  parallel_for(nd_range<Dims> Range, RestT &&...Rest) {
    return parallel_for<KernelName>(
        Range, ext::oneapi::experimental::empty_properties_t{}, Rest...);
  }

  /// parallel_for version with a kernel represented as a lambda + nd_range that
  /// specifies global, local sizes and offset.
  ///
  /// \param Range specifies the global and local work spaces of the kernel
  /// \param DepEvent is an event that specifies the kernel dependencies
  /// \param Rest acts as-if: "ReductionTypes&&... Reductions,
  /// const KernelType &KernelFunc".
  template <typename KernelName = detail::auto_name, int Dims,
            typename... RestT>
  event parallel_for(nd_range<Dims> Range, event DepEvent, RestT &&...Rest) {
    using KI = sycl::detail::KernelInfo<KernelName>;
    constexpr detail::code_location CodeLoc(
        KI::getFileName(), KI::getFunctionName(), KI::getLineNumber(),
        KI::getColumnNumber());
    detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
    return submit(
        [&](handler &CGH) {
          CGH.depends_on(DepEvent);
          CGH.template parallel_for<KernelName>(Range, Rest...);
        },
        CodeLoc);
  }

  /// parallel_for version with a kernel represented as a lambda + nd_range that
  /// specifies global, local sizes and offset.
  ///
  /// \param Range specifies the global and local work spaces of the kernel
  /// \param DepEvents is a vector of events that specifies the kernel
  /// dependencies
  /// \param Rest acts as-if: "ReductionTypes&&... Reductions,
  /// const KernelType &KernelFunc".
  template <typename KernelName = detail::auto_name, int Dims,
            typename... RestT>
  event parallel_for(nd_range<Dims> Range, const std::vector<event> &DepEvents,
                     RestT &&...Rest) {
    using KI = sycl::detail::KernelInfo<KernelName>;
    constexpr detail::code_location CodeLoc(
        KI::getFileName(), KI::getFunctionName(), KI::getLineNumber(),
        KI::getColumnNumber());
    detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
    return submit(
        [&](handler &CGH) {
          CGH.depends_on(DepEvents);
          CGH.template parallel_for<KernelName>(Range, Rest...);
        },
        CodeLoc);
  }

  /// Copies data from a memory region pointed to by a placeholder accessor to
  /// another memory region pointed to by a shared_ptr.
  ///
  /// \param Src is a placeholder accessor to the source memory.
  /// \param Dest is a shared_ptr to the destination memory.
  /// \return an event representing copy operation.
  template <typename SrcT, int SrcDims, access_mode SrcMode, target SrcTgt,
            access::placeholder IsPlaceholder, typename DestT>
  event copy(
      accessor<SrcT, SrcDims, SrcMode, SrcTgt, IsPlaceholder> Src,
      std::shared_ptr<DestT> Dest,
      const detail::code_location &CodeLoc = detail::code_location::current()) {
    return submit(
        [&](handler &CGH) {
          CGH.require(Src);
          CGH.copy(Src, Dest);
        },
        CodeLoc);
  }

  /// Copies data from a memory region pointed to by a shared_ptr to another
  /// memory region pointed to by a placeholder accessor.
  ///
  /// \param Src is a shared_ptr to the source memory.
  /// \param Dest is a placeholder accessor to the destination memory.
  /// \return an event representing copy operation.
  template <typename SrcT, typename DestT, int DestDims, access_mode DestMode,
            target DestTgt, access::placeholder IsPlaceholder>
  event copy(
      std::shared_ptr<SrcT> Src,
      accessor<DestT, DestDims, DestMode, DestTgt, IsPlaceholder> Dest,
      const detail::code_location &CodeLoc = detail::code_location::current()) {
    return submit(
        [&](handler &CGH) {
          CGH.require(Dest);
          CGH.copy(Src, Dest);
        },
        CodeLoc);
  }

  /// Copies data from a memory region pointed to by a placeholder accessor to
  /// another memory region pointed to by a raw pointer.
  ///
  /// \param Src is a placeholder accessor to the source memory.
  /// \param Dest is a raw pointer to the destination memory.
  /// \return an event representing copy operation.
  template <typename SrcT, int SrcDims, access_mode SrcMode, target SrcTgt,
            access::placeholder IsPlaceholder, typename DestT>
  event copy(
      accessor<SrcT, SrcDims, SrcMode, SrcTgt, IsPlaceholder> Src, DestT *Dest,
      const detail::code_location &CodeLoc = detail::code_location::current()) {
    return submit(
        [&](handler &CGH) {
          CGH.require(Src);
          CGH.copy(Src, Dest);
        },
        CodeLoc);
  }

  /// Copies data from a memory region pointed to by a raw pointer to another
  /// memory region pointed to by a placeholder accessor.
  ///
  /// \param Src is a raw pointer to the source memory.
  /// \param Dest is a placeholder accessor to the destination memory.
  /// \return an event representing copy operation.
  template <typename SrcT, typename DestT, int DestDims, access_mode DestMode,
            target DestTgt, access::placeholder IsPlaceholder>
  event copy(
      const SrcT *Src,
      accessor<DestT, DestDims, DestMode, DestTgt, IsPlaceholder> Dest,
      const detail::code_location &CodeLoc = detail::code_location::current()) {
    return submit(
        [&](handler &CGH) {
          CGH.require(Dest);
          CGH.copy(Src, Dest);
        },
        CodeLoc);
  }

  /// Copies data from one memory region to another, both pointed by placeholder
  /// accessors.
  ///
  /// \param Src is a placeholder accessor to the source memory.
  /// \param Dest is a placeholder accessor to the destination memory.
  /// \return an event representing copy operation.
  template <typename SrcT, int SrcDims, access_mode SrcMode, target SrcTgt,
            access::placeholder IsSrcPlaceholder, typename DestT, int DestDims,
            access_mode DestMode, target DestTgt,
            access::placeholder IsDestPlaceholder>
  event copy(
      accessor<SrcT, SrcDims, SrcMode, SrcTgt, IsSrcPlaceholder> Src,
      accessor<DestT, DestDims, DestMode, DestTgt, IsDestPlaceholder> Dest,
      const detail::code_location &CodeLoc = detail::code_location::current()) {
    return submit(
        [&](handler &CGH) {
          CGH.require(Src);
          CGH.require(Dest);
          CGH.copy(Src, Dest);
        },
        CodeLoc);
  }

  /// Provides guarantees that the memory object accessed via Acc is updated
  /// on the host after operation is complete.
  ///
  /// \param Acc is a SYCL accessor that needs to be updated on host.
  /// \return an event representing update_host operation.
  template <typename T, int Dims, access_mode Mode, target Tgt,
            access::placeholder IsPlaceholder>
  event update_host(
      accessor<T, Dims, Mode, Tgt, IsPlaceholder> Acc,
      const detail::code_location &CodeLoc = detail::code_location::current()) {
    return submit(
        [&](handler &CGH) {
          CGH.require(Acc);
          CGH.update_host(Acc);
        },
        CodeLoc);
  }

  /// Fills the specified memory with the specified data.
  ///
  /// \param Dest is the placeholder accessor to the memory to fill.
  /// \param Src is the data to fill the memory with. T should be
  /// trivially copyable.
  /// \return an event representing fill operation.
  template <typename T, int Dims, access_mode Mode, target Tgt,
            access::placeholder IsPlaceholder>
  event fill(
      accessor<T, Dims, Mode, Tgt, IsPlaceholder> Dest, const T &Src,
      const detail::code_location &CodeLoc = detail::code_location::current()) {
    return submit(
        [&](handler &CGH) {
          CGH.require(Dest);
          CGH.fill<T>(Dest, Src);
        },
        CodeLoc);
  }

  /// @brief Returns true if the queue was created with the
  /// ext::codeplay::experimental::property::queue::enable_fusion property.
  ///
  /// Equivalent to
  /// `has_property<ext::codeplay::experimental::property::queue::enable_fusion>()`.
  ///
  bool ext_codeplay_supports_fusion() const;

// Clean KERNELFUNC macros.
#undef _KERNELFUNCPARAM

  /// Shortcut for executing a graph of commands.
  ///
  /// \param Graph the graph of commands to execute
  /// \return an event representing graph execution operation.
  event ext_oneapi_graph(
      ext::oneapi::experimental::command_graph<
          ext::oneapi::experimental::graph_state::executable>
          Graph,
      const detail::code_location &CodeLoc = detail::code_location::current()) {
    return submit([&](handler &CGH) { CGH.ext_oneapi_graph(Graph); }, CodeLoc);
  }

  /// Shortcut for executing a graph of commands with a single dependency.
  ///
  /// \param Graph the graph of commands to execute
  /// \param DepEvent is an event that specifies the graph execution
  /// dependencies.
  /// \return an event representing graph execution operation.
  event ext_oneapi_graph(
      ext::oneapi::experimental::command_graph<
          ext::oneapi::experimental::graph_state::executable>
          Graph,
      event DepEvent,
      const detail::code_location &CodeLoc = detail::code_location::current()) {
    return submit(
        [&](handler &CGH) {
          CGH.depends_on(DepEvent);
          CGH.ext_oneapi_graph(Graph);
        },
        CodeLoc);
  }

  /// Shortcut for executing a graph of commands with multiple dependencies.
  ///
  /// \param Graph the graph of commands to execute
  /// \param DepEvents is a vector of events that specifies the graph
  /// execution dependencies.
  /// \return an event representing graph execution operation.
  event ext_oneapi_graph(
      ext::oneapi::experimental::command_graph<
          ext::oneapi::experimental::graph_state::executable>
          Graph,
      const std::vector<event> &DepEvents,
      const detail::code_location &CodeLoc = detail::code_location::current()) {
    return submit(
        [&](handler &CGH) {
          CGH.depends_on(DepEvents);
          CGH.ext_oneapi_graph(Graph);
        },
        CodeLoc);
  }

  /// Returns whether the queue is in order or OoO
  ///
  /// Equivalent to has_property<property::queue::in_order>()
  bool is_in_order() const;

  /// Returns the backend associated with this queue.
  ///
  /// \return the backend associated with this queue.
  backend get_backend() const noexcept;

  /// Allows to check status of the queue (completed vs noncompleted).
  ///
  /// \return returns true if all enqueued commands in the queue have been
  /// completed, otherwise returns false.
  bool ext_oneapi_empty() const;

  pi_native_handle getNative(int32_t &NativeHandleDesc) const;

  event ext_oneapi_get_last_event() const;

  void ext_oneapi_set_external_event(const event &external_event);

private:
  std::shared_ptr<detail::queue_impl> impl;
  queue(std::shared_ptr<detail::queue_impl> impl) : impl(impl) {}

  template <class Obj>
  friend decltype(Obj::impl) detail::getSyclObjImpl(const Obj &SyclObject);
  template <class T>
  friend T detail::createSyclObjFromImpl(decltype(T::impl) ImplObj);

  template <backend BackendName, class SyclObjectT>
  friend auto get_native(const SyclObjectT &Obj)
      -> backend_return_t<BackendName, SyclObjectT>;

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
  /// TODO: move to impl class in the next ABI Breaking window
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
  /// \param Range specifies the global work space of the kernel
  /// \param Properties is the kernel properties.
  /// \param KernelFunc is the Kernel functor or lambda
  template <typename KernelName, int Dims, typename PropertiesT,
            typename... RestT>
  std::enable_if_t<
      detail::AreAllButLastReductions<RestT...>::value &&
          ext::oneapi::experimental::is_property_list<PropertiesT>::value,
      event>
  parallel_for_impl(range<Dims> Range, PropertiesT Properties,
                    RestT &&...Rest) {
    using KI = sycl::detail::KernelInfo<KernelName>;
    constexpr detail::code_location CodeLoc(
        KI::getFileName(), KI::getFunctionName(), KI::getLineNumber(),
        KI::getColumnNumber());
    detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
    return submit(
        [&](handler &CGH) {
          CGH.template parallel_for<KernelName>(Range, Properties, Rest...);
        },
        CodeLoc);
  }

  /// parallel_for_impl with a kernel represented as a lambda + range that
  /// specifies global size only.
  ///
  /// \param Range specifies the global work space of the kernel
  /// \param KernelFunc is the Kernel functor or lambda
  template <typename KernelName, int Dims, typename... RestT>
  std::enable_if_t<detail::AreAllButLastReductions<RestT...>::value, event>
  parallel_for_impl(range<Dims> Range, RestT &&...Rest) {
    return parallel_for_impl<KernelName>(
        Range, ext::oneapi::experimental::empty_properties_t{}, Rest...);
  }

  /// parallel_for_impl with a kernel represented as a lambda + range that
  /// specifies global size only.
  ///
  /// \param Range specifies the global work space of the kernel
  /// \param DepEvent is an event that specifies the kernel dependencies
  /// \param Properties is the kernel properties.
  /// \param KernelFunc is the Kernel functor or lambda
  template <typename KernelName, int Dims, typename PropertiesT,
            typename... RestT>
  std::enable_if_t<
      ext::oneapi::experimental::is_property_list<PropertiesT>::value, event>
  parallel_for_impl(range<Dims> Range, event DepEvent, PropertiesT Properties,
                    RestT &&...Rest) {
    using KI = sycl::detail::KernelInfo<KernelName>;
    constexpr detail::code_location CodeLoc(
        KI::getFileName(), KI::getFunctionName(), KI::getLineNumber(),
        KI::getColumnNumber());
    detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
    return submit(
        [&](handler &CGH) {
          CGH.depends_on(DepEvent);
          CGH.template parallel_for<KernelName>(Range, Properties, Rest...);
        },
        CodeLoc);
  }

  /// parallel_for_impl with a kernel represented as a lambda + range that
  /// specifies global size only.
  ///
  /// \param Range specifies the global work space of the kernel
  /// \param DepEvent is an event that specifies the kernel dependencies
  /// \param KernelFunc is the Kernel functor or lambda
  template <typename KernelName, int Dims, typename... RestT>
  event parallel_for_impl(range<Dims> Range, event DepEvent, RestT &&...Rest) {
    return parallel_for_impl<KernelName>(
        Range, DepEvent, ext::oneapi::experimental::empty_properties_t{},
        Rest...);
  }

  /// parallel_for_impl version with a kernel represented as a lambda + range
  /// that specifies global size only.
  ///
  /// \param Range specifies the global work space of the kernel
  /// \param DepEvents is a vector of events that specifies the kernel
  /// dependencies
  /// \param Properties is the kernel properties.
  /// \param KernelFunc is the Kernel functor or lambda
  template <typename KernelName, int Dims, typename PropertiesT,
            typename... RestT>
  std::enable_if_t<
      ext::oneapi::experimental::is_property_list<PropertiesT>::value, event>
  parallel_for_impl(range<Dims> Range, const std::vector<event> &DepEvents,
                    PropertiesT Properties, RestT &&...Rest) {
    using KI = sycl::detail::KernelInfo<KernelName>;
    constexpr detail::code_location CodeLoc(
        KI::getFileName(), KI::getFunctionName(), KI::getLineNumber(),
        KI::getColumnNumber());
    detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
    return submit(
        [&](handler &CGH) {
          CGH.depends_on(DepEvents);
          CGH.template parallel_for<KernelName>(Range, Properties, Rest...);
        },
        CodeLoc);
  }

  /// parallel_for_impl version with a kernel represented as a lambda + range
  /// that specifies global size only.
  ///
  /// \param Range specifies the global work space of the kernel
  /// \param DepEvents is a vector of events that specifies the kernel
  /// dependencies
  /// \param KernelFunc is the Kernel functor or lambda
  template <typename KernelName, int Dims, typename... RestT>
  event parallel_for_impl(range<Dims> Range,
                          const std::vector<event> &DepEvents,
                          RestT &&...Rest) {
    return parallel_for_impl<KernelName>(
        Range, DepEvents, ext::oneapi::experimental::empty_properties_t{},
        Rest...);
  }

#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
  buffer<detail::AssertHappened, 1> &getAssertHappenedBuffer();
#endif

  event memcpyToDeviceGlobal(void *DeviceGlobalPtr, const void *Src,
                             bool IsDeviceImageScope, size_t NumBytes,
                             size_t Offset,
                             const std::vector<event> &DepEvents);
  event memcpyFromDeviceGlobal(void *Dest, const void *DeviceGlobalPtr,
                               bool IsDeviceImageScope, size_t NumBytes,
                               size_t Offset,
                               const std::vector<event> &DepEvents);
};

} // namespace _V1
} // namespace sycl

namespace std {
template <> struct __SYCL_EXPORT hash<sycl::queue> {
  size_t operator()(const sycl::queue &Q) const;
};
} // namespace std

#if __SYCL_USE_FALLBACK_ASSERT
// Explicitly request format macros
#ifndef __STDC_FORMAT_MACROS
#define __STDC_FORMAT_MACROS 1
#endif
#include <cinttypes>

namespace sycl {
inline namespace _V1 {

namespace detail {
#define __SYCL_ASSERT_START 1

namespace __sycl_service_kernel__ {
class AssertInfoCopier;
} // namespace __sycl_service_kernel__

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
  buffer<detail::AssertHappened, 1> Buffer{1};

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
        throw sycl::exception(
            make_error_code(errc::invalid),
            "Internal Error. Invalid value in assert description.");
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
        fflush(stderr);
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
} // namespace detail

} // namespace _V1
} // namespace sycl
#endif // __SYCL_USE_FALLBACK_ASSERT

#undef __SYCL_USE_FALLBACK_ASSERT
