//==---------------- event.hpp --- SYCL event ------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/backend_types.hpp>
#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/export.hpp>
#include <CL/sycl/info/info_desc.hpp>
#include <CL/sycl/stl.hpp>

#include <memory>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
// Forward declaration
class context;
namespace detail {
class event_impl;
}

/// An event object can be used to synchronize memory transfers, enqueues of
/// kernels and signaling barriers.
///
/// \ingroup sycl_api
class __SYCL_EXPORT event {
public:
  /// Constructs a ready SYCL event.
  ///
  /// If the constructed SYCL event is waited on it will complete immediately.
  event();

  /// Constructs a SYCL event instance from an OpenCL cl_event.
  ///
  /// The SyclContext must match the OpenCL context associated with the ClEvent.
  ///
  /// \param ClEvent is a valid instance of OpenCL cl_event.
  /// \param SyclContext is an instance of SYCL context.
  __SYCL2020_DEPRECATED("OpenCL interop APIs are deprecated")
  event(cl_event ClEvent, const context &SyclContext);

  event(const event &rhs) = default;

  event(event &&rhs) = default;

  event &operator=(const event &rhs) = default;

  event &operator=(event &&rhs) = default;

  bool operator==(const event &rhs) const;

  bool operator!=(const event &rhs) const;

  /// Returns a valid OpenCL event interoperability handle.
  ///
  /// \return a valid instance of OpenCL cl_event.
  __SYCL2020_DEPRECATED("OpenCL interop APIs are deprecated")
  cl_event get() const;

  /// Checks if this event is a SYCL host event.
  ///
  /// \return true if this event is a SYCL host event.
  bool is_host() const;

  /// Return the list of events that this event waits for.
  ///
  /// Only direct dependencies are returned. Already completed events are not
  /// included in the returned vector.
  ///
  /// \return a vector of SYCL events.
  vector_class<event> get_wait_list();

  /// Wait for the event.
  void wait();

  /// Synchronously wait on a list of events.
  ///
  /// \param EventList is a vector of SYCL events.
  static void wait(const vector_class<event> &EventList);

  /// Wait for the event.
  ///
  /// If any uncaught asynchronous errors occurred on the context that the event
  /// is waiting on executions from, then call that context's asynchronous error
  /// handler with those errors.
  void wait_and_throw();

  /// Synchronously wait on a list of events.
  ///
  /// If any uncaught asynchronous errors occurred on the context that the
  /// events are waiting on executions from, then call those contexts'
  /// asynchronous error handlers with those errors.
  ///
  /// \param EventList is a vector of SYCL events.
  static void wait_and_throw(const vector_class<event> &EventList);

  /// Queries this SYCL event for information.
  ///
  /// \return depends on the information being requested.
  template <info::event param>
  typename info::param_traits<info::event, param>::return_type get_info() const;

  /// Queries this SYCL event for profiling information.
  ///
  /// If the requested info is not available when this member function is called
  /// due to incompletion of command groups associated with the event, then the
  /// call to this member function will block until the requested info is
  /// available. If the queue which submitted the command group this event is
  /// associated with was not constructed with the
  /// property::queue::enable_profiling property, an invalid_object_error SYCL
  /// exception is thrown.
  ///
  /// \return depends on template parameter.
  template <info::event_profiling param>
  typename info::param_traits<info::event_profiling, param>::return_type
  get_profiling_info() const;

  /// Returns the backend associated with this platform.
  ///
  /// \return the backend associated with this platform
  backend get_backend() const noexcept;

  /// Gets the native handle of the SYCL event.
  ///
  /// \return a native handle, the type of which defined by the backend.
  template <backend BackendName>
  auto get_native() const -> typename interop<BackendName, event>::type {
    return reinterpret_cast<typename interop<BackendName, event>::type>(
        getNative());
  }

private:
  event(shared_ptr_class<detail::event_impl> EventImpl);

  pi_native_handle getNative() const;

  shared_ptr_class<detail::event_impl> impl;

  template <class Obj>
  friend decltype(Obj::impl) detail::getSyclObjImpl(const Obj &SyclObject);

  template <class T>
  friend T detail::createSyclObjFromImpl(decltype(T::impl) ImplObj);
};

} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)

namespace std {
template <> struct hash<cl::sycl::event> {
  size_t operator()(const cl::sycl::event &e) const {
    return hash<cl::sycl::shared_ptr_class<cl::sycl::detail::event_impl>>()(
        cl::sycl::detail::getSyclObjImpl(e));
  }
};
} // namespace std
