//==---------------- event.hpp --- SYCL event ------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/backend_types.hpp>             // for backend, backend_return_t
#include <sycl/detail/defines_elementary.hpp> // for __SYCL2020_DEPRECATED
#include <sycl/detail/export.hpp>             // for __SYCL_EXPORT
#include <sycl/detail/info_desc_helpers.hpp>  // for is_event_info_desc, is_...
#include <sycl/detail/owner_less_base.hpp>    // for OwnerLessBase
#include <sycl/detail/pi.h>                   // for pi_native_handle

#ifdef __SYCL_INTERNAL_API
#include <sycl/detail/cl.h>
#endif

#include <cstddef> // for size_t
#include <memory>  // for shared_ptr, hash
#include <variant> // for hash
#include <vector>  // for vector

namespace sycl {
inline namespace _V1 {
// Forward declaration
class context;

template <backend BackendName, class SyclObjectT>
auto get_native(const SyclObjectT &Obj)
    -> backend_return_t<BackendName, SyclObjectT>;

namespace detail {
class event_impl;
}

/// An event object can be used to synchronize memory transfers, enqueues of
/// kernels and signaling barriers.
///
/// \ingroup sycl_api
class __SYCL_EXPORT event : public detail::OwnerLessBase<event> {
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
#ifdef __SYCL_INTERNAL_API
  event(cl_event ClEvent, const context &SyclContext);
#endif

  event(const event &rhs) = default;

  event(event &&rhs) = default;

  event &operator=(const event &rhs) = default;

  event &operator=(event &&rhs) = default;

  bool operator==(const event &rhs) const;

  bool operator!=(const event &rhs) const;

  /// Return the list of events that this event waits for.
  ///
  /// Only direct dependencies are returned. Already completed events are not
  /// included in the returned vector.
  ///
  /// \return a vector of SYCL events.
  std::vector<event> get_wait_list();

  /// Wait for the event.
  void wait();

  /// Synchronously wait on a list of events.
  ///
  /// \param EventList is a vector of SYCL events.
  static void wait(const std::vector<event> &EventList);

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
  static void wait_and_throw(const std::vector<event> &EventList);

  /// Queries this SYCL event for information.
  ///
  /// \return depends on the information being requested.
  template <typename Param>
  typename detail::is_event_info_desc<Param>::return_type get_info() const;

  /// Queries this SYCL event for SYCL backend-specific information.
  ///
  /// \return depends on information being queried.
  template <typename Param>
  typename detail::is_backend_info_desc<Param>::return_type
  get_backend_info() const;

  /// Queries this SYCL event for profiling information.
  ///
  /// If the requested info is not available when this member function is called
  /// due to incompletion of command groups associated with the event, then the
  /// call to this member function will block until the requested info is
  /// available. If the queue which submitted the command group this event is
  /// associated with was not constructed with the
  /// property::queue::enable_profiling property, an a SYCL exception with
  /// errc::invalid error code is thrown.
  ///
  /// \return depends on template parameter.
  template <typename Param>
  typename detail::is_event_profiling_info_desc<Param>::return_type
  get_profiling_info() const;

  /// Returns the backend associated with this platform.
  ///
  /// \return the backend associated with this platform
  backend get_backend() const noexcept;

private:
  event(std::shared_ptr<detail::event_impl> EventImpl);

  pi_native_handle getNative() const;

  std::vector<pi_native_handle> getNativeVector() const;

  std::shared_ptr<detail::event_impl> impl;

  template <class Obj>
  friend const decltype(Obj::impl) &
  detail::getSyclObjImpl(const Obj &SyclObject);

  template <class T>
  friend T detail::createSyclObjFromImpl(decltype(T::impl) ImplObj);

  template <backend BackendName, class SyclObjectT>
  friend auto get_native(const SyclObjectT &Obj)
      -> backend_return_t<BackendName, SyclObjectT>;
};

} // namespace _V1
} // namespace sycl

namespace std {
template <> struct hash<sycl::event> {
  size_t operator()(const sycl::event &e) const {
    return hash<std::shared_ptr<sycl::detail::event_impl>>()(
        sycl::detail::getSyclObjImpl(e));
  }
};
} // namespace std
