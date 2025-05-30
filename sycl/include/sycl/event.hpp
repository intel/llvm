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
#include <ur_api.h>                           // for ur_native_handle_t

#ifdef __SYCL_INTERNAL_API
#include <sycl/detail/cl.h>
#endif

#include <cstddef> // for size_t
#include <memory>  // for shared_ptr, hash
#include <variant> // for hash
#include <vector>  // for vector

#ifdef _WIN32
#include <intrin.h>
#endif
// also defined in event_imp.hpp. probably need to move it elsewhere
//#define CP_LOG_EVENT_LIFECYCLE 1

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

#ifdef CP_LOG_EVENT_LIFECYCLE
	// Copy Constructor          // event(const event &rhs) = default;
    event(const event &rhs) : impl(rhs.impl) { // Calls std::shared_ptr's copy constructor
        std::cout << "EVENT: Copy Constructor (of " << this << ") from " << &rhs << " - new impl: " << impl.get() << " (use_count: " << impl.use_count() << ")" << std::endl;
		__debugbreak();
    }

    // Move Constructor          // event(event &&rhs) = default;
    event(event &&rhs) noexcept : impl(std::move(rhs.impl)) { // Calls std::shared_ptr's move constructor
        std::cout << "EVENT: Move Constructor (of " << this << ") from " << &rhs << " - new impl: " << impl.get() << " (use_count: " << impl.use_count() << ")" << std::endl;
		__debugbreak();
    }

    // Copy Assignment Operator  //event &operator=(const event &rhs) = default;
    event &operator=(const event &rhs) {
        if (this != &rhs) { 
            impl = rhs.impl; // Calls std::shared_ptr's copy assignment operator
        }
        std::cout << "EVENT: Copy Assignment (of " << this << ") from " << &rhs << " - new impl: " << impl.get() << " (use_count: " << impl.use_count() << ")" << std::endl;
        __debugbreak();
		return *this;
    }

    // Move Assignment Operator // event &operator=(event &&rhs) = default;
    event &operator=(event &&rhs) noexcept {
        if (this != &rhs) { 
            impl = std::move(rhs.impl); // Calls std::shared_ptr's move assignment operator
        }
        std::cout << "EVENT: Move Assignment (of " << this << ") from " << &rhs << " - new impl: " << impl.get() << " (use_count: " << impl.use_count() << ")" << std::endl;
        __debugbreak();
		return *this;
    }

    // Destructor
    ~event() {
        std::cout << "EVENT: Destructor (of " << this << ") - impl: " << impl.get() << " (use_count: " << impl.use_count() << ")" << std::endl;
    }

#else
  event(const event &rhs) = default;

  event(event &&rhs) = default;

  event &operator=(const event &rhs) = default;

  event &operator=(event &&rhs) = default;
  
  ~event() = default; // CP
#endif

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
  template <typename Param
#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
#if defined(_GLIBCXX_USE_CXX11_ABI) && _GLIBCXX_USE_CXX11_ABI == 0
            ,
            int = detail::emit_get_backend_info_error<event, Param>()
#endif
#endif
            >
#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
  __SYCL_DEPRECATED(
      "All current implementations of get_backend_info() are to be removed. "
      "Use respective variants of get_info() instead.")
#endif
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

  ur_native_handle_t getNative() const;

  std::vector<ur_native_handle_t> getNativeVector() const;

  std::shared_ptr<detail::event_impl> impl;

  template <class Obj>
  friend const decltype(Obj::impl) &
  detail::getSyclObjImpl(const Obj &SyclObject);

  template <class T>
  friend T detail::createSyclObjFromImpl(
      std::add_rvalue_reference_t<decltype(T::impl)> ImplObj);
  template <class T>
  friend T detail::createSyclObjFromImpl(
      std::add_lvalue_reference_t<const decltype(T::impl)> ImplObj);

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
