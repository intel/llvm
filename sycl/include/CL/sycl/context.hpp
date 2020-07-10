//==---------------- context.hpp - SYCL context ----------------------------==//
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
#include <CL/sycl/exception_list.hpp>
#include <CL/sycl/info/info_desc.hpp>
#include <CL/sycl/stl.hpp>

#include <type_traits>
// 4.6.2 Context class

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
// Forward declarations
class device;
class platform;
namespace detail {
class context_impl;
class buffer_impl;
}

/// The context class represents a SYCL context on which kernel functions may
/// be executed.
///
/// \ingroup sycl_api
class __SYCL_EXPORT context {
public:
  /// Constructs a SYCL context instance using an instance of default_selector.
  ///
  /// The instance of default_selector is used to select the associated platform
  /// and device(s).
  /// The constructed SYCL context will use the AsyncHandler parameter to handle
  /// exceptions.
  ///
  /// \param AsyncHandler is an instance of async_handler.
  /// \param UseCUDAPrimaryContext is a bool determining whether to use the
  ///        primary context in the CUDA backend.
  explicit context(const async_handler &AsyncHandler = {},
                   bool UseCUDAPrimaryContext = false);

  /// Constructs a SYCL context instance using the provided device.
  ///
  /// Newly created context is associated with the Device and the SYCL platform
  /// that is associated with the Device.
  /// The constructed SYCL context will use the AsyncHandler parameter to handle
  /// exceptions.
  ///
  /// \param Device is an instance of SYCL device.
  /// \param AsyncHandler is an instance of async_handler.
  /// \param UseCUDAPrimaryContext is a bool determining whether to use the
  ///        primary context in the CUDA backend.
  explicit context(const device &Device, async_handler AsyncHandler = {},
                   bool UseCUDAPrimaryContext = false);

  /// Constructs a SYCL context instance using the provided platform.
  ///
  /// Newly created context is associated with the Platform and with each
  /// SYCL device that is associated with the Platform.
  /// The constructed SYCL context will use the AsyncHandler parameter to handle
  /// exceptions.
  ///
  /// \param Platform is an instance of SYCL platform.
  /// \param AsyncHandler is an instance of async_handler.
  /// \param UseCUDAPrimaryContext is a bool determining whether to use the
  ///        primary context in the CUDA backend.
  explicit context(const platform &Platform, async_handler AsyncHandler = {},
                   bool UseCUDAPrimaryContext = false);

  /// Constructs a SYCL context instance using list of devices.
  ///
  /// Newly created context will be associated with each SYCL device in the
  /// DeviceList. This requires that all SYCL devices in the list have the same
  /// associated SYCL platform.
  /// The constructed SYCL context will use the AsyncHandler parameter to handle
  /// exceptions.
  ///
  /// \param DeviceList is a list of SYCL device instances.
  /// \param AsyncHandler is an instance of async_handler.
  /// \param UseCUDAPrimaryContext is a bool determining whether to use the
  ///        primary context in the CUDA backend.
  explicit context(const vector_class<device> &DeviceList,
                   async_handler AsyncHandler = {},
                   bool UseCUDAPrimaryContext = false);

  /// Constructs a SYCL context instance from OpenCL cl_context.
  ///
  /// ClContext is retained on SYCL context instantiation.
  /// The constructed SYCL context will use the AsyncHandler parameter to handle
  /// exceptions.
  ///
  /// \param ClContext is an instance of OpenCL cl_context.
  /// \param AsyncHandler is an instance of async_handler.
  context(cl_context ClContext, async_handler AsyncHandler = {});

  /// Queries this SYCL context for information.
  ///
  /// The return type depends on information being queried.
  template <info::context param>
  typename info::param_traits<info::context, param>::return_type
  get_info() const;

  context(const context &rhs) = default;

  context(context &&rhs) = default;

  context &operator=(const context &rhs) = default;

  context &operator=(context &&rhs) = default;

  bool operator==(const context &rhs) const { return impl == rhs.impl; }

  bool operator!=(const context &rhs) const { return !(*this == rhs); }

  /// Gets OpenCL interoperability context.
  ///
  /// The OpenCL cl_context handle is retained on return.
  ///
  /// \return a valid instance of OpenCL cl_context.
  cl_context get() const;

  /// Checks if this context is a SYCL host context.
  ///
  /// \return true if this context is a SYCL host context.
  bool is_host() const;

  /// Gets platform associated with this SYCL context.
  ///
  /// \return a valid instance of SYCL platform.
  platform get_platform() const;

  /// Gets devices associated with this SYCL context.
  ///
  /// \return a vector of valid SYCL device instances.
  vector_class<device> get_devices() const;

  /// Gets the native handle of the SYCL context.
  ///
  /// \return a native handle, the type of which defined by the backend.
  template <backend BackendName>
  auto get_native() const -> typename interop<BackendName, context>::type {
    return reinterpret_cast<typename interop<BackendName, context>::type>(
        getNative());
  }

private:
  /// Constructs a SYCL context object from a valid context_impl instance.
  context(shared_ptr_class<detail::context_impl> Impl);

  pi_native_handle getNative() const;

  shared_ptr_class<detail::context_impl> impl;
  template <class Obj>
  friend decltype(Obj::impl) detail::getSyclObjImpl(const Obj &SyclObject);

  template <class T>
  friend
      typename std::add_pointer<typename decltype(T::impl)::element_type>::type
      detail::getRawSyclObjImpl(const T &SyclObject);

  template <class T>
  friend T detail::createSyclObjFromImpl(decltype(T::impl) ImplObj);

  friend class detail::buffer_impl;
};

} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)

namespace std {
template <> struct hash<cl::sycl::context> {
  size_t operator()(const cl::sycl::context &Context) const {
    return hash<cl::sycl::shared_ptr_class<cl::sycl::detail::context_impl>>()(
        cl::sycl::detail::getSyclObjImpl(Context));
  }
};
} // namespace std
