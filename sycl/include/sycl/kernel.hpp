//==--------------- kernel.hpp --- SYCL kernel -----------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/cl.h>
#include <sycl/detail/common.hpp>
#include <sycl/detail/export.hpp>
#include <sycl/detail/info_desc_helpers.hpp>
#include <sycl/detail/pi.h>
#include <sycl/info/info_desc.hpp>
#include <sycl/kernel_bundle_enums.hpp>
#include <sycl/stl.hpp>

#include <memory>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
// Forward declaration
class context;
template <backend Backend> class backend_traits;
template <bundle_state State> class kernel_bundle;
template <backend BackendName, class SyclObjectT>
auto get_native(const SyclObjectT &Obj)
    -> backend_return_t<BackendName, SyclObjectT>;

namespace detail {
class kernel_impl;

/// This class is the default KernelName template parameter type for kernel
/// invocation APIs such as single_task.
class auto_name {};

/// Helper struct to get a kernel name type based on given \c Name and \c Type
/// types: if \c Name is undefined (is a \c auto_name) then \c Type becomes
/// the \c Name.
template <typename Name, typename Type> struct get_kernel_name_t {
  using name = Name;
  static_assert(
      !std::is_same<Name, auto_name>::value,
      "No kernel name provided without -fsycl-unnamed-lambda enabled!");
};

#ifdef __SYCL_UNNAMED_LAMBDA__
/// Specialization for the case when \c Name is undefined.
/// This is only legal with our compiler with the unnamed lambda
/// extension, so make sure the specialiation isn't available in that case: the
/// lack of specialization allows us to trigger static_assert from the primary
/// definition.
template <typename Type> struct get_kernel_name_t<detail::auto_name, Type> {
  using name = Type;
};
#endif // __SYCL_UNNAMED_LAMBDA__

} // namespace detail

/// Provides an abstraction of a SYCL kernel.
///
/// \sa sycl_api_exec
/// \sa program
/// \sa queue
///
/// \ingroup sycl_api
class __SYCL_EXPORT kernel {
public:
  /// Constructs a SYCL kernel instance from an OpenCL cl_kernel
  ///
  /// The requirements for this constructor are described in section 4.3.1
  /// of the SYCL specification.
  ///
  /// \param ClKernel is a valid OpenCL cl_kernel instance
  /// \param SyclContext is a valid SYCL context
#ifdef __SYCL_INTERNAL_API
  kernel(cl_kernel ClKernel, const context &SyclContext);
#endif

  kernel(const kernel &RHS) = default;

  kernel(kernel &&RHS) = default;

  kernel &operator=(const kernel &RHS) = default;

  kernel &operator=(kernel &&RHS) = default;

  bool operator==(const kernel &RHS) const { return impl == RHS.impl; }

  bool operator!=(const kernel &RHS) const { return !operator==(RHS); }

  /// Get a valid OpenCL kernel handle
  ///
  /// If this kernel encapsulates an instance of OpenCL kernel, a valid
  /// cl_kernel will be returned. If this kernel is a host kernel,
  /// an invalid_object_error exception will be thrown.
  ///
  /// \return a valid cl_kernel instance
#ifdef __SYCL_INTERNAL_API
  cl_kernel get() const;
#endif

  /// Check if the associated SYCL context is a SYCL host context.
  ///
  /// \return true if this SYCL kernel is a host kernel.
  __SYCL2020_DEPRECATED(
      "is_host() is deprecated as the host device is no longer supported.")
  bool is_host() const;

  /// Get the context that this kernel is defined for.
  ///
  /// The value returned must be equal to that returned by
  /// get_info<info::kernel::context>().
  ///
  /// \return a valid SYCL context
  context get_context() const;

  /// Returns the backend associated with this kernel.
  ///
  /// \return the backend associated with this kernel.
  backend get_backend() const noexcept;

  /// Get the kernel_bundle associated with this kernel.
  ///
  /// \return a valid kernel_bundle<bundle_state::executable>
  kernel_bundle<bundle_state::executable> get_kernel_bundle() const;

  /// Query information from the kernel object using the info::kernel_info
  /// descriptor.
  ///
  /// \return depends on information being queried.
  template <typename Param>
  typename detail::is_kernel_info_desc<Param>::return_type get_info() const;

  /// Query device-specific information from the kernel object using the
  /// info::kernel_device_specific descriptor.
  ///
  /// \param Device is a valid SYCL device to query info for.
  /// \return depends on information being queried.
  template <typename Param>
  typename detail::is_kernel_device_specific_info_desc<Param>::return_type
  get_info(const device &Device) const;

  /// Query device-specific information from a kernel using the
  /// info::kernel_device_specific descriptor for a specific device and value.
  /// max_sub_group_size is the only valid descriptor for this function.
  ///
  /// \param Device is a valid SYCL device.
  /// \param WGSize is the work-group size the sub-group size is requested for.
  /// \return depends on information being queried.
  template <typename Param>
  __SYCL2020_DEPRECATED("Use the overload without the second parameter")
  typename detail::is_kernel_device_specific_info_desc<Param>::return_type
      get_info(const device &Device, const range<3> &WGSize) const;

private:
  /// Constructs a SYCL kernel object from a valid kernel_impl instance.
  kernel(std::shared_ptr<detail::kernel_impl> Impl);

  pi_native_handle getNative() const;

  __SYCL_DEPRECATED("Use getNative() member function")
  pi_native_handle getNativeImpl() const;

  std::shared_ptr<detail::kernel_impl> impl;

  template <class Obj>
  friend decltype(Obj::impl) detail::getSyclObjImpl(const Obj &SyclObject);
  template <class T>
  friend T detail::createSyclObjFromImpl(decltype(T::impl) ImplObj);
  template <backend BackendName, class SyclObjectT>
  friend auto get_native(const SyclObjectT &Obj)
      -> backend_return_t<BackendName, SyclObjectT>;
};
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl

namespace std {
template <> struct hash<sycl::kernel> {
  size_t operator()(const sycl::kernel &Kernel) const {
    return hash<std::shared_ptr<sycl::detail::kernel_impl>>()(
        sycl::detail::getSyclObjImpl(Kernel));
  }
};
} // namespace std
