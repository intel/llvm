//==--------------- kernel.hpp --- SYCL kernel -----------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/export.hpp>
#include <CL/sycl/detail/pi.h>
#include <CL/sycl/info/info_desc.hpp>
#include <CL/sycl/kernel_bundle_enums.hpp>
#include <CL/sycl/stl.hpp>

#include <memory>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
// Forward declaration
class program;
class context;
template <backend Backend> class backend_traits;
template <bundle_state State> class kernel_bundle;

namespace detail {
class kernel_impl;

/// This class is the default KernelName template parameter type for kernel
/// invocation APIs such as single_task.
class auto_name {};

// Helper for the auto_name specialization to ensure that the 'B' is evaluted.
template <typename Type, bool B> struct get_kernel_name_t_helper {
  using name = Type;
};

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
  // We need to mark 'Type' as kernel here so FE will apply proper mangling for
  // it. The reason for that is that when with range rounding enabled, we
  // evaluate __builtin_sycl_unique_stable_name before instantiating the kernel,
  // which leads to different results of built-in evaluation before and after
  // kernel instantiation, which is illegal as it changes the result of
  // previously evaluated constant expression.
  using name =
      typename get_kernel_name_t_helper<Type, __builtin_sycl_mark_kernel_name(
                                                  Type)>::name;
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
  __SYCL2020_DEPRECATED(
      "OpenCL interop constructors are deprecated, use make_kernel() instead")
  kernel(cl_kernel ClKernel, const context &SyclContext);

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
  __SYCL2020_DEPRECATED(
      "OpenCL interop get() functions are deprecated, use get_native() instead")
  cl_kernel get() const;

  /// Check if the associated SYCL context is a SYCL host context.
  ///
  /// \return true if this SYCL kernel is a host kernel.
  bool is_host() const;

  /// Get the context that this kernel is defined for.
  ///
  /// The value returned must be equal to that returned by
  /// get_info<info::kernel::context>().
  ///
  /// \return a valid SYCL context
  context get_context() const;

  /// Get the kernel_bundle associated with this kernel.
  ///
  /// \return a valid kernel_bundle<bundle_state::executable>
  kernel_bundle<bundle_state::executable> get_kernel_bundle() const;

  /// Get the program that this kernel is defined for.
  ///
  /// The value returned must be equal to that returned by
  /// get_info<info::kernel::program>().
  ///
  /// \return a valid SYCL program
  program get_program() const;

  /// Query information from the kernel object using the info::kernel_info
  /// descriptor.
  ///
  /// \return depends on information being queried.
  template <info::kernel param>
  typename info::param_traits<info::kernel, param>::return_type
  get_info() const;

  /// Query device-specific information from the kernel object using the
  /// info::kernel_device_specific descriptor.
  ///
  /// \param Device is a valid SYCL device to query info for.
  /// \return depends on information being queried.
  template <info::kernel_device_specific param>
  typename info::param_traits<info::kernel_device_specific, param>::return_type
  get_info(const device &Device) const;

  /// Query device-specific information from a kernel using the
  /// info::kernel_device_specific descriptor for a specific device and value.
  ///
  /// \param Device is a valid SYCL device.
  /// \param Value depends on information being queried.
  /// \return depends on information being queried.
  template <info::kernel_device_specific param>
  typename info::param_traits<info::kernel_device_specific, param>::return_type
  get_info(const device &Device,
           typename info::param_traits<info::kernel_device_specific,
                                       param>::input_type Value) const;

  /// Query work-group information from a kernel using the
  /// info::kernel_work_group descriptor for a specific device.
  ///
  /// \param Device is a valid SYCL device.
  /// \return depends on information being queried.
  template <info::kernel_work_group param>
  typename info::param_traits<info::kernel_work_group, param>::return_type
  get_work_group_info(const device &Device) const;

  /// Query sub-group information from a kernel using the
  /// info::kernel_sub_group descriptor for a specific device.
  ///
  /// \param Device is a valid SYCL device.
  /// \return depends on information being queried.
  template <info::kernel_sub_group param>
  // clang-format off
  __SYCL_DEPRECATED("Use get_info with info::kernel_device_specific instead.")
  typename info::param_traits<info::kernel_sub_group, param>::return_type
  get_sub_group_info(const device &Device) const;
  // clang-format on

  /// Query sub-group information from a kernel using the
  /// info::kernel_sub_group descriptor for a specific device and value.
  ///
  /// \param Device is a valid SYCL device.
  /// \param Value depends on information being queried.
  /// \return depends on information being queried.
  template <info::kernel_sub_group param>
  // clang-format off
  __SYCL_DEPRECATED("Use get_info with info::kernel_device_specific instead.")
  typename info::param_traits<info::kernel_sub_group, param>::return_type
  get_sub_group_info(const device &Device,
                     typename info::param_traits<info::kernel_sub_group,
                     param>::input_type Value) const;
  // clang-format on

  template <backend Backend>
  typename backend_traits<Backend>::template return_type<kernel>
  get_native() const {
    return detail::pi::cast<
        typename backend_traits<Backend>::template return_type<kernel>>(
        getNativeImpl());
  }

private:
  /// Constructs a SYCL kernel object from a valid kernel_impl instance.
  kernel(std::shared_ptr<detail::kernel_impl> Impl);

  pi_native_handle getNativeImpl() const;

  std::shared_ptr<detail::kernel_impl> impl;

  template <class Obj>
  friend decltype(Obj::impl) detail::getSyclObjImpl(const Obj &SyclObject);
  template <class T>
  friend T detail::createSyclObjFromImpl(decltype(T::impl) ImplObj);
};
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)

namespace std {
template <> struct hash<cl::sycl::kernel> {
  size_t operator()(const cl::sycl::kernel &Kernel) const {
    return hash<std::shared_ptr<cl::sycl::detail::kernel_impl>>()(
        cl::sycl::detail::getSyclObjImpl(Kernel));
  }
};
} // namespace std
