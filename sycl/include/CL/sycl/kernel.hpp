//==--------------- kernel.hpp --- SYCL kernel -----------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/info/info_desc.hpp>
#include <CL/sycl/stl.hpp>

#include <memory>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
// Forward declaration
class program;
class context;
namespace detail {
class kernel_impl;
}

class kernel {
public:
  /// Constructs a SYCL kernel instance from an OpenCL cl_kernel
  ///
  /// The requirements for this constructor are described in section 4.3.1
  /// of the SYCL specification.
  ///
  /// \param ClKernel is a valid OpenCL cl_kernel instance
  /// \param SyclContext is a valid SYCL context
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
  typename info::param_traits<info::kernel_sub_group, param>::return_type
  get_sub_group_info(const device &Device) const;

  /// Query sub-group information from a kernel using the
  /// info::kernel_sub_group descriptor for a specific device and value.
  ///
  /// \param Device is a valid SYCL device.
  /// \param Value depends on information being queried.
  /// \return depends on information being queried.
  template <info::kernel_sub_group param>
  typename info::param_traits<info::kernel_sub_group, param>::return_type
  get_sub_group_info(
      const device &Device,
      typename info::param_traits<info::kernel_sub_group, param>::input_type
          Value) const;

private:
  /// Constructs a SYCL kernel object from a valid kernel_impl instance.
  kernel(std::shared_ptr<detail::kernel_impl> Impl);

  shared_ptr_class<detail::kernel_impl> impl;

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
