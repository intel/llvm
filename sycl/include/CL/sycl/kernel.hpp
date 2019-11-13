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

namespace cl {
namespace sycl {
// Forward declaration
class program;
class context;
namespace detail {
class kernel_impl;
}

class kernel {
  template <class Obj>
  friend decltype(Obj::impl) detail::getSyclObjImpl(const Obj &SyclObject);
  template <class T>
  friend T detail::createSyclObjFromImpl(decltype(T::impl) ImplObj);

public:
  /// Constructs a SYCL kernel instance from an OpenCL cl_kernel
  ///
  /// The requirements for this constructor are described in section 4.3.1
  /// of the SYCL specification.
  ///
  /// @param ClKernel is a valid OpenCL cl_kernel instance
  /// @param SyclContext is a valid SYCL context
  kernel(cl_kernel ClKernel, const context &SyclContext);

  kernel(const kernel &RHS) = default;

  kernel(kernel &&RHS) = default;

  kernel &operator=(const kernel &RHS) = default;

  kernel &operator=(kernel &&RHS) = default;

  bool operator==(const kernel &RHS) const;

  bool operator!=(const kernel &RHS) const;

  /// Get a valid OpenCL interoperability kernel
  ///
  /// The requirements for this method are described in section 4.3.1
  /// of the SYCL specification.
  ///
  /// @return a valid cl_kernel instance
  cl_kernel get() const;

  /// Check if this kernel is a SYCL host device kernel
  ///
  /// @return true if a kernel is a valid SYCL host device kernel
  bool is_host() const;

  /// Get the context that this kernel is defined for.
  ///
  /// The value returned must be equal to that returned by
  /// get_info<info::kernel::context>().
  ///
  /// @return a valid SYCL context
  context get_context() const;

  /// Get the program that this kernel is defined for.
  ///
  /// The value returned must be equal to that returned by
  /// get_info<info::kernel::program>().
  ///
  /// @return a valid SYCL program
  program get_program() const;

  /// Query information from the kernel object using the info::kernel_info
  /// descriptor.
  ///
  /// Valid template parameters are described in Table 4.84 of the SYCL
  /// specification.
  template <info::kernel param>
  typename info::param_traits<info::kernel, param>::return_type
  get_info() const;

  /// Query information from the work-group from a kernel using the
  /// info::kernel_work_group descriptor for a specific device.
  ///
  /// Valid template parameters are described in Table 4.85 of the SYCL
  /// specification.
  template <info::kernel_work_group param>
  typename info::param_traits<info::kernel_work_group, param>::return_type
  get_work_group_info(const device &Device) const;

  /// Query information from the sub-group from a kernel using the
  /// info::kernel_sub_group descriptor for a specific device.
  template <info::kernel_sub_group param>
  typename info::param_traits<info::kernel_sub_group, param>::return_type
  get_sub_group_info(const device &Device) const;

  /// Query information from the sub-group from a kernel using the
  /// info::kernel_sub_group descriptor for a specific device.
  template <info::kernel_sub_group param>
  typename info::param_traits<info::kernel_sub_group, param>::return_type
  get_sub_group_info(
      const device &Device,
      typename info::param_traits<info::kernel_sub_group, param>::input_type
          Value) const;

private:
  /// Constructs a SYCL kernel object from a valid kernel_impl instance.
  kernel(std::shared_ptr<detail::kernel_impl> Impl);

  std::shared_ptr<detail::kernel_impl> impl;
};
} // namespace sycl
} // namespace cl

namespace std {
template <> struct hash<cl::sycl::kernel> {
  size_t operator()(const cl::sycl::kernel &Kernel) const {
    return hash<std::shared_ptr<cl::sycl::detail::kernel_impl>>()(
        cl::sycl::detail::getSyclObjImpl(Kernel));
  }
};
} // namespace std
