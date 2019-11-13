//==------- kernel_impl.hpp --- SYCL kernel implementation -----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/context.hpp>
#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/device_impl.hpp>
#include <CL/sycl/detail/pi.hpp>
#include <CL/sycl/device.hpp>
#include <CL/sycl/info/info_desc.hpp>

#include <cassert>
#include <memory>

namespace cl {
namespace sycl {
// Forward declaration
class program;

namespace detail {
class program_impl;

class kernel_impl {
public:
  /// Constructs a SYCL kernel instance from a PiKernel
  ///
  /// @param Kernel is a valid PiKernel instance
  /// @param SyclContext is a valid SYCL context
  kernel_impl(RT::PiKernel Kernel, const context &SyclContext);

  /// Constructs a SYCL kernel instance from a SYCL program
  ///
  /// @param Kernel is a valid PiKernel instance
  /// @param SyclContext is a valid SYCL context
  /// @param ProgramImpl is a valid instance of program_impl
  /// @param IsCreatedFromSource is a flag that indicates whether program
  /// is created from source code
  kernel_impl(RT::PiKernel Kernel, const context &SyclContext,
              std::shared_ptr<program_impl> ProgramImpl,
              bool IsCreatedFromSource);

  /// Constructs a SYCL kernel for host device
  ///
  /// @param SyclContext is a valid SYCL context
  /// @param ProgramImpl is a valid instance of program_impl
  kernel_impl(const context &SyclContext,
              std::shared_ptr<program_impl> ProgramImpl);

  ~kernel_impl();

  /// Get a valid OpenCL interoperability kernel
  ///
  /// The requirements for this method are described in section 4.3.1
  /// of the SYCL specification.
  ///
  /// @return a valid cl_kernel instance
  cl_kernel get() const {
    if (is_host()) {
      throw invalid_object_error("This instance of kernel is a host instance");
    }
    PI_CALL(RT::piKernelRetain, MKernel);
    return pi::cast<cl_kernel>(MKernel);
  }

  /// Check if this kernel is a SYCL host device kernel
  ///
  /// @return true if a kernel is a valid SYCL host device kernel
  bool is_host() const { return MContext.is_host(); }

  /// Get the context that this kernel is defined for.
  ///
  /// The value returned must be equal to that returned by
  /// get_info<info::kernel::context>().
  ///
  /// @return a valid SYCL context
  context get_context() const { return MContext; }

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

  /// Get a reference to a raw kernel object.
  ///
  /// @return a reference to ai valid PiKernel instance with raw kernel object.
  RT::PiKernel &getHandleRef() { return MKernel; }
  /// Get a constant reference to a raw kernel object.
  ///
  /// @return a constant reference to ai valid PiKernel instance with raw kernel
  /// object.
  const RT::PiKernel &getHandleRef() const { return MKernel; }

  /// Check if kernel was created from a program that had been created from
  /// source.
  ///
  /// @return true if kernel was created from source.
  bool isCreatedFromSource() const;

private:
  RT::PiKernel MKernel;
  context MContext;
  std::shared_ptr<program_impl> MProgramImpl;
  bool MIsCreatedFromSource = true;
};

} // namespace detail
} // namespace sycl
} // namespace cl
