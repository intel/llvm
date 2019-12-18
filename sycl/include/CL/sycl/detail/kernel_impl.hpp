//==------- kernel_impl.hpp --- SYCL kernel implementation -----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/context_impl.hpp>
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

using ContextImplPtr = std::shared_ptr<detail::context_impl>;
using ProgramImplPtr = std::shared_ptr<program_impl>;
class kernel_impl {
public:
  /// Constructs a SYCL kernel instance from a PiKernel
  ///
  /// This constructor is used for plug-in interoperability. It always marks
  /// kernel as being created from source and creates a new program_impl
  /// instance.
  ///
  /// @param Kernel is a valid PiKernel instance
  /// @param SyclContext is a valid SYCL context
  kernel_impl(RT::PiKernel Kernel, ContextImplPtr Context);

  /// Constructs a SYCL kernel instance from a SYCL program and a PiKernel
  ///
  /// This constructor creates a new instance from PiKernel and saves
  /// the provided SYCL program. If context of PiKernel differs from
  /// context of the SYCL program, an invalid_parameter_error exception is
  /// thrown.
  ///
  /// @param Kernel is a valid PiKernel instance
  /// @param SyclContext is a valid SYCL context
  /// @param ProgramImpl is a valid instance of program_impl
  /// @param IsCreatedFromSource is a flag that indicates whether program
  /// is created from source code
  kernel_impl(RT::PiKernel Kernel, ContextImplPtr ContextImpl,
              ProgramImplPtr ProgramImpl,
              bool IsCreatedFromSource);

  /// Constructs a SYCL kernel for host device
  ///
  /// @param SyclContext is a valid SYCL context
  /// @param ProgramImpl is a valid instance of program_impl
  kernel_impl(ContextImplPtr Context,
              ProgramImplPtr ProgramImpl);

  ~kernel_impl();

  /// Gets a valid OpenCL kernel handle
  ///
  /// If this kernel encapsulates an instance of OpenCL kernel, a valid
  /// cl_kernel will be returned. If this kernel is a host kernel,
  /// an invalid_object_error exception will be thrown.
  ///
  /// @return a valid cl_kernel instance
  cl_kernel get() const {
    if (is_host())
      throw invalid_object_error("This instance of kernel is a host instance");
    PI_CALL(piKernelRetain)(MKernel);
    return pi::cast<cl_kernel>(MKernel);
  }

  /// Check if the associated SYCL context is a SYCL host context.
  ///
  /// @return true if this SYCL kernel is a host kernel.
  bool is_host() const { return MContext->is_host(); }

  /// Query information from the kernel object using the info::kernel_info
  /// descriptor.
  ///
  /// @return depends on information being queried.
  template <info::kernel param>
  typename info::param_traits<info::kernel, param>::return_type
  get_info() const;

  /// Query work-group information from a kernel using the
  /// info::kernel_work_group descriptor for a specific device.
  ///
  /// @param Device is a valid SYCL device.
  /// @return depends on information being queried.
  template <info::kernel_work_group param>
  typename info::param_traits<info::kernel_work_group, param>::return_type
  get_work_group_info(const device &Device) const;

  /// Query sub-group information from a kernel using the
  /// info::kernel_sub_group descriptor for a specific device.
  ///
  /// @param Device is a valid SYCL device
  template <info::kernel_sub_group param>
  typename info::param_traits<info::kernel_sub_group, param>::return_type
  get_sub_group_info(const device &Device) const;

  /// Query sub-group information from a kernel using the
  /// info::kernel_sub_group descriptor for a specific device and value.
  ///
  /// @param Device is a valid SYCL device.
  /// @param Value depends on information being queried.
  /// @return depends on information being queried.
  template <info::kernel_sub_group param>
  typename info::param_traits<info::kernel_sub_group, param>::return_type
  get_sub_group_info(
      const device &Device,
      typename info::param_traits<info::kernel_sub_group, param>::input_type
          Value) const;

  /// Get a reference to a raw kernel object.
  ///
  /// @return a reference to a valid PiKernel instance with raw kernel object.
  RT::PiKernel &getHandleRef() { return MKernel; }
  /// Get a constant reference to a raw kernel object.
  ///
  /// @return a constant reference to a valid PiKernel instance with raw kernel
  /// object.
  const RT::PiKernel &getHandleRef() const { return MKernel; }

  /// Check if kernel was created from a program that had been created from
  /// source.
  ///
  /// @return true if kernel was created from source.
  bool isCreatedFromSource() const;

private:
  RT::PiKernel MKernel;
  const ContextImplPtr MContext;
  const ProgramImplPtr MProgramImpl;
  bool MCreatedFromSource = true;
};

} // namespace detail
} // namespace sycl
} // namespace cl
