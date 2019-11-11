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
  kernel_impl(RT::PiKernel Kernel, const context &SyclContext);

  kernel_impl(RT::PiKernel Kernel, const context &SyclContext,
              std::shared_ptr<program_impl> ProgramImpl,
              bool IsCreatedFromSource);

  // Host kernel constructor
  kernel_impl(const context &SyclContext,
              std::shared_ptr<program_impl> ProgramImpl);

  ~kernel_impl();

  cl_kernel get() const {
    if (is_host()) {
      throw invalid_object_error("This instance of kernel is a host instance");
    }
    PI_CALL(RT::piKernelRetain, Kernel);
    return pi::cast<cl_kernel>(Kernel);
  }

  bool is_host() const { return Context.is_host(); }

  context get_context() const { return Context; }

  program get_program() const;

  template <info::kernel param>
  typename info::param_traits<info::kernel, param>::return_type
  get_info() const;

  template <info::kernel_work_group param>
  typename info::param_traits<info::kernel_work_group, param>::return_type
  get_work_group_info(const device &Device) const;

  template <info::kernel_sub_group param>
  typename info::param_traits<info::kernel_sub_group, param>::return_type
  get_sub_group_info(const device &Device) const;

  template <info::kernel_sub_group param>
  typename info::param_traits<info::kernel_sub_group, param>::return_type
  get_sub_group_info(
      const device &Device,
      typename info::param_traits<info::kernel_sub_group, param>::input_type
          Value) const;

  RT::PiKernel &getHandleRef() { return Kernel; }
  const RT::PiKernel &getHandleRef() const { return Kernel; }

  bool isCreatedFromSource() const;

private:
  RT::PiKernel Kernel;
  context Context;
  std::shared_ptr<program_impl> ProgramImpl;
  bool IsCreatedFromSource = true;
};

} // namespace detail
} // namespace sycl
} // namespace cl
