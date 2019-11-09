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
#include <CL/sycl/detail/kernel_info.hpp>
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
              bool IsCreatedFromSource)
      : Kernel(Kernel), Context(SyclContext), ProgramImpl(ProgramImpl),
        IsCreatedFromSource(IsCreatedFromSource) {

    RT::PiContext Context = nullptr;
    PI_CALL(RT::piKernelGetInfo(
        Kernel, CL_KERNEL_CONTEXT, sizeof(Context), &Context, nullptr));
    auto ContextImpl = detail::getSyclObjImpl(SyclContext);
    if (ContextImpl->getHandleRef() != Context)
      throw cl::sycl::invalid_parameter_error(
          "Input context must be the same as the context of cl_kernel");
    PI_CALL(RT::piKernelRetain(Kernel));
  }

  // Host kernel constructor
  kernel_impl(const context &SyclContext,
              std::shared_ptr<program_impl> ProgramImpl)
      : Context(SyclContext), ProgramImpl(ProgramImpl) {}

  ~kernel_impl() {
    // TODO catch an exception and put it to list of asynchronous exceptions
    if (!is_host()) {
      PI_CALL(RT::piKernelRelease(Kernel));
    }
  }

  cl_kernel get() const {
    if (is_host()) {
      throw invalid_object_error("This instance of kernel is a host instance");
    }
    PI_CALL(RT::piKernelRetain(Kernel));
    return pi::cast<cl_kernel>(Kernel);
  }

  bool is_host() const { return Context.is_host(); }

  context get_context() const { return Context; }

  program get_program() const;

  template <info::kernel param>
  typename info::param_traits<info::kernel, param>::return_type
  get_info() const {
    if (is_host()) {
      // TODO implement
      assert(0 && "Not implemented");
    }
    return get_kernel_info<
        typename info::param_traits<info::kernel, param>::return_type,
        param>::_(this->getHandleRef());
  }

  template <info::kernel_work_group param>
  typename info::param_traits<info::kernel_work_group, param>::return_type
  get_work_group_info(const device &Device) const {
    if (is_host()) {
      return get_kernel_work_group_info_host<param>(Device);
    }
    return get_kernel_work_group_info<
        typename info::param_traits<info::kernel_work_group,
                                    param>::return_type,
        param>::_(this->getHandleRef(),
                  getSyclObjImpl(Device)->getHandleRef());
  }

  template <info::kernel_sub_group param>
  typename info::param_traits<info::kernel_sub_group, param>::return_type
  get_sub_group_info(const device &Device) const {
    if (is_host()) {
      throw runtime_error("Sub-group feature is not supported on HOST device.");
    }
    return get_kernel_sub_group_info<
        typename info::param_traits<info::kernel_sub_group,
                                    param>::return_type, param>::_(
            this->getHandleRef(),
            getSyclObjImpl(Device)->getHandleRef());
  }

  template <info::kernel_sub_group param>
  typename info::param_traits<info::kernel_sub_group, param>::return_type
  get_sub_group_info(
      const device &Device,
      typename info::param_traits<info::kernel_sub_group, param>::input_type
          Value) const {
    if (is_host()) {
      throw runtime_error("Sub-group feature is not supported on HOST device.");
    }
    return get_kernel_sub_group_info_with_input<
        typename info::param_traits<info::kernel_sub_group, param>::return_type,
        param,
        typename info::param_traits<info::kernel_sub_group, param>::input_type>::_(
            this->getHandleRef(),
            getSyclObjImpl(Device)->getHandleRef(), Value);
  }

  RT::PiKernel &getHandleRef() { return Kernel; }
  const RT::PiKernel &getHandleRef() const { return Kernel; }

  bool isCreatedFromSource() const;

private:
  RT::PiKernel Kernel;
  context Context;
  std::shared_ptr<program_impl> ProgramImpl;
  bool IsCreatedFromSource = true;
};

template <> context kernel_impl::get_info<info::kernel::context>() const;

template <> program kernel_impl::get_info<info::kernel::program>() const;

} // namespace detail
} // namespace sycl
} // namespace cl
