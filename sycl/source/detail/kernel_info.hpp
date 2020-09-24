//==-------- kernel_info.hpp - SYCL kernel info methods --------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/common_info.hpp>
#include <CL/sycl/detail/pi.hpp>
#include <CL/sycl/device.hpp>
#include <CL/sycl/info/info_desc.hpp>
#include <detail/kernel_impl.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

// OpenCL kernel information methods
template <typename T, info::kernel Param> struct get_kernel_info {};

template <info::kernel Param> struct get_kernel_info<string_class, Param> {
  static string_class get(RT::PiKernel Kernel, const plugin &Plugin) {
    size_t ResultSize;

    // TODO catch an exception and put it to list of asynchronous exceptions
    Plugin.call<PiApiKind::piKernelGetInfo>(Kernel, pi_kernel_info(Param), 0,
                                            nullptr, &ResultSize);
    if (ResultSize == 0) {
      return "";
    }
    vector_class<char> Result(ResultSize);
    // TODO catch an exception and put it to list of asynchronous exceptions
    Plugin.call<PiApiKind::piKernelGetInfo>(Kernel, pi_kernel_info(Param),
                                            ResultSize, Result.data(), nullptr);
    return string_class(Result.data());
  }
};

template <info::kernel Param> struct get_kernel_info<cl_uint, Param> {
  static cl_uint get(RT::PiKernel Kernel, const plugin &Plugin) {
    cl_uint Result;

    // TODO catch an exception and put it to list of asynchronous exceptions
    Plugin.call<PiApiKind::piKernelGetInfo>(Kernel, pi_kernel_info(Param),
                                            sizeof(cl_uint), &Result, nullptr);
    return Result;
  }
};

// OpenCL kernel work-group methods

template <typename T, info::kernel_work_group Param>
struct get_kernel_work_group_info {
  static T get(RT::PiKernel Kernel, RT::PiDevice Device, const plugin &Plugin) {
    T Result;
    // TODO catch an exception and put it to list of asynchronous exceptions
    Plugin.call<PiApiKind::piKernelGetGroupInfo>(
        Kernel, Device, pi::cast<pi_kernel_group_info>(Param), sizeof(T),
        &Result, nullptr);
    return Result;
  }
};

template <info::kernel_work_group Param>
struct get_kernel_work_group_info<cl::sycl::range<3>, Param> {
  static cl::sycl::range<3> get(RT::PiKernel Kernel, RT::PiDevice Device,
                                const plugin &Plugin) {
    size_t Result[3];
    // TODO catch an exception and put it to list of asynchronous exceptions
    Plugin.call<PiApiKind::piKernelGetGroupInfo>(
        Kernel, Device, pi::cast<pi_kernel_group_info>(Param),
        sizeof(size_t) * 3, Result, nullptr);
    return cl::sycl::range<3>(Result[0], Result[1], Result[2]);
  }
};

template <info::kernel_work_group Param>
inline typename info::param_traits<info::kernel_work_group, Param>::return_type
get_kernel_work_group_info_host(const cl::sycl::device &Device);

template <>
inline cl::sycl::range<3>
get_kernel_work_group_info_host<info::kernel_work_group::global_work_size>(
    const cl::sycl::device &) {
  throw invalid_object_error("This instance of kernel is a host instance",
                             PI_INVALID_KERNEL);
}

template <>
inline size_t
get_kernel_work_group_info_host<info::kernel_work_group::work_group_size>(
    const cl::sycl::device &Dev) {
  return Dev.get_info<info::device::max_work_group_size>();
}

template <>
inline cl::sycl::range<3> get_kernel_work_group_info_host<
    info::kernel_work_group::compile_work_group_size>(
    const cl::sycl::device &) {
  return {0, 0, 0};
}

template <>
inline size_t get_kernel_work_group_info_host<
    info::kernel_work_group::preferred_work_group_size_multiple>(
    const cl::sycl::device &Dev) {
  return get_kernel_work_group_info_host<
      info::kernel_work_group::work_group_size>(Dev);
}

template <>
inline cl_ulong
get_kernel_work_group_info_host<info::kernel_work_group::private_mem_size>(
    const cl::sycl::device &) {
  return 0;
}
// The kernel sub-group methods

template <info::kernel_sub_group Param> struct get_kernel_sub_group_info {
  static uint32_t get(RT::PiKernel Kernel, RT::PiDevice Device,
                      const plugin &Plugin) {
    uint32_t Result;
    // TODO catch an exception and put it to list of asynchronous exceptions
    Plugin.call<PiApiKind::piKernelGetSubGroupInfo>(
        Kernel, Device, pi_kernel_sub_group_info(Param), 0, nullptr,
        sizeof(uint32_t), &Result, nullptr);

    return Result;
  }
};

template <info::kernel_sub_group Param>
struct get_kernel_sub_group_info_with_input {
  static uint32_t get(RT::PiKernel Kernel, RT::PiDevice Device,
                      cl::sycl::range<3> In, const plugin &Plugin) {
    size_t Input[3] = {In[0], In[1], In[2]};
    uint32_t Result;
    // TODO catch an exception and put it to list of asynchronous exceptions
    Plugin.call<PiApiKind::piKernelGetSubGroupInfo>(
        Kernel, Device, pi_kernel_sub_group_info(Param), sizeof(size_t) * 3,
        Input, sizeof(uint32_t), &Result, nullptr);

    return Result;
  }
};
} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
