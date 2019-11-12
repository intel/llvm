//==--------------- kernel.hpp --- SYCL kernel -----------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/stl.hpp>
#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/info/info_desc.hpp>

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
  kernel(cl_kernel ClKernel, const context &SyclContext);

  kernel(const kernel &RHS) = default;

  kernel(kernel &&RHS) = default;

  kernel &operator=(const kernel &RHS) = default;

  kernel &operator=(kernel &&RHS) = default;

  bool operator==(const kernel &RHS) const;

  bool operator!=(const kernel &RHS) const;

  cl_kernel get() const;

  bool is_host() const;

  context get_context() const;

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
  get_sub_group_info(const device &Device,
                     typename info::param_traits<info::kernel_sub_group,
                                                 param>::input_type Value) const;

private:
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
