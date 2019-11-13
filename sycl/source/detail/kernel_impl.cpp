//==------- kernel_impl.cpp --- SYCL kernel implementation -----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/kernel_impl.hpp>
#include <CL/sycl/detail/kernel_info.hpp>
#include <CL/sycl/info/info_desc.hpp>
#include <CL/sycl/program.hpp>

#include <memory>

namespace cl {
namespace sycl {
namespace detail {

kernel_impl::kernel_impl(RT::PiKernel Kernel, const context &SyclContext)
    : kernel_impl(Kernel, SyclContext,
                  std::make_shared<program_impl>(SyclContext, Kernel),
                  /*IsCreatedFromSource*/ true) {}

kernel_impl::kernel_impl(RT::PiKernel Kernel, const context &SyclContext,
                         std::shared_ptr<program_impl> ProgramImpl,
                         bool IsCreatedFromSource)
    : MKernel(Kernel), MContext(SyclContext), MProgramImpl(ProgramImpl),
      MIsCreatedFromSource(IsCreatedFromSource) {

  RT::PiContext Context = nullptr;
  PI_CALL(RT::piKernelGetInfo, MKernel, CL_KERNEL_CONTEXT, sizeof(Context),
          &Context, nullptr);
  auto ContextImpl = detail::getSyclObjImpl(SyclContext);
  if (ContextImpl->getHandleRef() != Context)
    throw cl::sycl::invalid_parameter_error(
        "Input context must be the same as the context of cl_kernel");
  PI_CALL(RT::piKernelRetain, MKernel);
}

kernel_impl::kernel_impl(const context &SyclContext,
                         std::shared_ptr<program_impl> ProgramImpl)
    : MContext(SyclContext), MProgramImpl(ProgramImpl) {}

kernel_impl::~kernel_impl() {
  // TODO catch an exception and put it to list of asynchronous exceptions
  if (!is_host()) {
    PI_CALL(RT::piKernelRelease, MKernel);
  }
}

program kernel_impl::get_program() const {
  return createSyclObjFromImpl<program>(MProgramImpl);
}

template <info::kernel param>
typename info::param_traits<info::kernel, param>::return_type
kernel_impl::get_info() const {
  if (is_host()) {
    // TODO implement
    assert(0 && "Not implemented");
  }
  return get_kernel_info<
      typename info::param_traits<info::kernel, param>::return_type,
      param>::_(this->getHandleRef());
}

template <> context kernel_impl::get_info<info::kernel::context>() const {
  return get_context();
}

template <> program kernel_impl::get_info<info::kernel::program>() const {
  return get_program();
}

template <info::kernel_work_group param>
typename info::param_traits<info::kernel_work_group, param>::return_type
kernel_impl::get_work_group_info(const device &Device) const {
  if (is_host()) {
    return get_kernel_work_group_info_host<param>(Device);
  }
  return get_kernel_work_group_info<
      typename info::param_traits<info::kernel_work_group, param>::return_type,
      param>::_(this->getHandleRef(), getSyclObjImpl(Device)->getHandleRef());
}

template <info::kernel_sub_group param>
typename info::param_traits<info::kernel_sub_group, param>::return_type
kernel_impl::get_sub_group_info(const device &Device) const {
  if (is_host()) {
    throw runtime_error("Sub-group feature is not supported on HOST device.");
  }
  return get_kernel_sub_group_info<
      typename info::param_traits<info::kernel_sub_group, param>::return_type,
      param>::_(this->getHandleRef(), getSyclObjImpl(Device)->getHandleRef());
}

template <info::kernel_sub_group param>
typename info::param_traits<info::kernel_sub_group, param>::return_type
kernel_impl::get_sub_group_info(
    const device &Device,
    typename info::param_traits<info::kernel_sub_group, param>::input_type
        Value) const {
  if (is_host()) {
    throw runtime_error("Sub-group feature is not supported on HOST device.");
  }
  return get_kernel_sub_group_info_with_input<
      typename info::param_traits<info::kernel_sub_group, param>::return_type,
      param,
      typename info::param_traits<info::kernel_sub_group, param>::input_type>::
      _(this->getHandleRef(), getSyclObjImpl(Device)->getHandleRef(), Value);
}

#define PARAM_TRAITS_SPEC(param_type, param, ret_type)                         \
  template ret_type kernel_impl::get_info<info::param_type::param>() const;

#include <CL/sycl/info/kernel_traits.def>

#undef PARAM_TRAITS_SPEC

#define PARAM_TRAITS_SPEC(param_type, param, ret_type)                         \
  template ret_type kernel_impl::get_work_group_info<info::param_type::param>( \
      const device &) const;

#include <CL/sycl/info/kernel_work_group_traits.def>

#undef PARAM_TRAITS_SPEC

#define PARAM_TRAITS_SPEC(param_type, param, ret_type)                         \
  template ret_type kernel_impl::get_sub_group_info<info::param_type::param>(  \
      const device &) const;
#define PARAM_TRAITS_SPEC_WITH_INPUT(param_type, param, ret_type, in_type)     \
  template ret_type kernel_impl::get_sub_group_info<info::param_type::param>(  \
      const device &, in_type) const;

#include <CL/sycl/info/kernel_sub_group_traits.def>

#undef PARAM_TRAITS_SPEC
#undef PARAM_TRAITS_SPEC_WITH_INPUT

bool kernel_impl::isCreatedFromSource() const {
  // TODO it is not clear how to understand whether the SYCL kernel is created
  // from source code or not when the SYCL kernel is created using
  // the interoperability constructor.
  // Here a strange case which does not work now:
  // context Context;
  // program Program(Context);
  // Program.build_with_kernel_type<class A>();
  // kernel FirstKernel= Program.get_kernel<class A>();
  // cl_kernel ClKernel = FirstKernel.get();
  // kernel SecondKernel = kernel(ClKernel, Context);
  // clReleaseKernel(ClKernel);
  // FirstKernel.isCreatedFromSource() != FirstKernel.isCreatedFromSource();
  return MIsCreatedFromSource;
}

} // namespace detail
} // namespace sycl
} // namespace cl
