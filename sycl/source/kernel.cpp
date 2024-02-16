//==--------------- kernel.cpp --- SYCL kernel -----------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/backend_impl.hpp>
#include <detail/kernel_bundle_impl.hpp>
#include <detail/kernel_impl.hpp>
#include <sycl/detail/export.hpp>
#include <sycl/detail/pi.h>
#include <sycl/kernel.hpp>

namespace sycl {
inline namespace _V1 {

kernel::kernel(cl_kernel ClKernel, const context &SyclContext) {
  try {
    sycl::detail::pi::PiKernel Kernel;
    auto Context = detail::getSyclObjImpl(SyclContext);
    auto Plugin = sycl::detail::pi::getPlugin<backend::opencl>();
    cl_program CLProgram;
    size_t Ret = clGetKernelInfo(ClKernel, CL_KERNEL_PROGRAM, sizeof(CLProgram),
                                 &CLProgram, nullptr);
    if (Ret != CL_SUCCESS) {
      throw runtime_error(
          "Failed to retrieve program associated with the kernel",
          PI_ERROR_INVALID_KERNEL);
    }
    sycl::detail::pi::PiProgram Program;
    Plugin->call<detail::PiApiKind::piextProgramCreateWithNativeHandle>(
        detail::pi::cast<pi_native_handle>(CLProgram), Context->getHandleRef(),
        false, &Program);

    Plugin->call<detail::PiApiKind::piextKernelCreateWithNativeHandle>(
        detail::pi::cast<pi_native_handle>(ClKernel), Context->getHandleRef(),
        Program, false, &Kernel);
    impl = std::make_shared<detail::kernel_impl>(Kernel, Context, nullptr,
                                                 nullptr);
  } catch (sycl::runtime_error &) {
    throw sycl::invalid_parameter_error(
        "Input context must be the same as the context of cl_kernel",
        PI_ERROR_INVALID_CONTEXT);
  }
}

cl_kernel kernel::get() const {
  return detail::pi::cast<cl_kernel>(impl->getNative());
}

bool kernel::is_host() const {
  bool IsHost = impl->is_host();
  assert(!IsHost && "kernel::is_host should not be called in implementation.");
  return IsHost;
}

context kernel::get_context() const {
  return impl->get_info<info::kernel::context>();
}

backend kernel::get_backend() const noexcept { return getImplBackend(impl); }

kernel_bundle<sycl::bundle_state::executable>
kernel::get_kernel_bundle() const {
  return detail::createSyclObjFromImpl<
      kernel_bundle<sycl::bundle_state::executable>>(impl->get_kernel_bundle());
}

template <typename Param>
typename detail::is_kernel_info_desc<Param>::return_type
kernel::get_info() const {
  return impl->get_info<Param>();
}

#define __SYCL_PARAM_TRAITS_SPEC(DescType, Desc, ReturnT, PiCode)              \
  template __SYCL_EXPORT ReturnT kernel::get_info<info::kernel::Desc>() const;

#include <sycl/info/kernel_traits.def>

#undef __SYCL_PARAM_TRAITS_SPEC

template <typename Param>
typename detail::is_kernel_device_specific_info_desc<Param>::return_type
kernel::get_info(const device &Dev) const {
  return impl->get_info<Param>(Dev);
}

// Deprecated overload for kernel_device_specific::max_sub_group_size taking
// an extra argument.
template <typename Param>
typename detail::is_kernel_device_specific_info_desc<Param>::return_type
kernel::get_info(const device &Device, const range<3> &WGSize) const {
  static_assert(
      std::is_same_v<Param, info::kernel_device_specific::max_sub_group_size>,
      "Unexpected param for kernel::get_info with range argument.");
  return impl->get_info<Param>(Device, WGSize);
}

#define __SYCL_PARAM_TRAITS_SPEC(DescType, Desc, ReturnT, PiCode)              \
  template __SYCL_EXPORT ReturnT kernel::get_info<info::DescType::Desc>(       \
      const device &) const;

#include <sycl/info/kernel_device_specific_traits.def>

#undef __SYCL_PARAM_TRAITS_SPEC

template __SYCL_EXPORT uint32_t
kernel::get_info<info::kernel_device_specific::max_sub_group_size>(
    const device &, const sycl::range<3> &) const;

kernel::kernel(std::shared_ptr<detail::kernel_impl> Impl) : impl(Impl) {}

pi_native_handle kernel::getNative() const { return impl->getNative(); }

pi_native_handle kernel::getNativeImpl() const { return impl->getNative(); }

} // namespace _V1
} // namespace sycl
