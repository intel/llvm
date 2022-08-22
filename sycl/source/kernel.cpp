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
#include <sycl/program.hpp>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {

kernel::kernel(cl_kernel ClKernel, const context &SyclContext)
    : impl(std::make_shared<detail::kernel_impl>(
          detail::pi::cast<detail::RT::PiKernel>(ClKernel),
          detail::getSyclObjImpl(SyclContext), nullptr)) {}

cl_kernel kernel::get() const { return impl->get(); }

bool kernel::is_host() const { return impl->is_host(); }

context kernel::get_context() const {
  return impl->get_info<info::kernel::context>();
}

backend kernel::get_backend() const noexcept { return getImplBackend(impl); }

kernel_bundle<sycl::bundle_state::executable>
kernel::get_kernel_bundle() const {
  return detail::createSyclObjFromImpl<
      kernel_bundle<sycl::bundle_state::executable>>(impl->get_kernel_bundle());
}

program kernel::get_program() const {
  return impl->get_info<info::kernel::program>();
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

template <typename Param>
typename detail::is_kernel_device_specific_info_desc<
    Param>::with_input_return_type
kernel::get_info(const device &Device, const range<3> &WGSize) const {
  return impl->get_info<Param>(Device, WGSize);
}

#define __SYCL_PARAM_TRAITS_SPEC(DescType, Desc, ReturnT, PiCode)              \
  template __SYCL_EXPORT ReturnT kernel::get_info<info::DescType::Desc>(       \
      const device &) const;
#define __SYCL_PARAM_TRAITS_SPEC_WITH_INPUT(DescType, Desc, ReturnT, InputT,   \
                                            PiCode)                            \
  template __SYCL_EXPORT ReturnT kernel::get_info<info::DescType::Desc>(       \
      const device &, const InputT &) const;

#include <sycl/info/kernel_device_specific_traits.def>

#undef __SYCL_PARAM_TRAITS_SPEC
#undef __SYCL_PARAM_TRAITS_SPEC_WITH_INPUT

kernel::kernel(std::shared_ptr<detail::kernel_impl> Impl) : impl(Impl) {}

pi_native_handle kernel::getNative() const { return impl->getNative(); }

pi_native_handle kernel::getNativeImpl() const { return impl->getNative(); }

} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
