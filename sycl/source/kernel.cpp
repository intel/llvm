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

kernel::kernel(cl_kernel ClKernel, const context &SyclContext)
    : impl(std::make_shared<detail::kernel_impl>(
          detail::pi::cast<sycl::detail::pi::PiKernel>(ClKernel),
          detail::getSyclObjImpl(SyclContext), nullptr, nullptr)) {
  // This is a special interop constructor for OpenCL, so the kernel must be
  // retained.
  if (get_backend() == backend::opencl) {
    impl->getPlugin()->call<detail::PiApiKind::piKernelRetain>(
        detail::pi::cast<sycl::detail::pi::PiKernel>(ClKernel));
  }
}

cl_kernel kernel::get() const { return impl->get(); }

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
detail::ABINeutralT_t<typename detail::is_kernel_info_desc<Param>::return_type>
kernel::get_info_impl() const {
  return detail::convert_to_abi_neutral(impl->template get_info<Param>());
}

#define __SYCL_PARAM_TRAITS_SPEC(DescType, Desc, ReturnT, PiCode)              \
  template __SYCL_EXPORT detail::ABINeutralT_t<ReturnT>                        \
  kernel::get_info_impl<info::kernel::Desc>() const;

#include <sycl/info/kernel_traits.def>

#undef __SYCL_PARAM_TRAITS_SPEC

template <typename Param>
typename detail::is_backend_info_desc<Param>::return_type
kernel::get_backend_info() const {
  return impl->get_backend_info<Param>();
}

#define __SYCL_PARAM_TRAITS_SPEC(DescType, Desc, ReturnT, Picode)              \
  template __SYCL_EXPORT ReturnT                                               \
  kernel::get_backend_info<info::DescType::Desc>() const;

#include <sycl/info/sycl_backend_traits.def>

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

template <typename Param>
typename Param::return_type
kernel::ext_oneapi_get_info(const queue &Queue) const {
  return impl->ext_oneapi_get_info<Param>(Queue);
}

template __SYCL_EXPORT typename ext::oneapi::experimental::info::
    kernel_queue_specific::max_num_work_group_sync::return_type
    kernel::ext_oneapi_get_info<
        ext::oneapi::experimental::info::kernel_queue_specific::
            max_num_work_group_sync>(const queue &Queue) const;

kernel::kernel(std::shared_ptr<detail::kernel_impl> Impl) : impl(Impl) {}

pi_native_handle kernel::getNative() const { return impl->getNative(); }

pi_native_handle kernel::getNativeImpl() const { return impl->getNative(); }

} // namespace _V1
} // namespace sycl
