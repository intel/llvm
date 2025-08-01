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
#include <detail/ur.hpp>
#include <sycl/detail/export.hpp>
#include <sycl/kernel.hpp>

namespace sycl {
inline namespace _V1 {

// TODO(pi2ur): Don't cast straight from cl_kernel below
kernel::kernel(cl_kernel ClKernel, const context &SyclContext) {
  using namespace sycl::detail;
  adapter_impl &Adapter = ur::getAdapter<backend::opencl>();
  Managed<ur_kernel_handle_t> hKernel{Adapter};
  ur_native_handle_t nativeHandle =
      reinterpret_cast<ur_native_handle_t>(ClKernel);
  Adapter.call<errc::invalid, UrApiKind::urKernelCreateWithNativeHandle>(
      nativeHandle, getSyclObjImpl(SyclContext)->getHandleRef(), nullptr,
      nullptr, &hKernel);
  impl = std::make_shared<kernel_impl>(
      std::move(hKernel), *getSyclObjImpl(SyclContext), nullptr, nullptr);
  // This is a special interop constructor for OpenCL, so the kernel must be
  // retained.
  if (get_backend() == backend::opencl) {
    __SYCL_OCL_CALL(clRetainKernel, ClKernel);
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

#define __SYCL_PARAM_TRAITS_SPEC(Namespace, DescType, Desc, ReturnT, UrCode)   \
  template __SYCL_EXPORT ReturnT                                               \
  kernel::get_info<Namespace::info::DescType::Desc>(const device &) const;

#include <sycl/info/ext_intel_kernel_info_traits.def>

#undef __SYCL_PARAM_TRAITS_SPEC

template __SYCL_EXPORT uint32_t
kernel::get_info<info::kernel_device_specific::max_sub_group_size>(
    const device &, const sycl::range<3> &) const;

#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
// This function is unused and should be removed in the next ABI-breaking
// window.
template <typename Param>
typename detail::is_kernel_queue_specific_info_desc<Param>::return_type
kernel::ext_oneapi_get_info(queue Queue) const {
  return impl->ext_oneapi_get_info<Param>(std::move(Queue));
}
#endif // __INTEL_PREVIEW_BREAKING_CHANGES

template <typename Param>
typename detail::is_kernel_queue_specific_info_desc<Param>::return_type
kernel::ext_oneapi_get_info(queue Queue, const range<1> &WorkGroupSize,
                            size_t DynamicLocalMemorySize) const {
  return impl->ext_oneapi_get_info<Param>(std::move(Queue), WorkGroupSize,
                                          DynamicLocalMemorySize);
}

template <typename Param>
typename detail::is_kernel_queue_specific_info_desc<Param>::return_type
kernel::ext_oneapi_get_info(queue Queue, const range<3> &WG) const {
  return impl->ext_oneapi_get_info<Param>(Queue, WG);
}

template <typename Param>
typename detail::is_kernel_queue_specific_info_desc<Param>::return_type
kernel::ext_oneapi_get_info(queue Queue, const range<2> &WorkGroupSize,
                            size_t DynamicLocalMemorySize) const {
  return impl->ext_oneapi_get_info<Param>(std::move(Queue), WorkGroupSize,
                                          DynamicLocalMemorySize);
}

template <typename Param>
typename detail::is_kernel_queue_specific_info_desc<Param>::return_type
kernel::ext_oneapi_get_info(queue Queue, const range<2> &WG) const {
  return impl->ext_oneapi_get_info<Param>(Queue, WG);
}

template <typename Param>
typename detail::is_kernel_queue_specific_info_desc<Param>::return_type
kernel::ext_oneapi_get_info(queue Queue, const range<1> &WG) const {
  return impl->ext_oneapi_get_info<Param>(Queue, WG);
}

template <typename Param>
typename detail::is_kernel_queue_specific_info_desc<Param>::return_type
kernel::ext_oneapi_get_info(queue Queue, const range<3> &WorkGroupSize,
                            size_t DynamicLocalMemorySize) const {
  return impl->ext_oneapi_get_info<Param>(std::move(Queue), WorkGroupSize,
                                          DynamicLocalMemorySize);
}

#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
// These functions are unused and should be removed in the next ABI-breaking
// window.

template __SYCL_EXPORT typename ext::oneapi::experimental::info::
    kernel_queue_specific::max_work_group_size::return_type
    kernel::ext_oneapi_get_info<ext::oneapi::experimental::info::
                                    kernel_queue_specific::max_work_group_size>(
        queue Queue) const;

template __SYCL_EXPORT typename ext::oneapi::experimental::info::
    kernel_queue_specific::max_work_item_sizes<1>::return_type
    kernel::ext_oneapi_get_info<
        ext::oneapi::experimental::info::kernel_queue_specific::
            max_work_item_sizes<1>>(queue Queue) const;

template __SYCL_EXPORT typename ext::oneapi::experimental::info::
    kernel_queue_specific::max_work_item_sizes<2>::return_type
    kernel::ext_oneapi_get_info<
        ext::oneapi::experimental::info::kernel_queue_specific::
            max_work_item_sizes<2>>(queue Queue) const;

template __SYCL_EXPORT typename ext::oneapi::experimental::info::
    kernel_queue_specific::max_work_item_sizes<3>::return_type
    kernel::ext_oneapi_get_info<
        ext::oneapi::experimental::info::kernel_queue_specific::
            max_work_item_sizes<3>>(queue Queue) const;

#endif // __INTEL_PREVIEW_BREAKING_CHANGES

template __SYCL_EXPORT typename ext::oneapi::experimental::info::
    kernel_queue_specific::max_sub_group_size::return_type
    kernel::ext_oneapi_get_info<ext::oneapi::experimental::info::
                                    kernel_queue_specific::max_sub_group_size>(
        queue Queue, const range<3> &) const;

template __SYCL_EXPORT typename ext::oneapi::experimental::info::
    kernel_queue_specific::max_sub_group_size::return_type
    kernel::ext_oneapi_get_info<ext::oneapi::experimental::info::
                                    kernel_queue_specific::max_sub_group_size>(
        queue Queue, const range<2> &) const;

template __SYCL_EXPORT typename ext::oneapi::experimental::info::
    kernel_queue_specific::max_sub_group_size::return_type
    kernel::ext_oneapi_get_info<ext::oneapi::experimental::info::
                                    kernel_queue_specific::max_sub_group_size>(
        queue Queue, const range<1> &) const;

template __SYCL_EXPORT typename ext::oneapi::experimental::info::
    kernel_queue_specific::num_sub_groups::return_type
    kernel::ext_oneapi_get_info<
        ext::oneapi::experimental::info::kernel_queue_specific::num_sub_groups>(
        queue Queue, const range<3> &) const;

template __SYCL_EXPORT typename ext::oneapi::experimental::info::
    kernel_queue_specific::num_sub_groups::return_type
    kernel::ext_oneapi_get_info<
        ext::oneapi::experimental::info::kernel_queue_specific::num_sub_groups>(
        queue Queue, const range<2> &) const;

template __SYCL_EXPORT typename ext::oneapi::experimental::info::
    kernel_queue_specific::num_sub_groups::return_type
    kernel::ext_oneapi_get_info<
        ext::oneapi::experimental::info::kernel_queue_specific::num_sub_groups>(
        queue Queue, const range<1> &) const;

#define __SYCL_PARAM_TRAITS_SPEC(Namespace, DescType, Desc, ReturnT)           \
  template __SYCL_EXPORT ReturnT                                               \
  kernel::ext_oneapi_get_info<Namespace::info::DescType::Desc>(                \
      queue, const range<1> &, size_t) const;                                  \
  template __SYCL_EXPORT ReturnT                                               \
  kernel::ext_oneapi_get_info<Namespace::info::DescType::Desc>(                \
      queue, const range<2> &, size_t) const;                                  \
  template __SYCL_EXPORT ReturnT                                               \
  kernel::ext_oneapi_get_info<Namespace::info::DescType::Desc>(                \
      queue, const range<3> &, size_t) const;
// Not including "ext_oneapi_kernel_queue_specific_traits.def" because not all
// kernel_queue_specific queries require the above-defined get_info interface.
// clang-format off
__SYCL_PARAM_TRAITS_SPEC(ext::oneapi::experimental, kernel_queue_specific, max_num_work_groups, size_t)
// clang-format on
#undef __SYCL_PARAM_TRAITS_SPEC

kernel::kernel(std::shared_ptr<detail::kernel_impl> Impl) : impl(Impl) {}

ur_native_handle_t kernel::getNative() const { return impl->getNative(); }

ur_native_handle_t kernel::getNativeImpl() const { return impl->getNative(); }

#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
// The following query was deprecated since it doesn't include a way to specify
// the invdividual dimensions of the work group. All of the contents of this
// #ifndef block should be removed during the next ABI breaking window.
namespace ext::oneapi::experimental::info::kernel_queue_specific {
struct max_num_work_group_sync {
  using return_type = size_t;
};
} // namespace ext::oneapi::experimental::info::kernel_queue_specific
template <>
struct detail::is_kernel_queue_specific_info_desc<
    ext::oneapi::experimental::info::kernel_queue_specific::
        max_num_work_group_sync> : std::true_type {
  using return_type = ext::oneapi::experimental::info::kernel_queue_specific::
      max_num_work_group_sync::return_type;
};
template <>
__SYCL2020_DEPRECATED(
    "The 'max_num_work_group_sync' query is deprecated. See "
    "'sycl_ext_oneapi_launch_queries' for the new 'max_num_work_groups' query.")
__SYCL_EXPORT typename ext::oneapi::experimental::info::kernel_queue_specific::
    max_num_work_group_sync::return_type kernel::ext_oneapi_get_info<
        ext::oneapi::experimental::info::kernel_queue_specific::
            max_num_work_group_sync>(queue Queue, const range<3> &WorkGroupSize,
                                     size_t DynamicLocalMemorySize) const {
  return ext_oneapi_get_info<ext::oneapi::experimental::info::
                                 kernel_queue_specific::max_num_work_groups>(
      std::move(Queue), WorkGroupSize, DynamicLocalMemorySize);
}
template <>
__SYCL2020_DEPRECATED(
    "The 'max_num_work_group_sync' query is deprecated. See "
    "'sycl_ext_oneapi_launch_queries' for the new 'max_num_work_groups' query.")
__SYCL_EXPORT typename ext::oneapi::experimental::info::kernel_queue_specific::
    max_num_work_group_sync::return_type kernel::ext_oneapi_get_info<
        ext::oneapi::experimental::info::kernel_queue_specific::
            max_num_work_group_sync>(queue Queue) const {
  auto Device = Queue.get_device();
  const auto MaxWorkGroupSize =
      get_info<info::kernel_device_specific::work_group_size>(Device);
  const sycl::range<3> WorkGroupSize{MaxWorkGroupSize, 1, 1};
  return ext_oneapi_get_info<ext::oneapi::experimental::info::
                                 kernel_queue_specific::max_num_work_groups>(
      std::move(Queue), WorkGroupSize,
      /* DynamicLocalMemorySize */ 0);
}
#endif

} // namespace _V1
} // namespace sycl
