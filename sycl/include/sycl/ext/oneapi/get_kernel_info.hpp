//==----- get_kernel_info.hpp --- SYCL get_kernel_info extension -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------------------------------------------------------------===//

#pragma once
#include <sycl/context.hpp>
#include <sycl/detail/export.hpp>
#include <sycl/detail/get_device_kernel_info.hpp>
#include <sycl/detail/get_kernel_info_impl.hpp>
#include <sycl/detail/impl_utils.hpp>
#include <sycl/device.hpp>
#include <sycl/ext/oneapi/experimental/free_function_traits.hpp>
#include <sycl/info/kernel.hpp>
#include <sycl/kernel_bundle.hpp>
#include <sycl/kernel_bundle_enums.hpp>
#include <sycl/queue.hpp>
#include <sycl/kernel.hpp>

#include <vector>

namespace sycl {
inline namespace _V1 {

template <bundle_state State> class kernel_bundle;

template <typename KernelName, bundle_state State>
kernel_bundle<State> get_kernel_bundle(const context &,
                                       const std::vector<device> &);

namespace ext::oneapi {

// Non-device-specific query - keep inline fallback via kernel_bundle.
template <typename KernelName, typename Param>
typename sycl::detail::is_kernel_info_desc<Param>::return_type
get_kernel_info(const context &Ctx) {
  auto Bundle =
      sycl::get_kernel_bundle<KernelName, sycl::bundle_state::executable>(Ctx);
  return Bundle.template get_kernel<KernelName>().template get_info<Param>();
}

// Device-specific query - uses fast kernel cache (O(1) lookup) via the
// ABI-exported get_kernel_info_impl, which performs spec-mandated validation
// and dispatches to get_kernel_device_specific_info inside the library.
template <typename KernelName, typename Param>
typename sycl::detail::is_kernel_device_specific_info_desc<Param>::return_type
get_kernel_info(const context &Ctx, const device &Dev) {
  static_assert(sycl::detail::is_kernel_device_specific_info_desc<Param>::value,
                "Invalid kernel_device_specific information descriptor");
  auto &CtxImpl = *sycl::detail::getSyclObjImpl(Ctx);
  auto &DevImpl = *sycl::detail::getSyclObjImpl(Dev);
  sycl::detail::DeviceKernelInfo &DKI =
      sycl::detail::getDeviceKernelInfo<KernelName>();
  return sycl::detail::get_kernel_info_impl<Param>(CtxImpl, DevImpl, DKI);
}


// Queue variant - delegates to context+device
template <typename KernelName, typename Param>
typename sycl::detail::is_kernel_device_specific_info_desc<Param>::return_type
get_kernel_info(const queue &Q) {
  return get_kernel_info<KernelName, Param>(Q.get_context(), Q.get_device());
}

namespace experimental {

// Param must be max_num_work_groups_sync.
template <typename KernelName, typename Param,
          typename LaunchProperties = empty_properties_t>
typename std::enable_if_t<
    std::is_same_v<Param, sycl::ext::oneapi::experimental::info::kernel::
                              max_num_work_groups_sync>,
    typename Param::return_type>
get_kernel_info(const context &ctxt, const device &dev, sycl::range<1> r,
                LaunchProperties props = {}, size_t bytes = 0) {
  auto bundle =
      sycl::get_kernel_bundle<KernelName, sycl::bundle_state::executable>(ctxt);
  sycl::kernel k = bundle.template get_kernel<KernelName>();
  auto ret = k.ext_oneapi_get_info<Param>(dev, r, props, bytes);
}

// Param must be max_num_work_groups_sync.
template <typename KernelName, typename Param,
          typename LaunchProperties = empty_properties_t>
typename std::enable_if_t<
    std::is_same_v<Param, sycl::ext::oneapi::experimental::info::kernel::
                              max_num_work_groups_sync>,
    typename Param::return_type>
get_kernel_info(const context &ctxt, const device &dev, sycl::range<2> r,
                LaunchProperties props = {}, size_t bytes = 0) {
  auto bundle =
      sycl::get_kernel_bundle<KernelName, sycl::bundle_state::executable>(ctxt);
  sycl::kernel k = bundle.template get_kernel<KernelName>();
  auto ret = k.ext_oneapi_get_info<Param>(dev, r, props, bytes);
}

// Param must be max_num_work_groups_sync.
template <typename KernelName, typename Param,
          typename LaunchProperties = empty_properties_t>
typename std::enable_if_t<
    std::is_same_v<Param, sycl::ext::oneapi::experimental::info::kernel::
                              max_num_work_groups_sync>,
    typename Param::return_type>
get_kernel_info(const context &ctxt, const device &dev, sycl::range<3> r,
                LaunchProperties props = {}, size_t bytes = 0) {
  auto bundle =
      sycl::get_kernel_bundle<KernelName, sycl::bundle_state::executable>(ctxt);
  sycl::kernel k = bundle.template get_kernel<KernelName>();
  auto ret = k.ext_oneapi_get_info<Param>(dev, r, props, bytes);
}

// Param must be max_num_work_groups_sync.
template <typename KernelName, typename Param,
          typename LaunchProperties = empty_properties_t>
typename std::enable_if_t<
    std::is_same_v<Param, sycl::ext::oneapi::experimental::info::kernel::
                              max_num_work_groups_sync>,
    typename Param::return_type>

get_kernel_info(const queue &q, sycl::range<1> r,

                LaunchProperties props = {}, size_t bytes = 0) {
  sycl::context ctxt = q.get_context();
  sycl::device dev = q.get_device();
  auto bundle =
      sycl::get_kernel_bundle<KernelName, sycl::bundle_state::executable>(ctxt);
  sycl::kernel k = bundle.template get_kernel<KernelName>();
  auto ret = k.ext_oneapi_get_info<Param>(dev, r, props, bytes);
}

// Param must be max_num_work_groups_sync.
template <typename KernelName, typename Param,
          typename LaunchProperties = empty_properties_t>
typename std::enable_if_t<
    std::is_same_v<Param, sycl::ext::oneapi::experimental::info::kernel::
                              max_num_work_groups_sync>,
    typename Param::return_type>

get_kernel_info(const queue &q, sycl::range<2> r, LaunchProperties props = {},
                size_t bytes = 0) {
  sycl::context ctxt = q.get_context();
  sycl::device dev = q.get_device();
  auto bundle =
      sycl::get_kernel_bundle<KernelName, sycl::bundle_state::executable>(ctxt);
  sycl::kernel k = bundle.template get_kernel<KernelName>();
  auto ret = k.ext_oneapi_get_info<Param>(dev, r, props, bytes);
}

// Param must be max_num_work_groups_sync.
template <typename KernelName, typename Param,
          typename LaunchProperties = empty_properties_t>
typename std::enable_if_t<
    std::is_same_v<Param, sycl::ext::oneapi::experimental::info::kernel::
                              max_num_work_groups_sync>,
    typename Param::return_type>

get_kernel_info(const queue &q, sycl::range<3> r, LaunchProperties props = {},
                size_t bytes = 0) {
  sycl::context ctxt = q.get_context();
  sycl::device dev = q.get_device();
  auto bundle =
      sycl::get_kernel_bundle<KernelName, sycl::bundle_state::executable>(ctxt);
  sycl::kernel k = bundle.template get_kernel<KernelName>();
  auto ret = k.ext_oneapi_get_info<Param>(dev, r, props, bytes);
}

// For free functions.
template <auto *Func, typename Param>
std::enable_if_t<ext::oneapi::experimental::is_kernel_v<Func>,
                 typename sycl::detail::is_kernel_info_desc<Param>::return_type>
get_kernel_info(const context &ctxt) {
  auto Bundle = sycl::ext::oneapi::experimental::get_kernel_bundle<
      Func, sycl::bundle_state::executable>(ctxt);
  return Bundle.template ext_oneapi_get_kernel<Func>()
      .template get_info<Param>();
}

template <auto *Func, typename Param>
std::enable_if_t<ext::oneapi::experimental::is_kernel_v<Func>,
                 typename sycl::detail::is_kernel_device_specific_info_desc<
                     Param>::return_type>
get_kernel_info(const context &ctxt, const device &dev) {
  static_assert(sycl::detail::is_kernel_device_specific_info_desc<Param>::value,
                "Invalid kernel_device_specific information descriptor");
  auto &CtxImpl = *sycl::detail::getSyclObjImpl(ctxt);
  auto &DevImpl = *sycl::detail::getSyclObjImpl(dev);
  sycl::detail::DeviceKernelInfo &DKI =
      sycl::detail::getDeviceKernelInfo<Func>();
  return sycl::detail::get_kernel_info_impl<Param>(CtxImpl, DevImpl, DKI);
}

template <auto *Func, typename Param>
std::enable_if_t<ext::oneapi::experimental::is_kernel_v<Func>,
                 typename sycl::detail::is_kernel_device_specific_info_desc<
                     Param>::return_type>
get_kernel_info(const queue &q) {
  return get_kernel_info<Func, Param>(q.get_context(), q.get_device());
}

template <auto *Func, typename Param>
std::enable_if_t<ext::oneapi::experimental::is_kernel_v<Func> &&
                     std::is_same_v<Param, sycl::info::kernel::num_args>,
                 typename sycl::detail::is_kernel_info_desc<Param>::return_type>
get_kernel_info(const context &, const device &) {
  return sycl::detail::FreeFunctionInfoData<Func>::getNumParams();
}

// Param must be equal to max_num_work_groups_sync.
template <auto *Func, typename Param,
          typename LaunchProperties = empty_properties_t>
std::enable_if_t<
    sycl::ext::oneapi::experimental::is_kernel_v<Func> &&
        std::is_same_v<Param, sycl::ext::oneapi::experimental::info::kernel::
                                  max_num_work_groups_sync>,
    typename Param::return_type>
get_kernel_info(const context &ctxt, const device &dev, sycl::range<1> r,
                LaunchProperties props = {}, size_t bytes = 0) {
  auto bundle = sycl::ext::oneapi::experimental::get_kernel_bundle<
      Func, sycl::bundle_state::executable>(ctxt);
  sycl::kernel k = bundle.template ext_oneapi_get_kernel<Func>();
  auto ret = k.ext_oneapi_get_info<Param>(dev, r, props, bytes);
}

// Param must be equal to max_num_work_groups_sync.
template <auto *Func, typename Param,
          typename LaunchProperties = empty_properties_t>
std::enable_if_t<
    sycl::ext::oneapi::experimental::is_kernel_v<Func> &&
        std::is_same_v<Param, sycl::ext::oneapi::experimental::info::kernel::
                                  max_num_work_groups_sync>,
    typename Param::return_type>

get_kernel_info(const context &ctxt, const device &dev, sycl::range<2> r,
                LaunchProperties props = {}, size_t bytes = 0) {
  auto bundle = sycl::ext::oneapi::experimental::get_kernel_bundle<
      Func, sycl::bundle_state::executable>(ctxt);
  sycl::kernel k = bundle.template ext_oneapi_get_kernel<Func>();
  auto ret = k.ext_oneapi_get_info<Param>(dev, r, props, bytes);
}

// Param must be equal to max_num_work_groups_sync.
template <auto *Func, typename Param,
          typename LaunchProperties = empty_properties_t>
std::enable_if_t<
    sycl::ext::oneapi::experimental::is_kernel_v<Func> &&
        std::is_same_v<Param, sycl::ext::oneapi::experimental::info::kernel::
                                  max_num_work_groups_sync>,
    typename Param::return_type>

get_kernel_info(const context &ctxt, const device &dev, sycl::range<3> r,
                LaunchProperties props = {}, size_t bytes = 0) {
  auto bundle = sycl::ext::oneapi::experimental::get_kernel_bundle<
      Func, sycl::bundle_state::executable>(ctxt);
  sycl::kernel k = bundle.template ext_oneapi_get_kernel<Func>();
  auto ret = k.ext_oneapi_get_info<Param>(dev, r, props, bytes);
}

// Param must be equal to max_num_work_groups_sync.
template <auto *Func, typename Param,
          typename LaunchProperties = empty_properties_t>
std::enable_if_t<
    sycl::ext::oneapi::experimental::is_kernel_v<Func> &&
        std::is_same_v<Param, sycl::ext::oneapi::experimental::info::kernel::
                                  max_num_work_groups_sync>,
    typename Param::return_type>
get_kernel_info(const queue &q, sycl::range<1> r, LaunchProperties props = {},
                size_t bytes = 0) {
  sycl::context ctxt = q.get_context();
  sycl::device dev = q.get_device();
  auto bundle = sycl::ext::oneapi::experimental::get_kernel_bundle<
      Func, sycl::bundle_state::executable>(ctxt);
  sycl::kernel k = bundle.template ext_oneapi_get_kernel<Func>();
  auto ret = k.ext_oneapi_get_info<Param>(dev, r, props, bytes);
}

// Param must be equal to max_num_work_groups_sync.
template <auto *Func, typename Param,
          typename LaunchProperties = empty_properties_t>
std::enable_if_t<
    sycl::ext::oneapi::experimental::is_kernel_v<Func> &&
        std::is_same_v<Param, sycl::ext::oneapi::experimental::info::kernel::
                                  max_num_work_groups_sync>,
    typename Param::return_type>
get_kernel_info(const queue &q, sycl::range<2> r, LaunchProperties props = {},
                size_t bytes = 0) {
  sycl::context ctxt = q.get_context();
  sycl::device dev = q.get_device();
  auto bundle = sycl::ext::oneapi::experimental::get_kernel_bundle<
      Func, sycl::bundle_state::executable>(ctxt);
  sycl::kernel k = bundle.template ext_oneapi_get_kernel<Func>();
  auto ret = k.ext_oneapi_get_info<Param>(dev, r, props, bytes);
}

// Param must be equal to max_num_work_groups_sync.
template <auto *Func, typename Param,
          typename LaunchProperties = empty_properties_t>
std::enable_if_t<
    sycl::ext::oneapi::experimental::is_kernel_v<Func> &&
        std::is_same_v<Param, sycl::ext::oneapi::experimental::info::kernel::
                                  max_num_work_groups_sync>,
    typename Param::return_type>
get_kernel_info(const queue &q, sycl::range<3> r, LaunchProperties props = {},
                size_t bytes = 0) {
  sycl::context ctxt = q.get_context();
  sycl::device dev = q.get_device();
  auto bundle = sycl::ext::oneapi::experimental::get_kernel_bundle<
      Func, sycl::bundle_state::executable>(ctxt);
  sycl::kernel k = bundle.template ext_oneapi_get_kernel<Func>();
  auto ret = k.ext_oneapi_get_info<Param>(dev, r, props, bytes);
}

} // namespace experimental
} // namespace ext::oneapi
} // namespace _V1
} // namespace sycl
