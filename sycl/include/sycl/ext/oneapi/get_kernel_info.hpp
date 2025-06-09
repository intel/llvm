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
#include <sycl/detail/info_desc_helpers.hpp>
#include <sycl/device.hpp>
#include <sycl/ext/oneapi/experimental/free_function_traits.hpp>
#include <sycl/kernel_bundle_enums.hpp>
#include <sycl/queue.hpp>
#include <sycl/kernel_bundle.hpp>

#include <vector>

namespace sycl {
inline namespace _V1 {

template <bundle_state State> class kernel_bundle;

template <typename KernelName, bundle_state State>
kernel_bundle<State> get_kernel_bundle(const context &,
                                       const std::vector<device> &);

namespace ext::oneapi {

template <typename KernelName, typename Param>
typename sycl::detail::is_kernel_info_desc<Param>::return_type
get_kernel_info(const context &Ctx) {
  auto Bundle =
      sycl::get_kernel_bundle<KernelName, sycl::bundle_state::executable>(Ctx);
  return Bundle.template get_kernel<KernelName>().template get_info<Param>();
}

template <typename KernelName, typename Param>
typename sycl::detail::is_kernel_device_specific_info_desc<Param>::return_type
get_kernel_info(const context &Ctx, const device &Dev) {
  auto Bundle =
      sycl::get_kernel_bundle<KernelName, sycl::bundle_state::executable>(Ctx);
  return Bundle.template get_kernel<KernelName>().template get_info<Param>(Dev);
}

template <typename KernelName, typename Param>
typename sycl::detail::is_kernel_device_specific_info_desc<Param>::return_type
get_kernel_info(const queue &Q) {
  auto Bundle =
      sycl::get_kernel_bundle<KernelName, sycl::bundle_state::executable>(
          Q.get_context());
  return Bundle.template get_kernel<KernelName>().template get_info<Param>(
      Q.get_device());
}

} // namespace ext::oneapi
} // namespace _V1
} // namespace sycl

// For free functions.

namespace sycl::ext::oneapi::experimental {

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
  auto Bundle = sycl::get_kernel_bundle<sycl::bundle_state::executable>(
      ctxt, std::vector<sycl::device>{dev});
  auto kid = sycl::ext::oneapi::experimental::get_kernel_id<Func>();
  return Bundle.get_kernel(kid).template get_info<Param>(dev);
}

template <auto *Func, typename Param>
std::enable_if_t<ext::oneapi::experimental::is_kernel_v<Func>,
                 typename sycl::detail::is_kernel_device_specific_info_desc<
                     Param>::return_type>
get_kernel_info(const queue &q) {

  const sycl::device &dev = q.get_device();
  const sycl::context &ctxt = q.get_context();
  auto Bundle = sycl::get_kernel_bundle<sycl::bundle_state::executable>(
      ctxt, std::vector<sycl::device>{dev});
  auto kid = sycl::ext::oneapi::experimental::get_kernel_id<Func>();
  return Bundle.get_kernel(kid).template get_info<Param>(dev);
}
} // namespace sycl::ext::oneapi::experimental
