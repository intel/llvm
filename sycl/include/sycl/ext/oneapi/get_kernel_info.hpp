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
#include <sycl/detail/kernel_cache.hpp>
#include <sycl/device.hpp>
#include <sycl/kernel_bundle.hpp>
#include <sycl/kernel_bundle_enums.hpp>
#include <sycl/queue.hpp>

#include <vector>

namespace sycl {
inline namespace _V1 {

template <bundle_state State> class kernel_bundle;

template <typename KernelName, bundle_state State>
kernel_bundle<State> get_kernel_bundle(const context &,
                                       const std::vector<device> &);

namespace ext::oneapi {

// Non-device-specific query - keep inline fallback
template <typename KernelName, typename Param>
typename sycl::detail::is_kernel_info_desc<Param>::return_type
get_kernel_info(const context &Ctx) {
  auto Bundle =
      sycl::get_kernel_bundle<KernelName, sycl::bundle_state::executable>(Ctx);
  return Bundle.template get_kernel<KernelName>().template get_info<Param>();
}

// Device-specific query - uses fast kernel cache (O(1) lookup)
template <typename KernelName, typename Param>
typename sycl::detail::is_kernel_device_specific_info_desc<Param>::return_type
get_kernel_info(const context &Ctx, const device &Dev) {
  // Get implementation objects
  auto &CtxImpl = *sycl::detail::getSyclObjImpl(Ctx);
  auto &DevImpl = *sycl::detail::getSyclObjImpl(Dev);

  // Get kernel metadata
  sycl::detail::DeviceKernelInfo &DKI =
      sycl::detail::getDeviceKernelInfo<KernelName>();

  // Dispatch to appropriate fast cache function based on query type
  if constexpr (std::is_same_v<
                    Param,
                    sycl::info::kernel_device_specific::work_group_size>) {
    return sycl::detail::getKernelWorkGroupSize(CtxImpl, DevImpl, DKI);
  } else if constexpr (std::is_same_v<Param,
                                      sycl::info::kernel_device_specific::
                                          compile_work_group_size>) {
    return sycl::detail::getKernelCompileWorkGroupSize(CtxImpl, DevImpl, DKI);
  } else if constexpr (std::is_same_v<Param,
                                      sycl::info::kernel_device_specific::
                                          preferred_work_group_size_multiple>) {
    return sycl::detail::getKernelPreferredWorkGroupSizeMultiple(CtxImpl,
                                                                 DevImpl, DKI);
  } else if constexpr (std::is_same_v<Param,
                                      sycl::info::kernel_device_specific::
                                          private_mem_size>) {
    return sycl::detail::getKernelPrivateMemSize(CtxImpl, DevImpl, DKI);
  } else if constexpr (std::is_same_v<Param,
                                      sycl::info::kernel_device_specific::
                                          max_sub_group_size>) {
    return sycl::detail::getKernelMaxSubGroupSize(CtxImpl, DevImpl, DKI);
  } else if constexpr (std::is_same_v<Param,
                                      sycl::info::kernel_device_specific::
                                          compile_num_sub_groups>) {
    return sycl::detail::getKernelCompileNumSubGroups(CtxImpl, DevImpl, DKI);
  } else if constexpr (std::is_same_v<Param,
                                      sycl::info::kernel_device_specific::
                                          compile_sub_group_size>) {
    return sycl::detail::getKernelCompileSubGroupSize(CtxImpl, DevImpl, DKI);
  } else {
    // Fallback for any other queries
    auto Bundle =
        sycl::get_kernel_bundle<KernelName, sycl::bundle_state::executable>(
            Ctx, {Dev});
    return Bundle.template get_kernel<KernelName>().template get_info<Param>(
        Dev);
  }
}

// Queue variant - delegates to context+device
template <typename KernelName, typename Param>
typename sycl::detail::is_kernel_device_specific_info_desc<Param>::return_type
get_kernel_info(const queue &Q) {
  return get_kernel_info<KernelName, Param>(Q.get_context(), Q.get_device());
}

// For free functions.
namespace experimental {

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
  // TODO: Optimize free function kernel path
  // Free functions have compiler-generated kernel types that are harder to
  // access For now, use the existing get_kernel_bundle path
  auto Bundle = sycl::ext::oneapi::experimental::get_kernel_bundle<
      Func, sycl::bundle_state::executable>(ctxt, {dev});
  return Bundle.template ext_oneapi_get_kernel<Func>().template get_info<Param>(
      dev);
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

} // namespace experimental
} // namespace ext::oneapi
} // namespace _V1
} // namespace sycl
