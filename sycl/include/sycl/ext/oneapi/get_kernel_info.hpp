//==----- get_kernel_info.hpp --- SYCL get_kernel_info extension -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------------------------------------------------------------===//

#pragma once
#include <sycl/aspects.hpp>
#include <sycl/context.hpp>
#include <sycl/detail/export.hpp>
#include <sycl/detail/get_device_kernel_info.hpp>
#include <sycl/detail/impl_utils.hpp>
#include <sycl/detail/info_desc_helpers.hpp>
#include <sycl/device.hpp>
#include <sycl/exception.hpp>
#include <sycl/ext/oneapi/experimental/free_function_traits.hpp>
#include <sycl/info/info_desc.hpp>
#include <sycl/kernel_bundle.hpp>
#include <sycl/kernel_bundle_enums.hpp>
#include <sycl/queue.hpp>
#include <sycl/range.hpp>

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace sycl {
inline namespace _V1 {

template <bundle_state State> class kernel_bundle;

template <typename KernelName, bundle_state State>
kernel_bundle<State> get_kernel_bundle(const context &,
                                       const std::vector<device> &);

namespace ext::oneapi {
namespace detail {

// Per-Param UR info code + dispatch category, built from the same .def files
// the runtime uses. We can't include <detail/ur_info_code.hpp> here because it
// lives outside the public include path, so this local trait mirrors it and
// also encodes the category via the enum type of `value`.
template <typename Param> struct FastKernelInfoCode;

#define __SYCL_PARAM_TRAITS_SPEC(DescType, Desc, ReturnT, UrCode)              \
  template <> struct FastKernelInfoCode<sycl::info::DescType::Desc> {          \
    static constexpr auto value = UrCode;                                      \
  };
#include <sycl/info/kernel_device_specific_traits.def>
#undef __SYCL_PARAM_TRAITS_SPEC

#define __SYCL_PARAM_TRAITS_SPEC(Namespace, DescType, Desc, ReturnT, UrCode)   \
  template <> struct FastKernelInfoCode<Namespace::info::DescType::Desc> {     \
    static constexpr auto value = UrCode;                                      \
  };
#include <sycl/info/ext_intel_kernel_info_traits.def>
#undef __SYCL_PARAM_TRAITS_SPEC

// Dispatch a device-specific info query through the fast kernel cache,
// skipping the sycl::kernel wrapper. Param::return_type is returned by value.
// The dispatch category is chosen from the enum type of the UR info code:
// ur_kernel_sub_group_info_t / ur_kernel_info_t / ur_kernel_group_info_t.
template <typename Param>
inline typename Param::return_type
fastGetKernelDeviceSpecificInfo(sycl::detail::context_impl &CtxImpl,
                                sycl::detail::device_impl &DevImpl,
                                sycl::detail::DeviceKernelInfo &DKI) {
  using ReturnT = typename Param::return_type;
  using CodeT = decltype(FastKernelInfoCode<Param>::value);
  constexpr auto Code = FastKernelInfoCode<Param>::value;

  if constexpr (std::is_same_v<Param, ext::intel::info::kernel_device_specific::
                                          spill_memory_size>) {
    return sycl::detail::queryCachedKernelSpillMemSize(CtxImpl, DevImpl, DKI);
  } else if constexpr (std::is_same_v<CodeT, ur_kernel_sub_group_info_t>) {
    ReturnT Result{};
    sycl::detail::queryCachedKernelSubGroupInfo(CtxImpl, DevImpl, DKI, Code,
                                                sizeof(ReturnT), &Result);
    return Result;
  } else if constexpr (std::is_same_v<CodeT, ur_kernel_info_t>) {
    ReturnT Result{};
    sycl::detail::queryCachedKernelUrKernelInfo(CtxImpl, DevImpl, DKI, Code,
                                                sizeof(ReturnT), &Result);
    return Result;
  } else if constexpr (std::is_same_v<ReturnT, sycl::range<3>>) {
    size_t Raw[3] = {0, 0, 0};
    sycl::detail::queryCachedKernelGroupInfo(CtxImpl, DevImpl, DKI, Code,
                                             sizeof(Raw), Raw);
    return sycl::range<3>(Raw[0], Raw[1], Raw[2]);
  } else {
    ReturnT Result{};
    sycl::detail::queryCachedKernelGroupInfo(CtxImpl, DevImpl, DKI, Code,
                                             sizeof(ReturnT), &Result);
    return Result;
  }
}

// Spec-mandated pre-query checks that kernel_impl::get_info would do for us,
// but which we must now replicate since we bypass the wrapper.
template <typename Param>
inline void validateDeviceSpecificQuery(const device &Dev) {
  if constexpr (std::is_same_v<
                    Param,
                    sycl::info::kernel_device_specific::global_work_size>) {
    if (Dev.get_info<sycl::info::device::device_type>() !=
        sycl::info::device_type::custom)
      throw exception(
          sycl::make_error_code(errc::invalid),
          "info::kernel_device_specific::global_work_size descriptor may only "
          "be used if the device type is device_type::custom or if the "
          "kernel is a built-in kernel.");
  } else if constexpr (std::is_same_v<Param,
                                      ext::intel::info::kernel_device_specific::
                                          spill_memory_size>) {
    if (!Dev.has(aspect::ext_intel_spill_memory_size))
      throw exception(
          sycl::make_error_code(errc::feature_not_supported),
          "This device does not have the ext_intel_spill_memory_size aspect");
  }
}

} // namespace detail

// Non-device-specific query - keep inline fallback via kernel_bundle.
template <typename KernelName, typename Param>
typename sycl::detail::is_kernel_info_desc<Param>::return_type
get_kernel_info(const context &Ctx) {
  auto Bundle =
      sycl::get_kernel_bundle<KernelName, sycl::bundle_state::executable>(Ctx);
  return Bundle.template get_kernel<KernelName>().template get_info<Param>();
}

// Device-specific query - uses fast kernel cache (O(1) lookup) with direct
// UR dispatch, skipping the sycl::kernel wrapper.
template <typename KernelName, typename Param>
typename sycl::detail::is_kernel_device_specific_info_desc<Param>::return_type
get_kernel_info(const context &Ctx, const device &Dev) {
  detail::validateDeviceSpecificQuery<Param>(Dev);
  auto &CtxImpl = *sycl::detail::getSyclObjImpl(Ctx);
  auto &DevImpl = *sycl::detail::getSyclObjImpl(Dev);
  sycl::detail::DeviceKernelInfo &DKI =
      sycl::detail::getDeviceKernelInfo<KernelName>();
  return detail::fastGetKernelDeviceSpecificInfo<Param>(CtxImpl, DevImpl, DKI);
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
  ext::oneapi::detail::validateDeviceSpecificQuery<Param>(dev);
  auto &CtxImpl = *sycl::detail::getSyclObjImpl(ctxt);
  auto &DevImpl = *sycl::detail::getSyclObjImpl(dev);
  sycl::detail::DeviceKernelInfo &DKI =
      sycl::detail::getDeviceKernelInfo<Func>();
  return ext::oneapi::detail::fastGetKernelDeviceSpecificInfo<Param>(
      CtxImpl, DevImpl, DKI);
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
