//==-------------------- get_kernel_info_impl.cpp ----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/detail/get_kernel_info_impl.hpp>
#include <sycl/detail/kernel_desc.hpp>

#include <detail/context_impl.hpp>
#include <detail/device_impl.hpp>
#include <detail/kernel_info.hpp>
#include <detail/program_manager/program_manager.hpp>

#include <algorithm>
#include <string_view>

namespace sycl {
inline namespace _V1 {
namespace detail {

// Replicate the spec-mandated pre-query checks from kernel_impl::get_info.
// Canonical versions live in kernel_impl.hpp (kernel_impl::get_info(const
// device &) for global_work_size, and the spill_memory_size specialization
// immediately below it). If you change those, change these too.
template <typename Param>
static void validateDeviceSpecificQuery(const device_impl &DevImpl,
                                        const DeviceKernelInfo &KernelInfo) {
  if constexpr (std::is_same_v<
                    Param, info::kernel_device_specific::global_work_size>) {
    bool IsCustom = DevImpl.get_info<info::device::device_type>() ==
                    info::device_type::custom;
    if (!IsCustom) {
      std::string_view KernelName{KernelInfo.Name};
      auto BuiltIns = DevImpl.get_info<info::device::built_in_kernel_ids>();
      bool IsBuiltIn =
          std::any_of(BuiltIns.begin(), BuiltIns.end(), [&](kernel_id &Id) {
            return KernelName == Id.get_name();
          });
      if (!IsBuiltIn)
        throw exception(
            sycl::make_error_code(errc::invalid),
            "info::kernel_device_specific::global_work_size descriptor may "
            "only be used if the device type is device_type::custom or if the "
            "kernel is a built-in kernel.");
    }
  } else if constexpr (std::is_same_v<Param,
                                      ext::intel::info::kernel_device_specific::
                                          spill_memory_size>) {
    if (!DevImpl.has(aspect::ext_intel_spill_memory_size))
      throw exception(
          make_error_code(errc::feature_not_supported),
          "This device does not have the ext_intel_spill_memory_size aspect");
  }
}

template <typename Param>
typename Param::return_type get_kernel_info_impl(context_impl &CtxImpl,
                                                 device_impl &DevImpl,
                                                 DeviceKernelInfo &KernelInfo) {
  validateDeviceSpecificQuery<Param>(DevImpl, KernelInfo);
  NDRDescT NDRDesc{};
  FastKernelCacheValPtr KernelCacheVal =
      ProgramManager::getInstance().getOrCreateKernel(CtxImpl, DevImpl,
                                                      KernelInfo, NDRDesc);
  return get_kernel_device_specific_info<Param>(KernelCacheVal->MKernelHandle,
                                                DevImpl.getHandleRef(),
                                                CtxImpl.getAdapter());
}

#define __SYCL_PARAM_TRAITS_SPEC(DescType, Desc, ReturnT, UrCode)              \
  template __SYCL_EXPORT ReturnT get_kernel_info_impl<info::DescType::Desc>(   \
      context_impl &, device_impl &, DeviceKernelInfo &);
#include <sycl/info/kernel_device_specific_traits.def>
#undef __SYCL_PARAM_TRAITS_SPEC

#define __SYCL_PARAM_TRAITS_SPEC(Namespace, DescType, Desc, ReturnT, UrCode)   \
  template __SYCL_EXPORT ReturnT                                               \
  get_kernel_info_impl<Namespace::info::DescType::Desc>(                       \
      context_impl &, device_impl &, DeviceKernelInfo &);
#include <sycl/info/ext_intel_kernel_info_traits.def>
#undef __SYCL_PARAM_TRAITS_SPEC

} // namespace detail
} // namespace _V1
} // namespace sycl
