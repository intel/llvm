//==-- get_kernel_info.cpp - SYCL get_kernel_info implementation ----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SYCL-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/ext/oneapi/get_kernel_info.hpp>

// Internal headers - OK to include in .cpp file
#include <detail/context_impl.hpp>
#include <detail/device_impl.hpp>
#include <detail/kernel_impl.hpp>
#include <detail/program_manager/program_manager.hpp>
#include <detail/ur_info_code.hpp>
#include <sycl/context.hpp>
#include <sycl/detail/get_device_kernel_info.hpp>
#include <sycl/device.hpp>
#include <sycl/queue.hpp>

// Specialized by KernelName, directly get the kernel info. Otherwise, fallback
// to get_kernel_bundle, which is O(N).

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi {
namespace detail {

// Optimized implementation used by public API
// Uses getOrCreateKernel for O(1) cache lookup instead of O(N) device image
// search
template <typename KernelName, typename Param>
typename sycl::detail::is_kernel_device_specific_info_desc<Param>::return_type
get_kernel_info_impl(const context &Ctx, const device &Dev) {
  // Get implementation objects
  auto &CtxImpl = *sycl::detail::getSyclObjImpl(Ctx);
  auto &DevImpl = *sycl::detail::getSyclObjImpl(Dev);

  // Get kernel info (contains name and other metadata)
  sycl::detail::DeviceKernelInfo &DKI =
      sycl::detail::getDeviceKernelInfo<KernelName>();

  // Empty NDRDesc is fine for info queries
  sycl::detail::NDRDescT NDRDesc{};

  // Use the fast kernel cache (same path as kernel submission!)
  sycl::detail::FastKernelCacheValPtr KernelCacheVal =
      sycl::detail::ProgramManager::getInstance().getOrCreateKernel(
          CtxImpl, DevImpl, DKI, NDRDesc);

  ur_kernel_handle_t Kernel = KernelCacheVal->MKernelHandle;
  ur_device_handle_t Device = DevImpl.getHandleRef();
  sycl::detail::adapter_impl &Adapter = CtxImpl.getAdapter();

  // Query info directly from UR kernel based on Param type
  if constexpr (std::is_same_v<
                    Param,
                    sycl::info::kernel_device_specific::work_group_size>) {
    size_t result;
    Adapter.call<sycl::detail::UrApiKind::urKernelGetGroupInfo>(
        Kernel, Device, UR_KERNEL_GROUP_INFO_WORK_GROUP_SIZE, sizeof(size_t),
        &result, nullptr);
    return result;
  } else if constexpr (std::is_same_v<Param,
                                      sycl::info::kernel_device_specific::
                                          compile_work_group_size>) {
    size_t result[3];
    Adapter.call<sycl::detail::UrApiKind::urKernelGetGroupInfo>(
        Kernel, Device, UR_KERNEL_GROUP_INFO_COMPILE_WORK_GROUP_SIZE,
        sizeof(result), result, nullptr);
    return range<3>(result[0], result[1], result[2]);
  } else if constexpr (std::is_same_v<Param,
                                      sycl::info::kernel_device_specific::
                                          preferred_work_group_size_multiple>) {
    size_t result;
    Adapter.call<sycl::detail::UrApiKind::urKernelGetGroupInfo>(
        Kernel, Device, UR_KERNEL_GROUP_INFO_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
        sizeof(size_t), &result, nullptr);
    return result;
  } else if constexpr (std::is_same_v<Param,
                                      sycl::info::kernel_device_specific::
                                          private_mem_size>) {
    size_t result;
    Adapter.call<sycl::detail::UrApiKind::urKernelGetGroupInfo>(
        Kernel, Device, UR_KERNEL_GROUP_INFO_PRIVATE_MEM_SIZE, sizeof(size_t),
        &result, nullptr);
    return result;
  } else if constexpr (std::is_same_v<Param,
                                      sycl::info::kernel_device_specific::
                                          max_sub_group_size>) {
    uint32_t result;
    Adapter.call<sycl::detail::UrApiKind::urKernelGetSubGroupInfo>(
        Kernel, Device, UR_KERNEL_SUB_GROUP_INFO_MAX_SUB_GROUP_SIZE,
        sizeof(uint32_t), &result, nullptr);
    return result;
  } else if constexpr (std::is_same_v<Param,
                                      sycl::info::kernel_device_specific::
                                          compile_num_sub_groups>) {
    uint32_t result;
    Adapter.call<sycl::detail::UrApiKind::urKernelGetSubGroupInfo>(
        Kernel, Device, UR_KERNEL_SUB_GROUP_INFO_COMPILE_NUM_SUB_GROUPS,
        sizeof(uint32_t), &result, nullptr);
    return result;
  } else if constexpr (std::is_same_v<Param,
                                      sycl::info::kernel_device_specific::
                                          compile_sub_group_size>) {
    uint32_t result;
    Adapter.call<sycl::detail::UrApiKind::urKernelGetSubGroupInfo>(
        Kernel, Device, UR_KERNEL_SUB_GROUP_INFO_SUB_GROUP_SIZE_INTEL,
        sizeof(uint32_t), &result, nullptr);
    return result;
  } else {
    // For any other query, fall back to the original path
    auto Bundle =
        sycl::get_kernel_bundle<KernelName, sycl::bundle_state::executable>(
            Ctx, {Dev});
    return Bundle.template get_kernel<KernelName>().template get_info<Param>(
        Dev);
  }
}

} // namespace detail
} // namespace ext::oneapi
} // namespace _V1
} // namespace sycl
