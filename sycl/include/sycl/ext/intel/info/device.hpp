//==-- device.hpp - Intel device extension info traits ---------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/device_info_types.hpp>
#include <sycl/detail/info_desc_traits.hpp>
#include <unified-runtime/ur_api.h>

#include <cstdint>
#include <string>
#include <vector>

namespace sycl {
inline namespace _V1 {
namespace ext::intel {

enum class throttle_reason {
  power_cap,
  current_limit,
  thermal_limit,
  psu_alert,
  sw_range,
  hw_range,
  other
};

namespace info::device {

#define __SYCL_INTEL_DEVICE_TRAIT(NAME, RETURN_T, UR_CODE)                     \
  struct NAME {                                                                \
    using return_type = RETURN_T;                                              \
    using info_class = sycl::detail::info_class::device;                       \
    static constexpr ur_device_info_t ur_code = UR_CODE;                       \
  };

__SYCL_INTEL_DEVICE_TRAIT(device_id, uint32_t, UR_DEVICE_INFO_DEVICE_ID)
__SYCL_INTEL_DEVICE_TRAIT(pci_address, std::string, UR_DEVICE_INFO_PCI_ADDRESS)
__SYCL_INTEL_DEVICE_TRAIT(gpu_eu_count, uint32_t, UR_DEVICE_INFO_GPU_EU_COUNT)
__SYCL_INTEL_DEVICE_TRAIT(gpu_eu_simd_width, uint32_t,
                          UR_DEVICE_INFO_GPU_EU_SIMD_WIDTH)
__SYCL_INTEL_DEVICE_TRAIT(gpu_slices, uint32_t, UR_DEVICE_INFO_GPU_EU_SLICES)
__SYCL_INTEL_DEVICE_TRAIT(gpu_subslices_per_slice, uint32_t,
                          UR_DEVICE_INFO_GPU_SUBSLICES_PER_SLICE)
__SYCL_INTEL_DEVICE_TRAIT(gpu_eu_count_per_subslice, uint32_t,
                          UR_DEVICE_INFO_GPU_EU_COUNT_PER_SUBSLICE)
__SYCL_INTEL_DEVICE_TRAIT(gpu_hw_threads_per_eu, uint32_t,
                          UR_DEVICE_INFO_GPU_HW_THREADS_PER_EU)
__SYCL_INTEL_DEVICE_TRAIT(max_mem_bandwidth, uint64_t,
                          UR_DEVICE_INFO_MAX_MEMORY_BANDWIDTH)
__SYCL_INTEL_DEVICE_TRAIT(uuid, sycl::detail::uuid_type, UR_DEVICE_INFO_UUID)
__SYCL_INTEL_DEVICE_TRAIT(free_memory, uint64_t,
                          UR_DEVICE_INFO_GLOBAL_MEM_FREE)
__SYCL_INTEL_DEVICE_TRAIT(memory_clock_rate, uint32_t,
                          UR_DEVICE_INFO_MEMORY_CLOCK_RATE)
__SYCL_INTEL_DEVICE_TRAIT(memory_bus_width, uint32_t,
                          UR_DEVICE_INFO_MEMORY_BUS_WIDTH)
__SYCL_INTEL_DEVICE_TRAIT(max_compute_queue_indices, int32_t,
                          UR_DEVICE_INFO_MAX_COMPUTE_QUEUE_INDICES)
__SYCL_INTEL_DEVICE_TRAIT(current_clock_throttle_reasons,
                          std::vector<sycl::ext::intel::throttle_reason>,
                          UR_DEVICE_INFO_CURRENT_CLOCK_THROTTLE_REASONS)
__SYCL_INTEL_DEVICE_TRAIT(fan_speed, int32_t, UR_DEVICE_INFO_FAN_SPEED)
__SYCL_INTEL_DEVICE_TRAIT(min_power_limit, int32_t,
                          UR_DEVICE_INFO_MIN_POWER_LIMIT)
__SYCL_INTEL_DEVICE_TRAIT(max_power_limit, int32_t,
                          UR_DEVICE_INFO_MAX_POWER_LIMIT)
__SYCL_INTEL_DEVICE_TRAIT(xe_stack_count, uint32_t,
                          UR_DEVICE_INFO_XE_STACK_COUNT)
__SYCL_INTEL_DEVICE_TRAIT(xe_regions_per_stack, uint32_t,
                          UR_DEVICE_INFO_XE_REGIONS_PER_STACK)
__SYCL_INTEL_DEVICE_TRAIT(xe_clusters_per_region, uint32_t,
                          UR_DEVICE_INFO_XE_CLUSTERS_PER_REGION)
__SYCL_INTEL_DEVICE_TRAIT(xe_cores_per_cluster, uint32_t,
                          UR_DEVICE_INFO_XE_CORES_PER_CLUSTER)
__SYCL_INTEL_DEVICE_TRAIT(eus_per_xe_core, uint32_t,
                          UR_DEVICE_INFO_EUS_PER_XE_CORE)
__SYCL_INTEL_DEVICE_TRAIT(max_lanes_per_hw_thread, uint32_t,
                          UR_DEVICE_INFO_MAX_LANES_PER_HW_THREAD)

// __SYCL_TRAIT_HANDLED_IN_RT: dispatched via switch in device_impl.hpp,
// not via UR enum lookup. ur_code value is unused for these traits.
struct luid {
  using return_type = sycl::detail::luid_type;
  using info_class = sycl::detail::info_class::device;
  static constexpr ur_device_info_t ur_code = ur_device_info_t(0);
};

struct node_mask {
  using return_type = uint32_t;
  using info_class = sycl::detail::info_class::device;
  static constexpr ur_device_info_t ur_code = ur_device_info_t(0);
};

#undef __SYCL_INTEL_DEVICE_TRAIT

} // namespace info::device

namespace esimd::info::device {
struct has_2d_block_io_support {
  using return_type = bool;
  using info_class = sycl::detail::info_class::device;
  static constexpr ur_device_info_t ur_code =
      UR_DEVICE_INFO_2D_BLOCK_ARRAY_CAPABILITIES_EXP;
};
} // namespace esimd::info::device

} // namespace ext::intel
} // namespace _V1
} // namespace sycl
