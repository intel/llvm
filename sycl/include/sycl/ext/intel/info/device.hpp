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

namespace info {

// The IP version of a GPU device, as returned by the `ip_version` information
// descriptor, packs three components, from the most significant bit down:
//
//    31        22 21    14 13       6 5        0
//   +------------+--------+----------+----------+
//   |    major   |  minor | reserved | revision |
//   +------------+--------+----------+----------+
//      10 bits    8 bits    8 bits     6 bits
//
// The reserved bits carry no information.
inline uint32_t get_gpu_ip_version_major(uint32_t IPVersion) {
  return IPVersion >> 22;
}
inline uint32_t get_gpu_ip_version_minor(uint32_t IPVersion) {
  // 0xff is 0b11111111, the 8 bits the minor component occupies.
  return (IPVersion >> 14) & 0xff;
}
inline uint32_t get_gpu_ip_version_revision(uint32_t IPVersion) {
  // 0x3f is 0b111111, the 6 bits the revision component occupies.
  return IPVersion & 0x3f;
}

constexpr uint64_t igca_feature_set_render = (1 << 0);
constexpr uint64_t igca_feature_set_compute = (1 << 1);

struct igca {
  int target;
  uint64_t feature_sets;
};

namespace device {

template <ur_device_info_t UrCode>
using device_traits =
    sycl::detail::ur_traits_base<sycl::detail::info_class::device, UrCode>;
using device_runtime_traits =
    sycl::detail::rt_traits_base<sycl::detail::info_class::device>;

struct device_id : device_traits<UR_DEVICE_INFO_DEVICE_ID> {
  using return_type = uint32_t;
};
struct pci_address : device_traits<UR_DEVICE_INFO_PCI_ADDRESS> {
  using return_type = std::string;
};
struct gpu_eu_count : device_traits<UR_DEVICE_INFO_GPU_EU_COUNT> {
  using return_type = uint32_t;
};
struct gpu_eu_simd_width : device_traits<UR_DEVICE_INFO_GPU_EU_SIMD_WIDTH> {
  using return_type = uint32_t;
};
struct gpu_slices : device_traits<UR_DEVICE_INFO_GPU_EU_SLICES> {
  using return_type = uint32_t;
};
struct gpu_subslices_per_slice
    : device_traits<UR_DEVICE_INFO_GPU_SUBSLICES_PER_SLICE> {
  using return_type = uint32_t;
};
struct gpu_eu_count_per_subslice
    : device_traits<UR_DEVICE_INFO_GPU_EU_COUNT_PER_SUBSLICE> {
  using return_type = uint32_t;
};
struct gpu_hw_threads_per_eu
    : device_traits<UR_DEVICE_INFO_GPU_HW_THREADS_PER_EU> {
  using return_type = uint32_t;
};
struct max_mem_bandwidth : device_traits<UR_DEVICE_INFO_MAX_MEMORY_BANDWIDTH> {
  using return_type = uint64_t;
};
struct uuid : device_traits<UR_DEVICE_INFO_UUID> {
  using return_type = sycl::detail::uuid_type;
};
struct free_memory : device_traits<UR_DEVICE_INFO_GLOBAL_MEM_FREE> {
  using return_type = uint64_t;
};
struct memory_clock_rate : device_traits<UR_DEVICE_INFO_MEMORY_CLOCK_RATE> {
  using return_type = uint32_t;
};
struct memory_bus_width : device_traits<UR_DEVICE_INFO_MEMORY_BUS_WIDTH> {
  using return_type = uint32_t;
};
struct max_compute_queue_indices
    : device_traits<UR_DEVICE_INFO_MAX_COMPUTE_QUEUE_INDICES> {
  using return_type = int32_t;
};
struct current_clock_throttle_reasons
    : device_traits<UR_DEVICE_INFO_CURRENT_CLOCK_THROTTLE_REASONS> {
  using return_type = std::vector<sycl::ext::intel::throttle_reason>;
};
struct fan_speed : device_traits<UR_DEVICE_INFO_FAN_SPEED> {
  using return_type = int32_t;
};
struct min_power_limit : device_traits<UR_DEVICE_INFO_MIN_POWER_LIMIT> {
  using return_type = int32_t;
};
struct max_power_limit : device_traits<UR_DEVICE_INFO_MAX_POWER_LIMIT> {
  using return_type = int32_t;
};
struct xe_stack_count : device_traits<UR_DEVICE_INFO_XE_STACK_COUNT> {
  using return_type = uint32_t;
};
struct xe_regions_per_stack
    : device_traits<UR_DEVICE_INFO_XE_REGIONS_PER_STACK> {
  using return_type = uint32_t;
};
struct xe_clusters_per_region
    : device_traits<UR_DEVICE_INFO_XE_CLUSTERS_PER_REGION> {
  using return_type = uint32_t;
};
struct xe_cores_per_cluster
    : device_traits<UR_DEVICE_INFO_XE_CORES_PER_CLUSTER> {
  using return_type = uint32_t;
};
struct eus_per_xe_core : device_traits<UR_DEVICE_INFO_EUS_PER_XE_CORE> {
  using return_type = uint32_t;
};
struct max_lanes_per_hw_thread
    : device_traits<UR_DEVICE_INFO_MAX_LANES_PER_HW_THREAD> {
  using return_type = uint32_t;
};
struct ip_version : device_traits<UR_DEVICE_INFO_IP_VERSION> {
  using return_type = uint32_t;
};
struct igca : device_traits<UR_DEVICE_INFO_IGCA_TARGET> {
  using return_type = info::igca;
};

// RT-only: dispatched via explicit CASE in device_impl.hpp; no UR enum.
struct luid : device_runtime_traits {
  using return_type = sycl::detail::luid_type;
};

struct node_mask : device_runtime_traits {
  using return_type = uint32_t;
};

} // namespace device
} // namespace info

namespace esimd::info::device {
struct has_2d_block_io_support
    : sycl::detail::ur_traits_base<
          sycl::detail::info_class::device,
          UR_DEVICE_INFO_2D_BLOCK_ARRAY_CAPABILITIES_EXP> {
  using return_type = bool;
};
} // namespace esimd::info::device

} // namespace ext::intel
} // namespace _V1
} // namespace sycl
