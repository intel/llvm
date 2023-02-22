//==------------------- device_aspect_traits.hpp - SYCL device -------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "device_aspect_macros.hpp"

#pragma once

namespace sycl {
template <aspect Aspect> struct all_devices_have;
template <>
struct all_devices_have<aspect::host>
    : std::bool_constant<__SYCL_ALL_DEVICES_HAVE_0__> {};
template <>
struct all_devices_have<aspect::cpu>
    : std::bool_constant<__SYCL_ALL_DEVICES_HAVE_1__> {};
template <>
struct all_devices_have<aspect::gpu>
    : std::bool_constant<__SYCL_ALL_DEVICES_HAVE_2__> {};
template <>
struct all_devices_have<aspect::accelerator>
    : std::bool_constant<__SYCL_ALL_DEVICES_HAVE_3__> {};
template <>
struct all_devices_have<aspect::custom>
    : std::bool_constant<__SYCL_ALL_DEVICES_HAVE_4__> {};
template <>
struct all_devices_have<aspect::fp16>
    : std::bool_constant<__SYCL_ALL_DEVICES_HAVE_5__> {};
template <>
struct all_devices_have<aspect::fp64>
    : std::bool_constant<__SYCL_ALL_DEVICES_HAVE_6__> {};
template <>
struct all_devices_have<aspect::int64_base_atomics>
    : std::bool_constant<__SYCL_ALL_DEVICES_HAVE_7__> {};
template <>
struct all_devices_have<aspect::int64_extended_atomics>
    : std::bool_constant<__SYCL_ALL_DEVICES_HAVE_8__> {};
template <>
struct all_devices_have<aspect::image>
    : std::bool_constant<__SYCL_ALL_DEVICES_HAVE_9__> {};
template <>
struct all_devices_have<aspect::online_compiler>
    : std::bool_constant<__SYCL_ALL_DEVICES_HAVE_10__> {};
template <>
struct all_devices_have<aspect::online_linker>
    : std::bool_constant<__SYCL_ALL_DEVICES_HAVE_11__> {};
template <>
struct all_devices_have<aspect::queue_profiling>
    : std::bool_constant<__SYCL_ALL_DEVICES_HAVE_12__> {};
template <>
struct all_devices_have<aspect::usm_device_allocations>
    : std::bool_constant<__SYCL_ALL_DEVICES_HAVE_13__> {};
template <>
struct all_devices_have<aspect::usm_host_allocations>
    : std::bool_constant<__SYCL_ALL_DEVICES_HAVE_14__> {};
template <>
struct all_devices_have<aspect::usm_shared_allocations>
    : std::bool_constant<__SYCL_ALL_DEVICES_HAVE_15__> {};
template <>
struct all_devices_have<aspect::usm_restricted_shared_allocations>
    : std::bool_constant<__SYCL_ALL_DEVICES_HAVE_16__> {};
template <>
struct all_devices_have<aspect::usm_system_allocations>
    : std::bool_constant<__SYCL_ALL_DEVICES_HAVE_17__> {};
template <>
struct all_devices_have<aspect::ext_intel_pci_address>
    : std::bool_constant<__SYCL_ALL_DEVICES_HAVE_18__> {};
template <>
struct all_devices_have<aspect::ext_intel_gpu_eu_count>
    : std::bool_constant<__SYCL_ALL_DEVICES_HAVE_19__> {};
template <>
struct all_devices_have<aspect::ext_intel_gpu_eu_simd_width>
    : std::bool_constant<__SYCL_ALL_DEVICES_HAVE_20__> {};
template <>
struct all_devices_have<aspect::ext_intel_gpu_slices>
    : std::bool_constant<__SYCL_ALL_DEVICES_HAVE_21__> {};
template <>
struct all_devices_have<aspect::ext_intel_gpu_subslices_per_slice>
    : std::bool_constant<__SYCL_ALL_DEVICES_HAVE_22__> {};
template <>
struct all_devices_have<aspect::ext_intel_gpu_eu_count_per_subslice>
    : std::bool_constant<__SYCL_ALL_DEVICES_HAVE_23__> {};
template <>
struct all_devices_have<aspect::ext_intel_max_mem_bandwidth>
    : std::bool_constant<__SYCL_ALL_DEVICES_HAVE_24__> {};
template <>
struct all_devices_have<aspect::ext_intel_mem_channel>
    : std::bool_constant<__SYCL_ALL_DEVICES_HAVE_25__> {};
template <>
struct all_devices_have<aspect::usm_atomic_host_allocations>
    : std::bool_constant<__SYCL_ALL_DEVICES_HAVE_26__> {};
template <>
struct all_devices_have<aspect::usm_atomic_shared_allocations>
    : std::bool_constant<__SYCL_ALL_DEVICES_HAVE_27__> {};
template <>
struct all_devices_have<aspect::atomic64>
    : std::bool_constant<__SYCL_ALL_DEVICES_HAVE_28__> {};
template <>
struct all_devices_have<aspect::ext_intel_device_info_uuid>
    : std::bool_constant<__SYCL_ALL_DEVICES_HAVE_29__> {};
template <>
struct all_devices_have<aspect::ext_oneapi_srgb>
    : std::bool_constant<__SYCL_ALL_DEVICES_HAVE_30__> {};
template <>
struct all_devices_have<aspect::ext_oneapi_native_assert>
    : std::bool_constant<__SYCL_ALL_DEVICES_HAVE_31__> {};
template <>
struct all_devices_have<aspect::host_debuggable>
    : std::bool_constant<__SYCL_ALL_DEVICES_HAVE_32__> {};
template <>
struct all_devices_have<aspect::ext_intel_gpu_hw_threads_per_eu>
    : std::bool_constant<__SYCL_ALL_DEVICES_HAVE_33__> {};
template <>
struct all_devices_have<aspect::ext_oneapi_cuda_async_barrier>
    : std::bool_constant<__SYCL_ALL_DEVICES_HAVE_34__> {};
template <>
struct all_devices_have<aspect::ext_oneapi_bfloat16_math_functions>
    : std::bool_constant<__SYCL_ALL_DEVICES_HAVE_35__> {};
template <>
struct all_devices_have<aspect::ext_intel_free_memory>
    : std::bool_constant<__SYCL_ALL_DEVICES_HAVE_36__> {};
template <>
struct all_devices_have<aspect::ext_intel_device_id>
    : std::bool_constant<__SYCL_ALL_DEVICES_HAVE_37__> {};
template <>
struct all_devices_have<aspect::ext_intel_memory_clock_rate>
    : std::bool_constant<__SYCL_ALL_DEVICES_HAVE_38__> {};
template <>
struct all_devices_have<aspect::ext_intel_memory_bus_width>
    : std::bool_constant<__SYCL_ALL_DEVICES_HAVE_39__> {};

#ifdef __SYCL_ANY_DEVICE_HAS_ANY_ASPECT__
// Special case where any_device_has is trivially true.
template <aspect Aspect> struct any_device_has : std::true_t {};
#else
template <aspect Aspect> struct any_device_has;
template <>
struct any_device_has<aspect::host>
    : std::bool_constant<__SYCL_ANY_DEVICE_HAS_0__> {};
template <>
struct any_device_has<aspect::cpu>
    : std::bool_constant<__SYCL_ANY_DEVICE_HAS_1__> {};
template <>
struct any_device_has<aspect::gpu>
    : std::bool_constant<__SYCL_ANY_DEVICE_HAS_2__> {};
template <>
struct any_device_has<aspect::accelerator>
    : std::bool_constant<__SYCL_ANY_DEVICE_HAS_3__> {};
template <>
struct any_device_has<aspect::custom>
    : std::bool_constant<__SYCL_ANY_DEVICE_HAS_4__> {};
template <>
struct any_device_has<aspect::fp16>
    : std::bool_constant<__SYCL_ANY_DEVICE_HAS_5__> {};
template <>
struct any_device_has<aspect::fp64>
    : std::bool_constant<__SYCL_ANY_DEVICE_HAS_6__> {};
template <>
struct any_device_has<aspect::int64_base_atomics>
    : std::bool_constant<__SYCL_ANY_DEVICE_HAS_7__> {};
template <>
struct any_device_has<aspect::int64_extended_atomics>
    : std::bool_constant<__SYCL_ANY_DEVICE_HAS_8__> {};
template <>
struct any_device_has<aspect::image>
    : std::bool_constant<__SYCL_ANY_DEVICE_HAS_9__> {};
template <>
struct any_device_has<aspect::online_compiler>
    : std::bool_constant<__SYCL_ANY_DEVICE_HAS_10__> {};
template <>
struct any_device_has<aspect::online_linker>
    : std::bool_constant<__SYCL_ANY_DEVICE_HAS_11__> {};
template <>
struct any_device_has<aspect::queue_profiling>
    : std::bool_constant<__SYCL_ANY_DEVICE_HAS_12__> {};
template <>
struct any_device_has<aspect::usm_device_allocations>
    : std::bool_constant<__SYCL_ANY_DEVICE_HAS_13__> {};
template <>
struct any_device_has<aspect::usm_host_allocations>
    : std::bool_constant<__SYCL_ANY_DEVICE_HAS_14__> {};
template <>
struct any_device_has<aspect::usm_shared_allocations>
    : std::bool_constant<__SYCL_ANY_DEVICE_HAS_15__> {};
template <>
struct any_device_has<aspect::usm_restricted_shared_allocations>
    : std::bool_constant<__SYCL_ANY_DEVICE_HAS_16__> {};
template <>
struct any_device_has<aspect::usm_system_allocations>
    : std::bool_constant<__SYCL_ANY_DEVICE_HAS_17__> {};
template <>
struct any_device_has<aspect::ext_intel_pci_address>
    : std::bool_constant<__SYCL_ANY_DEVICE_HAS_18__> {};
template <>
struct any_device_has<aspect::ext_intel_gpu_eu_count>
    : std::bool_constant<__SYCL_ANY_DEVICE_HAS_19__> {};
template <>
struct any_device_has<aspect::ext_intel_gpu_eu_simd_width>
    : std::bool_constant<__SYCL_ANY_DEVICE_HAS_20__> {};
template <>
struct any_device_has<aspect::ext_intel_gpu_slices>
    : std::bool_constant<__SYCL_ANY_DEVICE_HAS_21__> {};
template <>
struct any_device_has<aspect::ext_intel_gpu_subslices_per_slice>
    : std::bool_constant<__SYCL_ANY_DEVICE_HAS_22__> {};
template <>
struct any_device_has<aspect::ext_intel_gpu_eu_count_per_subslice>
    : std::bool_constant<__SYCL_ANY_DEVICE_HAS_23__> {};
template <>
struct any_device_has<aspect::ext_intel_max_mem_bandwidth>
    : std::bool_constant<__SYCL_ANY_DEVICE_HAS_24__> {};
template <>
struct any_device_has<aspect::ext_intel_mem_channel>
    : std::bool_constant<__SYCL_ANY_DEVICE_HAS_25__> {};
template <>
struct any_device_has<aspect::usm_atomic_host_allocations>
    : std::bool_constant<__SYCL_ANY_DEVICE_HAS_26__> {};
template <>
struct any_device_has<aspect::usm_atomic_shared_allocations>
    : std::bool_constant<__SYCL_ANY_DEVICE_HAS_27__> {};
template <>
struct any_device_has<aspect::atomic64>
    : std::bool_constant<__SYCL_ANY_DEVICE_HAS_28__> {};
template <>
struct any_device_has<aspect::ext_intel_device_info_uuid>
    : std::bool_constant<__SYCL_ANY_DEVICE_HAS_29__> {};
template <>
struct any_device_has<aspect::ext_oneapi_srgb>
    : std::bool_constant<__SYCL_ANY_DEVICE_HAS_30__> {};
template <>
struct any_device_has<aspect::ext_oneapi_native_assert>
    : std::bool_constant<__SYCL_ANY_DEVICE_HAS_31__> {};
template <>
struct any_device_has<aspect::host_debuggable>
    : std::bool_constant<__SYCL_ANY_DEVICE_HAS_32__> {};
template <>
struct any_device_has<aspect::ext_intel_gpu_hw_threads_per_eu>
    : std::bool_constant<__SYCL_ANY_DEVICE_HAS_33__> {};
template <>
struct any_device_has<aspect::ext_oneapi_cuda_async_barrier>
    : std::bool_constant<__SYCL_ANY_DEVICE_HAS_34__> {};
template <>
struct any_device_has<aspect::ext_oneapi_bfloat16_math_functions>
    : std::bool_constant<__SYCL_ANY_DEVICE_HAS_35__> {};
template <>
struct any_device_has<aspect::ext_intel_free_memory>
    : std::bool_constant<__SYCL_ANY_DEVICE_HAS_36__> {};
template <>
struct any_device_has<aspect::ext_intel_device_id>
    : std::bool_constant<__SYCL_ANY_DEVICE_HAS_37__> {};
template <>
struct any_device_has<aspect::ext_intel_memory_clock_rate>
    : std::bool_constant<__SYCL_ANY_DEVICE_HAS_38__> {};
template <>
struct any_device_has<aspect::ext_intel_memory_bus_width>
    : std::bool_constant<__SYCL_ANY_DEVICE_HAS_39__> {};
#endif // __SYCL_ANY_DEVICE_HAS_ANY_ASPECT__

template <aspect Aspect>
constexpr bool all_devices_have_v = all_devices_have<Aspect>::value;
template <aspect Aspect>
constexpr bool any_device_has_v = any_device_has<Aspect>::value;
} // namespace sycl
