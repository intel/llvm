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
    : std::bool_constant<__SYCL_ALL_DEVICES_HAVE_host__> {};
template <>
struct all_devices_have<aspect::cpu>
    : std::bool_constant<__SYCL_ALL_DEVICES_HAVE_cpu__> {};
template <>
struct all_devices_have<aspect::gpu>
    : std::bool_constant<__SYCL_ALL_DEVICES_HAVE_gpu__> {};
template <>
struct all_devices_have<aspect::accelerator>
    : std::bool_constant<__SYCL_ALL_DEVICES_HAVE_accelerator__> {};
template <>
struct all_devices_have<aspect::custom>
    : std::bool_constant<__SYCL_ALL_DEVICES_HAVE_custom__> {};
template <>
struct all_devices_have<aspect::fp16>
    : std::bool_constant<__SYCL_ALL_DEVICES_HAVE_fp16__> {};
template <>
struct all_devices_have<aspect::fp64>
    : std::bool_constant<__SYCL_ALL_DEVICES_HAVE_fp64__> {};
template <>
struct all_devices_have<aspect::int64_base_atomics>
    : std::bool_constant<__SYCL_ALL_DEVICES_HAVE_int64_base_atomics__> {};
template <>
struct all_devices_have<aspect::int64_extended_atomics>
    : std::bool_constant<__SYCL_ALL_DEVICES_HAVE_int64_extended_atomics__> {};
template <>
struct all_devices_have<aspect::image>
    : std::bool_constant<__SYCL_ALL_DEVICES_HAVE_image__> {};
template <>
struct all_devices_have<aspect::online_compiler>
    : std::bool_constant<__SYCL_ALL_DEVICES_HAVE_online_compiler__> {};
template <>
struct all_devices_have<aspect::online_linker>
    : std::bool_constant<__SYCL_ALL_DEVICES_HAVE_online_linker__> {};
template <>
struct all_devices_have<aspect::queue_profiling>
    : std::bool_constant<__SYCL_ALL_DEVICES_HAVE_queue_profiling__> {};
template <>
struct all_devices_have<aspect::usm_device_allocations>
    : std::bool_constant<__SYCL_ALL_DEVICES_HAVE_usm_device_allocations__> {};
template <>
struct all_devices_have<aspect::usm_host_allocations>
    : std::bool_constant<__SYCL_ALL_DEVICES_HAVE_usm_host_allocations__> {};
template <>
struct all_devices_have<aspect::usm_shared_allocations>
    : std::bool_constant<__SYCL_ALL_DEVICES_HAVE_usm_shared_allocations__> {};
template <>
struct all_devices_have<aspect::usm_restricted_shared_allocations>
    : std::bool_constant<
          __SYCL_ALL_DEVICES_HAVE_usm_restricted_shared_allocations__> {};
template <>
struct all_devices_have<aspect::usm_system_allocations>
    : std::bool_constant<__SYCL_ALL_DEVICES_HAVE_usm_system_allocations__> {};
template <>
struct all_devices_have<aspect::ext_intel_pci_address>
    : std::bool_constant<__SYCL_ALL_DEVICES_HAVE_ext_intel_pci_address__> {};
template <>
struct all_devices_have<aspect::ext_intel_gpu_eu_count>
    : std::bool_constant<__SYCL_ALL_DEVICES_HAVE_ext_intel_gpu_eu_count__> {};
template <>
struct all_devices_have<aspect::ext_intel_gpu_eu_simd_width>
    : std::bool_constant<
          __SYCL_ALL_DEVICES_HAVE_ext_intel_gpu_eu_simd_width__> {};
template <>
struct all_devices_have<aspect::ext_intel_gpu_slices>
    : std::bool_constant<__SYCL_ALL_DEVICES_HAVE_ext_intel_gpu_slices__> {};
template <>
struct all_devices_have<aspect::ext_intel_gpu_subslices_per_slice>
    : std::bool_constant<
          __SYCL_ALL_DEVICES_HAVE_ext_intel_gpu_subslices_per_slice__> {};
template <>
struct all_devices_have<aspect::ext_intel_gpu_eu_count_per_subslice>
    : std::bool_constant<
          __SYCL_ALL_DEVICES_HAVE_ext_intel_gpu_eu_count_per_subslice__> {};
template <>
struct all_devices_have<aspect::ext_intel_max_mem_bandwidth>
    : std::bool_constant<
          __SYCL_ALL_DEVICES_HAVE_ext_intel_max_mem_bandwidth__> {};
template <>
struct all_devices_have<aspect::ext_intel_mem_channel>
    : std::bool_constant<__SYCL_ALL_DEVICES_HAVE_ext_intel_mem_channel__> {};
template <>
struct all_devices_have<aspect::usm_atomic_host_allocations>
    : std::bool_constant<
          __SYCL_ALL_DEVICES_HAVE_usm_atomic_host_allocations__> {};
template <>
struct all_devices_have<aspect::usm_atomic_shared_allocations>
    : std::bool_constant<
          __SYCL_ALL_DEVICES_HAVE_usm_atomic_shared_allocations__> {};
template <>
struct all_devices_have<aspect::atomic64>
    : std::bool_constant<__SYCL_ALL_DEVICES_HAVE_atomic64__> {};
template <>
struct all_devices_have<aspect::ext_intel_device_info_uuid>
    : std::bool_constant<__SYCL_ALL_DEVICES_HAVE_ext_intel_device_info_uuid__> {
};
template <>
struct all_devices_have<aspect::ext_oneapi_srgb>
    : std::bool_constant<__SYCL_ALL_DEVICES_HAVE_ext_oneapi_srgb__> {};
template <>
struct all_devices_have<aspect::ext_oneapi_native_assert>
    : std::bool_constant<__SYCL_ALL_DEVICES_HAVE_ext_oneapi_native_assert__> {};
template <>
struct all_devices_have<aspect::host_debuggable>
    : std::bool_constant<__SYCL_ALL_DEVICES_HAVE_host_debuggable__> {};
template <>
struct all_devices_have<aspect::ext_intel_gpu_hw_threads_per_eu>
    : std::bool_constant<
          __SYCL_ALL_DEVICES_HAVE_ext_intel_gpu_hw_threads_per_eu__> {};
template <>
struct all_devices_have<aspect::ext_oneapi_cuda_async_barrier>
    : std::bool_constant<
          __SYCL_ALL_DEVICES_HAVE_ext_oneapi_cuda_async_barrier__> {};
template <>
struct all_devices_have<aspect::ext_oneapi_bfloat16_math_functions>
    : std::bool_constant<
          __SYCL_ALL_DEVICES_HAVE_ext_oneapi_bfloat16_math_functions__> {};
template <>
struct all_devices_have<aspect::ext_intel_free_memory>
    : std::bool_constant<__SYCL_ALL_DEVICES_HAVE_ext_intel_free_memory__> {};
template <>
struct all_devices_have<aspect::ext_intel_device_id>
    : std::bool_constant<__SYCL_ALL_DEVICES_HAVE_ext_intel_device_id__> {};
template <>
struct all_devices_have<aspect::ext_intel_memory_clock_rate>
    : std::bool_constant<
          __SYCL_ALL_DEVICES_HAVE_ext_intel_memory_clock_rate__> {};
template <>
struct all_devices_have<aspect::ext_intel_memory_bus_width>
    : std::bool_constant<__SYCL_ALL_DEVICES_HAVE_ext_intel_memory_bus_width__> {
};
template <>
struct all_devices_have<aspect::emulated>
    : std::bool_constant<__SYCL_ALL_DEVICES_HAVE_emulated__> {};
template <>
struct all_devices_have<aspect::ext_intel_legacy_image>
    : std::bool_constant<__SYCL_ALL_DEVICES_HAVE_ext_intel_legacy_image__> {};

#ifdef __SYCL_ANY_DEVICE_HAS_ANY_ASPECT__
// Special case where any_device_has is trivially true.
template <aspect Aspect> struct any_device_has : std::true_type {};
#else
template <aspect Aspect> struct any_device_has;
template <>
struct any_device_has<aspect::host>
    : std::bool_constant<__SYCL_ANY_DEVICE_HAS_host__> {};
template <>
struct any_device_has<aspect::cpu>
    : std::bool_constant<__SYCL_ANY_DEVICE_HAS_cpu__> {};
template <>
struct any_device_has<aspect::gpu>
    : std::bool_constant<__SYCL_ANY_DEVICE_HAS_gpu__> {};
template <>
struct any_device_has<aspect::accelerator>
    : std::bool_constant<__SYCL_ANY_DEVICE_HAS_accelerator__> {};
template <>
struct any_device_has<aspect::custom>
    : std::bool_constant<__SYCL_ANY_DEVICE_HAS_custom__> {};
template <>
struct any_device_has<aspect::fp16>
    : std::bool_constant<__SYCL_ANY_DEVICE_HAS_fp16__> {};
template <>
struct any_device_has<aspect::fp64>
    : std::bool_constant<__SYCL_ANY_DEVICE_HAS_fp64__> {};
template <>
struct any_device_has<aspect::int64_base_atomics>
    : std::bool_constant<__SYCL_ANY_DEVICE_HAS_int64_base_atomics__> {};
template <>
struct any_device_has<aspect::int64_extended_atomics>
    : std::bool_constant<__SYCL_ANY_DEVICE_HAS_int64_extended_atomics__> {};
template <>
struct any_device_has<aspect::image>
    : std::bool_constant<__SYCL_ANY_DEVICE_HAS_image__> {};
template <>
struct any_device_has<aspect::online_compiler>
    : std::bool_constant<__SYCL_ANY_DEVICE_HAS_online_compiler__> {};
template <>
struct any_device_has<aspect::online_linker>
    : std::bool_constant<__SYCL_ANY_DEVICE_HAS_online_linker__> {};
template <>
struct any_device_has<aspect::queue_profiling>
    : std::bool_constant<__SYCL_ANY_DEVICE_HAS_queue_profiling__> {};
template <>
struct any_device_has<aspect::usm_device_allocations>
    : std::bool_constant<__SYCL_ANY_DEVICE_HAS_usm_device_allocations__> {};
template <>
struct any_device_has<aspect::usm_host_allocations>
    : std::bool_constant<__SYCL_ANY_DEVICE_HAS_usm_host_allocations__> {};
template <>
struct any_device_has<aspect::usm_shared_allocations>
    : std::bool_constant<__SYCL_ANY_DEVICE_HAS_usm_shared_allocations__> {};
template <>
struct any_device_has<aspect::usm_restricted_shared_allocations>
    : std::bool_constant<
          __SYCL_ANY_DEVICE_HAS_usm_restricted_shared_allocations__> {};
template <>
struct any_device_has<aspect::usm_system_allocations>
    : std::bool_constant<__SYCL_ANY_DEVICE_HAS_usm_system_allocations__> {};
template <>
struct any_device_has<aspect::ext_intel_pci_address>
    : std::bool_constant<__SYCL_ANY_DEVICE_HAS_ext_intel_pci_address__> {};
template <>
struct any_device_has<aspect::ext_intel_gpu_eu_count>
    : std::bool_constant<__SYCL_ANY_DEVICE_HAS_ext_intel_gpu_eu_count__> {};
template <>
struct any_device_has<aspect::ext_intel_gpu_eu_simd_width>
    : std::bool_constant<__SYCL_ANY_DEVICE_HAS_ext_intel_gpu_eu_simd_width__> {
};
template <>
struct any_device_has<aspect::ext_intel_gpu_slices>
    : std::bool_constant<__SYCL_ANY_DEVICE_HAS_ext_intel_gpu_slices__> {};
template <>
struct any_device_has<aspect::ext_intel_gpu_subslices_per_slice>
    : std::bool_constant<
          __SYCL_ANY_DEVICE_HAS_ext_intel_gpu_subslices_per_slice__> {};
template <>
struct any_device_has<aspect::ext_intel_gpu_eu_count_per_subslice>
    : std::bool_constant<
          __SYCL_ANY_DEVICE_HAS_ext_intel_gpu_eu_count_per_subslice__> {};
template <>
struct any_device_has<aspect::ext_intel_max_mem_bandwidth>
    : std::bool_constant<__SYCL_ANY_DEVICE_HAS_ext_intel_max_mem_bandwidth__> {
};
template <>
struct any_device_has<aspect::ext_intel_mem_channel>
    : std::bool_constant<__SYCL_ANY_DEVICE_HAS_ext_intel_mem_channel__> {};
template <>
struct any_device_has<aspect::usm_atomic_host_allocations>
    : std::bool_constant<__SYCL_ANY_DEVICE_HAS_usm_atomic_host_allocations__> {
};
template <>
struct any_device_has<aspect::usm_atomic_shared_allocations>
    : std::bool_constant<
          __SYCL_ANY_DEVICE_HAS_usm_atomic_shared_allocations__> {};
template <>
struct any_device_has<aspect::atomic64>
    : std::bool_constant<__SYCL_ANY_DEVICE_HAS_atomic64__> {};
template <>
struct any_device_has<aspect::ext_intel_device_info_uuid>
    : std::bool_constant<__SYCL_ANY_DEVICE_HAS_ext_intel_device_info_uuid__> {};
template <>
struct any_device_has<aspect::ext_oneapi_srgb>
    : std::bool_constant<__SYCL_ANY_DEVICE_HAS_ext_oneapi_srgb__> {};
template <>
struct any_device_has<aspect::ext_oneapi_native_assert>
    : std::bool_constant<__SYCL_ANY_DEVICE_HAS_ext_oneapi_native_assert__> {};
template <>
struct any_device_has<aspect::host_debuggable>
    : std::bool_constant<__SYCL_ANY_DEVICE_HAS_host_debuggable__> {};
template <>
struct any_device_has<aspect::ext_intel_gpu_hw_threads_per_eu>
    : std::bool_constant<
          __SYCL_ANY_DEVICE_HAS_ext_intel_gpu_hw_threads_per_eu__> {};
template <>
struct any_device_has<aspect::ext_oneapi_cuda_async_barrier>
    : std::bool_constant<
          __SYCL_ANY_DEVICE_HAS_ext_oneapi_cuda_async_barrier__> {};
template <>
struct any_device_has<aspect::ext_oneapi_bfloat16_math_functions>
    : std::bool_constant<
          __SYCL_ANY_DEVICE_HAS_ext_oneapi_bfloat16_math_functions__> {};
template <>
struct any_device_has<aspect::ext_intel_free_memory>
    : std::bool_constant<__SYCL_ANY_DEVICE_HAS_ext_intel_free_memory__> {};
template <>
struct any_device_has<aspect::ext_intel_device_id>
    : std::bool_constant<__SYCL_ANY_DEVICE_HAS_ext_intel_device_id__> {};
template <>
struct any_device_has<aspect::ext_intel_memory_clock_rate>
    : std::bool_constant<__SYCL_ANY_DEVICE_HAS_ext_intel_memory_clock_rate__> {
};
template <>
struct any_device_has<aspect::ext_intel_memory_bus_width>
    : std::bool_constant<__SYCL_ANY_DEVICE_HAS_ext_intel_memory_bus_width__> {};
template <>
struct any_device_has<aspect::emulated>
    : std::bool_constant<__SYCL_ANY_DEVICE_HAS_emulated__> {};
template <>
struct any_device_has<aspect::ext_intel_legacy_image>
    : std::bool_constant<__SYCL_ANY_DEVICE_HAS_ext_intel_legacy_image__> {};
#endif // __SYCL_ANY_DEVICE_HAS_ANY_ASPECT__

template <aspect Aspect>
constexpr bool all_devices_have_v = all_devices_have<Aspect>::value;
template <aspect Aspect>
constexpr bool any_device_has_v = any_device_has<Aspect>::value;
} // namespace sycl
