//==--------------- aspects.hpp - SYCL Aspect Enums ------------*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //
#pragma once

#include <CL/sycl/detail/defines.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {

enum class aspect {
  host = 0,
  cpu = 1,
  gpu = 2,
  accelerator = 3,
  custom = 4,
  fp16 = 5,
  fp64 = 6,
  int64_base_atomics __SYCL2020_DEPRECATED("use atomic64 instead") = 7,
  int64_extended_atomics __SYCL2020_DEPRECATED("use atomic64 instead") = 8,
  image = 9,
  online_compiler = 10,
  online_linker = 11,
  queue_profiling = 12,
  usm_device_allocations = 13,
  usm_host_allocations = 14,
  usm_shared_allocations = 15,
  usm_restricted_shared_allocations = 16,
  usm_system_allocations = 17,
  usm_system_allocator __SYCL2020_DEPRECATED(
      "use usm_system_allocations instead") = usm_system_allocations,
  ext_intel_pci_address = 18,
  ext_intel_gpu_eu_count = 19,
  ext_intel_gpu_eu_simd_width = 20,
  ext_intel_gpu_slices = 21,
  ext_intel_gpu_subslices_per_slice = 22,
  ext_intel_gpu_eu_count_per_subslice = 23,
  ext_intel_max_mem_bandwidth = 24,
  ext_intel_mem_channel = 25,
  usm_atomic_host_allocations = 26,
  usm_atomic_shared_allocations = 27,
  atomic64 = 28,
  ext_intel_device_info_uuid = 29,
  ext_oneapi_srgb = 30,
  ext_oneapi_native_assert = 31,
  host_debuggable = 32,
  ext_intel_gpu_hw_threads_per_eu = 33,
  ext_intel_buffer_location = 34,
};

} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
