//==-------------- aspects.hpp - SYCL Aspect Enums ------------*- C++ -*---==//
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
  int64_base_atomics = 7,
  int64_extended_atomics = 8,
  image = 9,
  online_compiler = 10,
  online_linker = 11,
  queue_profiling = 12,
  usm_device_allocations = 13,
  usm_host_allocations = 14,
  usm_shared_allocations = 15,
  usm_restricted_shared_allocations = 16,
  usm_system_allocator = 17,
  ext_intel_pci_address = 18,
  ext_intel_gpu_eu_count = 19,
  ext_intel_gpu_eu_simd_width = 20,
  ext_intel_gpu_slices = 21,
  ext_intel_gpu_subslices_per_slice = 22,
  ext_intel_gpu_eu_count_per_subslice = 23,
  ext_intel_max_mem_bandwidth = 24,
  ext_intel_mem_channel = 25
};

} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
