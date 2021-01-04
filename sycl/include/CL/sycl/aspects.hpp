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
  host,
  cpu,
  gpu,
  accelerator,
  custom,
  fp16,
  fp64,
  int64_base_atomics,
  int64_extended_atomics,
  image,
  online_compiler,
  online_linker,
  queue_profiling,
  usm_device_allocations,
  usm_host_allocations,
  usm_shared_allocations,
  usm_restricted_shared_allocations,
  usm_system_allocator,
  ext_intel_pci_address,
  ext_intel_gpu_eu_count,
  ext_intel_gpu_eu_simd_width,
  ext_intel_gpu_slices,
  ext_intel_gpu_subslices_per_slice,
  ext_intel_gpu_eu_count_per_subslice,
  ext_intel_max_mem_bandwidth,
  ext_intel_mem_channel
};

} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
