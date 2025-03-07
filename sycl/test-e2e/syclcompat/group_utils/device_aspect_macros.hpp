//
//
// Modifications, Copyright (C) 2024 Intel Corporation
//
// This software and the related documents are Intel copyrighted materials, and
// your use of them is governed by the express license under which they were
// provided to you ("License"). Unless the License provides otherwise, you may
// not use, modify, copy, publish, distribute, disclose or transmit this
// software or the related documents without Intel's prior written permission.
//
// This software and the related documents are provided as is, with no express
// or implied warranties, other than those that are expressly stated in the
// License.
//
//==------------------- device_aspect_macros.hpp - SYCL device -------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//
// IMPORTANT: device_aspect_macros.hpp is a generated file - DO NOT EDIT
//            original definitions are in aspects.def & aspects_deprecated.def
//

#pragma once

// __SYCL_ASPECT_DEPRECATED(host, 0)
#ifndef __SYCL_ALL_DEVICES_HAVE_host__
#define __SYCL_ALL_DEVICES_HAVE_host__ 0
#endif
#ifndef __SYCL_ANY_DEVICE_HAS_host__
#define __SYCL_ANY_DEVICE_HAS_host__ 0
#endif

// __SYCL_ASPECT_DEPRECATED(int64_base_atomics, 7)
#ifndef __SYCL_ALL_DEVICES_HAVE_int64_base_atomics__
#define __SYCL_ALL_DEVICES_HAVE_int64_base_atomics__ 0
#endif
#ifndef __SYCL_ANY_DEVICE_HAS_int64_base_atomics__
#define __SYCL_ANY_DEVICE_HAS_int64_base_atomics__ 0
#endif

// __SYCL_ASPECT_DEPRECATED(int64_extended_atomics, 8)
#ifndef __SYCL_ALL_DEVICES_HAVE_int64_extended_atomics__
#define __SYCL_ALL_DEVICES_HAVE_int64_extended_atomics__ 0
#endif
#ifndef __SYCL_ANY_DEVICE_HAS_int64_extended_atomics__
#define __SYCL_ANY_DEVICE_HAS_int64_extended_atomics__ 0
#endif

// __SYCL_ASPECT_DEPRECATED(usm_restricted_shared_allocations, 16)
#ifndef __SYCL_ALL_DEVICES_HAVE_usm_restricted_shared_allocations__
#define __SYCL_ALL_DEVICES_HAVE_usm_restricted_shared_allocations__ 0
#endif
#ifndef __SYCL_ANY_DEVICE_HAS_usm_restricted_shared_allocations__
#define __SYCL_ANY_DEVICE_HAS_usm_restricted_shared_allocations__ 0
#endif

// __SYCL_ASPECT(cpu, 1)
#ifndef __SYCL_ALL_DEVICES_HAVE_cpu__
#define __SYCL_ALL_DEVICES_HAVE_cpu__ 0
#endif
#ifndef __SYCL_ANY_DEVICE_HAS_cpu__
#define __SYCL_ANY_DEVICE_HAS_cpu__ 0
#endif

// __SYCL_ASPECT(gpu, 2)
#ifndef __SYCL_ALL_DEVICES_HAVE_gpu__
#define __SYCL_ALL_DEVICES_HAVE_gpu__ 0
#endif
#ifndef __SYCL_ANY_DEVICE_HAS_gpu__
#define __SYCL_ANY_DEVICE_HAS_gpu__ 0
#endif

// __SYCL_ASPECT(accelerator, 3)
#ifndef __SYCL_ALL_DEVICES_HAVE_accelerator__
#define __SYCL_ALL_DEVICES_HAVE_accelerator__ 0
#endif
#ifndef __SYCL_ANY_DEVICE_HAS_accelerator__
#define __SYCL_ANY_DEVICE_HAS_accelerator__ 0
#endif

// __SYCL_ASPECT(custom, 4)
#ifndef __SYCL_ALL_DEVICES_HAVE_custom__
#define __SYCL_ALL_DEVICES_HAVE_custom__ 0
#endif
#ifndef __SYCL_ANY_DEVICE_HAS_custom__
#define __SYCL_ANY_DEVICE_HAS_custom__ 0
#endif

// __SYCL_ASPECT(fp16, 5)
#ifndef __SYCL_ALL_DEVICES_HAVE_fp16__
#define __SYCL_ALL_DEVICES_HAVE_fp16__ 0
#endif
#ifndef __SYCL_ANY_DEVICE_HAS_fp16__
#define __SYCL_ANY_DEVICE_HAS_fp16__ 0
#endif

// __SYCL_ASPECT(fp64, 6)
#ifndef __SYCL_ALL_DEVICES_HAVE_fp64__
#define __SYCL_ALL_DEVICES_HAVE_fp64__ 0
#endif
#ifndef __SYCL_ANY_DEVICE_HAS_fp64__
#define __SYCL_ANY_DEVICE_HAS_fp64__ 0
#endif

// __SYCL_ASPECT(image, 9)
#ifndef __SYCL_ALL_DEVICES_HAVE_image__
#define __SYCL_ALL_DEVICES_HAVE_image__ 0
#endif
#ifndef __SYCL_ANY_DEVICE_HAS_image__
#define __SYCL_ANY_DEVICE_HAS_image__ 0
#endif

// __SYCL_ASPECT(online_compiler, 10)
#ifndef __SYCL_ALL_DEVICES_HAVE_online_compiler__
#define __SYCL_ALL_DEVICES_HAVE_online_compiler__ 0
#endif
#ifndef __SYCL_ANY_DEVICE_HAS_online_compiler__
#define __SYCL_ANY_DEVICE_HAS_online_compiler__ 0
#endif

// __SYCL_ASPECT(online_linker, 11)
#ifndef __SYCL_ALL_DEVICES_HAVE_online_linker__
#define __SYCL_ALL_DEVICES_HAVE_online_linker__ 0
#endif
#ifndef __SYCL_ANY_DEVICE_HAS_online_linker__
#define __SYCL_ANY_DEVICE_HAS_online_linker__ 0
#endif

// __SYCL_ASPECT(queue_profiling, 12)
#ifndef __SYCL_ALL_DEVICES_HAVE_queue_profiling__
#define __SYCL_ALL_DEVICES_HAVE_queue_profiling__ 0
#endif
#ifndef __SYCL_ANY_DEVICE_HAS_queue_profiling__
#define __SYCL_ANY_DEVICE_HAS_queue_profiling__ 0
#endif

// __SYCL_ASPECT(usm_device_allocations, 13)
#ifndef __SYCL_ALL_DEVICES_HAVE_usm_device_allocations__
#define __SYCL_ALL_DEVICES_HAVE_usm_device_allocations__ 0
#endif
#ifndef __SYCL_ANY_DEVICE_HAS_usm_device_allocations__
#define __SYCL_ANY_DEVICE_HAS_usm_device_allocations__ 0
#endif

// __SYCL_ASPECT(usm_host_allocations, 14)
#ifndef __SYCL_ALL_DEVICES_HAVE_usm_host_allocations__
#define __SYCL_ALL_DEVICES_HAVE_usm_host_allocations__ 0
#endif
#ifndef __SYCL_ANY_DEVICE_HAS_usm_host_allocations__
#define __SYCL_ANY_DEVICE_HAS_usm_host_allocations__ 0
#endif

// __SYCL_ASPECT(usm_shared_allocations, 15)
#ifndef __SYCL_ALL_DEVICES_HAVE_usm_shared_allocations__
#define __SYCL_ALL_DEVICES_HAVE_usm_shared_allocations__ 0
#endif
#ifndef __SYCL_ANY_DEVICE_HAS_usm_shared_allocations__
#define __SYCL_ANY_DEVICE_HAS_usm_shared_allocations__ 0
#endif

// __SYCL_ASPECT(usm_system_allocations, 17)
#ifndef __SYCL_ALL_DEVICES_HAVE_usm_system_allocations__
#define __SYCL_ALL_DEVICES_HAVE_usm_system_allocations__ 0
#endif
#ifndef __SYCL_ANY_DEVICE_HAS_usm_system_allocations__
#define __SYCL_ANY_DEVICE_HAS_usm_system_allocations__ 0
#endif

// __SYCL_ASPECT(ext_intel_pci_address, 18)
#ifndef __SYCL_ALL_DEVICES_HAVE_ext_intel_pci_address__
#define __SYCL_ALL_DEVICES_HAVE_ext_intel_pci_address__ 0
#endif
#ifndef __SYCL_ANY_DEVICE_HAS_ext_intel_pci_address__
#define __SYCL_ANY_DEVICE_HAS_ext_intel_pci_address__ 0
#endif

// __SYCL_ASPECT(ext_intel_gpu_eu_count, 19)
#ifndef __SYCL_ALL_DEVICES_HAVE_ext_intel_gpu_eu_count__
#define __SYCL_ALL_DEVICES_HAVE_ext_intel_gpu_eu_count__ 0
#endif
#ifndef __SYCL_ANY_DEVICE_HAS_ext_intel_gpu_eu_count__
#define __SYCL_ANY_DEVICE_HAS_ext_intel_gpu_eu_count__ 0
#endif

// __SYCL_ASPECT(ext_intel_gpu_eu_simd_width, 20)
#ifndef __SYCL_ALL_DEVICES_HAVE_ext_intel_gpu_eu_simd_width__
#define __SYCL_ALL_DEVICES_HAVE_ext_intel_gpu_eu_simd_width__ 0
#endif
#ifndef __SYCL_ANY_DEVICE_HAS_ext_intel_gpu_eu_simd_width__
#define __SYCL_ANY_DEVICE_HAS_ext_intel_gpu_eu_simd_width__ 0
#endif

// __SYCL_ASPECT(ext_intel_gpu_slices, 21)
#ifndef __SYCL_ALL_DEVICES_HAVE_ext_intel_gpu_slices__
#define __SYCL_ALL_DEVICES_HAVE_ext_intel_gpu_slices__ 0
#endif
#ifndef __SYCL_ANY_DEVICE_HAS_ext_intel_gpu_slices__
#define __SYCL_ANY_DEVICE_HAS_ext_intel_gpu_slices__ 0
#endif

// __SYCL_ASPECT(ext_intel_gpu_subslices_per_slice, 22)
#ifndef __SYCL_ALL_DEVICES_HAVE_ext_intel_gpu_subslices_per_slice__
#define __SYCL_ALL_DEVICES_HAVE_ext_intel_gpu_subslices_per_slice__ 0
#endif
#ifndef __SYCL_ANY_DEVICE_HAS_ext_intel_gpu_subslices_per_slice__
#define __SYCL_ANY_DEVICE_HAS_ext_intel_gpu_subslices_per_slice__ 0
#endif

// __SYCL_ASPECT(ext_intel_gpu_eu_count_per_subslice, 23)
#ifndef __SYCL_ALL_DEVICES_HAVE_ext_intel_gpu_eu_count_per_subslice__
#define __SYCL_ALL_DEVICES_HAVE_ext_intel_gpu_eu_count_per_subslice__ 0
#endif
#ifndef __SYCL_ANY_DEVICE_HAS_ext_intel_gpu_eu_count_per_subslice__
#define __SYCL_ANY_DEVICE_HAS_ext_intel_gpu_eu_count_per_subslice__ 0
#endif

// __SYCL_ASPECT(ext_intel_max_mem_bandwidth, 24)
#ifndef __SYCL_ALL_DEVICES_HAVE_ext_intel_max_mem_bandwidth__
#define __SYCL_ALL_DEVICES_HAVE_ext_intel_max_mem_bandwidth__ 0
#endif
#ifndef __SYCL_ANY_DEVICE_HAS_ext_intel_max_mem_bandwidth__
#define __SYCL_ANY_DEVICE_HAS_ext_intel_max_mem_bandwidth__ 0
#endif

// __SYCL_ASPECT(ext_intel_mem_channel, 25)
#ifndef __SYCL_ALL_DEVICES_HAVE_ext_intel_mem_channel__
#define __SYCL_ALL_DEVICES_HAVE_ext_intel_mem_channel__ 0
#endif
#ifndef __SYCL_ANY_DEVICE_HAS_ext_intel_mem_channel__
#define __SYCL_ANY_DEVICE_HAS_ext_intel_mem_channel__ 0
#endif

// __SYCL_ASPECT(usm_atomic_host_allocations, 26)
#ifndef __SYCL_ALL_DEVICES_HAVE_usm_atomic_host_allocations__
#define __SYCL_ALL_DEVICES_HAVE_usm_atomic_host_allocations__ 0
#endif
#ifndef __SYCL_ANY_DEVICE_HAS_usm_atomic_host_allocations__
#define __SYCL_ANY_DEVICE_HAS_usm_atomic_host_allocations__ 0
#endif

// __SYCL_ASPECT(usm_atomic_shared_allocations, 27)
#ifndef __SYCL_ALL_DEVICES_HAVE_usm_atomic_shared_allocations__
#define __SYCL_ALL_DEVICES_HAVE_usm_atomic_shared_allocations__ 0
#endif
#ifndef __SYCL_ANY_DEVICE_HAS_usm_atomic_shared_allocations__
#define __SYCL_ANY_DEVICE_HAS_usm_atomic_shared_allocations__ 0
#endif

// __SYCL_ASPECT(atomic64, 28)
#ifndef __SYCL_ALL_DEVICES_HAVE_atomic64__
#define __SYCL_ALL_DEVICES_HAVE_atomic64__ 0
#endif
#ifndef __SYCL_ANY_DEVICE_HAS_atomic64__
#define __SYCL_ANY_DEVICE_HAS_atomic64__ 0
#endif

// __SYCL_ASPECT(ext_intel_device_info_uuid, 29)
#ifndef __SYCL_ALL_DEVICES_HAVE_ext_intel_device_info_uuid__
#define __SYCL_ALL_DEVICES_HAVE_ext_intel_device_info_uuid__ 0
#endif
#ifndef __SYCL_ANY_DEVICE_HAS_ext_intel_device_info_uuid__
#define __SYCL_ANY_DEVICE_HAS_ext_intel_device_info_uuid__ 0
#endif

// __SYCL_ASPECT(ext_oneapi_srgb, 30)
#ifndef __SYCL_ALL_DEVICES_HAVE_ext_oneapi_srgb__
#define __SYCL_ALL_DEVICES_HAVE_ext_oneapi_srgb__ 0
#endif
#ifndef __SYCL_ANY_DEVICE_HAS_ext_oneapi_srgb__
#define __SYCL_ANY_DEVICE_HAS_ext_oneapi_srgb__ 0
#endif

// __SYCL_ASPECT(ext_oneapi_native_assert, 31)
#ifndef __SYCL_ALL_DEVICES_HAVE_ext_oneapi_native_assert__
#define __SYCL_ALL_DEVICES_HAVE_ext_oneapi_native_assert__ 0
#endif
#ifndef __SYCL_ANY_DEVICE_HAS_ext_oneapi_native_assert__
#define __SYCL_ANY_DEVICE_HAS_ext_oneapi_native_assert__ 0
#endif

// __SYCL_ASPECT(host_debuggable, 32)
#ifndef __SYCL_ALL_DEVICES_HAVE_host_debuggable__
#define __SYCL_ALL_DEVICES_HAVE_host_debuggable__ 0
#endif
#ifndef __SYCL_ANY_DEVICE_HAS_host_debuggable__
#define __SYCL_ANY_DEVICE_HAS_host_debuggable__ 0
#endif

// __SYCL_ASPECT(ext_intel_gpu_hw_threads_per_eu, 33)
#ifndef __SYCL_ALL_DEVICES_HAVE_ext_intel_gpu_hw_threads_per_eu__
#define __SYCL_ALL_DEVICES_HAVE_ext_intel_gpu_hw_threads_per_eu__ 0
#endif
#ifndef __SYCL_ANY_DEVICE_HAS_ext_intel_gpu_hw_threads_per_eu__
#define __SYCL_ANY_DEVICE_HAS_ext_intel_gpu_hw_threads_per_eu__ 0
#endif

// __SYCL_ASPECT(ext_oneapi_cuda_async_barrier, 34)
#ifndef __SYCL_ALL_DEVICES_HAVE_ext_oneapi_cuda_async_barrier__
#define __SYCL_ALL_DEVICES_HAVE_ext_oneapi_cuda_async_barrier__ 0
#endif
#ifndef __SYCL_ANY_DEVICE_HAS_ext_oneapi_cuda_async_barrier__
#define __SYCL_ANY_DEVICE_HAS_ext_oneapi_cuda_async_barrier__ 0
#endif

// __SYCL_ASPECT(ext_intel_free_memory, 36)
#ifndef __SYCL_ALL_DEVICES_HAVE_ext_intel_free_memory__
#define __SYCL_ALL_DEVICES_HAVE_ext_intel_free_memory__ 0
#endif
#ifndef __SYCL_ANY_DEVICE_HAS_ext_intel_free_memory__
#define __SYCL_ANY_DEVICE_HAS_ext_intel_free_memory__ 0
#endif

// __SYCL_ASPECT(ext_intel_device_id, 37)
#ifndef __SYCL_ALL_DEVICES_HAVE_ext_intel_device_id__
#define __SYCL_ALL_DEVICES_HAVE_ext_intel_device_id__ 0
#endif
#ifndef __SYCL_ANY_DEVICE_HAS_ext_intel_device_id__
#define __SYCL_ANY_DEVICE_HAS_ext_intel_device_id__ 0
#endif

// __SYCL_ASPECT(ext_intel_memory_clock_rate, 38)
#ifndef __SYCL_ALL_DEVICES_HAVE_ext_intel_memory_clock_rate__
#define __SYCL_ALL_DEVICES_HAVE_ext_intel_memory_clock_rate__ 0
#endif
#ifndef __SYCL_ANY_DEVICE_HAS_ext_intel_memory_clock_rate__
#define __SYCL_ANY_DEVICE_HAS_ext_intel_memory_clock_rate__ 0
#endif

// __SYCL_ASPECT(ext_intel_memory_bus_width, 39)
#ifndef __SYCL_ALL_DEVICES_HAVE_ext_intel_memory_bus_width__
#define __SYCL_ALL_DEVICES_HAVE_ext_intel_memory_bus_width__ 0
#endif
#ifndef __SYCL_ANY_DEVICE_HAS_ext_intel_memory_bus_width__
#define __SYCL_ANY_DEVICE_HAS_ext_intel_memory_bus_width__ 0
#endif

// __SYCL_ASPECT(emulated, 40)
#ifndef __SYCL_ALL_DEVICES_HAVE_emulated__
#define __SYCL_ALL_DEVICES_HAVE_emulated__ 0
#endif
#ifndef __SYCL_ANY_DEVICE_HAS_emulated__
#define __SYCL_ANY_DEVICE_HAS_emulated__ 0
#endif

// __SYCL_ASPECT(ext_intel_legacy_image, 41)
#ifndef __SYCL_ALL_DEVICES_HAVE_ext_intel_legacy_image__
#define __SYCL_ALL_DEVICES_HAVE_ext_intel_legacy_image__ 0
#endif
#ifndef __SYCL_ANY_DEVICE_HAS_ext_intel_legacy_image__
#define __SYCL_ANY_DEVICE_HAS_ext_intel_legacy_image__ 0
#endif

// __SYCL_ASPECT(ext_oneapi_bindless_images, 42)
#ifndef __SYCL_ALL_DEVICES_HAVE_ext_oneapi_bindless_images__
#define __SYCL_ALL_DEVICES_HAVE_ext_oneapi_bindless_images__ 0
#endif
#ifndef __SYCL_ANY_DEVICE_HAS_ext_oneapi_bindless_images__
#define __SYCL_ANY_DEVICE_HAS_ext_oneapi_bindless_images__ 0
#endif

// __SYCL_ASPECT(ext_oneapi_bindless_images_shared_usm, 43)
#ifndef __SYCL_ALL_DEVICES_HAVE_ext_oneapi_bindless_images_shared_usm__
#define __SYCL_ALL_DEVICES_HAVE_ext_oneapi_bindless_images_shared_usm__ 0
#endif
#ifndef __SYCL_ANY_DEVICE_HAS_ext_oneapi_bindless_images_shared_usm__
#define __SYCL_ANY_DEVICE_HAS_ext_oneapi_bindless_images_shared_usm__ 0
#endif

// __SYCL_ASPECT(ext_oneapi_bindless_images_1d_usm, 44)
#ifndef __SYCL_ALL_DEVICES_HAVE_ext_oneapi_bindless_images_1d_usm__
#define __SYCL_ALL_DEVICES_HAVE_ext_oneapi_bindless_images_1d_usm__ 0
#endif
#ifndef __SYCL_ANY_DEVICE_HAS_ext_oneapi_bindless_images_1d_usm__
#define __SYCL_ANY_DEVICE_HAS_ext_oneapi_bindless_images_1d_usm__ 0
#endif

// __SYCL_ASPECT(ext_oneapi_bindless_images_2d_usm, 45)
#ifndef __SYCL_ALL_DEVICES_HAVE_ext_oneapi_bindless_images_2d_usm__
#define __SYCL_ALL_DEVICES_HAVE_ext_oneapi_bindless_images_2d_usm__ 0
#endif
#ifndef __SYCL_ANY_DEVICE_HAS_ext_oneapi_bindless_images_2d_usm__
#define __SYCL_ANY_DEVICE_HAS_ext_oneapi_bindless_images_2d_usm__ 0
#endif

// __SYCL_ASPECT(ext_oneapi_external_memory_import, 46)
#ifndef __SYCL_ALL_DEVICES_HAVE_ext_oneapi_external_memory_import__
#define __SYCL_ALL_DEVICES_HAVE_ext_oneapi_external_memory_import__ 0
#endif
#ifndef __SYCL_ANY_DEVICE_HAS_ext_oneapi_external_memory_import__
#define __SYCL_ANY_DEVICE_HAS_ext_oneapi_external_memory_import__ 0
#endif

// __SYCL_ASPECT(ext_oneapi_external_semaphore_import, 48)
#ifndef __SYCL_ALL_DEVICES_HAVE_ext_oneapi_external_semaphore_import__
#define __SYCL_ALL_DEVICES_HAVE_ext_oneapi_external_semaphore_import__ 0
#endif
#ifndef __SYCL_ANY_DEVICE_HAS_ext_oneapi_external_semaphore_import__
#define __SYCL_ANY_DEVICE_HAS_ext_oneapi_external_semaphore_import__ 0
#endif

// __SYCL_ASPECT(ext_oneapi_mipmap, 50)
#ifndef __SYCL_ALL_DEVICES_HAVE_ext_oneapi_mipmap__
#define __SYCL_ALL_DEVICES_HAVE_ext_oneapi_mipmap__ 0
#endif
#ifndef __SYCL_ANY_DEVICE_HAS_ext_oneapi_mipmap__
#define __SYCL_ANY_DEVICE_HAS_ext_oneapi_mipmap__ 0
#endif

// __SYCL_ASPECT(ext_oneapi_mipmap_anisotropy, 51)
#ifndef __SYCL_ALL_DEVICES_HAVE_ext_oneapi_mipmap_anisotropy__
#define __SYCL_ALL_DEVICES_HAVE_ext_oneapi_mipmap_anisotropy__ 0
#endif
#ifndef __SYCL_ANY_DEVICE_HAS_ext_oneapi_mipmap_anisotropy__
#define __SYCL_ANY_DEVICE_HAS_ext_oneapi_mipmap_anisotropy__ 0
#endif

// __SYCL_ASPECT(ext_oneapi_mipmap_level_reference, 52)
#ifndef __SYCL_ALL_DEVICES_HAVE_ext_oneapi_mipmap_level_reference__
#define __SYCL_ALL_DEVICES_HAVE_ext_oneapi_mipmap_level_reference__ 0
#endif
#ifndef __SYCL_ANY_DEVICE_HAS_ext_oneapi_mipmap_level_reference__
#define __SYCL_ANY_DEVICE_HAS_ext_oneapi_mipmap_level_reference__ 0
#endif

// __SYCL_ASPECT(ext_intel_esimd, 53)
#ifndef __SYCL_ALL_DEVICES_HAVE_ext_intel_esimd__
#define __SYCL_ALL_DEVICES_HAVE_ext_intel_esimd__ 0
#endif
#ifndef __SYCL_ANY_DEVICE_HAS_ext_intel_esimd__
#define __SYCL_ANY_DEVICE_HAS_ext_intel_esimd__ 0
#endif

// __SYCL_ASPECT(ext_oneapi_ballot_group, 54)
#ifndef __SYCL_ALL_DEVICES_HAVE_ext_oneapi_ballot_group__
#define __SYCL_ALL_DEVICES_HAVE_ext_oneapi_ballot_group__ 0
#endif
#ifndef __SYCL_ANY_DEVICE_HAS_ext_oneapi_ballot_group__
#define __SYCL_ANY_DEVICE_HAS_ext_oneapi_ballot_group__ 0
#endif

// __SYCL_ASPECT(ext_oneapi_fixed_size_group, 55)
#ifndef __SYCL_ALL_DEVICES_HAVE_ext_oneapi_fixed_size_group__
#define __SYCL_ALL_DEVICES_HAVE_ext_oneapi_fixed_size_group__ 0
#endif
#ifndef __SYCL_ANY_DEVICE_HAS_ext_oneapi_fixed_size_group__
#define __SYCL_ANY_DEVICE_HAS_ext_oneapi_fixed_size_group__ 0
#endif

// __SYCL_ASPECT(ext_oneapi_opportunistic_group, 56)
#ifndef __SYCL_ALL_DEVICES_HAVE_ext_oneapi_opportunistic_group__
#define __SYCL_ALL_DEVICES_HAVE_ext_oneapi_opportunistic_group__ 0
#endif
#ifndef __SYCL_ANY_DEVICE_HAS_ext_oneapi_opportunistic_group__
#define __SYCL_ANY_DEVICE_HAS_ext_oneapi_opportunistic_group__ 0
#endif

// __SYCL_ASPECT(ext_oneapi_tangle_group, 57)
#ifndef __SYCL_ALL_DEVICES_HAVE_ext_oneapi_tangle_group__
#define __SYCL_ALL_DEVICES_HAVE_ext_oneapi_tangle_group__ 0
#endif
#ifndef __SYCL_ANY_DEVICE_HAS_ext_oneapi_tangle_group__
#define __SYCL_ANY_DEVICE_HAS_ext_oneapi_tangle_group__ 0
#endif

// __SYCL_ASPECT(ext_intel_matrix, 58)
#ifndef __SYCL_ALL_DEVICES_HAVE_ext_intel_matrix__
#define __SYCL_ALL_DEVICES_HAVE_ext_intel_matrix__ 0
#endif
#ifndef __SYCL_ANY_DEVICE_HAS_ext_intel_matrix__
#define __SYCL_ANY_DEVICE_HAS_ext_intel_matrix__ 0
#endif

// __SYCL_ASPECT(ext_oneapi_is_composite, 59)
#ifndef __SYCL_ALL_DEVICES_HAVE_ext_oneapi_is_composite__
#define __SYCL_ALL_DEVICES_HAVE_ext_oneapi_is_composite__ 0
#endif
#ifndef __SYCL_ANY_DEVICE_HAS_ext_oneapi_is_composite__
#define __SYCL_ANY_DEVICE_HAS_ext_oneapi_is_composite__ 0
#endif

// __SYCL_ASPECT(ext_oneapi_is_component, 60)
#ifndef __SYCL_ALL_DEVICES_HAVE_ext_oneapi_is_component__
#define __SYCL_ALL_DEVICES_HAVE_ext_oneapi_is_component__ 0
#endif
#ifndef __SYCL_ANY_DEVICE_HAS_ext_oneapi_is_component__
#define __SYCL_ANY_DEVICE_HAS_ext_oneapi_is_component__ 0
#endif

// __SYCL_ASPECT(ext_oneapi_graph, 61)
#ifndef __SYCL_ALL_DEVICES_HAVE_ext_oneapi_graph__
#define __SYCL_ALL_DEVICES_HAVE_ext_oneapi_graph__ 0
#endif
#ifndef __SYCL_ANY_DEVICE_HAS_ext_oneapi_graph__
#define __SYCL_ANY_DEVICE_HAS_ext_oneapi_graph__ 0
#endif

// __SYCL_ASPECT(ext_intel_fpga_task_sequence, 62)
#ifndef __SYCL_ALL_DEVICES_HAVE_ext_intel_fpga_task_sequence__
#define __SYCL_ALL_DEVICES_HAVE_ext_intel_fpga_task_sequence__ 0
#endif
#ifndef __SYCL_ANY_DEVICE_HAS_ext_intel_fpga_task_sequence__
#define __SYCL_ANY_DEVICE_HAS_ext_intel_fpga_task_sequence__ 0
#endif

// __SYCL_ASPECT(ext_oneapi_limited_graph, 63)
#ifndef __SYCL_ALL_DEVICES_HAVE_ext_oneapi_limited_graph__
#define __SYCL_ALL_DEVICES_HAVE_ext_oneapi_limited_graph__ 0
#endif
#ifndef __SYCL_ANY_DEVICE_HAS_ext_oneapi_limited_graph__
#define __SYCL_ANY_DEVICE_HAS_ext_oneapi_limited_graph__ 0
#endif

// __SYCL_ASPECT(ext_oneapi_private_alloca, 64)
#ifndef __SYCL_ALL_DEVICES_HAVE_ext_oneapi_private_alloca__
#define __SYCL_ALL_DEVICES_HAVE_ext_oneapi_private_alloca__ 0
#endif
#ifndef __SYCL_ANY_DEVICE_HAS_ext_oneapi_private_alloca__
#define __SYCL_ANY_DEVICE_HAS_ext_oneapi_private_alloca__ 0
#endif

// __SYCL_ASPECT(ext_oneapi_cubemap, 65)
#ifndef __SYCL_ALL_DEVICES_HAVE_ext_oneapi_cubemap__
#define __SYCL_ALL_DEVICES_HAVE_ext_oneapi_cubemap__ 0
#endif
#ifndef __SYCL_ANY_DEVICE_HAS_ext_oneapi_cubemap__
#define __SYCL_ANY_DEVICE_HAS_ext_oneapi_cubemap__ 0
#endif

// __SYCL_ASPECT(ext_oneapi_cubemap_seamless_filtering, 66)
#ifndef __SYCL_ALL_DEVICES_HAVE_ext_oneapi_cubemap_seamless_filtering__
#define __SYCL_ALL_DEVICES_HAVE_ext_oneapi_cubemap_seamless_filtering__ 0
#endif
#ifndef __SYCL_ANY_DEVICE_HAS_ext_oneapi_cubemap_seamless_filtering__
#define __SYCL_ANY_DEVICE_HAS_ext_oneapi_cubemap_seamless_filtering__ 0
#endif

// __SYCL_ASPECT(ext_oneapi_bindless_sampled_image_fetch_1d_usm, 67)
#ifndef __SYCL_ALL_DEVICES_HAVE_ext_oneapi_bindless_sampled_image_fetch_1d_usm__
#define __SYCL_ALL_DEVICES_HAVE_ext_oneapi_bindless_sampled_image_fetch_1d_usm__ 0
#endif
#ifndef __SYCL_ANY_DEVICE_HAS_ext_oneapi_bindless_sampled_image_fetch_1d_usm__
#define __SYCL_ANY_DEVICE_HAS_ext_oneapi_bindless_sampled_image_fetch_1d_usm__ 0
#endif

// __SYCL_ASPECT(ext_oneapi_bindless_sampled_image_fetch_1d, 68)
#ifndef __SYCL_ALL_DEVICES_HAVE_ext_oneapi_bindless_sampled_image_fetch_1d__
#define __SYCL_ALL_DEVICES_HAVE_ext_oneapi_bindless_sampled_image_fetch_1d__ 0
#endif
#ifndef __SYCL_ANY_DEVICE_HAS_ext_oneapi_bindless_sampled_image_fetch_1d__
#define __SYCL_ANY_DEVICE_HAS_ext_oneapi_bindless_sampled_image_fetch_1d__ 0
#endif

// __SYCL_ASPECT(ext_oneapi_bindless_sampled_image_fetch_2d_usm, 69)
#ifndef __SYCL_ALL_DEVICES_HAVE_ext_oneapi_bindless_sampled_image_fetch_2d_usm__
#define __SYCL_ALL_DEVICES_HAVE_ext_oneapi_bindless_sampled_image_fetch_2d_usm__ 0
#endif
#ifndef __SYCL_ANY_DEVICE_HAS_ext_oneapi_bindless_sampled_image_fetch_2d_usm__
#define __SYCL_ANY_DEVICE_HAS_ext_oneapi_bindless_sampled_image_fetch_2d_usm__ 0
#endif

// __SYCL_ASPECT(ext_oneapi_bindless_sampled_image_fetch_2d, 70)
#ifndef __SYCL_ALL_DEVICES_HAVE_ext_oneapi_bindless_sampled_image_fetch_2d__
#define __SYCL_ALL_DEVICES_HAVE_ext_oneapi_bindless_sampled_image_fetch_2d__ 0
#endif
#ifndef __SYCL_ANY_DEVICE_HAS_ext_oneapi_bindless_sampled_image_fetch_2d__
#define __SYCL_ANY_DEVICE_HAS_ext_oneapi_bindless_sampled_image_fetch_2d__ 0
#endif

// __SYCL_ASPECT(ext_oneapi_bindless_sampled_image_fetch_3d, 72)
#ifndef __SYCL_ALL_DEVICES_HAVE_ext_oneapi_bindless_sampled_image_fetch_3d__
#define __SYCL_ALL_DEVICES_HAVE_ext_oneapi_bindless_sampled_image_fetch_3d__ 0
#endif
#ifndef __SYCL_ANY_DEVICE_HAS_ext_oneapi_bindless_sampled_image_fetch_3d__
#define __SYCL_ANY_DEVICE_HAS_ext_oneapi_bindless_sampled_image_fetch_3d__ 0
#endif

// __SYCL_ASPECT(ext_oneapi_queue_profiling_tag, 73)
#ifndef __SYCL_ALL_DEVICES_HAVE_ext_oneapi_queue_profiling_tag__
#define __SYCL_ALL_DEVICES_HAVE_ext_oneapi_queue_profiling_tag__ 0
#endif
#ifndef __SYCL_ANY_DEVICE_HAS_ext_oneapi_queue_profiling_tag__
#define __SYCL_ANY_DEVICE_HAS_ext_oneapi_queue_profiling_tag__ 0
#endif

// __SYCL_ASPECT(ext_oneapi_virtual_mem, 74)
#ifndef __SYCL_ALL_DEVICES_HAVE_ext_oneapi_virtual_mem__
#define __SYCL_ALL_DEVICES_HAVE_ext_oneapi_virtual_mem__ 0
#endif
#ifndef __SYCL_ANY_DEVICE_HAS_ext_oneapi_virtual_mem__
#define __SYCL_ANY_DEVICE_HAS_ext_oneapi_virtual_mem__ 0
#endif

// __SYCL_ASPECT(ext_oneapi_cuda_cluster_group, 75)
#ifndef __SYCL_ALL_DEVICES_HAVE_ext_oneapi_cuda_cluster_group__
#define __SYCL_ALL_DEVICES_HAVE_ext_oneapi_cuda_cluster_group__ 0
#endif
#ifndef __SYCL_ANY_DEVICE_HAS_ext_oneapi_cuda_cluster_group__
#define __SYCL_ANY_DEVICE_HAS_ext_oneapi_cuda_cluster_group__ 0
#endif

// __SYCL_ASPECT(ext_oneapi_image_array, 76)
#ifndef __SYCL_ALL_DEVICES_HAVE_ext_oneapi_image_array__
#define __SYCL_ALL_DEVICES_HAVE_ext_oneapi_image_array__ 0
#endif
#ifndef __SYCL_ANY_DEVICE_HAS_ext_oneapi_image_array__
#define __SYCL_ANY_DEVICE_HAS_ext_oneapi_image_array__ 0
#endif

// __SYCL_ASPECT(ext_oneapi_unique_addressing_per_dim, 77)
#ifndef __SYCL_ALL_DEVICES_HAVE_ext_oneapi_unique_addressing_per_dim__
#define __SYCL_ALL_DEVICES_HAVE_ext_oneapi_unique_addressing_per_dim__ 0
#endif
#ifndef __SYCL_ANY_DEVICE_HAS_ext_oneapi_unique_addressing_per_dim__
#define __SYCL_ANY_DEVICE_HAS_ext_oneapi_unique_addressing_per_dim__ 0
#endif

// __SYCL_ASPECT(ext_oneapi_bindless_images_sample_1d_usm, 78)
#ifndef __SYCL_ALL_DEVICES_HAVE_ext_oneapi_bindless_images_sample_1d_usm__
#define __SYCL_ALL_DEVICES_HAVE_ext_oneapi_bindless_images_sample_1d_usm__ 0
#endif
#ifndef __SYCL_ANY_DEVICE_HAS_ext_oneapi_bindless_images_sample_1d_usm__
#define __SYCL_ANY_DEVICE_HAS_ext_oneapi_bindless_images_sample_1d_usm__ 0
#endif

// __SYCL_ASPECT(ext_oneapi_bindless_images_sample_2d_usm, 79)
#ifndef __SYCL_ALL_DEVICES_HAVE_ext_oneapi_bindless_images_sample_2d_usm__
#define __SYCL_ALL_DEVICES_HAVE_ext_oneapi_bindless_images_sample_2d_usm__ 0
#endif
#ifndef __SYCL_ANY_DEVICE_HAS_ext_oneapi_bindless_images_sample_2d_usm__
#define __SYCL_ANY_DEVICE_HAS_ext_oneapi_bindless_images_sample_2d_usm__ 0
#endif

// __SYCL_ASPECT(ext_oneapi_atomic16, 80)
#ifndef __SYCL_ALL_DEVICES_HAVE_ext_oneapi_atomic16__
#define __SYCL_ALL_DEVICES_HAVE_ext_oneapi_atomic16__ 0
#endif
#ifndef __SYCL_ANY_DEVICE_HAS_ext_oneapi_atomic16__
#define __SYCL_ANY_DEVICE_HAS_ext_oneapi_atomic16__ 0
#endif

// __SYCL_ASPECT(ext_oneapi_virtual_functions, 81)
#ifndef __SYCL_ALL_DEVICES_HAVE_ext_oneapi_virtual_functions__
#define __SYCL_ALL_DEVICES_HAVE_ext_oneapi_virtual_functions__ 0
#endif
#ifndef __SYCL_ANY_DEVICE_HAS_ext_oneapi_virtual_functions__
#define __SYCL_ANY_DEVICE_HAS_ext_oneapi_virtual_functions__ 0
#endif

