//==------------------- device_aspect_macros.hpp - SYCL device -------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#ifndef __SYCL_ALL_DEVICES_HAVE_host__
// __SYCL_ASPECT(host, 0)
#define __SYCL_ALL_DEVICES_HAVE_host__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_cpu__
// __SYCL_ASPECT(cpu, 1)
#define __SYCL_ALL_DEVICES_HAVE_cpu__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_gpu__
//__SYCL_ASPECT(gpu, 2)
#define __SYCL_ALL_DEVICES_HAVE_gpu__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_accelerator__
//__SYCL_ASPECT(accelerator, 3)
#define __SYCL_ALL_DEVICES_HAVE_accelerator__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_custom__
//__SYCL_ASPECT(custom, 4)
#define __SYCL_ALL_DEVICES_HAVE_custom__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_fp16__
// __SYCL_ASPECT(fp16, 5)
#define __SYCL_ALL_DEVICES_HAVE_fp16__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_fp64__
// __SYCL_ASPECT(fp64, 6)
#define __SYCL_ALL_DEVICES_HAVE_fp64__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_int64_base_atomics__
// __SYCL_ASPECT_DEPRECATED(int64_base_atomics, 7)
#define __SYCL_ALL_DEVICES_HAVE_int64_base_atomics__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_int64_extended_atomics__
// __SYCL_ASPECT_DEPRECATED(int64_extended_atomics, 8)
#define __SYCL_ALL_DEVICES_HAVE_int64_extended_atomics__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_image__
// __SYCL_ASPECT(image, 9)
#define __SYCL_ALL_DEVICES_HAVE_image__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_online_compiler__
// __SYCL_ASPECT(online_compiler, 10)
#define __SYCL_ALL_DEVICES_HAVE_online_compiler__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_online_linker__
// __SYCL_ASPECT(online_linker, 11)
#define __SYCL_ALL_DEVICES_HAVE_online_linker__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_queue_profiling__
// __SYCL_ASPECT(queue_profiling, 12)
#define __SYCL_ALL_DEVICES_HAVE_queue_profiling__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_usm_device_allocations__
// __SYCL_ASPECT(usm_device_allocations, 13)
#define __SYCL_ALL_DEVICES_HAVE_usm_device_allocations__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_usm_host_allocations__
// __SYCL_ASPECT(usm_host_allocations, 14)
#define __SYCL_ALL_DEVICES_HAVE_usm_host_allocations__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_usm_shared_allocations__
// __SYCL_ASPECT(usm_shared_allocations, 15)
#define __SYCL_ALL_DEVICES_HAVE_usm_shared_allocations__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_usm_restricted_shared_allocations__
// __SYCL_ASPECT(usm_restricted_shared_allocations, 16)
#define __SYCL_ALL_DEVICES_HAVE_usm_restricted_shared_allocations__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_usm_system_allocations__
// __SYCL_ASPECT(usm_system_allocations, 17)
#define __SYCL_ALL_DEVICES_HAVE_usm_system_allocations__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_ext_intel_pci_address__
// __SYCL_ASPECT(ext_intel_pci_address, 18)
#define __SYCL_ALL_DEVICES_HAVE_ext_intel_pci_address__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_ext_intel_gpu_eu_count__
// __SYCL_ASPECT(ext_intel_gpu_eu_count, 19)
#define __SYCL_ALL_DEVICES_HAVE_ext_intel_gpu_eu_count__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_ext_intel_gpu_eu_simd_width__
// __SYCL_ASPECT(ext_intel_gpu_eu_simd_width, 20)
#define __SYCL_ALL_DEVICES_HAVE_ext_intel_gpu_eu_simd_width__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_ext_intel_gpu_slices__
// __SYCL_ASPECT(ext_intel_gpu_slices, 21)
#define __SYCL_ALL_DEVICES_HAVE_ext_intel_gpu_slices__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_ext_intel_gpu_subslices_per_slice__
// __SYCL_ASPECT(ext_intel_gpu_subslices_per_slice, 22)
#define __SYCL_ALL_DEVICES_HAVE_ext_intel_gpu_subslices_per_slice__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_ext_intel_gpu_eu_count_per_subslice__
// __SYCL_ASPECT(ext_intel_gpu_eu_count_per_subslice, 23)
#define __SYCL_ALL_DEVICES_HAVE_ext_intel_gpu_eu_count_per_subslice__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_ext_intel_max_mem_bandwidth__
// __SYCL_ASPECT(ext_intel_max_mem_bandwidth, 24)
#define __SYCL_ALL_DEVICES_HAVE_ext_intel_max_mem_bandwidth__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_ext_intel_mem_channel__
// __SYCL_ASPECT(ext_intel_mem_channel, 25)
#define __SYCL_ALL_DEVICES_HAVE_ext_intel_mem_channel__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_usm_atomic_host_allocations__
// __SYCL_ASPECT(usm_atomic_host_allocations, 26)
#define __SYCL_ALL_DEVICES_HAVE_usm_atomic_host_allocations__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_usm_atomic_shared_allocations__
// __SYCL_ASPECT(usm_atomic_shared_allocations, 27)
#define __SYCL_ALL_DEVICES_HAVE_usm_atomic_shared_allocations__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_atomic64__
// __SYCL_ASPECT(atomic64, 28)
#define __SYCL_ALL_DEVICES_HAVE_atomic64__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_ext_intel_device_info_uuid__
// __SYCL_ASPECT(ext_intel_device_info_uuid, 29)
#define __SYCL_ALL_DEVICES_HAVE_ext_intel_device_info_uuid__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_ext_oneapi_srgb__
// __SYCL_ASPECT(ext_oneapi_srgb, 30)
#define __SYCL_ALL_DEVICES_HAVE_ext_oneapi_srgb__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_ext_oneapi_native_assert__
// __SYCL_ASPECT(ext_oneapi_native_assert, 31)
#define __SYCL_ALL_DEVICES_HAVE_ext_oneapi_native_assert__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_host_debuggable__
// __SYCL_ASPECT(host_debuggable, 32)
#define __SYCL_ALL_DEVICES_HAVE_host_debuggable__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_ext_intel_gpu_hw_threads_per_eu__
// __SYCL_ASPECT(ext_intel_gpu_hw_threads_per_eu, 33)
#define __SYCL_ALL_DEVICES_HAVE_ext_intel_gpu_hw_threads_per_eu__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_ext_oneapi_cuda_async_barrier__
// __SYCL_ASPECT(ext_oneapi_cuda_async_barrier, 34)
#define __SYCL_ALL_DEVICES_HAVE_ext_oneapi_cuda_async_barrier__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_ext_intel_free_memory__
// __SYCL_ASPECT(ext_intel_free_memory, 36)
#define __SYCL_ALL_DEVICES_HAVE_ext_intel_free_memory__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_ext_intel_device_id__
// __SYCL_ASPECT(ext_intel_device_id, 37)
#define __SYCL_ALL_DEVICES_HAVE_ext_intel_device_id__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_ext_intel_memory_clock_rate__
// __SYCL_ASPECT(ext_intel_memory_clock_rate, 38)
#define __SYCL_ALL_DEVICES_HAVE_ext_intel_memory_clock_rate__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_ext_intel_memory_bus_width__
// __SYCL_ASPECT(ext_intel_memory_bus_width, 39)
#define __SYCL_ALL_DEVICES_HAVE_ext_intel_memory_bus_width__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_emulated__
// __SYCL_ASPECT(emulated, 40)
#define __SYCL_ALL_DEVICES_HAVE_emulated__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_ext_intel_legacy_image__
// __SYCL_ASPECT(ext_intel_legacy_image, 41)
#define __SYCL_ALL_DEVICES_HAVE_ext_intel_legacy_image__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_ext_oneapi_bindless_images__
// __SYCL_ASPECT(ext_oneapi_bindless_images, 42)
#define __SYCL_ALL_DEVICES_HAVE_ext_oneapi_bindless_images__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_ext_oneapi_bindless_images_shared_usm__
//__SYCL_ASPECT(ext_oneapi_bindless_images_shared_usm, 43)
#define __SYCL_ALL_DEVICES_HAVE_ext_oneapi_bindless_images_shared_usm__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_ext_oneapi_bindless_images_1d_usm__
//__SYCL_ASPECT(ext_oneapi_bindless_images_1d_usm, 44)
#define __SYCL_ALL_DEVICES_HAVE_ext_oneapi_bindless_images_1d_usm__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_ext_oneapi_bindless_images_2d_usm__
//__SYCL_ASPECT(ext_oneapi_bindless_images_2d_usm, 45)
#define __SYCL_ALL_DEVICES_HAVE_ext_oneapi_bindless_images_2d_usm__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_ext_oneapi_external_memory_import__
//__SYCL_ASPECT(ext_oneapi_external_memory_import, 46)
#define __SYCL_ALL_DEVICES_HAVE_ext_oneapi_external_memory_import__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_ext_oneapi_external_semaphore_import__
//__SYCL_ASPECT(ext_oneapi_external_semaphore_import, 48)
#define __SYCL_ALL_DEVICES_HAVE_ext_oneapi_external_semaphore_import__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_ext_oneapi_mipmap__
//__SYCL_ASPECT(ext_oneapi_mipmap, 50)
#define __SYCL_ALL_DEVICES_HAVE_ext_oneapi_mipmap__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_ext_oneapi_mipmap_anisotropy__
//__SYCL_ASPECT(ext_oneapi_mipmap_anisotropy, 51)
#define __SYCL_ALL_DEVICES_HAVE_ext_oneapi_mipmap_anisotropy__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_ext_oneapi_mipmap_level_reference__
//__SYCL_ASPECT(ext_oneapi_mipmap_level_reference, 52)
#define __SYCL_ALL_DEVICES_HAVE_ext_oneapi_mipmap_level_reference__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_ext_intel_esimd__
//__SYCL_ASPECT(ext_intel_esimd, 53)
#define __SYCL_ALL_DEVICES_HAVE_ext_intel_esimd__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_ext_oneapi_ballot_group__
// __SYCL_ASPECT(ext_oneapi_ballot_group, 54)
#define __SYCL_ALL_DEVICES_HAVE_ext_oneapi_ballot_group__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_ext_oneapi_fixed_size_group__
// __SYCL_ASPECT(ext_oneapi_fixed_size_group, 55)
#define __SYCL_ALL_DEVICES_HAVE_ext_oneapi_fixed_size_group__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_ext_oneapi_opportunistic_group__
// __SYCL_ASPECT(ext_oneapi_opportunistic_group, 56)
#define __SYCL_ALL_DEVICES_HAVE_ext_oneapi_opportunistic_group__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_ext_oneapi_tangle_group__
// __SYCL_ASPECT(ext_oneapi_tangle_group, 57)
#define __SYCL_ALL_DEVICES_HAVE_ext_oneapi_tangle_group__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_ext_intel_matrix__
// __SYCL_ASPECT(ext_intel_matrix, 58)
#define __SYCL_ALL_DEVICES_HAVE_ext_intel_matrix__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_ext_oneapi_is_composite__
// __SYCL_ASPECT(ext_oneapi_is_composite, 59)
#define __SYCL_ALL_DEVICES_HAVE_ext_oneapi_is_composite__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_ext_oneapi_is_component__
// __SYCL_ASPECT(ext_oneapi_is_component, 60)
#define __SYCL_ALL_DEVICES_HAVE_ext_oneapi_is_component__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_ext_oneapi_graph__
// __SYCL_ASPECT(ext_oneapi_graph, 61)
#define __SYCL_ALL_DEVICES_HAVE_ext_oneapi_graph__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_ext_intel_fpga_task_sequence__
// __SYCL_ASPECT(ext_intel_fpga_task_sequence, 62)
#define __SYCL_ALL_DEVICES_HAVE_ext_intel_fpga_task_sequence__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_ext_oneapi_limited_graph__
// __SYCL_ASPECT(ext_oneapi_limited_graph, 63)
#define __SYCL_ALL_DEVICES_HAVE_ext_oneapi_limited_graph__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_ext_oneapi_private_alloca__
// __SYCL_ASPECT(ext_oneapi_private_alloca, 64)
#define __SYCL_ALL_DEVICES_HAVE_ext_oneapi_private_alloca__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_ext_oneapi_cubemap__
// __SYCL_ASPECT(ext_oneapi_cubemap, 65)
#define __SYCL_ALL_DEVICES_HAVE_ext_oneapi_cubemap__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_ext_oneapi_cubemap_seamless_filtering__
// __SYCL_ASPECT(ext_oneapi_cubemap_seamless_filtering, 66)
#define __SYCL_ALL_DEVICES_HAVE_ext_oneapi_cubemap_seamless_filtering__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_ext_oneapi_bindless_sampled_image_fetch_1d_usm__
//__SYCL_ASPECT(ext_oneapi_bindless_sampled_image_fetch_1d_usm, 67)
#define __SYCL_ALL_DEVICES_HAVE_ext_oneapi_bindless_sampled_image_fetch_1d_usm__ \
  0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_ext_oneapi_bindless_sampled_image_fetch_1d__
//__SYCL_ASPECT(ext_oneapi_bindless_sampled_image_fetch_1d, 68)
#define __SYCL_ALL_DEVICES_HAVE_ext_oneapi_bindless_sampled_image_fetch_1d__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_ext_oneapi_bindless_sampled_image_fetch_2d_usm__
//__SYCL_ASPECT(ext_oneapi_bindless_sampled_image_fetch_2d_usm, 69)
#define __SYCL_ALL_DEVICES_HAVE_ext_oneapi_bindless_sampled_image_fetch_2d_usm__ \
  0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_ext_oneapi_bindless_sampled_image_fetch_2d__
//__SYCL_ASPECT(ext_oneapi_bindless_sampled_image_fetch_2d, 70)
#define __SYCL_ALL_DEVICES_HAVE_ext_oneapi_bindless_sampled_image_fetch_2d__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_ext_oneapi_bindless_sampled_image_fetch_3d__
//__SYCL_ASPECT(ext_oneapi_bindless_sampled_image_fetch_3d, 72)
#define __SYCL_ALL_DEVICES_HAVE_ext_oneapi_bindless_sampled_image_fetch_3d__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_ext_oneapi_queue_profiling_tag__
// __SYCL_ASPECT(ext_oneapi_queue_profiling_tag, 73)
#define __SYCL_ALL_DEVICES_HAVE_ext_oneapi_queue_profiling_tag__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_ext_oneapi_virtual_mem__
// __SYCL_ASPECT(ext_oneapi_virtual_mem, 74)
#define __SYCL_ALL_DEVICES_HAVE_ext_oneapi_virtual_mem__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_ext_oneapi_cuda_cluster_group__
// __SYCL_ASPECT(ext_oneapi_cuda_cluster_group, 75)
#define __SYCL_ALL_DEVICES_HAVE_ext_oneapi_cuda_cluster_group__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_ext_oneapi_image_array__
//__SYCL_ASPECT(ext_oneapi_image_array, 76)
#define __SYCL_ALL_DEVICES_HAVE_ext_oneapi_image_array__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_ext_oneapi_unique_addressing_per_dim__
//__SYCL_ASPECT(ext_oneapi_unique_addressing_per_dim, 77)
#define __SYCL_ALL_DEVICES_HAVE_ext_oneapi_unique_addressing_per_dim__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_ext_oneapi_bindless_images_sample_1d_usm__
//__SYCL_ASPECT(ext_oneapi_bindless_images_sample_1d_usm, 78)
#define __SYCL_ALL_DEVICES_HAVE_ext_oneapi_bindless_images_sample_1d_usm__ 0
#endif

#ifndef __SYCL_ALL_DEVICES_HAVE_ext_oneapi_bindless_images_sample_2d_usm__
//__SYCL_ASPECT(ext_oneapi_bindless_images_sample_2d_usm, 79)
#define __SYCL_ALL_DEVICES_HAVE_ext_oneapi_bindless_images_sample_2d_usm__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_host__
// __SYCL_ASPECT(host, 0)
#define __SYCL_ANY_DEVICE_HAS_host__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_cpu__
// __SYCL_ASPECT(cpu, 1)
#define __SYCL_ANY_DEVICE_HAS_cpu__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_gpu__
//__SYCL_ASPECT(gpu, 2)
#define __SYCL_ANY_DEVICE_HAS_gpu__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_accelerator__
//__SYCL_ASPECT(accelerator, 3)
#define __SYCL_ANY_DEVICE_HAS_accelerator__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_custom__
//__SYCL_ASPECT(custom, 4)
#define __SYCL_ANY_DEVICE_HAS_custom__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_fp16__
// __SYCL_ASPECT(fp16, 5)
#define __SYCL_ANY_DEVICE_HAS_fp16__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_fp64__
// __SYCL_ASPECT(fp64, 6)
#define __SYCL_ANY_DEVICE_HAS_fp64__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_int64_base_atomics__
// __SYCL_ASPECT_DEPRECATED(int64_base_atomics, 7)
#define __SYCL_ANY_DEVICE_HAS_int64_base_atomics__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_int64_extended_atomics__
// __SYCL_ASPECT_DEPRECATED(int64_extended_atomics, 8)
#define __SYCL_ANY_DEVICE_HAS_int64_extended_atomics__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_image__
// __SYCL_ASPECT(image, 9)
#define __SYCL_ANY_DEVICE_HAS_image__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_online_compiler__
// __SYCL_ASPECT(online_compiler, 10)
#define __SYCL_ANY_DEVICE_HAS_online_compiler__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_online_linker__
// __SYCL_ASPECT(online_linker, 11)
#define __SYCL_ANY_DEVICE_HAS_online_linker__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_queue_profiling__
// __SYCL_ASPECT(queue_profiling, 12)
#define __SYCL_ANY_DEVICE_HAS_queue_profiling__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_usm_device_allocations__
// __SYCL_ASPECT(usm_device_allocations, 13)
#define __SYCL_ANY_DEVICE_HAS_usm_device_allocations__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_usm_host_allocations__
// __SYCL_ASPECT(usm_host_allocations, 14)
#define __SYCL_ANY_DEVICE_HAS_usm_host_allocations__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_usm_shared_allocations__
// __SYCL_ASPECT(usm_shared_allocations, 15)
#define __SYCL_ANY_DEVICE_HAS_usm_shared_allocations__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_usm_restricted_shared_allocations__
// __SYCL_ASPECT(usm_restricted_shared_allocations, 16)
#define __SYCL_ANY_DEVICE_HAS_usm_restricted_shared_allocations__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_usm_system_allocations__
// __SYCL_ASPECT(usm_system_allocations, 17)
#define __SYCL_ANY_DEVICE_HAS_usm_system_allocations__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_ext_intel_pci_address__
// __SYCL_ASPECT(ext_intel_pci_address, 18)
#define __SYCL_ANY_DEVICE_HAS_ext_intel_pci_address__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_ext_intel_gpu_eu_count__
// __SYCL_ASPECT(ext_intel_gpu_eu_count, 19)
#define __SYCL_ANY_DEVICE_HAS_ext_intel_gpu_eu_count__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_ext_intel_gpu_eu_simd_width__
// __SYCL_ASPECT(ext_intel_gpu_eu_simd_width, 20)
#define __SYCL_ANY_DEVICE_HAS_ext_intel_gpu_eu_simd_width__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_ext_intel_gpu_slices__
// __SYCL_ASPECT(ext_intel_gpu_slices, 21)
#define __SYCL_ANY_DEVICE_HAS_ext_intel_gpu_slices__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_ext_intel_gpu_subslices_per_slice__
// __SYCL_ASPECT(ext_intel_gpu_subslices_per_slice, 22)
#define __SYCL_ANY_DEVICE_HAS_ext_intel_gpu_subslices_per_slice__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_ext_intel_gpu_eu_count_per_subslice__
// __SYCL_ASPECT(ext_intel_gpu_eu_count_per_subslice, 23)
#define __SYCL_ANY_DEVICE_HAS_ext_intel_gpu_eu_count_per_subslice__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_ext_intel_max_mem_bandwidth__
// __SYCL_ASPECT(ext_intel_max_mem_bandwidth, 24)
#define __SYCL_ANY_DEVICE_HAS_ext_intel_max_mem_bandwidth__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_ext_intel_mem_channel__
// __SYCL_ASPECT(ext_intel_mem_channel, 25)
#define __SYCL_ANY_DEVICE_HAS_ext_intel_mem_channel__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_usm_atomic_host_allocations__
// __SYCL_ASPECT(usm_atomic_host_allocations, 26)
#define __SYCL_ANY_DEVICE_HAS_usm_atomic_host_allocations__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_usm_atomic_shared_allocations__
// __SYCL_ASPECT(usm_atomic_shared_allocations, 27)
#define __SYCL_ANY_DEVICE_HAS_usm_atomic_shared_allocations__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_atomic64__
// __SYCL_ASPECT(atomic64, 28)
#define __SYCL_ANY_DEVICE_HAS_atomic64__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_ext_intel_device_info_uuid__
// __SYCL_ASPECT(ext_intel_device_info_uuid, 29)
#define __SYCL_ANY_DEVICE_HAS_ext_intel_device_info_uuid__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_ext_oneapi_srgb__
// __SYCL_ASPECT(ext_oneapi_srgb, 30)
#define __SYCL_ANY_DEVICE_HAS_ext_oneapi_srgb__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_ext_oneapi_native_assert__
// __SYCL_ASPECT(ext_oneapi_native_assert, 31)
#define __SYCL_ANY_DEVICE_HAS_ext_oneapi_native_assert__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_host_debuggable__
// __SYCL_ASPECT(host_debuggable, 32)
#define __SYCL_ANY_DEVICE_HAS_host_debuggable__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_ext_intel_gpu_hw_threads_per_eu__
// __SYCL_ASPECT(ext_intel_gpu_hw_threads_per_eu, 33)
#define __SYCL_ANY_DEVICE_HAS_ext_intel_gpu_hw_threads_per_eu__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_ext_oneapi_cuda_async_barrier__
// __SYCL_ASPECT(ext_oneapi_cuda_async_barrier, 34)
#define __SYCL_ANY_DEVICE_HAS_ext_oneapi_cuda_async_barrier__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_ext_intel_free_memory__
// __SYCL_ASPECT(ext_intel_free_memory, 36)
#define __SYCL_ANY_DEVICE_HAS_ext_intel_free_memory__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_ext_intel_device_id__
// __SYCL_ASPECT(ext_intel_device_id, 37)
#define __SYCL_ANY_DEVICE_HAS_ext_intel_device_id__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_ext_intel_memory_clock_rate__
// __SYCL_ASPECT(ext_intel_memory_clock_rate, 38)
#define __SYCL_ANY_DEVICE_HAS_ext_intel_memory_clock_rate__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_ext_intel_memory_bus_width__
// __SYCL_ASPECT(ext_intel_memory_bus_width, 39)
#define __SYCL_ANY_DEVICE_HAS_ext_intel_memory_bus_width__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_emulated__
// __SYCL_ASPECT(emulated, 40)
#define __SYCL_ANY_DEVICE_HAS_emulated__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_ext_intel_legacy_image__
// __SYCL_ASPECT(ext_intel_legacy_image, 41)
#define __SYCL_ANY_DEVICE_HAS_ext_intel_legacy_image__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_ext_oneapi_bindless_images__
// __SYCL_ASPECT(ext_oneapi_bindless_images, 42)
#define __SYCL_ANY_DEVICE_HAS_ext_oneapi_bindless_images__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_ext_oneapi_bindless_images_shared_usm__
//__SYCL_ASPECT(ext_oneapi_bindless_images_shared_usm, 43)
#define __SYCL_ANY_DEVICE_HAS_ext_oneapi_bindless_images_shared_usm__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_ext_oneapi_bindless_images_1d_usm__
//__SYCL_ASPECT(ext_oneapi_bindless_images_1d_usm, 44)
#define __SYCL_ANY_DEVICE_HAS_ext_oneapi_bindless_images_1d_usm__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_ext_oneapi_bindless_images_2d_usm__
//__SYCL_ASPECT(ext_oneapi_bindless_images_2d_usm, 45)
#define __SYCL_ANY_DEVICE_HAS_ext_oneapi_bindless_images_2d_usm__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_ext_oneapi_external_memory_import__
//__SYCL_ASPECT(ext_oneapi_external_memory_import, 46)
#define __SYCL_ANY_DEVICE_HAS_ext_oneapi_external_memory_import__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_ext_oneapi_external_semaphore_import__
//__SYCL_ASPECT(ext_oneapi_external_semaphore_import, 48)
#define __SYCL_ANY_DEVICE_HAS_ext_oneapi_external_semaphore_import__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_ext_oneapi_mipmap__
//__SYCL_ASPECT(ext_oneapi_mipmap, 50)
#define __SYCL_ANY_DEVICE_HAS_ext_oneapi_mipmap__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_ext_oneapi_mipmap_anisotropy__
//__SYCL_ASPECT(ext_oneapi_mipmap_anisotropy, 51)
#define __SYCL_ANY_DEVICE_HAS_ext_oneapi_mipmap_anisotropy__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_ext_oneapi_mipmap_level_reference__
//__SYCL_ASPECT(ext_oneapi_mipmap_level_reference, 52)
#define __SYCL_ANY_DEVICE_HAS_ext_oneapi_mipmap_level_reference__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_ext_intel_esimd__
//__SYCL_ASPECT(ext_intel_esimd, 53)
#define __SYCL_ANY_DEVICE_HAS_ext_intel_esimd__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_ext_oneapi_ballot_group__
// __SYCL_ASPECT(ext_oneapi_ballot_group, 54)
#define __SYCL_ANY_DEVICE_HAS_ext_oneapi_ballot_group__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_ext_oneapi_fixed_size_group__
// __SYCL_ASPECT(ext_oneapi_fixed_size_group, 55)
#define __SYCL_ANY_DEVICE_HAS_ext_oneapi_fixed_size_group__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_ext_oneapi_opportunistic_group__
// __SYCL_ASPECT(ext_oneapi_opportunistic_group, 56)
#define __SYCL_ANY_DEVICE_HAS_ext_oneapi_opportunistic_group__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_ext_oneapi_tangle_group__
// __SYCL_ASPECT(ext_oneapi_tangle_group, 57)
#define __SYCL_ANY_DEVICE_HAS_ext_oneapi_tangle_group__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_ext_intel_matrix__
// __SYCL_ASPECT(ext_intel_matrix, 58)
#define __SYCL_ANY_DEVICE_HAS_ext_intel_matrix__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_ext_oneapi_is_composite__
// __SYCL_ASPECT(ext_oneapi_is_composite, 59)
#define __SYCL_ANY_DEVICE_HAS_ext_oneapi_is_composite__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_ext_oneapi_is_component__
// __SYCL_ASPECT(ext_oneapi_is_component, 60)
#define __SYCL_ANY_DEVICE_HAS_ext_oneapi_is_component__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_ext_oneapi_graph__
// __SYCL_ASPECT(ext_oneapi_graph, 61)
#define __SYCL_ANY_DEVICE_HAS_ext_oneapi_graph__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_ext_intel_fpga_task_sequence__
// __SYCL_ASPECT(ext_intel_fpga_task_sequence__, 62)
#define __SYCL_ANY_DEVICE_HAS_ext_intel_fpga_task_sequence__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_ext_oneapi_limited_graph__
// __SYCL_ASPECT(ext_oneapi_limited_graph, 63)
#define __SYCL_ANY_DEVICE_HAS_ext_oneapi_limited_graph__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_ext_oneapi_private_alloca__
// __SYCL_ASPECT(ext_oneapi_private_alloca, 64)
#define __SYCL_ANY_DEVICE_HAS_ext_oneapi_private_alloca__ 1
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_ext_oneapi_cubemap__
// __SYCL_ASPECT(ext_oneapi_cubemap, 65)
#define __SYCL_ANY_DEVICE_HAS_ext_oneapi_cubemap__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_ext_oneapi_cubemap_seamless_filtering__
// __SYCL_ASPECT(ext_oneapi_cubemap_seamless_filtering, 66)
#define __SYCL_ANY_DEVICE_HAS_ext_oneapi_cubemap_seamless_filtering__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_ext_oneapi_bindless_sampled_image_fetch_1d_usm__
//__SYCL_ASPECT(ext_oneapi_bindless_sampled_image_fetch_1d_usm, 67)
#define __SYCL_ANY_DEVICE_HAS_ext_oneapi_bindless_sampled_image_fetch_1d_usm__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_ext_oneapi_bindless_sampled_image_fetch_1d__
//__SYCL_ASPECT(ext_oneapi_bindless_sampled_image_fetch_1d, 68)
#define __SYCL_ANY_DEVICE_HAS_ext_oneapi_bindless_sampled_image_fetch_1d__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_ext_oneapi_bindless_sampled_image_fetch_2d_usm__
//__SYCL_ASPECT(ext_oneapi_bindless_sampled_image_fetch_2d_usm, 69)
#define __SYCL_ANY_DEVICE_HAS_ext_oneapi_bindless_sampled_image_fetch_2d_usm__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_ext_oneapi_bindless_sampled_image_fetch_2d__
//__SYCL_ASPECT(ext_oneapi_bindless_sampled_image_fetch_2d, 70)
#define __SYCL_ANY_DEVICE_HAS_ext_oneapi_bindless_sampled_image_fetch_2d__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_ext_oneapi_bindless_sampled_image_fetch_3d__
//__SYCL_ASPECT(ext_oneapi_bindless_sampled_image_fetch_3d, 72)
#define __SYCL_ANY_DEVICE_HAS_ext_oneapi_bindless_sampled_image_fetch_3d__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_ext_oneapi_queue_profiling_tag__
// __SYCL_ASPECT(ext_oneapi_queue_profiling_tag, 73)
#define __SYCL_ANY_DEVICE_HAS_ext_oneapi_queue_profiling_tag__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_ext_oneapi_virtual_mem__
// __SYCL_ASPECT(ext_oneapi_virtual_mem, 74)
#define __SYCL_ANY_DEVICE_HAS_ext_oneapi_virtual_mem__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_ext_oneapi_cuda_cluster_group__
// __SYCL_ASPECT(ext_oneapi_cuda_cluster_group, 75)
#define __SYCL_ANY_DEVICE_HAS_ext_oneapi_cuda_cluster_group__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_ext_oneapi_image_array__
//__SYCL_ASPECT(ext_oneapi_image_array, 76)
#define __SYCL_ANY_DEVICE_HAS_ext_oneapi_image_array__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_ext_oneapi_unique_addressing_per_dim__
//__SYCL_ASPECT(ext_oneapi_unique_addressing_per_dim, 77)
#define __SYCL_ANY_DEVICE_HAS_ext_oneapi_unique_addressing_per_dim__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_ext_oneapi_bindless_images_sample_1d_usm__
//__SYCL_ASPECT(ext_oneapi_bindless_images_sample_1d_usm, 78)
#define __SYCL_ANY_DEVICE_HAS_ext_oneapi_bindless_images_sample_1d_usm__ 0
#endif

#ifndef __SYCL_ANY_DEVICE_HAS_ext_oneapi_bindless_images_sample_2d_usm__
//__SYCL_ASPECT(ext_oneapi_bindless_images_sample_2d_usm, 79)
#define __SYCL_ANY_DEVICE_HAS_ext_oneapi_bindless_images_sample_2d_usm__ 0
#endif
