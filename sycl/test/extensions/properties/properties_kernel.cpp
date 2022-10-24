// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -fsyntax-only -Xclang -verify %s
// expected-no-diagnostics

#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental;

using device_has_all =
    decltype(device_has<
             aspect::host, aspect::cpu, aspect::gpu, aspect::accelerator,
             aspect::custom, aspect::fp16, aspect::fp64, aspect::image,
             aspect::online_compiler, aspect::online_linker,
             aspect::queue_profiling, aspect::usm_device_allocations,
             aspect::usm_host_allocations, aspect::usm_shared_allocations,
             aspect::usm_restricted_shared_allocations,
             aspect::usm_system_allocations, aspect::ext_intel_pci_address,
             aspect::ext_intel_gpu_eu_count,
             aspect::ext_intel_gpu_eu_simd_width, aspect::ext_intel_gpu_slices,
             aspect::ext_intel_gpu_subslices_per_slice,
             aspect::ext_intel_gpu_eu_count_per_subslice,
             aspect::ext_intel_max_mem_bandwidth, aspect::ext_intel_mem_channel,
             aspect::usm_atomic_host_allocations,
             aspect::usm_atomic_shared_allocations, aspect::atomic64,
             aspect::ext_intel_device_info_uuid, aspect::ext_oneapi_srgb,
             aspect::ext_oneapi_native_assert, aspect::host_debuggable,
             aspect::ext_intel_gpu_hw_threads_per_eu,
             aspect::ext_oneapi_cuda_async_barrier, aspect::ext_oneapi_bfloat16,
             aspect::ext_intel_free_memory, aspect::ext_intel_device_id>);

int main() {
  static_assert(is_property_key<work_group_size_key>::value);
  static_assert(is_property_key<work_group_size_hint_key>::value);
  static_assert(is_property_key<sub_group_size_key>::value);

  static_assert(is_property_value<decltype(work_group_size<1>)>::value);
  static_assert(is_property_value<decltype(work_group_size<2, 2>)>::value);
  static_assert(is_property_value<decltype(work_group_size<3, 3, 3>)>::value);
  static_assert(is_property_value<decltype(work_group_size_hint<4>)>::value);
  static_assert(is_property_value<decltype(work_group_size_hint<5, 5>)>::value);
  static_assert(
      is_property_value<decltype(work_group_size_hint<6, 6, 6>)>::value);
  static_assert(is_property_value<decltype(sub_group_size<7>)>::value);
  static_assert(is_property_value<decltype(sub_group_size<7>)>::value);
  static_assert(is_property_value<decltype(device_has<>)>::value);
  static_assert(is_property_value<decltype(device_has<aspect::host>)>::value);
  static_assert(is_property_value<decltype(device_has<aspect::cpu>)>::value);
  static_assert(is_property_value<decltype(device_has<aspect::gpu>)>::value);
  static_assert(
      is_property_value<decltype(device_has<aspect::accelerator>)>::value);
  static_assert(is_property_value<decltype(device_has<aspect::custom>)>::value);
  static_assert(is_property_value<decltype(device_has<aspect::fp16>)>::value);
  static_assert(is_property_value<decltype(device_has<aspect::fp64>)>::value);
  static_assert(is_property_value<decltype(device_has<aspect::image>)>::value);
  static_assert(
      is_property_value<decltype(device_has<aspect::online_compiler>)>::value);
  static_assert(
      is_property_value<decltype(device_has<aspect::online_linker>)>::value);
  static_assert(
      is_property_value<decltype(device_has<aspect::queue_profiling>)>::value);
  static_assert(is_property_value<
                decltype(device_has<aspect::usm_device_allocations>)>::value);
  static_assert(is_property_value<
                decltype(device_has<aspect::usm_host_allocations>)>::value);
  static_assert(is_property_value<
                decltype(device_has<aspect::usm_shared_allocations>)>::value);
  static_assert(
      is_property_value<
          decltype(device_has<aspect::usm_restricted_shared_allocations>)>::
          value);
  static_assert(is_property_value<
                decltype(device_has<aspect::usm_system_allocations>)>::value);
  static_assert(is_property_value<
                decltype(device_has<aspect::ext_intel_pci_address>)>::value);
  static_assert(is_property_value<
                decltype(device_has<aspect::ext_intel_gpu_eu_count>)>::value);
  static_assert(
      is_property_value<
          decltype(device_has<aspect::ext_intel_gpu_eu_simd_width>)>::value);
  static_assert(is_property_value<
                decltype(device_has<aspect::ext_intel_gpu_slices>)>::value);
  static_assert(
      is_property_value<
          decltype(device_has<aspect::ext_intel_gpu_subslices_per_slice>)>::
          value);
  static_assert(
      is_property_value<
          decltype(device_has<aspect::ext_intel_gpu_eu_count_per_subslice>)>::
          value);
  static_assert(
      is_property_value<
          decltype(device_has<aspect::ext_intel_max_mem_bandwidth>)>::value);
  static_assert(is_property_value<
                decltype(device_has<aspect::ext_intel_mem_channel>)>::value);
  static_assert(
      is_property_value<
          decltype(device_has<aspect::usm_atomic_host_allocations>)>::value);
  static_assert(
      is_property_value<
          decltype(device_has<aspect::usm_atomic_shared_allocations>)>::value);
  static_assert(
      is_property_value<decltype(device_has<aspect::atomic64>)>::value);
  static_assert(
      is_property_value<
          decltype(device_has<aspect::ext_intel_device_info_uuid>)>::value);
  static_assert(
      is_property_value<decltype(device_has<aspect::ext_oneapi_srgb>)>::value);
  static_assert(is_property_value<
                decltype(device_has<aspect::ext_oneapi_native_assert>)>::value);
  static_assert(
      is_property_value<decltype(device_has<aspect::host_debuggable>)>::value);
  static_assert(is_property_value<
                decltype(device_has<aspect::ext_intel_gpu_hw_threads_per_eu>)>::
                    value);
  static_assert(
      is_property_value<
          decltype(device_has<aspect::ext_oneapi_cuda_async_barrier>)>::value);
  static_assert(is_property_value<
                decltype(device_has<aspect::ext_oneapi_bfloat16>)>::value);
  static_assert(is_property_value<
                decltype(device_has<aspect::ext_intel_free_memory>)>::value);
  static_assert(is_property_value<
                decltype(device_has<aspect::ext_intel_device_id>)>::value);
  static_assert(is_property_value<device_has_all>::value);

  static_assert(
      std::is_same_v<work_group_size_key, decltype(work_group_size<8>)::key_t>);
  static_assert(std::is_same_v<work_group_size_key,
                               decltype(work_group_size<9, 9>)::key_t>);
  static_assert(std::is_same_v<work_group_size_key,
                               decltype(work_group_size<10, 10, 10>)::key_t>);
  static_assert(std::is_same_v<work_group_size_hint_key,
                               decltype(work_group_size_hint<11>)::key_t>);
  static_assert(std::is_same_v<work_group_size_hint_key,
                               decltype(work_group_size_hint<12, 12>)::key_t>);
  static_assert(
      std::is_same_v<work_group_size_hint_key,
                     decltype(work_group_size_hint<13, 13, 13>)::key_t>);
  static_assert(
      std::is_same_v<sub_group_size_key, decltype(sub_group_size<14>)::key_t>);
  static_assert(std::is_same_v<device_has_key, decltype(device_has<>)::key_t>);
  static_assert(std::is_same_v<device_has_key, decltype(device_has<>)::key_t>);
  static_assert(std::is_same_v<device_has_key,
                               decltype(device_has<aspect::host>)::key_t>);
  static_assert(
      std::is_same_v<device_has_key, decltype(device_has<aspect::cpu>)::key_t>);
  static_assert(
      std::is_same_v<device_has_key, decltype(device_has<aspect::gpu>)::key_t>);
  static_assert(
      std::is_same_v<device_has_key,
                     decltype(device_has<aspect::accelerator>)::key_t>);
  static_assert(std::is_same_v<device_has_key,
                               decltype(device_has<aspect::custom>)::key_t>);
  static_assert(std::is_same_v<device_has_key,
                               decltype(device_has<aspect::fp16>)::key_t>);
  static_assert(std::is_same_v<device_has_key,
                               decltype(device_has<aspect::fp64>)::key_t>);
  static_assert(std::is_same_v<device_has_key,
                               decltype(device_has<aspect::image>)::key_t>);
  static_assert(
      std::is_same_v<device_has_key,
                     decltype(device_has<aspect::online_compiler>)::key_t>);
  static_assert(
      std::is_same_v<device_has_key,
                     decltype(device_has<aspect::online_linker>)::key_t>);
  static_assert(
      std::is_same_v<device_has_key,
                     decltype(device_has<aspect::queue_profiling>)::key_t>);
  static_assert(std::is_same_v<
                device_has_key,
                decltype(device_has<aspect::usm_device_allocations>)::key_t>);
  static_assert(std::is_same_v<
                device_has_key,
                decltype(device_has<aspect::usm_host_allocations>)::key_t>);
  static_assert(std::is_same_v<
                device_has_key,
                decltype(device_has<aspect::usm_shared_allocations>)::key_t>);
  static_assert(std::is_same_v<
                device_has_key,
                decltype(device_has<
                         aspect::usm_restricted_shared_allocations>)::key_t>);
  static_assert(std::is_same_v<
                device_has_key,
                decltype(device_has<aspect::usm_system_allocations>)::key_t>);
  static_assert(std::is_same_v<
                device_has_key,
                decltype(device_has<aspect::ext_intel_pci_address>)::key_t>);
  static_assert(std::is_same_v<
                device_has_key,
                decltype(device_has<aspect::ext_intel_gpu_eu_count>)::key_t>);
  static_assert(
      std::is_same_v<
          device_has_key,
          decltype(device_has<aspect::ext_intel_gpu_eu_simd_width>)::key_t>);
  static_assert(std::is_same_v<
                device_has_key,
                decltype(device_has<aspect::ext_intel_gpu_slices>)::key_t>);
  static_assert(std::is_same_v<
                device_has_key,
                decltype(device_has<
                         aspect::ext_intel_gpu_subslices_per_slice>)::key_t>);
  static_assert(std::is_same_v<
                device_has_key,
                decltype(device_has<
                         aspect::ext_intel_gpu_eu_count_per_subslice>)::key_t>);
  static_assert(
      std::is_same_v<
          device_has_key,
          decltype(device_has<aspect::ext_intel_max_mem_bandwidth>)::key_t>);
  static_assert(std::is_same_v<
                device_has_key,
                decltype(device_has<aspect::ext_intel_mem_channel>)::key_t>);
  static_assert(
      std::is_same_v<
          device_has_key,
          decltype(device_has<aspect::usm_atomic_host_allocations>)::key_t>);
  static_assert(
      std::is_same_v<
          device_has_key,
          decltype(device_has<aspect::usm_atomic_shared_allocations>)::key_t>);
  static_assert(std::is_same_v<device_has_key,
                               decltype(device_has<aspect::atomic64>)::key_t>);
  static_assert(
      std::is_same_v<
          device_has_key,
          decltype(device_has<aspect::ext_intel_device_info_uuid>)::key_t>);
  static_assert(
      std::is_same_v<device_has_key,
                     decltype(device_has<aspect::ext_oneapi_srgb>)::key_t>);
  static_assert(std::is_same_v<
                device_has_key,
                decltype(device_has<aspect::ext_oneapi_native_assert>)::key_t>);
  static_assert(
      std::is_same_v<device_has_key,
                     decltype(device_has<aspect::host_debuggable>)::key_t>);
  static_assert(std::is_same_v<
                device_has_key,
                decltype(device_has<
                         aspect::ext_intel_gpu_hw_threads_per_eu>)::key_t>);
  static_assert(
      std::is_same_v<
          device_has_key,
          decltype(device_has<aspect::ext_oneapi_cuda_async_barrier>)::key_t>);
  static_assert(
      std::is_same_v<device_has_key,
                     decltype(device_has<aspect::ext_oneapi_bfloat16>)::key_t>);
  static_assert(std::is_same_v<
                device_has_key,
                decltype(device_has<aspect::ext_intel_free_memory>)::key_t>);
  static_assert(
      std::is_same_v<device_has_key,
                     decltype(device_has<aspect::ext_intel_device_id>)::key_t>);
  static_assert(std::is_same_v<device_has_key, device_has_all::key_t>);

  static_assert(work_group_size<15>[0] == 15);
  static_assert(work_group_size<16, 17>[0] == 16);
  static_assert(work_group_size<16, 17>[1] == 17);
  static_assert(work_group_size<18, 19, 20>[0] == 18);
  static_assert(work_group_size<18, 19, 20>[1] == 19);
  static_assert(work_group_size<18, 19, 20>[2] == 20);
  static_assert(work_group_size_hint<21>[0] == 21);
  static_assert(work_group_size_hint<22, 23>[0] == 22);
  static_assert(work_group_size_hint<22, 23>[1] == 23);
  static_assert(work_group_size_hint<24, 25, 26>[0] == 24);
  static_assert(work_group_size_hint<24, 25, 26>[1] == 25);
  static_assert(work_group_size_hint<24, 25, 26>[2] == 26);
  static_assert(sub_group_size<27>.value == 27);

  static_assert(decltype(device_has<>)::value.size() == 0);
  static_assert(decltype(device_has<aspect::host>)::value.size() == 1);
  static_assert(decltype(device_has<aspect::cpu>)::value.size() == 1);
  static_assert(decltype(device_has<aspect::gpu>)::value.size() == 1);
  static_assert(decltype(device_has<aspect::accelerator>)::value.size() == 1);
  static_assert(decltype(device_has<aspect::custom>)::value.size() == 1);
  static_assert(decltype(device_has<aspect::fp16>)::value.size() == 1);
  static_assert(decltype(device_has<aspect::fp64>)::value.size() == 1);
  static_assert(decltype(device_has<aspect::image>)::value.size() == 1);
  static_assert(decltype(device_has<aspect::online_compiler>)::value.size() ==
                1);
  static_assert(decltype(device_has<aspect::online_linker>)::value.size() == 1);
  static_assert(decltype(device_has<aspect::queue_profiling>)::value.size() ==
                1);
  static_assert(
      decltype(device_has<aspect::usm_device_allocations>)::value.size() == 1);
  static_assert(
      decltype(device_has<aspect::usm_host_allocations>)::value.size() == 1);
  static_assert(
      decltype(device_has<aspect::usm_shared_allocations>)::value.size() == 1);
  static_assert(
      decltype(device_has<aspect::usm_restricted_shared_allocations>)::value
          .size() == 1);
  static_assert(
      decltype(device_has<aspect::usm_system_allocations>)::value.size() == 1);
  static_assert(
      decltype(device_has<aspect::ext_intel_pci_address>)::value.size() == 1);
  static_assert(
      decltype(device_has<aspect::ext_intel_gpu_eu_count>)::value.size() == 1);
  static_assert(
      decltype(device_has<aspect::ext_intel_gpu_eu_simd_width>)::value.size() ==
      1);
  static_assert(
      decltype(device_has<aspect::ext_intel_gpu_slices>)::value.size() == 1);
  static_assert(
      decltype(device_has<aspect::ext_intel_gpu_subslices_per_slice>)::value
          .size() == 1);
  static_assert(
      decltype(device_has<aspect::ext_intel_gpu_eu_count_per_subslice>)::value
          .size() == 1);
  static_assert(
      decltype(device_has<aspect::ext_intel_max_mem_bandwidth>)::value.size() ==
      1);
  static_assert(
      decltype(device_has<aspect::ext_intel_mem_channel>)::value.size() == 1);
  static_assert(
      decltype(device_has<aspect::usm_atomic_host_allocations>)::value.size() ==
      1);
  static_assert(
      decltype(device_has<aspect::usm_atomic_shared_allocations>)::value
          .size() == 1);
  static_assert(decltype(device_has<aspect::atomic64>)::value.size() == 1);
  static_assert(
      decltype(device_has<aspect::ext_intel_device_info_uuid>)::value.size() ==
      1);
  static_assert(decltype(device_has<aspect::ext_oneapi_srgb>)::value.size() ==
                1);
  static_assert(
      decltype(device_has<aspect::ext_oneapi_native_assert>)::value.size() ==
      1);
  static_assert(decltype(device_has<aspect::host_debuggable>)::value.size() ==
                1);
  static_assert(
      decltype(device_has<aspect::ext_intel_gpu_hw_threads_per_eu>)::value
          .size() == 1);
  static_assert(
      decltype(device_has<aspect::ext_oneapi_cuda_async_barrier>)::value
          .size() == 1);
  static_assert(
      decltype(device_has<aspect::ext_oneapi_bfloat16>)::value.size() == 1);
  static_assert(
      decltype(device_has<aspect::ext_intel_free_memory>)::value.size() == 1);
  static_assert(
      decltype(device_has<aspect::ext_intel_device_id>)::value.size() == 1);
  static_assert(device_has_all::value.size() == 36);

  static_assert(decltype(device_has<aspect::host>)::value[0] == aspect::host);
  static_assert(decltype(device_has<aspect::cpu>)::value[0] == aspect::cpu);
  static_assert(decltype(device_has<aspect::gpu>)::value[0] == aspect::gpu);
  static_assert(decltype(device_has<aspect::accelerator>)::value[0] ==
                aspect::accelerator);
  static_assert(decltype(device_has<aspect::custom>)::value[0] ==
                aspect::custom);
  static_assert(decltype(device_has<aspect::fp16>)::value[0] == aspect::fp16);
  static_assert(decltype(device_has<aspect::fp64>)::value[0] == aspect::fp64);
  static_assert(decltype(device_has<aspect::image>)::value[0] == aspect::image);
  static_assert(decltype(device_has<aspect::online_compiler>)::value[0] ==
                aspect::online_compiler);
  static_assert(decltype(device_has<aspect::online_linker>)::value[0] ==
                aspect::online_linker);
  static_assert(decltype(device_has<aspect::queue_profiling>)::value[0] ==
                aspect::queue_profiling);
  static_assert(
      decltype(device_has<aspect::usm_device_allocations>)::value[0] ==
      aspect::usm_device_allocations);
  static_assert(decltype(device_has<aspect::usm_host_allocations>)::value[0] ==
                aspect::usm_host_allocations);
  static_assert(
      decltype(device_has<aspect::usm_shared_allocations>)::value[0] ==
      aspect::usm_shared_allocations);
  static_assert(
      decltype(device_has<
               aspect::usm_restricted_shared_allocations>)::value[0] ==
      aspect::usm_restricted_shared_allocations);
  static_assert(
      decltype(device_has<aspect::usm_system_allocations>)::value[0] ==
      aspect::usm_system_allocations);
  static_assert(decltype(device_has<aspect::ext_intel_pci_address>)::value[0] ==
                aspect::ext_intel_pci_address);
  static_assert(
      decltype(device_has<aspect::ext_intel_gpu_eu_count>)::value[0] ==
      aspect::ext_intel_gpu_eu_count);
  static_assert(
      decltype(device_has<aspect::ext_intel_gpu_eu_simd_width>)::value[0] ==
      aspect::ext_intel_gpu_eu_simd_width);
  static_assert(decltype(device_has<aspect::ext_intel_gpu_slices>)::value[0] ==
                aspect::ext_intel_gpu_slices);
  static_assert(
      decltype(device_has<
               aspect::ext_intel_gpu_subslices_per_slice>)::value[0] ==
      aspect::ext_intel_gpu_subslices_per_slice);
  static_assert(
      decltype(device_has<
               aspect::ext_intel_gpu_eu_count_per_subslice>)::value[0] ==
      aspect::ext_intel_gpu_eu_count_per_subslice);
  static_assert(
      decltype(device_has<aspect::ext_intel_max_mem_bandwidth>)::value[0] ==
      aspect::ext_intel_max_mem_bandwidth);
  static_assert(decltype(device_has<aspect::ext_intel_mem_channel>)::value[0] ==
                aspect::ext_intel_mem_channel);
  static_assert(
      decltype(device_has<aspect::usm_atomic_host_allocations>)::value[0] ==
      aspect::usm_atomic_host_allocations);
  static_assert(
      decltype(device_has<aspect::usm_atomic_shared_allocations>)::value[0] ==
      aspect::usm_atomic_shared_allocations);
  static_assert(decltype(device_has<aspect::atomic64>)::value[0] ==
                aspect::atomic64);
  static_assert(
      decltype(device_has<aspect::ext_intel_device_info_uuid>)::value[0] ==
      aspect::ext_intel_device_info_uuid);
  static_assert(decltype(device_has<aspect::ext_oneapi_srgb>)::value[0] ==
                aspect::ext_oneapi_srgb);
  static_assert(
      decltype(device_has<aspect::ext_oneapi_native_assert>)::value[0] ==
      aspect::ext_oneapi_native_assert);
  static_assert(decltype(device_has<aspect::host_debuggable>)::value[0] ==
                aspect::host_debuggable);
  static_assert(
      decltype(device_has<aspect::ext_intel_gpu_hw_threads_per_eu>)::value[0] ==
      aspect::ext_intel_gpu_hw_threads_per_eu);
  static_assert(
      decltype(device_has<aspect::ext_oneapi_cuda_async_barrier>)::value[0] ==
      aspect::ext_oneapi_cuda_async_barrier);
  static_assert(decltype(device_has<aspect::ext_oneapi_bfloat16>)::value[0] ==
                aspect::ext_oneapi_bfloat16);
  static_assert(decltype(device_has<aspect::ext_intel_free_memory>)::value[0] ==
                aspect::ext_intel_free_memory);
  static_assert(decltype(device_has<aspect::ext_intel_device_id>)::value[0] ==
                aspect::ext_intel_device_id);

  static_assert(device_has_all::value[0] == aspect::host);
  static_assert(device_has_all::value[1] == aspect::cpu);
  static_assert(device_has_all::value[2] == aspect::gpu);
  static_assert(device_has_all::value[3] == aspect::accelerator);
  static_assert(device_has_all::value[4] == aspect::custom);
  static_assert(device_has_all::value[5] == aspect::fp16);
  static_assert(device_has_all::value[6] == aspect::fp64);
  static_assert(device_has_all::value[7] == aspect::image);
  static_assert(device_has_all::value[8] == aspect::online_compiler);
  static_assert(device_has_all::value[9] == aspect::online_linker);
  static_assert(device_has_all::value[10] == aspect::queue_profiling);
  static_assert(device_has_all::value[11] == aspect::usm_device_allocations);
  static_assert(device_has_all::value[12] == aspect::usm_host_allocations);
  static_assert(device_has_all::value[13] == aspect::usm_shared_allocations);
  static_assert(device_has_all::value[14] ==
                aspect::usm_restricted_shared_allocations);
  static_assert(device_has_all::value[15] == aspect::usm_system_allocations);
  static_assert(device_has_all::value[16] == aspect::ext_intel_pci_address);
  static_assert(device_has_all::value[17] == aspect::ext_intel_gpu_eu_count);
  static_assert(device_has_all::value[18] ==
                aspect::ext_intel_gpu_eu_simd_width);
  static_assert(device_has_all::value[19] == aspect::ext_intel_gpu_slices);
  static_assert(device_has_all::value[20] ==
                aspect::ext_intel_gpu_subslices_per_slice);
  static_assert(device_has_all::value[21] ==
                aspect::ext_intel_gpu_eu_count_per_subslice);
  static_assert(device_has_all::value[22] ==
                aspect::ext_intel_max_mem_bandwidth);
  static_assert(device_has_all::value[23] == aspect::ext_intel_mem_channel);
  static_assert(device_has_all::value[24] ==
                aspect::usm_atomic_host_allocations);
  static_assert(device_has_all::value[25] ==
                aspect::usm_atomic_shared_allocations);
  static_assert(device_has_all::value[26] == aspect::atomic64);
  static_assert(device_has_all::value[27] ==
                aspect::ext_intel_device_info_uuid);
  static_assert(device_has_all::value[28] == aspect::ext_oneapi_srgb);
  static_assert(device_has_all::value[29] == aspect::ext_oneapi_native_assert);
  static_assert(device_has_all::value[30] == aspect::host_debuggable);
  static_assert(device_has_all::value[31] ==
                aspect::ext_intel_gpu_hw_threads_per_eu);
  static_assert(device_has_all::value[32] ==
                aspect::ext_oneapi_cuda_async_barrier);
  static_assert(device_has_all::value[33] == aspect::ext_oneapi_bfloat16);
  static_assert(device_has_all::value[34] == aspect::ext_intel_free_memory);
  static_assert(device_has_all::value[35] == aspect::ext_intel_device_id);

  static_assert(std::is_same_v<decltype(sub_group_size<28>)::value_t,
                               std::integral_constant<uint32_t, 28>>);

  return 0;
}
