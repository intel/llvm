// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -fsyntax-only -Xclang -verify %s
// expected-no-diagnostics

#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental;

using device_has_all =
    decltype(device_has<
             aspect::cpu, aspect::gpu, aspect::accelerator, aspect::custom,
             aspect::fp16, aspect::fp64, aspect::image, aspect::online_compiler,
             aspect::online_linker, aspect::queue_profiling,
             aspect::usm_device_allocations, aspect::usm_host_allocations,
             aspect::usm_shared_allocations, aspect::usm_system_allocations,
             aspect::ext_intel_pci_address, aspect::ext_intel_gpu_eu_count,
             aspect::ext_intel_gpu_eu_simd_width, aspect::ext_intel_gpu_slices,
             aspect::ext_intel_gpu_subslices_per_slice,
             aspect::ext_intel_gpu_eu_count_per_subslice,
             aspect::ext_intel_max_mem_bandwidth, aspect::ext_intel_mem_channel,
             aspect::usm_atomic_host_allocations,
             aspect::usm_atomic_shared_allocations, aspect::atomic64,
             aspect::ext_intel_device_info_uuid, aspect::ext_oneapi_srgb,
             aspect::ext_oneapi_native_assert, aspect::host_debuggable,
             aspect::ext_intel_gpu_hw_threads_per_eu,
             aspect::ext_oneapi_cuda_async_barrier,
             aspect::ext_intel_free_memory, aspect::ext_intel_device_id,
             aspect::ext_intel_memory_clock_rate,
             aspect::ext_intel_memory_bus_width>);

template <aspect Aspect> inline void singleAspectDeviceHasChecks() {
  static_assert(is_property_value<decltype(device_has<Aspect>)>::value);
  static_assert(std::is_same_v<device_has_key,
                               typename decltype(device_has<Aspect>)::key_t>);
  static_assert(decltype(device_has<Aspect>)::value.size() == 1);
  static_assert(decltype(device_has<Aspect>)::value[0] == Aspect);
}

int main() {
  static_assert(is_property_key<work_group_size_key>::value);
  static_assert(is_property_key<work_group_size_hint_key>::value);
  static_assert(is_property_key<sub_group_size_key>::value);
  static_assert(is_property_key<device_has_key>::value);

  static_assert(is_property_value<decltype(work_group_size<1>)>::value);
  static_assert(is_property_value<decltype(work_group_size<2, 2>)>::value);
  static_assert(is_property_value<decltype(work_group_size<3, 3, 3>)>::value);
  static_assert(is_property_value<decltype(work_group_size_hint<4>)>::value);
  static_assert(is_property_value<decltype(work_group_size_hint<5, 5>)>::value);
  static_assert(
      is_property_value<decltype(work_group_size_hint<6, 6, 6>)>::value);
  static_assert(is_property_value<decltype(sub_group_size<7>)>::value);

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

  static_assert(std::is_same_v<decltype(sub_group_size<28>)::value_t,
                               std::integral_constant<uint32_t, 28>>);

  singleAspectDeviceHasChecks<aspect::cpu>();
  singleAspectDeviceHasChecks<aspect::gpu>();
  singleAspectDeviceHasChecks<aspect::accelerator>();
  singleAspectDeviceHasChecks<aspect::custom>();
  singleAspectDeviceHasChecks<aspect::fp16>();
  singleAspectDeviceHasChecks<aspect::fp64>();
  singleAspectDeviceHasChecks<aspect::image>();
  singleAspectDeviceHasChecks<aspect::online_compiler>();
  singleAspectDeviceHasChecks<aspect::online_linker>();
  singleAspectDeviceHasChecks<aspect::queue_profiling>();
  singleAspectDeviceHasChecks<aspect::usm_device_allocations>();
  singleAspectDeviceHasChecks<aspect::usm_host_allocations>();
  singleAspectDeviceHasChecks<aspect::usm_shared_allocations>();
  singleAspectDeviceHasChecks<aspect::usm_system_allocations>();
  singleAspectDeviceHasChecks<aspect::ext_intel_pci_address>();
  singleAspectDeviceHasChecks<aspect::ext_intel_gpu_eu_count>();
  singleAspectDeviceHasChecks<aspect::ext_intel_gpu_eu_simd_width>();
  singleAspectDeviceHasChecks<aspect::ext_intel_gpu_slices>();
  singleAspectDeviceHasChecks<aspect::ext_intel_gpu_subslices_per_slice>();
  singleAspectDeviceHasChecks<aspect::ext_intel_gpu_eu_count_per_subslice>();
  singleAspectDeviceHasChecks<aspect::ext_intel_max_mem_bandwidth>();
  singleAspectDeviceHasChecks<aspect::ext_intel_mem_channel>();
  singleAspectDeviceHasChecks<aspect::usm_atomic_host_allocations>();
  singleAspectDeviceHasChecks<aspect::usm_atomic_shared_allocations>();
  singleAspectDeviceHasChecks<aspect::atomic64>();
  singleAspectDeviceHasChecks<aspect::ext_intel_device_info_uuid>();
  singleAspectDeviceHasChecks<aspect::ext_oneapi_srgb>();
  singleAspectDeviceHasChecks<aspect::ext_oneapi_native_assert>();
  singleAspectDeviceHasChecks<aspect::host_debuggable>();
  singleAspectDeviceHasChecks<aspect::ext_intel_gpu_hw_threads_per_eu>();
  singleAspectDeviceHasChecks<aspect::ext_oneapi_cuda_async_barrier>();
  singleAspectDeviceHasChecks<aspect::ext_intel_free_memory>();
  singleAspectDeviceHasChecks<aspect::ext_intel_device_id>();
  singleAspectDeviceHasChecks<aspect::ext_intel_memory_clock_rate>();
  singleAspectDeviceHasChecks<aspect::ext_intel_memory_bus_width>();

  static_assert(is_property_value<decltype(device_has<>)>::value);
  static_assert(std::is_same_v<device_has_key, decltype(device_has<>)::key_t>);
  static_assert(decltype(device_has<>)::value.size() == 0);

  static_assert(is_property_value<device_has_all>::value);
  static_assert(std::is_same_v<device_has_key, device_has_all::key_t>);
  static_assert(device_has_all::value.size() == 35);
  static_assert(device_has_all::value[0] == aspect::cpu);
  static_assert(device_has_all::value[1] == aspect::gpu);
  static_assert(device_has_all::value[2] == aspect::accelerator);
  static_assert(device_has_all::value[3] == aspect::custom);
  static_assert(device_has_all::value[4] == aspect::fp16);
  static_assert(device_has_all::value[5] == aspect::fp64);
  static_assert(device_has_all::value[6] == aspect::image);
  static_assert(device_has_all::value[7] == aspect::online_compiler);
  static_assert(device_has_all::value[8] == aspect::online_linker);
  static_assert(device_has_all::value[9] == aspect::queue_profiling);
  static_assert(device_has_all::value[10] == aspect::usm_device_allocations);
  static_assert(device_has_all::value[11] == aspect::usm_host_allocations);
  static_assert(device_has_all::value[12] == aspect::usm_shared_allocations);
  static_assert(device_has_all::value[13] == aspect::usm_system_allocations);
  static_assert(device_has_all::value[14] == aspect::ext_intel_pci_address);
  static_assert(device_has_all::value[15] == aspect::ext_intel_gpu_eu_count);
  static_assert(device_has_all::value[16] ==
                aspect::ext_intel_gpu_eu_simd_width);
  static_assert(device_has_all::value[17] == aspect::ext_intel_gpu_slices);
  static_assert(device_has_all::value[18] ==
                aspect::ext_intel_gpu_subslices_per_slice);
  static_assert(device_has_all::value[19] ==
                aspect::ext_intel_gpu_eu_count_per_subslice);
  static_assert(device_has_all::value[20] ==
                aspect::ext_intel_max_mem_bandwidth);
  static_assert(device_has_all::value[21] == aspect::ext_intel_mem_channel);
  static_assert(device_has_all::value[22] ==
                aspect::usm_atomic_host_allocations);
  static_assert(device_has_all::value[23] ==
                aspect::usm_atomic_shared_allocations);
  static_assert(device_has_all::value[24] == aspect::atomic64);
  static_assert(device_has_all::value[25] ==
                aspect::ext_intel_device_info_uuid);
  static_assert(device_has_all::value[26] == aspect::ext_oneapi_srgb);
  static_assert(device_has_all::value[27] == aspect::ext_oneapi_native_assert);
  static_assert(device_has_all::value[28] == aspect::host_debuggable);
  static_assert(device_has_all::value[29] ==
                aspect::ext_intel_gpu_hw_threads_per_eu);
  static_assert(device_has_all::value[30] ==
                aspect::ext_oneapi_cuda_async_barrier);
  static_assert(device_has_all::value[31] == aspect::ext_intel_free_memory);
  static_assert(device_has_all::value[32] == aspect::ext_intel_device_id);
  static_assert(device_has_all::value[33] ==
                aspect::ext_intel_memory_clock_rate);
  static_assert(device_has_all::value[34] ==
                aspect::ext_intel_memory_bus_width);
  return 0;
}
