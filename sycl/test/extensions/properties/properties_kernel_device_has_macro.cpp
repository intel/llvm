// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify %s
// expected-no-diagnostics

#include <sycl/sycl.hpp>

using namespace sycl;
using namespace ext::oneapi::experimental;

static constexpr auto device_has_all = device_has<
    aspect::ext_oneapi_cuda_async_barrier,
    aspect::ext_oneapi_bfloat16_math_functions, aspect::custom, aspect::fp16,
    aspect::fp64, aspect::image, aspect::online_compiler, aspect::online_linker,
    aspect::queue_profiling, aspect::usm_device_allocations,
    aspect::usm_system_allocations, aspect::ext_intel_pci_address, aspect::cpu,
    aspect::gpu, aspect::accelerator, aspect::ext_intel_gpu_eu_count,
    aspect::ext_intel_gpu_subslices_per_slice,
    aspect::ext_intel_gpu_eu_count_per_subslice,
    aspect::ext_intel_max_mem_bandwidth, aspect::ext_intel_mem_channel,
    aspect::usm_atomic_host_allocations, aspect::usm_atomic_shared_allocations,
    aspect::atomic64, aspect::ext_intel_device_info_uuid,
    aspect::ext_oneapi_srgb, aspect::ext_intel_gpu_eu_simd_width,
    aspect::ext_intel_gpu_slices, aspect::ext_oneapi_native_assert,
    aspect::host_debuggable, aspect::ext_intel_gpu_hw_threads_per_eu,
    aspect::usm_host_allocations, aspect::usm_shared_allocations,
    aspect::ext_intel_free_memory, aspect::ext_intel_device_id,
    aspect::ext_intel_memory_clock_rate, aspect::ext_intel_memory_bus_width>;

SYCL_EXTERNAL SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(device_has_all) void Func0() {}

SYCL_EXTERNAL SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((device_has<>)) void Func1() {}

SYCL_EXTERNAL SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(
    (device_has<aspect::fp16, aspect::atomic64>)) void Func2() {}

int main() {
  queue Q;
  Q.single_task([]() {
    Func0();
    Func1();
    Func2();
  });
  return 0;
}
