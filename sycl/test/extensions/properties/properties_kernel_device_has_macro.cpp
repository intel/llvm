// RUN: %clangxx -fsycl-device-only -S -Xclang -emit-llvm -Xclang -disable-llvm-passes %s -o - | FileCheck %s --check-prefix CHECK-IR
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

// CHECK-IR: spir_func void @{{.*}}Func0{{.*}}(){{.*}} #[[DHAttr1:[0-9]+]]
SYCL_EXTERNAL SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(device_has_all) void Func0() {}

// CHECK-IR: spir_func void @{{.*}}Func1{{.*}}(){{.*}} #[[DHAttr2:[0-9]+]]
SYCL_EXTERNAL SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((device_has<>)) void Func1() {}

// CHECK-IR: spir_func void @{{.*}}Func2{{.*}}(){{.*}} #[[DHAttr3:[0-9]+]]
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

// CHECK-IR-DAG: attributes #[[DHAttr1]] = { {{.*}}"sycl-device-has"="34,35,4,5,6,9,10,11,12,13,17,18,1,2,3,19,22,23,24,25,26,27,28,29,30,20,21,31,32,33,14,15,36,37,38,39"
// CHECK-IR-DAG: attributes #[[DHAttr2]] = { {{.*}}"sycl-device-has" {{.*}}
// CHECK-IR-DAG: attributes #[[DHAttr3]] = { {{.*}}"sycl-device-has"="5,28"
