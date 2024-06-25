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

// CHECK-IR-DAG: !{{[0-9]+}} = !{!"ext_oneapi_cuda_async_barrier", i32 [[ext_oneapi_cuda_async_barrier_ASPECT_MD:[0-9]+]]}
// CHECK-IR-DAG: !{{[0-9]+}} = !{!"ext_oneapi_bfloat16_math_functions", i32 [[ext_oneapi_bfloat16_math_functions_ASPECT_MD:[0-9]+]]}
// CHECK-IR-DAG: !{{[0-9]+}} = !{!"custom", i32 [[custom_ASPECT_MD:[0-9]+]]}
// CHECK-IR-DAG: !{{[0-9]+}} = !{!"fp16", i32 [[fp16_ASPECT_MD:[0-9]+]]}
// CHECK-IR-DAG: !{{[0-9]+}} = !{!"fp64", i32 [[fp64_ASPECT_MD:[0-9]+]]}
// CHECK-IR-DAG: !{{[0-9]+}} = !{!"image", i32 [[image_ASPECT_MD:[0-9]+]]}
// CHECK-IR-DAG: !{{[0-9]+}} = !{!"online_compiler", i32 [[online_compiler_ASPECT_MD:[0-9]+]]}
// CHECK-IR-DAG: !{{[0-9]+}} = !{!"online_linker", i32 [[online_linker_ASPECT_MD:[0-9]+]]}
// CHECK-IR-DAG: !{{[0-9]+}} = !{!"queue_profiling", i32 [[queue_profiling_ASPECT_MD:[0-9]+]]}
// CHECK-IR-DAG: !{{[0-9]+}} = !{!"usm_device_allocations", i32 [[usm_device_allocations_ASPECT_MD:[0-9]+]]}
// CHECK-IR-DAG: !{{[0-9]+}} = !{!"usm_system_allocations", i32 [[usm_system_allocations_ASPECT_MD:[0-9]+]]}
// CHECK-IR-DAG: !{{[0-9]+}} = !{!"ext_intel_pci_address", i32 [[ext_intel_pci_address_ASPECT_MD:[0-9]+]]}
// CHECK-IR-DAG: !{{[0-9]+}} = !{!"cpu", i32 [[cpu_ASPECT_MD:[0-9]+]]}
// CHECK-IR-DAG: !{{[0-9]+}} = !{!"gpu", i32 [[gpu_ASPECT_MD:[0-9]+]]}
// CHECK-IR-DAG: !{{[0-9]+}} = !{!"accelerator", i32 [[accelerator_ASPECT_MD:[0-9]+]]}
// CHECK-IR-DAG: !{{[0-9]+}} = !{!"ext_intel_gpu_eu_count", i32 [[ext_intel_gpu_eu_count_ASPECT_MD:[0-9]+]]}
// CHECK-IR-DAG: !{{[0-9]+}} = !{!"ext_intel_gpu_subslices_per_slice", i32 [[ext_intel_gpu_subslices_per_slice_ASPECT_MD:[0-9]+]]}
// CHECK-IR-DAG: !{{[0-9]+}} = !{!"ext_intel_gpu_eu_count_per_subslice", i32 [[ext_intel_gpu_eu_count_per_subslice_ASPECT_MD:[0-9]+]]}
// CHECK-IR-DAG: !{{[0-9]+}} = !{!"ext_intel_max_mem_bandwidth", i32 [[ext_intel_max_mem_bandwidth_ASPECT_MD:[0-9]+]]}
// CHECK-IR-DAG: !{{[0-9]+}} = !{!"ext_intel_mem_channel", i32 [[ext_intel_mem_channel_ASPECT_MD:[0-9]+]]}
// CHECK-IR-DAG: !{{[0-9]+}} = !{!"usm_atomic_host_allocations", i32 [[usm_atomic_host_allocations_ASPECT_MD:[0-9]+]]}
// CHECK-IR-DAG: !{{[0-9]+}} = !{!"usm_atomic_shared_allocations", i32 [[usm_atomic_shared_allocations_ASPECT_MD:[0-9]+]]}
// CHECK-IR-DAG: !{{[0-9]+}} = !{!"atomic64", i32 [[atomic64_ASPECT_MD:[0-9]+]]}
// CHECK-IR-DAG: !{{[0-9]+}} = !{!"ext_intel_device_info_uuid", i32 [[ext_intel_device_info_uuid_ASPECT_MD:[0-9]+]]}
// CHECK-IR-DAG: !{{[0-9]+}} = !{!"ext_oneapi_srgb", i32 [[ext_oneapi_srgb_ASPECT_MD:[0-9]+]]}
// CHECK-IR-DAG: !{{[0-9]+}} = !{!"ext_intel_gpu_eu_simd_width", i32 [[ext_intel_gpu_eu_simd_width_ASPECT_MD:[0-9]+]]}
// CHECK-IR-DAG: !{{[0-9]+}} = !{!"ext_intel_gpu_slices", i32 [[ext_intel_gpu_slices_ASPECT_MD:[0-9]+]]}
// CHECK-IR-DAG: !{{[0-9]+}} = !{!"ext_oneapi_native_assert", i32 [[ext_oneapi_native_assert_ASPECT_MD:[0-9]+]]}
// CHECK-IR-DAG: !{{[0-9]+}} = !{!"host_debuggable", i32 [[host_debuggable_ASPECT_MD:[0-9]+]]}
// CHECK-IR-DAG: !{{[0-9]+}} = !{!"ext_intel_gpu_hw_threads_per_eu", i32 [[ext_intel_gpu_hw_threads_per_eu_ASPECT_MD:[0-9]+]]}
// CHECK-IR-DAG: !{{[0-9]+}} = !{!"usm_host_allocations", i32 [[usm_host_allocations_ASPECT_MD:[0-9]+]]}
// CHECK-IR-DAG: !{{[0-9]+}} = !{!"usm_shared_allocations", i32 [[usm_shared_allocations_ASPECT_MD:[0-9]+]]}
// CHECK-IR-DAG: !{{[0-9]+}} = !{!"ext_intel_free_memory", i32 [[ext_intel_free_memory_ASPECT_MD:[0-9]+]]}
// CHECK-IR-DAG: !{{[0-9]+}} = !{!"ext_intel_device_id", i32 [[ext_intel_device_id_ASPECT_MD:[0-9]+]]}
// CHECK-IR-DAG: !{{[0-9]+}} = !{!"ext_intel_memory_clock_rate", i32 [[ext_intel_memory_clock_rate_ASPECT_MD:[0-9]+]]}
// CHECK-IR-DAG: !{{[0-9]+}} = !{!"ext_intel_memory_bus_width", i32 [[ext_intel_memory_bus_width_ASPECT_MD:[0-9]+]]}

// CHECK-IR-DAG: attributes #[[DHAttr1]] = { {{.*}}"sycl-device-has"="[[ext_oneapi_cuda_async_barrier_ASPECT_MD]],[[ext_oneapi_bfloat16_math_functions_ASPECT_MD]],[[custom_ASPECT_MD]],[[fp16_ASPECT_MD]],[[fp64_ASPECT_MD]],[[image_ASPECT_MD]],[[online_compiler_ASPECT_MD]],[[online_linker_ASPECT_MD]],[[queue_profiling_ASPECT_MD]],[[usm_device_allocations_ASPECT_MD]],[[usm_system_allocations_ASPECT_MD]],[[ext_intel_pci_address_ASPECT_MD]],[[cpu_ASPECT_MD]],[[gpu_ASPECT_MD]],[[accelerator_ASPECT_MD]],[[ext_intel_gpu_eu_count_ASPECT_MD]],[[ext_intel_gpu_subslices_per_slice_ASPECT_MD]],[[ext_intel_gpu_eu_count_per_subslice_ASPECT_MD]],[[ext_intel_max_mem_bandwidth_ASPECT_MD]],[[ext_intel_mem_channel_ASPECT_MD]],[[usm_atomic_host_allocations_ASPECT_MD]],[[usm_atomic_shared_allocations_ASPECT_MD]],[[atomic64_ASPECT_MD]],[[ext_intel_device_info_uuid_ASPECT_MD]],[[ext_oneapi_srgb_ASPECT_MD]],[[ext_intel_gpu_eu_simd_width_ASPECT_MD]],[[ext_intel_gpu_slices_ASPECT_MD]],[[ext_oneapi_native_assert_ASPECT_MD]],[[host_debuggable_ASPECT_MD]],[[ext_intel_gpu_hw_threads_per_eu_ASPECT_MD]],[[usm_host_allocations_ASPECT_MD]],[[usm_shared_allocations_ASPECT_MD]],[[ext_intel_free_memory_ASPECT_MD]],[[ext_intel_device_id_ASPECT_MD]],[[ext_intel_memory_clock_rate_ASPECT_MD]],[[ext_intel_memory_bus_width_ASPECT_MD]]"
// CHECK-IR-DAG: attributes #[[DHAttr2]] = { {{.*}}"sycl-device-has" {{.*}}
// CHECK-IR-DAG: attributes #[[DHAttr3]] = { {{.*}}"sycl-device-has"="[[fp16_ASPECT_MD]],[[atomic64_ASPECT_MD]]"
