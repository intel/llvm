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
    aspect::ext_intel_free_memory, aspect::ext_intel_device_id>;

int main() {
  queue Q;
  event Ev;

  range<1> R1{1};
  nd_range<1> NDR1{R1, R1};

  constexpr auto Props = properties{device_has_all};

  auto Redu1 = reduction<int>(nullptr, plus<int>());
  auto Redu2 = reduction<float>(nullptr, multiplies<float>());

  // CHECK-IR: spir_kernel void @{{.*}}WGSizeKernel0(){{.*}} #[[DHAttr1:[0-9]+]]
  Q.single_task<class WGSizeKernel0>(Props, []() {});
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeKernel1(){{.*}} #[[DHAttr1]]
  Q.single_task<class WGSizeKernel1>(Ev, Props, []() {});
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeKernel2(){{.*}} #[[DHAttr1]]
  Q.single_task<class WGSizeKernel2>({Ev}, Props, []() {});

  // CHECK-IR: spir_kernel void @{{.*}}WGSizeKernel3(){{.*}} #[[DHAttr2:[0-9]+]]
  Q.parallel_for<class WGSizeKernel3>(R1, Props, [](id<1>) {});
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeKernel4(){{.*}} #[[DHAttr2]]
  Q.parallel_for<class WGSizeKernel4>(R1, Ev, Props, [](id<1>) {});
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeKernel5(){{.*}} #[[DHAttr2]]
  Q.parallel_for<class WGSizeKernel5>(R1, {Ev}, Props, [](id<1>) {});

  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeKernel6{{.*}}{{.*}} #[[DHAttr2:[0-9]+]]
  Q.parallel_for<class WGSizeKernel6>(R1, Props, Redu1, [](id<1>, auto &) {});
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeKernel7{{.*}}{{.*}} #[[DHAttr2]]
  Q.parallel_for<class WGSizeKernel7>(R1, Ev, Props, Redu1,
                                      [](id<1>, auto &) {});
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeKernel8{{.*}}{{.*}} #[[DHAttr2]]
  Q.parallel_for<class WGSizeKernel8>(R1, {Ev}, Props, Redu1,
                                      [](id<1>, auto &) {});

  // CHECK-IR: spir_kernel void @{{.*}}WGSizeKernel9(){{.*}} #[[DHAttr2]]
  Q.parallel_for<class WGSizeKernel9>(NDR1, Props, [](nd_item<1>) {});
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeKernel10(){{.*}} #[[DHAttr2]]
  Q.parallel_for<class WGSizeKernel10>(NDR1, Ev, Props, [](nd_item<1>) {});
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeKernel11(){{.*}} #[[DHAttr2]]
  Q.parallel_for<class WGSizeKernel11>(NDR1, {Ev}, Props, [](nd_item<1>) {});

  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeKernel12{{.*}}{{.*}} #[[DHAttr2]]
  Q.parallel_for<class WGSizeKernel12>(NDR1, Props, Redu1,
                                       [](nd_item<1>, auto &) {});
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeKernel13{{.*}}{{.*}} #[[DHAttr2]]
  Q.parallel_for<class WGSizeKernel13>(NDR1, Ev, Props, Redu1,
                                       [](nd_item<1>, auto &) {});
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeKernel14{{.*}}{{.*}} #[[DHAttr2]]
  Q.parallel_for<class WGSizeKernel14>(NDR1, {Ev}, Props, Redu1,
                                       [](nd_item<1>, auto &) {});

  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeKernel15{{.*}}{{.*}} #[[DHAttr2]]
  Q.parallel_for<class WGSizeKernel15>(NDR1, Props, Redu1, Redu2,
                                       [](nd_item<1>, auto &, auto &) {});
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeKernel16{{.*}}{{.*}} #[[DHAttr2]]
  Q.parallel_for<class WGSizeKernel16>(NDR1, Ev, Props, Redu1, Redu2,
                                       [](nd_item<1>, auto &, auto &) {});
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeKernel17{{.*}}{{.*}} #[[DHAttr2]]
  Q.parallel_for<class WGSizeKernel17>(NDR1, {Ev}, Props, Redu1, Redu2,
                                       [](nd_item<1>, auto &, auto &) {});

  // CHECK-IR: spir_kernel void @{{.*}}WGSizeKernel18(){{.*}} #[[DHAttr1]]
  Q.submit([&](handler &CGH) {
    CGH.single_task<class WGSizeKernel18>(Props, []() {});
  });

  // CHECK-IR: spir_kernel void @{{.*}}WGSizeKernel19(){{.*}} #[[DHAttr2]]
  Q.submit([&](handler &CGH) {
    CGH.parallel_for<class WGSizeKernel19>(R1, Props, [](id<1>) {});
  });

  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeKernel20{{.*}}{{.*}} #[[DHAttr2]]
  Q.submit([&](handler &CGH) {
    CGH.parallel_for<class WGSizeKernel20>(R1, Props, Redu1,
                                           [](id<1>, auto &) {});
  });

  // CHECK-IR: spir_kernel void @{{.*}}WGSizeKernel21(){{.*}} #[[DHAttr2]]
  Q.submit([&](handler &CGH) {
    CGH.parallel_for<class WGSizeKernel21>(NDR1, Props, [](nd_item<1>) {});
  });

  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeKernel22{{.*}}{{.*}} #[[DHAttr2]]
  Q.submit([&](handler &CGH) {
    CGH.parallel_for<class WGSizeKernel22>(NDR1, Props, Redu1,
                                           [](nd_item<1>, auto &) {});
  });

  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeKernel23{{.*}}{{.*}} #[[DHAttr2]]
  Q.submit([&](handler &CGH) {
    CGH.parallel_for<class WGSizeKernel23>(NDR1, Props, Redu1, Redu2,
                                           [](nd_item<1>, auto &, auto &) {});
  });

  // CHECK-IR: spir_kernel void @{{.*}}WGSizeKernel24(){{.*}} #[[DHAttr2]]
  Q.submit([&](handler &CGH) {
    CGH.parallel_for_work_group<class WGSizeKernel24>(
        R1, Props,
        [](group<1> G) { G.parallel_for_work_item([&](h_item<1>) {}); });
  });

  return 0;
}

// CHECK-IR-DAG: attributes #[[DHAttr1]] = { {{.*}}"sycl-device-has"="34,35,4,5,6,9,10,11,12,13,17,18,1,2,3,19,22,23,24,25,26,27,28,29,30,20,21,31,32,33,14,15,36,37"
// CHECK-IR-DAG: attributes #[[DHAttr2]] = { {{.*}}"sycl-device-has"="34,35,4,5,6,9,10,11,12,13,17,18,1,2,3,19,22,23,24,25,26,27,28,29,30,20,21,31,32,33,14,15,36,37"
