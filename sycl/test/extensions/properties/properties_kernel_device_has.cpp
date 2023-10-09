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

  Q.single_task<class WGSizeKernel0>(Props, []() {});
  Q.single_task<class WGSizeKernel1>(Ev, Props, []() {});
  Q.single_task<class WGSizeKernel2>({Ev}, Props, []() {});

  Q.parallel_for<class WGSizeKernel3>(R1, Props, [](id<1>) {});
  Q.parallel_for<class WGSizeKernel4>(R1, Ev, Props, [](id<1>) {});
  Q.parallel_for<class WGSizeKernel5>(R1, {Ev}, Props, [](id<1>) {});

  Q.parallel_for<class WGSizeKernel6>(R1, Props, Redu1, [](id<1>, auto &) {});
  Q.parallel_for<class WGSizeKernel7>(R1, Ev, Props, Redu1,
                                      [](id<1>, auto &) {});
  Q.parallel_for<class WGSizeKernel8>(R1, {Ev}, Props, Redu1,
                                      [](id<1>, auto &) {});

  Q.parallel_for<class WGSizeKernel9>(NDR1, Props, [](nd_item<1>) {});
  Q.parallel_for<class WGSizeKernel10>(NDR1, Ev, Props, [](nd_item<1>) {});
  Q.parallel_for<class WGSizeKernel11>(NDR1, {Ev}, Props, [](nd_item<1>) {});

  Q.parallel_for<class WGSizeKernel12>(NDR1, Props, Redu1,
                                       [](nd_item<1>, auto &) {});
  Q.parallel_for<class WGSizeKernel13>(NDR1, Ev, Props, Redu1,
                                       [](nd_item<1>, auto &) {});
  Q.parallel_for<class WGSizeKernel14>(NDR1, {Ev}, Props, Redu1,
                                       [](nd_item<1>, auto &) {});

  Q.parallel_for<class WGSizeKernel15>(NDR1, Props, Redu1, Redu2,
                                       [](nd_item<1>, auto &, auto &) {});
  Q.parallel_for<class WGSizeKernel16>(NDR1, Ev, Props, Redu1, Redu2,
                                       [](nd_item<1>, auto &, auto &) {});
  Q.parallel_for<class WGSizeKernel17>(NDR1, {Ev}, Props, Redu1, Redu2,
                                       [](nd_item<1>, auto &, auto &) {});

  Q.submit([&](handler &CGH) {
    CGH.single_task<class WGSizeKernel18>(Props, []() {});
  });

  Q.submit([&](handler &CGH) {
    CGH.parallel_for<class WGSizeKernel19>(R1, Props, [](id<1>) {});
  });

  Q.submit([&](handler &CGH) {
    CGH.parallel_for<class WGSizeKernel20>(R1, Props, Redu1,
                                           [](id<1>, auto &) {});
  });

  Q.submit([&](handler &CGH) {
    CGH.parallel_for<class WGSizeKernel21>(NDR1, Props, [](nd_item<1>) {});
  });

  Q.submit([&](handler &CGH) {
    CGH.parallel_for<class WGSizeKernel22>(NDR1, Props, Redu1,
                                           [](nd_item<1>, auto &) {});
  });

  Q.submit([&](handler &CGH) {
    CGH.parallel_for<class WGSizeKernel23>(NDR1, Props, Redu1, Redu2,
                                           [](nd_item<1>, auto &, auto &) {});
  });

  Q.submit([&](handler &CGH) {
    CGH.parallel_for_work_group<class WGSizeKernel24>(
        R1, Props,
        [](group<1> G) { G.parallel_for_work_item([&](h_item<1>) {}); });
  });

  return 0;
}
