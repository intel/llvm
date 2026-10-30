// REQUIRES: linux, cpu || (gpu && level_zero)
// RUN: %{build} %device_asan_flags -Xarch_device -fsanitize-recover=address -O2 -g -DUSER_CODE_1 -c -o %t1.o
// RUN: %{build} %device_asan_flags -Xarch_device -fsanitize-recover=address -O2 -g -DUSER_CODE_2 -c -o %t2.o
// RUN: %clangxx -fsycl %{sycl_target_opts} %device_asan_flags -O2 -g -fsycl-device-code-split=per_source %t1.o %t2.o -o %t_per_source.out
// RUN: %{run} %t_per_source.out 2>&1 | FileCheck %s
// RUN: %clangxx -fsycl %{sycl_target_opts} %device_asan_flags -O2 -g -fsycl-device-code-split=per_kernel %t1.o %t2.o -o %t_per_kernel.out
// RUN: %{run} %t_per_kernel.out 2>&1 | FileCheck %s

// Checks that when a program built from several translation units is split
// into several device images, the sanitizer kernel metadata (a device global
// copied into every image) has the same size in all of them, and that each
// image still describes its own kernels, so the errors of every kernel are
// reported.

#include <sycl/detail/core.hpp>
#include <sycl/usm.hpp>

constexpr std::size_t N = 4;

#ifdef USER_CODE_1

void runKernelsA(sycl::queue &Q, int *Array) {
  Q.submit([&](sycl::handler &h) {
     h.parallel_for<class KernelA1>(
         sycl::nd_range<1>(N + 1, 1),
         [=](sycl::nd_item<1> item) { ++Array[item.get_global_id(0)]; });
   }).wait();
  // CHECK: ERROR: DeviceSanitizer: out-of-bounds-access on Device USM
  // CHECK: {{READ of size 4 at kernel <.*KernelA1> LID\(0, 0, 0\) GID\(4, 0, 0\)}}
  // CHECK: {{  #0 .* .*device-code-split.cpp:}}[[@LINE-4]]

  Q.submit([&](sycl::handler &h) {
     h.parallel_for<class KernelA2>(
         sycl::nd_range<1>(N + 1, 1),
         [=](sycl::nd_item<1> item) { ++Array[item.get_global_id(0)]; });
   }).wait();
  // CHECK: ERROR: DeviceSanitizer: out-of-bounds-access on Device USM
  // CHECK: {{READ of size 4 at kernel <.*KernelA2> LID\(0, 0, 0\) GID\(4, 0, 0\)}}
  // CHECK: {{  #0 .* .*device-code-split.cpp:}}[[@LINE-4]]
}

#else // USER_CODE_2

void runKernelsA(sycl::queue &Q, int *Array);

int main() {
  sycl::queue Q;
  auto *Array = sycl::malloc_device<int>(N, Q);

  runKernelsA(Q, Array);

  Q.submit([&](sycl::handler &h) {
     h.parallel_for<class KernelB1>(
         sycl::nd_range<1>(N + 1, 1),
         [=](sycl::nd_item<1> item) { ++Array[item.get_global_id(0)]; });
   }).wait();
  // CHECK: ERROR: DeviceSanitizer: out-of-bounds-access on Device USM
  // CHECK: {{READ of size 4 at kernel <.*KernelB1> LID\(0, 0, 0\) GID\(4, 0, 0\)}}
  // CHECK: {{  #0 .* .*device-code-split.cpp:}}[[@LINE-4]]

  Q.submit([&](sycl::handler &h) {
     h.parallel_for<class KernelB2>(
         sycl::nd_range<1>(N + 1, 1),
         [=](sycl::nd_item<1> item) { ++Array[item.get_global_id(0)]; });
   }).wait();
  // CHECK: ERROR: DeviceSanitizer: out-of-bounds-access on Device USM
  // CHECK: {{READ of size 4 at kernel <.*KernelB2> LID\(0, 0, 0\) GID\(4, 0, 0\)}}
  // CHECK: {{  #0 .* .*device-code-split.cpp:}}[[@LINE-4]]

  sycl::free(Array, Q);
  return 0;
}

#endif
