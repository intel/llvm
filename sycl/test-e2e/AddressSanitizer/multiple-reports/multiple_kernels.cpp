// REQUIRES: linux
// RUN: %{build} %device_asan_flags -Xarch_device -fsanitize-recover=address -O2 -g -o %t
// RUN: env SYCL_PREFER_UR=1 %{run} %t 2>&1 | FileCheck %s

#include <sycl/detail/core.hpp>
#include <sycl/usm.hpp>

int main() {
  sycl::queue Q;
  constexpr std::size_t N = 4;
  auto *array = sycl::malloc_device<int>(N, Q);

  Q.submit([&](sycl::handler &h) {
     h.parallel_for<class Kernel1>(
         sycl::nd_range<1>(N + 1, 1),
         [=](sycl::nd_item<1> item) { ++array[item.get_global_id(0)]; });
   }).wait();
  // CHECK: ERROR: DeviceSanitizer: out-of-bounds-access on Device USM
  // CHECK: {{READ of size 4 at kernel <.*Kernel1> LID\(0, 0, 0\) GID\(4, 0, 0\)}}
  // CHECK: {{  #0 .* .*multiple_kernels.cpp:}}[[@LINE-4]]

  Q.submit([&](sycl::handler &h) {
     h.parallel_for<class Kernel2>(
         sycl::nd_range<1>(N + 1, 1),
         [=](sycl::nd_item<1> item) { ++array[item.get_global_id(0)]; });
   }).wait();
  // CHECK: ERROR: DeviceSanitizer: out-of-bounds-access on Device USM
  // CHECK: {{READ of size 4 at kernel <.*Kernel2> LID\(0, 0, 0\) GID\(4, 0, 0\)}}
  // CHECK: {{  #0 .* .*multiple_kernels.cpp:}}[[@LINE-4]]

  Q.submit([&](sycl::handler &h) {
     h.parallel_for<class Kernel3>(
         sycl::nd_range<1>(N + 1, 1),
         [=](sycl::nd_item<1> item) { ++array[item.get_global_id(0)]; });
   }).wait();
  // CHECK: ERROR: DeviceSanitizer: out-of-bounds-access on Device USM
  // CHECK: {{READ of size 4 at kernel <.*Kernel3> LID\(0, 0, 0\) GID\(4, 0, 0\)}}
  // CHECK: {{  #0 .* .*multiple_kernels.cpp:}}[[@LINE-4]]

  Q.submit([&](sycl::handler &h) {
     h.parallel_for<class Kernel4>(
         sycl::nd_range<1>(N + 1, 1),
         [=](sycl::nd_item<1> item) { ++array[item.get_global_id(0)]; });
   }).wait();
  // CHECK: ERROR: DeviceSanitizer: out-of-bounds-access on Device USM
  // CHECK: {{READ of size 4 at kernel <.*Kernel4> LID\(0, 0, 0\) GID\(4, 0, 0\)}}
  // CHECK: {{  #0 .* .*multiple_kernels.cpp:}}[[@LINE-4]]

  return 0;
}
