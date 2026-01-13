// REQUIRES: linux, cpu || (gpu && level_zero)
// RUN: %{build} %device_asan_flags -O0 -g -o %t.out
// RUN: %{run} not %t.out memset 2>&1 | FileCheck --check-prefixes CHECK-MEMSET %s
// RUN: %{run} not %t.out memcpy src 2>&1 | FileCheck --check-prefixes CHECK-MEMCPY-SRC %s
// RUN: %{run} not %t.out memcpy dst 2>&1 | FileCheck --check-prefixes CHECK-MEMCPY-DST %s
// RUN: %{run} not %t.out memmove src 2>&1 | FileCheck --check-prefixes CHECK-MEMMOVE-SRC %s
// RUN: %{run} not %t.out memmove dst 2>&1 | FileCheck --check-prefixes CHECK-MEMMOVE-DST %s
#include <sycl/detail/core.hpp>

#include <sycl/usm.hpp>

constexpr size_t N = 12;

void test_memset(sycl::queue &Q) {
  auto *ptr = sycl::malloc_device<char>(N, Q);

  Q.submit([&](sycl::handler &h) {
    h.parallel_for<class test_memset>(
        sycl::nd_range<1>(1, 1),
        [=](sycl::nd_item<1>) { memset(ptr, 1, N + 1); });
  });
  Q.wait();
  // CHECK-MEMSET: ERROR: DeviceSanitizer: out-of-bounds-access on Device USM
  // CHECK-MEMSET: {{WRITE of size 13 at kernel <.*test_memset>}}
  // CHECK-MEMSET: {{  #0 .* .*string_func.cpp:}}[[@LINE-5]]

  sycl::free(ptr, Q);
}

void test_memcpy(sycl::queue &Q, bool is_src_oob) {
  auto *dst = sycl::malloc_device<char>(N, Q);
  auto *src = sycl::malloc_device<char>(N, Q);

  if (is_src_oob)
    Q.submit([&](sycl::handler &h) {
      h.parallel_for<class test_memcpy_src>(
          sycl::nd_range<1>(1, 1),
          [=](sycl::nd_item<1>) { memcpy(dst, src + 1, N); });
    });
  else
    Q.submit([&](sycl::handler &h) {
      h.parallel_for<class test_memcpy_dst>(
          sycl::nd_range<1>(1, 1),
          [=](sycl::nd_item<1>) { memcpy(dst + 1, src, N); });
    });
  Q.wait();
  // CHECK-MEMCPY-SRC: ERROR: DeviceSanitizer: out-of-bounds-access on Device USM
  // CHECK-MEMCPY-SRC: {{READ of size 12 at kernel <.*test_memcpy_src>}}
  // CHECK-MEMCPY-SRC: {{  #0 .* .*string_func.cpp:}}[[@LINE-11]]
  // CHECK-MEMCPY-DST: ERROR: DeviceSanitizer: out-of-bounds-access on Device USM
  // CHECK-MEMCPY-DST: {{WRITE of size 12 at kernel <.*test_memcpy_dst>}}
  // CHECK-MEMCPY-DST: {{  #0 .* .*string_func.cpp:}}[[@LINE-8]]

  sycl::free(dst, Q);
  sycl::free(src, Q);
}

void test_memmove(sycl::queue &Q, bool is_src_oob) {
  auto *dst = sycl::malloc_device<char>(N, Q);
  auto *src = sycl::malloc_device<char>(N, Q);

  if (is_src_oob)
    Q.submit([&](sycl::handler &h) {
      h.parallel_for<class test_memmove_src>(
          sycl::nd_range<1>(1, 1),
          [=](sycl::nd_item<1>) { memmove(dst, src + 1, N); });
    });
  else
    Q.submit([&](sycl::handler &h) {
      h.parallel_for<class test_memmove_dst>(
          sycl::nd_range<1>(1, 1),
          [=](sycl::nd_item<1>) { memmove(dst + 1, src, N); });
    });
  Q.wait();
  // CHECK-MEMMOVE-SRC: ERROR: DeviceSanitizer: out-of-bounds-access on Device USM
  // CHECK-MEMMOVE-SRC: {{READ of size 12 at kernel <.*test_memmove_src>}}
  // CHECK-MEMMOVE-SRC: {{  #0 .* .*string_func.cpp:}}[[@LINE-11]]
  // CHECK-MEMMOVE-DST: ERROR: DeviceSanitizer: out-of-bounds-access on Device USM
  // CHECK-MEMMOVE-DST: {{WRITE of size 12 at kernel <.*test_memmove_dst>}}
  // CHECK-MEMMOVE-DST: {{  #0 .* .*string_func.cpp:}}[[@LINE-8]]

  sycl::free(dst, Q);
  sycl::free(src, Q);
}

int main(int argc, char **argv) {
  assert(argc > 1 && "test is not specified");
  sycl::queue Q;

  if (!strcmp(argv[1], "memset")) {
    test_memset(Q);
  } else if (!strcmp(argv[1], "memcpy")) {
    if (!strcmp(argv[2], "src"))
      test_memcpy(Q, true);
    else
      test_memcpy(Q, false);
  } else if (!strcmp(argv[1], "memmove")) {
    if (!strcmp(argv[2], "src"))
      test_memmove(Q, true);
    else
      test_memmove(Q, false);
  }
  return 0;
}
