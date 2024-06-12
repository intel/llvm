// REQUIRES: linux, cpu
// RUN: %{build} %device_asan_flags -DTEST1 -O0 -g -o %t
// RUN: env SYCL_PREFER_UR=1 %{run} not %t 2>&1 | FileCheck --check-prefixes CHECK,CHECK1 %s
// RUN: %{build} %device_asan_flags -DTEST2 -O0 -g -o %t
// RUN: env SYCL_PREFER_UR=1 %{run} not %t 2>&1 | FileCheck --check-prefixes CHECK,CHECK2 %s

#include <sycl/detail/core.hpp>

#include <sycl/usm.hpp>

static constexpr std::size_t ASAN_SHADOW_SCALE = 4;
static constexpr std::size_t ASAN_SHADOW_GRANULARITY = 1 << ASAN_SHADOW_SCALE;

#ifdef TEST1
typedef uint64_t TestType;
#elif TEST2
typedef unsigned _BitInt(128) TestType;
#endif

int main() {
  sycl::queue Q;
  constexpr std::size_t size = 128 + (ASAN_SHADOW_GRANULARITY - 1);
  TestType *array = (TestType *)sycl::malloc_device<char>(size, Q);

  Q.submit([&](sycl::handler &h) {
    h.parallel_for<class MyKernelR_4>(
        sycl::nd_range<1>(size / sizeof(TestType) + 1, 1),
        [=](sycl::nd_item<1> item) { ++array[item.get_global_id(0)]; });
  });
  Q.wait();
  // CHECK: ERROR: DeviceSanitizer: out-of-bounds-access on Device USM
  // CHECK1: {{READ of size 8 at kernel <.*MyKernelR_4> LID\(0, 0, 0\) GID\(17, 0, 0\)}}
  // CHECK1: {{  #0 .* .*unaligned_shadow_memory.cpp:}}[[@LINE-5]]
  // CHECK2: {{READ of size 16 at kernel <.*MyKernelR_4> LID\(0, 0, 0\) GID\(8, 0, 0\)}}
  // CHECK2: {{  #0 .* .*unaligned_shadow_memory.cpp:}}[[@LINE-7]]

  return 0;
}
