// RUN: %clangxx -fsycl -fsycl-unnamed-lambda -fsycl-targets=%sycl_triple %s -o %t.out \
// RUN: -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_60
// RUN: %HOST_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

// XFAIL: hip
// Expected failure because hip does not have atomic64 check implementation

#include "sub.h"
#include <iostream>
using namespace sycl;

// Floating-point types do not support pre- or post-decrement
template <> void sub_test<double>(queue q, size_t N) {
  sub_fetch_test<::sycl::ext::oneapi::atomic_ref,
                 access::address_space::global_space, double>(q, N);
  sub_fetch_test<::sycl::atomic_ref, access::address_space::global_space,
                 double>(q, N);
  sub_plus_equal_test<::sycl::ext::oneapi::atomic_ref,
                      access::address_space::global_space, double>(q, N);
  sub_plus_equal_test<::sycl::atomic_ref, access::address_space::global_space,
                      double>(q, N);
}

int main() {
  queue q;

  if (!q.get_device().has(aspect::atomic64)) {
    std::cout << "Skipping test\n";
    return 0;
  }

  constexpr int N = 32;
  sub_test<double>(q, N);

  // Include long tests if they are 64 bits wide
  if constexpr (sizeof(long) == 8) {
    sub_test<long>(q, N);
    sub_test<unsigned long>(q, N);
  }

  // Include long long tests if they are 64 bits wide
  if constexpr (sizeof(long long) == 8) {
    sub_test<long long>(q, N);
    sub_test<unsigned long long>(q, N);
  }

  // Include pointer tests if they are 64 bits wide
  if constexpr (sizeof(char *) == 8) {
    sub_test<char *, ptrdiff_t>(q, N);
  }

  std::cout << "Test passed." << std::endl;
}
