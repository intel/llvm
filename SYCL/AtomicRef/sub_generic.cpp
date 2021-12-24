// RUN: %clangxx -fsycl -fsycl-unnamed-lambda -fsycl-targets=%sycl_triple %s -o %t.out \
// RUN: -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_60
// RUN: %HOST_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

// CUDA backend has had no support for the generic address space yet
// XFAIL: cuda || hip

#include "sub.h"
#include <iostream>
using namespace sycl;

// Floating-point types do not support pre- or post-decrement
template <> void sub_generic_test<float>(queue q, size_t N) {
  sub_fetch_test<::sycl::atomic_ref, access::address_space::generic_space,
                 float>(q, N);
  sub_plus_equal_test<::sycl::atomic_ref, access::address_space::generic_space,
                      float>(q, N);
}

int main() {
  queue q;

  constexpr int N = 32;
  sub_generic_test<int>(q, N);
  sub_generic_test<unsigned int>(q, N);
  sub_generic_test<float>(q, N);

  // Include long tests if they are 32 bits wide
  if constexpr (sizeof(long) == 4) {
    sub_generic_test<long>(q, N);
    sub_generic_test<unsigned long>(q, N);
  }

  // Include pointer tests if they are 32 bits wide
  if constexpr (sizeof(char *) == 4) {
    sub_generic_test<char *, ptrdiff_t>(q, N);
  }

  std::cout << "Test passed." << std::endl;
}
