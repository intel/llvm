// RUN: %clangxx -fsycl -fsycl-unnamed-lambda -fsycl-targets=%sycl_triple %s -o %t.out \
// RUN: -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_60
// RUN: %HOST_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

#include "sub.h"
#include <iostream>
using namespace sycl;

// Floating-point types do not support pre- or post-decrement
template <> void sub_test<float>(queue q, size_t N) {
  sub_fetch_test<float>(q, N);
  sub_plus_equal_test<float>(q, N);
}

int main() {
  queue q;

  constexpr int N = 32;
  sub_test<int>(q, N);
  sub_test<unsigned int>(q, N);
  sub_test<float>(q, N);

  // Include long tests if they are 32 bits wide
  if constexpr (sizeof(long) == 4) {
    sub_test<long>(q, N);
    sub_test<unsigned long>(q, N);
  }

  // Include pointer tests if they are 32 bits wide
  if constexpr (sizeof(char *) == 4) {
    sub_test<char *, ptrdiff_t>(q, N);
  }

  std::cout << "Test passed." << std::endl;
}
