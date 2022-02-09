// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -O0 -o %t.out
// RUN: %HOST_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// UNSUPPORTED: cuda
// Disable test due to flaky failure on CUDA(issue #387)

// NOTE: The libclc target used by the CUDA backend used to generate atomic load
//       variants that were unsupported by NVPTX. Even if they were not used
//       directly, sycl::stream and other operations would keep the invalid
//       operations in when optimizations were disabled.

#include <sycl/sycl.hpp>

int main() {
  sycl::queue q;
  q.submit([&](sycl::handler &cgh) {
    sycl::stream os(1024, 256, cgh);
    cgh.single_task([=]() { os << "test"; });
  });
  q.wait();
  return 0;
}
