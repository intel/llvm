// Test hangs on AMD with https://github.com/intel/llvm/pull/8412
// UNSUPPORTED: hip_amd

// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -O0 -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

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
