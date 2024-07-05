// This test ensures that a program that has a kernel
// using fp64 can be compiled AOT.

// REQUIRES: ocloc
// RUN: %clangxx -fsycl -fsycl-targets=intel_gpu_tgllp -o %t.tgllp.out %s
// RUN: %clangxx -fsycl -fsycl-targets=intel_gpu_pvc -o %t.pvc.out %s
// RUN: %clangxx -fsycl -fsycl-targets=intel_gpu_cfl -o %t.cfl.out %s

#include <sycl/detail/core.hpp>

using namespace sycl;

int main() {
  queue q;
  if (q.get_device().has(aspect::fp64)) {
    double d = 2.5;
    {
      buffer<double, 1> buf(&d, 1);
      q.submit([&](handler &cgh) {
        accessor acc{buf, cgh};
        cgh.single_task([=] { acc[0] *= 2; });
      });
    }
    std::cout << d << "\n";
  }
}
