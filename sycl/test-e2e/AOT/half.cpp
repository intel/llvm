// This test ensures that a program that has a kernel
// using fp16 can be compiled AOT.

// REQUIRES: ocloc, opencl-aot, any-device-is-cpu
// RUN: %clangxx -fsycl -fsycl-targets=intel_gpu_tgllp -o %t.tgllp.out %s
// RUN: %clangxx -fsycl -fsycl-targets=spir64_x86_64 -o %t.x86.out %s

// ocloc on windows does not have support for PVC/CFL, so this command will
// result in an error when on windows. (In general, there is no support
// for pvc/cfl on windows.)
// RUN: %if !windows %{ %clangxx -fsycl -fsycl-targets=intel_gpu_cfl -o %t.cfl.out %s %}
// RUN: %if !windows %{ %clangxx -fsycl -fsycl-targets=intel_gpu_pvc -o %t.pvc.out %s %}

#include <sycl/detail/core.hpp>

using namespace sycl;

int main() {
  queue q;
  if (q.get_device().has(aspect::fp16)) {
    sycl::half h = 2.5;
    {
      buffer<sycl::half, 1> buf(&h, 1);
      q.submit([&](handler &cgh) {
        accessor acc{buf, cgh};
        cgh.single_task([=] { acc[0] *= 2; });
      });
    }
    std::cout << h << "\n";
  }
}
