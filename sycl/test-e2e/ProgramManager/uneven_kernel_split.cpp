// REQUIRES: gpu, opencl-aot, ocloc
// UNSUPPORTED: cuda, hip

// RUN: %clangxx -fsycl -fsycl-targets=spir64_x86_64 -I %S/Inputs/ %S/uneven_kernel_split.cpp -c -o %t.o
// RUN: %clangxx -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen %gpu_aot_target_opts -I %S/Inputs/ %S/Inputs/gpu_kernel1.cpp -c -o %t1.o
// RUN: %clangxx -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen %gpu_aot_target_opts -I %S/Inputs/ %S/Inputs/gpu_kernel2.cpp -c -o %t2.o
// RUN: %clangxx -fsycl -fsycl-targets=spir64_x86_64,spir64_gen -Xsycl-target-backend=spir64_gen %gpu_aot_target_opts %t.o %t1.o %t2.o -o %t.out
// RUN: %{run} %t.out

// Test require the following device image structure: cpu target device image
// contains kernel 1 and kernel 2. gpu target device image contains kernel1 and
// another gpu target device image contains kernel2. Kernel names must be the
// same for both targets. Checks validity of device image search.

#include "inc.hpp"
#include <sycl/properties/all_properties.hpp>
#include <sycl/usm.hpp>

void host_foo(sycl::queue &queue, int *buf) {
  queue.submit([&](sycl::handler &h) {
    h.single_task<KernelTest1>([=]() { buf[0] = buf[1] - 5; });
  });
}

void host_bar(sycl::queue &queue, int *buf) {
  queue.submit([&](sycl::handler &h) {
    h.single_task<KernelTest2>([=]() { buf[0] = buf[1] + 25; });
  });
}

int main() {
  sycl::queue q{sycl::device{}, sycl::property::queue::in_order()};

  auto buf = sycl::malloc_shared<int>(10, q);
  buf[0] = -1;
  buf[1] = 2;

  if (q.get_device().is_gpu()) {
    gpu_bar(q, buf);
    gpu_foo(q, buf);
  } else {
    host_bar(q, buf);
    host_foo(q, buf);
  }
  q.wait();
  sycl::free(buf, q);
  return 0;
}
