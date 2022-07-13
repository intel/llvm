// UNSUPPORTED: hip_amd
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %HOST_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

#include <stdlib.h>
#include <sycl/sycl.hpp>

int main() {

  cl::sycl::device dev{cl::sycl::default_selector{}};
  cl::sycl::queue q{dev};

  q.submit(
      [&](sycl::handler &cgh) { cgh.single_task<class kernel1>([]() {}); });

  cl::sycl::event e = q.ext_oneapi_submit_barrier();
  e.wait_and_throw();

  return 0;
}
