// REQUIRES: linux
// UNSUPPORTED: cuda || hip
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: env SYCL_PI_TRACE=-1 %t.out 2>&1 | FileCheck %s

// Checks pi traces when libpi_cuda and libpi_hip are not present
#include <CL/sycl.hpp>

using namespace cl::sycl;

int main() {
  // CHECK: SYCL_PI_TRACE[-1]: dlopen(libpi_cuda.so) failed with <libpi_cuda.so: cannot open shared object file: No such file or directory>
  // CHECK: SYCL_PI_TRACE[-1]: dlopen(libpi_hip.so) failed with <libpi_hip.so: cannot open shared object file: No such file or directory>
  queue q;
  q.submit([&](handler &cgh) {});
}
