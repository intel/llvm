// REQUIRES: linux
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: env SYCL_PI_TRACE=-1 %t.out 2>&1 | FileCheck %s

// Checks pi traces on library loading
#include <CL/sycl.hpp>

using namespace cl::sycl;

int main() {
  // CHECK: {{(SYCL_PI_TRACE\[-1\]: dlopen\(libpi_cuda.so\) failed with)|(SYCL_PI_TRACE\[basic\]: Plugin found and successfully loaded: libpi_cuda.so)}}
  // CHECK: {{(SYCL_PI_TRACE\[-1\]: dlopen\(libpi_hip.so\) failed with)|(SYCL_PI_TRACE\[basic\]: Plugin found and successfully loaded: libpi_hip.so)}}
  // CHECK: {{(SYCL_PI_TRACE\[-1\]: dlopen\(libpi_esimd_emulator.so\) failed with)|(SYCL_PI_TRACE\[basic\]: Plugin found and successfully loaded: libpi_esimd_emulator.so)}}
  queue q;
  q.submit([&](handler &cgh) {});
}
