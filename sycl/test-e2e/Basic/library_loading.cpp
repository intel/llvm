// REQUIRES: linux
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: env SYCL_PI_TRACE=-1 %t.out &> %t_trace_no_filter.txt || true
// RUN: FileCheck --input-file=%t_trace_no_filter.txt --check-prefix=CHECK-NO-FILTER %s
// RUN: env SYCL_PI_TRACE=-1 ONEAPI_DEVICE_SELECTOR='esimd_emulator:*' %t.out &> %t_trace_esimd_filter.txt || true
// RUN: FileCheck --input-file=%t_trace_esimd_filter.txt --check-prefix=CHECK-ESIMD-FILTER %s
// Checks pi traces on library loading

#include <sycl/sycl.hpp>

using namespace sycl;

int main() {
  // CHECK-NO-FILTER: {{(SYCL_PI_TRACE\[-1\]: dlopen\(.*/libpi_cuda.so\) failed with)|(SYCL_PI_TRACE\[basic\]: Plugin found and successfully loaded: libpi_cuda.so)}}
  // CHECK-NO-FILTER-NOT: {{(SYCL_PI_TRACE\[-1\]: dlopen\(.*/libpi_esimd_emulator.so\) failed with)|(SYCL_PI_TRACE\[basic\]: Plugin found and successfully loaded: libpi_esimd_emulator.so)}}
  // CHECK-NO-FILTER: {{(SYCL_PI_TRACE\[-1\]: dlopen\(.*/libpi_hip.so\) failed with)|(SYCL_PI_TRACE\[basic\]: Plugin found and successfully loaded: libpi_hip.so)}}
  // CHECK-ESIMD-FILTER: {{(SYCL_PI_TRACE\[-1\]: dlopen\(.*/libpi_esimd_emulator.so\) failed with)|(SYCL_PI_TRACE\[basic\]: Plugin found and successfully loaded: libpi_esimd_emulator.so)}}
  queue q;
  q.submit([&](handler &cgh) {});
}
