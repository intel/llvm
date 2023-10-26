// REQUIRES: linux
// RUN: %{build} -o %t.out
// RUN: env SYCL_PI_TRACE=-1 %{run-unfiltered-devices} %t.out &> %t_trace_no_filter.txt || true
// RUN: FileCheck --input-file=%t_trace_no_filter.txt --check-prefix=CHECK-NO-FILTER %s -dump-input=fail
// RUN: FileCheck --input-file=%t_trace_no_filter.txt --check-prefix=CHECK-ESIMD %s -dump-input=fail
// RUN: env SYCL_PI_TRACE=-1 ONEAPI_DEVICE_SELECTOR='esimd_emulator:*' %{run-unfiltered-devices} %t.out &> %t_trace_esimd_filter.txt || true
// RUN: FileCheck --input-file=%t_trace_esimd_filter.txt --check-prefix=CHECK-ESIMD-FILTER %s
// Checks pi traces on library loading

#include <sycl/sycl.hpp>

using namespace sycl;

int main() {
  // CHECK-NO-FILTER-DAG: {{(SYCL_PI_TRACE\[-1\]: dlopen\(.*/libpi_cuda.so\) failed with)|(SYCL_PI_TRACE\[basic\]: Plugin found and successfully loaded: libpi_cuda.so)}}
  // CHECK-NO-FILTER-DAG: {{(SYCL_PI_TRACE\[-1\]: dlopen\(.*/libpi_hip.so\) failed with)|(SYCL_PI_TRACE\[basic\]: Plugin found and successfully loaded: libpi_hip.so)}}
  // CHECK-ESIMD-NOT: {{(SYCL_PI_TRACE\[-1\]: dlopen\(.*/libpi_esimd_emulator.so\) failed with)|(SYCL_PI_TRACE\[basic\]: Plugin found and successfully loaded: libpi_esimd_emulator.so)}}
  // CHECK-ESIMD-FILTER: {{(SYCL_PI_TRACE\[-1\]: dlopen\(.*/libpi_esimd_emulator.so\) failed with)|(SYCL_PI_TRACE\[basic\]: Plugin found and successfully loaded: libpi_esimd_emulator.so)}}
  queue q;
  q.submit([&](handler &cgh) {});
}
