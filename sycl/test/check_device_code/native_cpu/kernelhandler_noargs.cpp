// RUN: %clangxx -fsycl-device-only -fsycl-targets=native_cpu -Xclang -fsycl-int-header=%t.h  -o %t.bc %s
// RUN: FileCheck -input-file=%t.h.hc %s --check-prefix=CHECK-HC
// Compiling generated main integration header to check correctness, -fsycl
// option used to find required includes
// RUN: %clangxx -fsycl -D __SYCL_NATIVE_CPU__ -c -x c++ %t.h

#include "sycl.hpp"
class Test1;
int main() {
  sycl::queue deviceQueue;
  sycl::accessor<int, 1, sycl::access::mode::write> acc;
  sycl::range<1> r(1);
  deviceQueue.submit([&](sycl::handler &h) {
    h.parallel_for<Test1>(r, [=](sycl::id<1> id) {
      acc[id[0]]; // all kernel arguments are removed
    });
  });
}

//CHECK-HC: #pragma once
//CHECK-HC-NEXT: #include <sycl/detail/native_cpu.hpp>
//CHECK-HC:extern "C" void _ZTS5Test1_NativeCPUKernelsubhandler(const sycl::detail::NativeCPUArgDesc *MArgs, __nativecpu_state *state);
