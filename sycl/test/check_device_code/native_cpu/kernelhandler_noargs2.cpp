// cmdline that used to fail in kernel handler emission
// RUN: %clangxx -fsycl-device-only  -fsycl-targets=native_cpu -sycl-std=2020 -Xclang -fsycl-int-header=%t.h -mllvm -sycl-opt -S -emit-llvm  -o - %s
// RUN: FileCheck -input-file=%t.h.hc %s --check-prefix=CHECK-HC
// RUN: %clangxx -fsycl -D __SYCL_NATIVE_CPU__ -c -x c++ %t.h
#include "sycl.hpp"

template <typename name, typename Func>
__attribute__((sycl_kernel)) void launch(const Func &kernelFunc) {
  kernelFunc();
}
int main() {
  launch<class TestKernel>([]() {});
  return 0;
}

//CHECK-HC: #pragma once
//CHECK-HC-NEXT: #include <sycl/detail/native_cpu.hpp>
//CHECK-HC:extern "C" void _ZTSZ4mainE10TestKernel_NativeCPUKernelsubhandler(const sycl::detail::NativeCPUArgDesc *MArgs, __nativecpu_state *state);
