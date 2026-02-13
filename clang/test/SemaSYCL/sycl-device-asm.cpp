// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple spir64 -sycl-std=2020 %s -verify

// This test checks that the device compiler does not issue an asm
// diagnostic about a register incorrect for it, unless it is a routine
// called on the device.
#include "sycl.hpp"

void non_device_func(int value) {
  // expected-no-diagnostic@+1
  { register int v asm ("eax") = value; }
}

void device_func(int value) {
  // expected-error@+1 {{unknown register name 'eax' in asm}}
  { register int v asm ("eax") = value; }
}

int main() {
  sycl::queue q;

  q.submit([&](sycl::handler &h) {
    // expected-note@#KernelSingleTaskKernelFuncCall {{called by 'kernel_single_task<kernel_wrapper}}
    h.single_task<class kernel_wrapper>(
        [=]() {
          // expected-note@+1 {{called by 'operator()'}}
          device_func(5);
        });
  });
}
