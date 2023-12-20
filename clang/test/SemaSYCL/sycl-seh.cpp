// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple spir64 -aux-triple x86_64 -fms-compatibility %s -verify
//
// This test checks that we issue a diagnostic about the use of SEH
// features only when they are used on the device.
//
#include "sycl.hpp"

void seh_okay_on_host() {
  __try {
  } __except(0) {
  }
}

void seh_not_okay_on_device() {
  // expected-error@+2 {{SEH '__try' is not supported on this target}}
  // expected-error@+1 {{SYCL kernel cannot use exceptions}}
  __try {
  } __except(0) {
  }
}

void foo() {
  sycl::queue q;

  seh_okay_on_host();
  q.submit([&](sycl::handler &h) {
    // expected-note@#KernelSingleTaskKernelFuncCall {{called by 'kernel_single_task<kernel_wrapper}}
    h.single_task<class kernel_wrapper>(
        [=]() {
          // expected-note@+1 {{called by 'operator()'}}
          seh_not_okay_on_device();
        });
  });
}
