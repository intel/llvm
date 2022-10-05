// RUN: %clang_cc1 -fsycl-is-device -verify=expected,fpga -fsyntax-only -fintelfpga -internal-isystem %S/Inputs %s
// RUN: %clang_cc1 -fsycl-is-device -verify -fsyntax-only -internal-isystem %S/Inputs %s

#include "sycl.hpp"

// This test - created largley from sycl-restrict.cpp - tests that
// when the 'SYCL kernel cannot use a non-const static data variable'
// error diagnostic is generated, and -fintelfpga option is enabled,
// we also generate the note that 'in the future, Intel FPGA compilations
// will support 'device_global' variables that retain state between
// invocations'

// This note is expected to be temporary and when the feature is supported
// in the future, this message and likely this test too, will be removed.

// This test also checks that this note is not generated (i.e. retains
// previous behavior) when -fintelfpga is not specified.

typedef struct A {
  static int stat_member;

  int fm(void) {
    // expected-error@+2{{SYCL kernel cannot use a non-const static data variable}}
    // fpga-note@+1{{in the future, Intel FPGA compilations will support 'device_global' variables that retain state between invocations}}
    return stat_member; 
  }
} a_type;

int use2(a_type ab, a_type *abp) {
  if (ab.fm()) // expected-note {{called by 'use2'}}
    return 0;

  // expected-error@+2{{SYCL kernel cannot use a non-const static data variable}}
  // fpga-note@+1{{in the future, Intel FPGA compilations will support 'device_global' variables that retain state between invocations}}
  if (ab.stat_member) 
    return 0;
  // expected-error@+2{{SYCL kernel cannot use a non-const static data variable}}
  // fpga-note@+1{{in the future, Intel FPGA compilations will support 'device_global' variables that retain state between invocations}}
  if (abp->stat_member)
    return 0;
}

// expected-note@#KernelSingleTaskKernelFuncCall 3{{called by 'kernel_single_task}}

int main() {
  sycl::handler h;
  a_type ab;

  h.single_task([=]() {
    a_type *p;
    // expected-error@+2{{SYCL kernel cannot use a non-const static data variable}}
    // fpga-note@+1{{in the future, Intel FPGA compilations will support 'device_global' variables that retain state between invocations}}
    static int i; 
    use2(ab, p); // expected-note 2{{called by 'operator()'}}
  });
  return 0;
}
