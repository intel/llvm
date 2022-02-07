// RUN: %clang_cc1 -internal-isystem %S/Inputs -fsycl-is-device -verify -fsyntax-only %s

// This test checks that a warning is emitted on attempt to cast from default
// address space to named address spaces.

#include "sycl.hpp"

using namespace sycl;

void foo(__attribute__((opencl_global)) int *A) {
}

void __attribute__((sycl_device)) onDeviceUsages(multi_ptr<int, access::address_space::global_space> F) {
  int *NoAS;
  // expected-warning@+1 {{explicit cast from 'int *' to '__local int *' potentially leads to an invalid address space cast in the resulting code}}
  __attribute__((opencl_local)) int *LL = (__attribute((opencl_local)) int *)NoAS;

  // expected-warning@Inputs/sycl.hpp:356 {{explicit cast from 'int *' to 'sycl::multi_ptr<int, sycl::access::address_space::private_space>::pointer_t' (aka '__private int *') potentially leads to an invalid address space cast in the resulting code}}
  //expected-note@+1 {{called by 'onDeviceUsages'}}
  auto P = multi_ptr<int, access::address_space::private_space>{F.get()};

  // expected-warning@+1 {{explicit cast from 'int *' to '__global int *' potentially leads to an invalid address space cast in the resulting code}}
  foo((__attribute((opencl_global)) int *)NoAS);
}
