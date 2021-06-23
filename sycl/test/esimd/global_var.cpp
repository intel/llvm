// RUN: %clangxx -fsycl -fsycl-device-only -fsyntax-only -Xclang -verify %s

#include <CL/sycl.hpp>
#include <sycl/ext/intel/experimental/esimd.hpp>

using namespace cl::sycl;

// This test checks usage of an ESIMD global in ESIMD(positive) and
// SYCL(negative) contexts.

ESIMD_PRIVATE ESIMD_REGISTER(17) int vc;

void func_that_uses_esimd_glob() {
  //expected-error@+1 2{{ESIMD globals cannot be used in a SYCL context}}
  vc = 0;
}

// ESIMD external function is allowed to use ESIMD global
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION void init_vc_esimd(int x) {
  func_that_uses_esimd_glob();
  vc = x;
}

SYCL_EXTERNAL void init_vc_sycl(int x) {
  //expected-note@+1{{called by}}
  func_that_uses_esimd_glob();
  // expected-error@+1{{ESIMD globals cannot be used in a SYCL context}}
  vc = x;
}

void kernel_call() {
  queue q;
  q.submit([&](cl::sycl::handler &cgh) {
    cgh.parallel_for<class Test>(nd_range<1>(1, 1),
                                 [=](nd_item<1> ndi) SYCL_ESIMD_KERNEL {
                                   // ESIMD kernel is allowed to use ESIMD
                                   // global
                                   vc = 0;
                                   func_that_uses_esimd_glob();
                                 });
  });

  q.submit([&](cl::sycl::handler &cgh) {
    cgh.parallel_for<class Test>(nd_range<1>(1, 1), [=](nd_item<1> ndi) {
      //expected-note@CL/sycl/handler.hpp:* 2{{called by 'kernel_parallel_for}}
      //expected-error@+1{{ESIMD globals cannot be used in a SYCL context}}
      vc = 0;
      //expected-note@+1{{called by}}
      func_that_uses_esimd_glob();
    });
  });
}
