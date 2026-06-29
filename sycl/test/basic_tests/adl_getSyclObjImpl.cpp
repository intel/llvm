// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note %s

// Tests that internal getSyclObjImpl API is not exposed via ADL
// Regression test for https://github.com/intel/llvm/issues/20820

#include <sycl/sycl.hpp>

void test_no_adl() {
  sycl::device d;
  // getSyclObjImpl is internal and should not be found via ADL
  auto id = getSyclObjImpl(
      d); // expected-error {{use of undeclared identifier 'getSyclObjImpl'}}

  sycl::queue q;
  auto iq = getSyclObjImpl(
      q); // expected-error {{use of undeclared identifier 'getSyclObjImpl'}}
}
