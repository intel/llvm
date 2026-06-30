// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note %s

// Tests that internal getSyclObjImpl API is not exposed via ADL for any of the
// user-facing SYCL objects (many of which inherit from helper bases living in
// the `sycl::detail` namespace, e.g. OwnerLessBase).
// Regression test for https://github.com/intel/llvm/issues/20820

#include <sycl/sycl.hpp>

void test_no_adl() {
  sycl::device d;
  // getSyclObjImpl is internal and should not be found via ADL
  // expected-error@+1 {{use of undeclared identifier 'getSyclObjImpl'}}
  auto id = getSyclObjImpl(d);

  sycl::queue q;
  // expected-error@+1 {{use of undeclared identifier 'getSyclObjImpl'}}
  auto iq = getSyclObjImpl(q);

  sycl::platform p;
  // expected-error@+1 {{use of undeclared identifier 'getSyclObjImpl'}}
  auto ip = getSyclObjImpl(p);

  sycl::context ctx;
  // expected-error@+1 {{use of undeclared identifier 'getSyclObjImpl'}}
  auto ictx = getSyclObjImpl(ctx);

  sycl::event e;
  // expected-error@+1 {{use of undeclared identifier 'getSyclObjImpl'}}
  auto ie = getSyclObjImpl(e);

  sycl::buffer<int, 1> buf{sycl::range<1>{1}};
  // expected-error@+1 {{use of undeclared identifier 'getSyclObjImpl'}}
  auto ibuf = getSyclObjImpl(buf);

  sycl::host_accessor acc{buf};
  // expected-error@+1 {{use of undeclared identifier 'getSyclObjImpl'}}
  auto iacc = getSyclObjImpl(acc);

  sycl::kernel_id kid = sycl::get_kernel_id<class SomeKernel>();
  // expected-error@+1 {{use of undeclared identifier 'getSyclObjImpl'}}
  auto ikid = getSyclObjImpl(kid);

  sycl::kernel_bundle kb =
      sycl::get_kernel_bundle<sycl::bundle_state::executable>(ctx);
  // expected-error@+1 {{use of undeclared identifier 'getSyclObjImpl'}}
  auto ikb = getSyclObjImpl(kb);
}
