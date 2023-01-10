// RUN: %clangxx -fsycl -fsycl-device-only -std=c++17 -fno-sycl-unnamed-lambda -isystem %sycl_include/sycl -Xclang -verify -fsyntax-only %s -Xclang -verify-ignore-unexpected=note
#include <sycl/sycl.hpp>

int main() {
  sycl::queue q;

  // expected-error@sycl/kernel.hpp:* {{No kernel name provided without -fsycl-unnamed-lambda enabled!}}
  // expected-note@+1 {{in instantiation of function template}}
  q.single_task([=](){});

  // expected-error@sycl/kernel.hpp:* {{No kernel name provided without -fsycl-unnamed-lambda enabled!}}
  // expected-note@+1 {{in instantiation of function template}}
  q.parallel_for(sycl::range<1>{1}, [=](sycl::item<1>) {});

  q.submit([&](sycl::handler &cgh) {
    // expected-error@sycl/kernel.hpp:* {{No kernel name provided without -fsycl-unnamed-lambda enabled!}}
    // expected-note@+1 {{in instantiation of function template}}
    cgh.single_task([=](){});

    // expected-error@sycl/kernel.hpp:* {{No kernel name provided without -fsycl-unnamed-lambda enabled!}}
    // expected-note@+1 {{in instantiation of function template}}
    cgh.parallel_for(sycl::range<1>{1}, [=](sycl::item<1>) {});

    // expected-error@sycl/kernel.hpp:* {{No kernel name provided without -fsycl-unnamed-lambda enabled!}}
    // expected-note@+1 {{in instantiation of function template}}
    cgh.parallel_for_work_group(sycl::range<1>{1}, [=](sycl::group<1>) {});
  });
}
