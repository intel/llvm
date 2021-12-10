// RUN: %clangxx -fsycl -fsycl-device-only -std=c++17 -fno-sycl-unnamed-lambda -isystem %sycl_include/sycl -Xclang -verify -fsyntax-only %s -Xclang -verify-ignore-unexpected=note
#include <CL/sycl.hpp>

int main() {
  cl::sycl::queue q;

  // expected-error@CL/sycl/kernel.hpp:* {{No kernel name provided without -fsycl-unnamed-lambda enabled!}}
  // expected-note@+1 {{in instantiation of function template}}
  q.single_task([=](){});

  // expected-error@CL/sycl/kernel.hpp:* {{No kernel name provided without -fsycl-unnamed-lambda enabled!}}
  // expected-note@+1 {{in instantiation of function template}}
  q.parallel_for(cl::sycl::range<1>{1}, [=](cl::sycl::item<1>){});

  q.submit([&](cl::sycl::handler &cgh) {
    // expected-error@CL/sycl/kernel.hpp:* {{No kernel name provided without -fsycl-unnamed-lambda enabled!}}
    // expected-note@+1 {{in instantiation of function template}}
    cgh.single_task([=](){});

    // expected-error@CL/sycl/kernel.hpp:* {{No kernel name provided without -fsycl-unnamed-lambda enabled!}}
    // expected-note@+1 {{in instantiation of function template}}
    cgh.parallel_for(cl::sycl::range<1>{1}, [=](cl::sycl::item<1>){});

    // expected-error@CL/sycl/kernel.hpp:* {{No kernel name provided without -fsycl-unnamed-lambda enabled!}}
    // expected-note@+1 {{in instantiation of function template}}
    cgh.parallel_for_work_group(cl::sycl::range<1>{1}, [=](cl::sycl::group<1>){});
  });
}
