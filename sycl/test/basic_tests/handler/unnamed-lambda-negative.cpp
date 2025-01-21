// RUN: %clangxx -fsycl -fsycl-device-only -std=c++17 -fno-sycl-unnamed-lambda -isystem %sycl_include/sycl -Xclang -verify -fsyntax-only %s -Xclang -verify-ignore-unexpected=note

// NOTE: Due to rounded kernels, some parallel_for cases may issue two error
//       diagnostics.

#include <sycl/sycl.hpp>

int main() {
  sycl::queue q;

  // expected-error-re@sycl/handler.hpp:* {{unnamed type '{{.*}}' is invalid; provide a kernel name, or use '-fsycl-unnamed-lambda' to enable unnamed kernel lambdas}}
  // expected-note@+1 {{in instantiation of function template}}
  q.single_task([=](){});

  // expected-error-re@sycl/handler.hpp:* {{unnamed type 'sycl::detail::RoundedRangeKernel<{{.*}}>' is invalid; provide a kernel name, or use '-fsycl-unnamed-lambda' to enable unnamed kernel lambdas}}
  // expected-error-re@sycl/handler.hpp:* {{unnamed type '{{.*}}' is invalid; provide a kernel name, or use '-fsycl-unnamed-lambda' to enable unnamed kernel lambdas}}
  // expected-note@+1 {{in instantiation of function template}}
  q.parallel_for(sycl::range<1>{1}, [=](sycl::item<1>) {});

  q.submit([&](sycl::handler &cgh) {
    // expected-error-re@sycl/handler.hpp:* {{unnamed type '{{.*}}' is invalid; provide a kernel name, or use '-fsycl-unnamed-lambda' to enable unnamed kernel lambdas}}
    // expected-note@+1 {{in instantiation of function template}}
    cgh.single_task([=](){});

    // expected-error-re@sycl/handler.hpp:* {{unnamed type 'sycl::detail::RoundedRangeKernel<{{.*}}>' is invalid; provide a kernel name, or use '-fsycl-unnamed-lambda' to enable unnamed kernel lambdas}}
    // expected-error-re@sycl/handler.hpp:* {{unnamed type '{{.*}}' is invalid; provide a kernel name, or use '-fsycl-unnamed-lambda' to enable unnamed kernel lambdas}}
    // expected-note@+1 {{in instantiation of function template}}
    cgh.parallel_for(sycl::range<1>{1}, [=](sycl::item<1>) {});

    // expected-error-re@sycl/handler.hpp:* {{unnamed type '{{.*}}' is invalid; provide a kernel name, or use '-fsycl-unnamed-lambda' to enable unnamed kernel lambdas}}
    // expected-note@+1 {{in instantiation of function template}}
    cgh.parallel_for_work_group(sycl::range<1>{1}, [=](sycl::group<1>) {});
  });
}
