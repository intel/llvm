// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs %s -verify

// Tests that an error diagnostic is issued instead of a crash (via an
// assertion) when some invalid cases are encountered in the processing
// of kernel_parallel_for_work_group.

#include "sycl.hpp"

int main() {
  sycl::queue queue;
  auto separate_lambda_item_arg = [](sycl::item<1>) {};
  queue.submit([&](sycl::handler &cgh) {
    //expected-error@#KernelPFWG2{{unable to find lambda or function object in the kernel parameter; perhaps it was invoked with the wrong signature?}}
    //expected-note@+1{{in instantiation of function template specialization}}
    cgh.parallel_for_work_group<class kernel>(
      sycl::range<1>{}, sycl::range<1>{}, separate_lambda_item_arg);
  });
  return 0;
}
