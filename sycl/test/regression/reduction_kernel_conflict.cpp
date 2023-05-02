// RUN: %clangxx -fsycl -fsyntax-only -sycl-std=2020 %s

// Regression test to make sure a certain configuration of reductions do not
// cause conflicting kernels to be generated.

#include <sycl/sycl.hpp>

int main() {
  sycl::queue Q;
  double *SumVar = sycl::malloc_shared<double>(1, Q);
  Q.submit([&](sycl::handler &CGH) {
     auto SumReduction = sycl::reduction(SumVar, sycl::plus<>());
     CGH.parallel_for<class kernel>(
         sycl::nd_range<1>{10, 10}, SumReduction,
         [=](sycl::nd_item<1>, auto &Sum) { Sum += 1; });
   }).wait();
  return 0;
}
