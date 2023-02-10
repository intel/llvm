// RUN: %clangxx -fsycl -fsyntax-only -sycl-std=2020 %s

// Regression test for ensuring that identityless reductions with accessors
// compile without error.

#include <sycl/sycl.hpp>

int main() {
  auto IdentitylessPlus = [](int a, int b) { return a + b; };

  sycl::queue Q;
  sycl::buffer<int, 1> ReduBuff{1};
  Q.submit([&](sycl::handler &CGH) {
     auto Reduction = sycl::reduction(ReduBuff, CGH, IdentitylessPlus);
     CGH.parallel_for(sycl::range<1>(10), Reduction,
                      [=](sycl::id<1>, auto &Redu) { Redu.combine(1); });
   }).wait();

  return 0;
}
