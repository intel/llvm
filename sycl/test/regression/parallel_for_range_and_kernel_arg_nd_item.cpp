// RUN: not %clangxx -fsycl -fsyntax-only %s 2>&1 | FileCheck %s

#include <sycl/sycl.hpp>

using namespace sycl;

int main() {
  queue q;
  q.submit([&](handler &cgh) {
    // CHECK: Kernel argument cannot have a sycl::nd_item type in sycl::parallel_for with sycl::range
    cgh.parallel_for<class MyKernel>(43, [=](nd_item<1> item) {});
  });
}
