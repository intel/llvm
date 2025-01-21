// RUN: %clangxx -fsycl -fsycl-device-only -std=c++17 -fno-sycl-unnamed-lambda -isystem %sycl_include/sycl -Xclang -verify -fsyntax-only %s -Xclang -verify-ignore-unexpected=note
// expected-no-diagnostics

// Tests that kernel functor objects are allowed through when unnamed lambdas
// are disabled.

#include <sycl/sycl.hpp>

struct SingleTaskKernel {
  void operator()() const {}
};

struct ParallelForKernel {
  void operator()(sycl::item<1> it) const {}
};

struct ParallelForWorkGroupKernel {
  void operator()(sycl::item<1> it) const {}
};

int main() {
  sycl::queue q;

  q.single_task(SingleTaskKernel{});

  q.parallel_for(sycl::range<1>{1}, ParallelForKernel{});

  q.submit([&](sycl::handler &cgh) {
    cgh.single_task(SingleTaskKernel{});
  });

  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::range<1>{1}, ParallelForKernel{});
  });

  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for_work_group(sycl::range<1>{1},
                                ParallelForWorkGroupKernel{});
  });
}
