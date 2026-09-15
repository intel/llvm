// An object built with '-fsycl -fno-sycl-rdc -c' carries the finalized device
// image and registers it with the SYCL runtime on its own, so it does not need
// an offloading aware link step.  Check that by linking it without '-fsycl'.

// REQUIRES: target-spir
// The self-contained object relies on a partial link, which the MSVC linker
// does not provide, and on '-fno-sycl-rdc' being given at the compile step,
// which is an Old Offload Model thing.
// UNSUPPORTED: windows, new-offload-model

// RUN: %clangxx -fsycl -fno-sycl-rdc %fPIC -c %s -o %t.o
// RUN: %clangxx %fPIC %t.o -lsycl -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>
#include <sycl/usm.hpp>

#include <iostream>

int main() {
  sycl::queue Q;
  int *Result = sycl::malloc_shared<int>(1, Q);
  *Result = 0;
  Q.single_task<class self_contained>([=] { *Result = 42; }).wait();
  int Value = *Result;
  sycl::free(Result, Q);

  if (Value != 42) {
    std::cout << "FAILED: got " << Value << ", expected 42\n";
    return 1;
  }
  std::cout << "PASSED\n";
  return 0;
}
