// REQUIRES: windows

// RUN: %clang_cl -fsycl /EHsc /MDd -QMT %t.out /Fo%t.obj -c %s
// RUN: %clang_cl -fsycl %t.obj -o %t.out
// RUN: %t.out

// The test aims to prevent regressions similar to one which caused by
// https://github.com/intel/llvm/pull/12793.
// If regression appears, the test should fail with abort() message during
// application run.

#include <sycl/sycl.hpp>

#include <iostream>

int main() {
  sycl::queue q;
  std::cout << "passed" << std::endl;
  return 0;
}
