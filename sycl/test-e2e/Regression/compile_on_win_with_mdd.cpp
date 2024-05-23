// REQUIRES: windows

// RUN: %clangxx --driver-mode=cl -fsycl /MDd -c %s -o %t.obj
// RUN: %clangxx --driver-mode=cl -fsycl %t.obj -o %t.out
// RUN: %{run} %t.out

// The test aims to prevent regressions similar to the one which caused by
// https://github.com/intel/llvm/pull/12793.
// The failure happens if perform separate compile and link, and pass /MDd to
// the compile line. In that case, user application will crash during launching
// with abort() message.

#include <sycl/queue.hpp>

#include <iostream>

int main() {
  sycl::queue q;
  std::cout << "passed" << std::endl;
  return 0;
}
