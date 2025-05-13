// REQUIRES: windows

// RUN: not clang-cl -fsycl -o %t.exe %s /Od /MDd /Zi /EHsc 2>&1 | FileCheck %s

// FIXME: This code should have compiled cleanly.
// CHECK: error: SYCL kernel cannot call an undefined function without SYCL_EXTERNAL attribute
// CHECK: note: '_invalid_parameter' declared here

#include <sycl/queue.hpp>

int main() {
  sycl::queue q;

  q.single_task([=]() {
    std::array<int, 5> arr = {1, 2, 0, 4, 5};
    arr[2] = 3;
  });

  return 0;
}
