// REQUIRES: windows

// RUN: clang-cl -fsycl -o %t.exe %s /Od /MDd /Zi /EHsc 2>&1 | FileCheck --allow-empty %s
 
// Check that std::array in device code does not result in compilation errors:

// CHECK-NOT: error: SYCL kernel cannot call an undefined function without SYCL_EXTERNAL attribute
// CHECK-NOT: note: '_invalid_parameter' declared here
// CHECK-NOT: Undefined function _invalid_parameter found in

// RUN: ${run} %t.exe

#include <sycl/queue.hpp>

int main() {
  sycl::queue q;

  q.single_task([=]() {
    std::array<int, 5> arr = {1, 2, 0, 4, 5};
    arr[2] = 3;
  });

  return 0;
}
