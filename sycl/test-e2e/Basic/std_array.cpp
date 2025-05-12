// Check that std::array in device code does not result in compilation errors

// REQUIRES: windows

// RUN: %clangxx --driver-mode=cl -fsycl -o %t.exe %s /Od /MDd /Zi /EHsc
// RUN: %{run} %t.exe

// Should not be producing errors such as:
// - error: SYCL kernel cannot call an undefined function without SYCL_EXTERNAL attribute
//   note: '_invalid_parameter' declared here
// - Undefined function _invalid_parameter found in

#include <sycl/queue.hpp>

int main() {
  sycl::queue q;

  q.single_task([=]() {
    std::array<int, 5> arr = {1, 2, 0, 4, 5};
    arr[2] = 3;
  });

  return 0;
}
