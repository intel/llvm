// Check that std::array is supported on device in debug mode on Windows.

// REQUIRES: windows
// XFAIL: windows && run-mode && cuda
// XFAIL-TRACKER: https://github.com/intel/llvm/issues/23547

// RUN: %clangxx --driver-mode=cl -fsycl -o %t.exe %s /Od /MDd /Zi /EHsc
// RUN: %{run} %t.exe

#include <sycl/queue.hpp>

int main() {
  sycl::queue q;

  q.single_task([=]() {
    std::array<int, 5> arr = {1, 2, 0, 4, 5};
    arr[2] = 3;
  });

  return 0;
}
