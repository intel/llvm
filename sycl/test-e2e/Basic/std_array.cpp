// Check that std::array is supported on device in debug mode on Windows.

// REQUIRES: windows

// RUN: %clangxx --driver-mode=cl -fsycl -o %t.exe %s /Od /MDd /Zi /EHsc
// RUN: %{run} %t.exe

#include <sycl/queue.hpp>

int main() {
  sycl::queue q;

  q.single_task([=]() {
    int tmp = 5;
    tmp += 5;
    // std::array<int, 5> arr;// = {1, 2, 0, 4, 5};
	//int tmp = arr[0];
    // arr[2] = 3;
  });

  return 0;
}
