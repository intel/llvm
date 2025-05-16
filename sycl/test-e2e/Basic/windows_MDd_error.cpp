// REQUIRES: windows

// RUN: %clangxx --driver-mode=cl -fsycl -o %t.exe %s /MDd
// RUN: %{run} %t.exe --expected-return-code=3221225477

#include <sycl/queue.hpp>

int main() {
  sycl::queue q;
}
