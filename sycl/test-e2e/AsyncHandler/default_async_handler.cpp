// RUN: %{build} -o %t.out
// RUN: %{run} %t.out &> %t.txt ; FileCheck %s --input-file %t.txt

#include <sycl/detail/core.hpp>

using namespace sycl;

int main() {
  queue Q;
  Q.submit([&](handler &CGH) {
     CGH.host_task([=]() {
       throw std::runtime_error("Exception thrown from host_task.");
     });
   }).wait_and_throw();
  return 0;
}

// CHECK:      Default async_handler caught exceptions:
// CHECK-NEXT: Exception thrown from host_task.
