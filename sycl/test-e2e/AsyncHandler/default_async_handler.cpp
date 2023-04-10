// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out &> %t.txt || true
// RUN: %CPU_RUN_PLACEHOLDER FileCheck %s --input-file %t.txt
// RUN: %GPU_RUN_PLACEHOLDER %t.out &> %t.txt || true
// RUN: %GPU_RUN_PLACEHOLDER FileCheck %s --input-file %t.txt
// RUN: %ACC_RUN_PLACEHOLDER %t.out &> %t.txt || true
// RUN: %ACC_RUN_PLACEHOLDER FileCheck %s --input-file %t.txt

#include <sycl/sycl.hpp>

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
