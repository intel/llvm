// REQUIRES: linux
// RUN: %clangxx -fsycl-device-only -fno-sycl-early-optimizations -Xarch_device -fsanitize=address -emit-llvm %s -S -o %t.ll
// RUN: FileCheck %s --input-file %t.ll

// Check asan ctor is not generated since device asan initialization is done in
// unified runtime rather than in ctor.

// CHECK-NOT: asan.module_ctor

#include <sycl/sycl.hpp>

using namespace sycl;

int main() {
  sycl::queue queue;
  {
    queue.submit(
        [&](sycl::handler &cgh) { cgh.single_task<class t>([=]() {}); });
    queue.wait();
  }

  return 0;
}
