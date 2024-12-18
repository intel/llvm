// RUN: %clangxx -fsycl -fsycl-device-only -S -emit-llvm %s -o - | FileCheck %s

// The test checks that multiple calls to the same template instantiation of
// syclcompat local_mem function result in separate allocations.

// CHECK: @WGLocalMem{{.*}} = internal addrspace(3) global [4 x i8] undef, align 4
// CHECK-NEXT: @WGLocalMem{{.*}} = internal addrspace(3) global [4 x i8] undef, align 4

#include <sycl/detail/core.hpp>
#include <syclcompat/memory.hpp>

using namespace sycl;

int main() {
  queue Q;

  int **Out = malloc_shared<int *>(4, Q);

  Q.submit([&](handler &Cgh) {
    Cgh.parallel_for(nd_range<1>({1}, {1}), [=](nd_item<1> Item) {
      auto Ptr0 = syclcompat::local_mem<int[1]>();
      auto Ptr1 = syclcompat::local_mem<int[1]>();
      Out[0] = Ptr0;
      Out[1] = Ptr1;
    });
  });
}
