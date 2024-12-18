// RUN: %clangxx -fsycl -fsycl-device-only -S -emit-llvm %s -o - | FileCheck %s
// RUN: %clangxx -fsycl -fsycl-device-only -S -emit-llvm %s -fno-sycl-early-optimizations -o - | FileCheck %s
// RUN: %clangxx -fsycl -fsycl-device-only -S -emit-llvm %s -O0 -o - | FileCheck %s

// The test checks that multiple calls to the same template instantiation of a
// group local memory function result in separate allocations.

// CHECK: @WGLocalMem{{.*}} = internal addrspace(3) global [4 x i8] undef, align 4
// CHECK-NEXT: @WGLocalMem{{.*}} = internal addrspace(3) global [4 x i8] undef, align 4
// CHECK-NEXT: @WGLocalMem{{.*}} = internal addrspace(3) global [4 x i8] undef, align 4
// CHECK-NEXT: @WGLocalMem{{.*}} = internal addrspace(3) global [4 x i8] undef, align 4

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/group_local_memory.hpp>
#include <sycl/usm.hpp>

using namespace sycl;

int main() {
  queue Q;

  int **Out = malloc_shared<int *>(4, Q);

  Q.submit([&](handler &Cgh) {
    Cgh.parallel_for(nd_range<1>({1}, {1}), [=](nd_item<1> Item) {
      auto Ptr0 =
          ext::oneapi::group_local_memory_for_overwrite<int>(Item.get_group());
      auto Ptr1 =
          ext::oneapi::group_local_memory_for_overwrite<int>(Item.get_group());
      auto Ptr2 = ext::oneapi::group_local_memory<int>(Item.get_group());
      auto Ptr3 = ext::oneapi::group_local_memory<int>(Item.get_group());
      Out[0] = Ptr0;
      Out[1] = Ptr1;
      Out[2] = Ptr2;
      Out[3] = Ptr3;
    });
  });
}
