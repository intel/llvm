// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown-sycldevice -disable-llvm-passes -S -emit-llvm %s -o - | FileCheck --check-prefixes=CHECK,CHECK-UNOPT %s
// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown-sycldevice -S -emit-llvm %s -o - | FileCheck %s

// CHECK: [[WGLOCALMEM_1:@WGLocalMem.*]] = internal addrspace(3) global [8 x i8] undef, align 8
// CHECK: [[WGLOCALMEM_2:@WGLocalMem.*]] = internal addrspace(3) global [4 x i8] undef, align 4
// CHECK: [[WGLOCALMEM_3:@WGLocalMem.*]] = internal addrspace(3) global [128 x i8] undef, align 4

// CHECK-NOT: __sycl_allocateLocalMemory

#include "Inputs/sycl.hpp"

constexpr size_t WgSize = 32;
constexpr size_t WgCount = 4;
constexpr size_t Size = WgSize * WgCount;

class KernelA;
class KernelB;

using namespace cl::sycl;

int main() {
  queue Q;
  {
    Q.submit([&](handler &cgh) {
      cgh.parallel_for<KernelA>(
          range<1>(Size), [=](item<1> Item) {
            auto *Ptr1 = group_local_memory<long>();
            // CHECK-UNOPT: i8 addrspace(3)* getelementptr inbounds ([8 x i8], [8 x i8] addrspace(3)* [[WGLOCALMEM_1]], i32 0, i32 0)
            auto *Ptr2 = group_local_memory<float>();
            // CHECK-UNOPT: i8 addrspace(3)* getelementptr inbounds ([4 x i8], [4 x i8] addrspace(3)* [[WGLOCALMEM_2]], i32 0, i32 0)
          });
    });
  }

  {
    Q.submit([&](handler &cgh) {
      cgh.parallel_for<KernelB>(
          range<1>(Size), [=](item<1> Item) {
            auto *Ptr3 = group_local_memory<int[WgSize]>();
            // CHECK-UNOPT: i8 addrspace(3)* getelementptr inbounds ([128 x i8], [128 x i8] addrspace(3)* [[WGLOCALMEM_3]], i32 0, i32 0)
          });
    });
  }
}
