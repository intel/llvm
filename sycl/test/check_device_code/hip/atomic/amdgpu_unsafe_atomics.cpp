// REQUIRES: hip
// RUN: %clangxx -fsycl -fsycl-targets=amd_gpu_gfx906 %s -S -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK,CHECK-SAFE
// RUN: %clangxx -fsycl -fsycl-targets=amd_gpu_gfx906 %s -mllvm --amdgpu-oclc-unsafe-int-atomics=true -S -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK,CHECK-UNSAFE

#include <sycl/sycl.hpp>

int main() {
  sycl::queue{}.single_task([=] {
    int a;
    sycl::atomic_ref<int, sycl::memory_order_relaxed, sycl::memory_scope_device>
        atomicInt(a);
    atomicInt.fetch_xor(1);
    atomicInt.fetch_and(1);
    atomicInt.fetch_or(1);
    // CHECK: __CLANG_OFFLOAD_BUNDLE____START__ sycl-amdgcn-amd-amdhsa-
    // CHECK-SAFE: cmpxchg volatile
    // CHECK-SAFE-NOT: atomicrmw
    // CHECK-UNSAFE: atomicrmw volatile xor
    // CHECK-UNSAFE: atomicrmw volatile and
    // CHECK-UNSAFE: atomicrmw volatile or
    // CHECK-UNSAFE-NOT: cmpxchg
    // CHECK: __CLANG_OFFLOAD_BUNDLE____END__ sycl-amdgcn-amd-amdhsa-
  });
}
