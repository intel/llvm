// REQUIRES: hip
// RUN: %clangxx -fsycl -fsycl-device-only -fsycl-targets=amdgcn-amd-amdhsa %s -S -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK,CHECK-SAFE
// RUN: %clangxx -fsycl -fsycl-device-only -fsycl-targets=amdgcn-amd-amdhsa %s -mllvm --amdgpu-oclc-unsafe-int-atomics=true -S -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK,CHECK-UNSAFE

#include <sycl/sycl.hpp>

SYCL_EXTERNAL void intAtomicFunc(int *i) {
  sycl::atomic_ref<int, sycl::memory_order_relaxed, sycl::memory_scope_device>
      atomicInt(*i);
  atomicInt.fetch_xor(1);
  atomicInt.fetch_and(1);
  atomicInt.fetch_or(1);
  // CHECK: void{{.*}}intAtomicFunc
  // CHECK-SAFE: cmpxchg
  // CHECK-SAFE-NOT: atomicrmw
  // CHECK-UNSAFE: atomicrmw xor
  // CHECK-UNSAFE: atomicrmw and
  // CHECK-UNSAFE: atomicrmw or
  // CHECK-UNSAFE-NOT: cmpxchg
}
