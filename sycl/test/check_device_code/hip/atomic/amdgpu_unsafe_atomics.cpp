// REQUIRES: hip
// RUN: %clangxx -fsycl -fsycl-targets=amd_gpu_gfx906 %s -S -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK,CHECK-SAFE
// RUN: %clangxx -fsycl -fsycl-targets=amd_gpu_gfx906 %s -mllvm --amdgpu-oclc-unsafe-int-atomics=true -S -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK,CHECK-UNSAFE
// RUN: %clangxx -fsycl -fsycl-targets=amd_gpu_gfx90a %s -mllvm --amdgpu-oclc-unsafe-fp-atomics=true  -mllvm --amdgpu-oclc-unsafe-int-atomics=true -S -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK,CHECK-UNSAFE-FP

#include <sycl/sycl.hpp>

SYCL_EXTERNAL void intAtomicFunc(int *i) {
  sycl::atomic_ref<int, sycl::memory_order_relaxed, sycl::memory_scope_device>
      atomicInt(*i);
  atomicInt.fetch_xor(1);
  atomicInt.fetch_and(1);
  atomicInt.fetch_or(1);
  // CHECK: void{{.*}}intAtomicFunc
  // CHECK-SAFE: cmpxchg volatile
  // CHECK-SAFE-NOT: atomicrmw
  // CHECK-UNSAFE: atomicrmw volatile xor
  // CHECK-UNSAFE: atomicrmw volatile and
  // CHECK-UNSAFE: atomicrmw volatile or
  // CHECK-UNSAFE-NOT: cmpxchg
}

SYCL_EXTERNAL void fpAtomicFunc(float *f, double *d) {
  sycl::atomic_ref<float, sycl::memory_order_relaxed, sycl::memory_scope_device,
                   sycl::access::address_space::global_space>(*f)
      .fetch_add(1.0f);
  // CHECK: void{{.*}}fpAtomicFunc
  // CHECK-SAFE: atomicrmw volatile fadd
  // CHECK-SAFE-NOT: llvm.amdgcn.global.atomic.fadd.f32
  // CHECK-UNSAFE-FP: llvm.amdgcn.global.atomic.fadd.f32
  // CHECK-UNSAFE-FP-NOT: atomicrmw volatile fadd
  sycl::atomic_ref<double, sycl::memory_order_relaxed,
                   sycl::memory_scope_device,
                   sycl::access::address_space::global_space>(*d)
      .fetch_add(1.0);
  // CHECK-SAFE: cmpxchg
  // CHECK-SAFE-NOT: llvm.amdgcn.global.atomic.fadd.f64
  // CHECK-UNSAFE-FP: llvm.amdgcn.global.atomic.fadd.f64
  // CHECK-UNSAFE-FP-NOT: cmpxchg
  // CHECK: __CLANG_OFFLOAD_BUNDLE____END__ sycl-amdgcn-amd-amdhsa-
}