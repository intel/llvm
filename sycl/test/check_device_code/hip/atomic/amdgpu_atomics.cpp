// REQUIRES: hip
// RUN: %clangxx -fsycl -fsycl-targets=amdgcn-amd-amdhsa %s -S -emit-llvm -o - | FileCheck %s

#include <sycl/sycl.hpp>

SYCL_EXTERNAL void intAtomicFunc(int *i) {
  sycl::atomic_ref<int, sycl::memory_order_relaxed, sycl::memory_scope_device>
      atomicInt(*i);
  atomicInt.fetch_xor(1);
  atomicInt.fetch_and(1);
  atomicInt.fetch_or(1);
  // CHECK: void{{.*}}intAtomicFunc
  // CHECK: atomicrmw xor
  // CHECK: atomicrmw and
  // CHECK: atomicrmw or
}

SYCL_EXTERNAL void fpAtomicFunc(float *f, double *d) {
  sycl::atomic_ref<float, sycl::memory_order_relaxed, sycl::memory_scope_device,
                   sycl::access::address_space::global_space>(*f)
      .fetch_add(1.0f);
  sycl::atomic_ref<double, sycl::memory_order_relaxed,
                   sycl::memory_scope_device,
                   sycl::access::address_space::global_space>(*d)
      .fetch_add(1.0);
  // CHECK: void{{.*}}fpAtomicFunc
  // CHECK: atomicrmw fadd
  // CHECK: atomicrmw fadd
}
