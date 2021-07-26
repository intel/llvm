// RUN: %clangxx -fsycl-device-only -fsycl-unnamed-lambda -S -Xclang -emit-llvm %s -o - | FileCheck %s

#include <sycl/sycl.hpp>

int main() {
  sycl::queue Q;

  Q.single_task([] {
    // CHECK: tail call spir_func void @_Z21__spirv_MemoryBarrierjj(i32 2, i32 896) #2
    sycl::atomic_fence(sycl::memory_order::relaxed,
                       sycl::memory_scope::work_group);
    // CHECK: tail call spir_func void @_Z21__spirv_MemoryBarrierjj(i32 2, i32 898) #2
    sycl::atomic_fence(sycl::memory_order::acquire,
                       sycl::memory_scope::work_group);
    // CHECK: tail call spir_func void @_Z21__spirv_MemoryBarrierjj(i32 2, i32 900) #2
    sycl::atomic_fence(sycl::memory_order::release,
                       sycl::memory_scope::work_group);
    // CHECK: tail call spir_func void @_Z21__spirv_MemoryBarrierjj(i32 2, i32 904) #2
    sycl::atomic_fence(sycl::memory_order::acq_rel,
                       sycl::memory_scope::work_group);
    // CHECK: tail call spir_func void @_Z21__spirv_MemoryBarrierjj(i32 2, i32 912) #2
    sycl::atomic_fence(sycl::memory_order::seq_cst,
                       sycl::memory_scope::work_group);
  });
  Q.wait();

  return 0;
}
