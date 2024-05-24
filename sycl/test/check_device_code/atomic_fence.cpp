// RUN: %clangxx -fsycl-device-only -fsycl-unnamed-lambda -S -Xclang -emit-llvm %s -o - | FileCheck %s

#include <sycl/sycl.hpp>

SYCL_EXTERNAL void atomic_fence() {
  // CHECK: tail call spir_func void @_Z21__spirv_MemoryBarrierjj(i32 noundef 2, i32 noundef 896) #{{.*}}
  sycl::atomic_fence(sycl::memory_order::relaxed,
                     sycl::memory_scope::work_group);
  // CHECK: tail call spir_func void @_Z21__spirv_MemoryBarrierjj(i32 noundef 2, i32 noundef 898) #{{.*}}
  sycl::atomic_fence(sycl::memory_order::acquire,
                     sycl::memory_scope::work_group);
  // CHECK: tail call spir_func void @_Z21__spirv_MemoryBarrierjj(i32 noundef 2, i32 noundef 900) #{{.*}}
  sycl::atomic_fence(sycl::memory_order::release,
                     sycl::memory_scope::work_group);
  // CHECK: tail call spir_func void @_Z21__spirv_MemoryBarrierjj(i32 noundef 2, i32 noundef 904) #{{.*}}
  sycl::atomic_fence(sycl::memory_order::acq_rel,
                     sycl::memory_scope::work_group);
  // CHECK: tail call spir_func void @_Z21__spirv_MemoryBarrierjj(i32 noundef 2, i32 noundef 912) #{{.*}}
  sycl::atomic_fence(sycl::memory_order::seq_cst,
                     sycl::memory_scope::work_group);
}