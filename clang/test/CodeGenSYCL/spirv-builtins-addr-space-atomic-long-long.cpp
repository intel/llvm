// RUN: %clang_cc1 %s -triple spir64 -fsycl-is-device -internal-isystem %S/Inputs -fdeclare-spirv-builtins -emit-llvm -o - -O0 | FileCheck %s

#include "Inputs/sycl.hpp"

void atomic_load_long_long_global() {
  __attribute__((opencl_global)) long long *ptr = nullptr;
  sycl::kernel_single_task<class atomic_load_long_long_global>(
      [=]() { (void)__spirv_AtomicLoad(ptr, 1, 16); });
}

void atomic_load_long_long_local() {
  __attribute__((opencl_local)) long long *ptr = nullptr;
  sycl::kernel_single_task<class atomic_load_long_long_local>(
      [=]() { (void)__spirv_AtomicLoad(ptr, 2, 8); });
}

void atomic_load_long_long_private() {
  __attribute__((opencl_private)) long long *ptr = nullptr;
  sycl::kernel_single_task<class atomic_load_long_long_private>(
      [=]() { (void)__spirv_AtomicLoad(ptr, 4, 4); });
}

void atomic_umin_unsigned_long_long_global() {
  __attribute__((opencl_global)) unsigned long long *ptr = nullptr;
  sycl::kernel_single_task<class atomic_umin_unsigned_long_long_global>(
      [=]() { (void)__spirv_AtomicUMin(ptr, 1, 16, 0ULL); });
}

void atomic_umin_unsigned_long_long_local() {
  __attribute__((opencl_local)) unsigned long long *ptr = nullptr;
  sycl::kernel_single_task<class atomic_umin_unsigned_long_long_local>(
      [=]() { (void)__spirv_AtomicUMin(ptr, 2, 8, 0ULL); });
}

void atomic_umin_unsigned_long_long_private() {
  __attribute__((opencl_private)) unsigned long long *ptr = nullptr;
  sycl::kernel_single_task<class atomic_umin_unsigned_long_long_private>(
      [=]() { (void)__spirv_AtomicUMin(ptr, 4, 4, 0ULL); });
}

// CHECK: __spirv_AtomicLoad{{.*}}AS1xii
// CHECK: __spirv_AtomicLoad{{.*}}AS3xii
// CHECK: __spirv_AtomicLoad{{.*}}AS0xii
// CHECK: __spirv_AtomicUMin{{.*}}AS1yiiy
// CHECK: __spirv_AtomicUMin{{.*}}AS3yiiy
// CHECK: __spirv_AtomicUMin{{.*}}AS0yiiy