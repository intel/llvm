// RUN: %clang_cc1 -internal-isystem %S/Inputs -fsycl-is-device -triple spir64-unknown-unknown -sycl-std=2020 -emit-llvm %s -o - | FileCheck %s --check-prefix CHECK-IR

#include "sycl.hpp"

// CHECK-IR-DAG: define dso_local spir_func noundef i32 @_Z15helper_restrictPKi(ptr addrspace(4) noalias
__attribute__((sycl_device)) int helper_restrict(const int *__restrict__ input) {
  return input[0];
}

// CHECK-IR-DAG: define dso_local spir_func void @_Z25helper_const_ptr_restrictPiPKi(ptr addrspace(4) noalias{{[^,]*}}, ptr addrspace(4) noalias
__attribute__((sycl_device)) void
helper_const_ptr_restrict(int *const __restrict__ output,
                          const int *const __restrict__ input) {
  output[0] = input[0];
}

// CHECK-IR: define dso_local spir_kernel void @_Z43__sycl_kernel_free_function_kernel_restrictPiPKiS_S_S1_PKv(ptr addrspace(1) noalias{{[^,]*}}, ptr addrspace(1) noalias{{[^,]*}}, ptr addrspace(1) {{[^,]*}}, ptr addrspace(1) noalias{{[^,]*}}, ptr addrspace(1) noalias{{[^,]*}}, ptr addrspace(1) noalias
[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel", 0)]]
void free_function_kernel_restrict(int *__restrict__ output,
                                   const int *__restrict__ input, int *plain,
                                   int *const __restrict__ output_const,
                                   const int *const __restrict__ input_const,
                                   const void *const __restrict__ opaque) {
  output_const[0] = plain[0];
  helper_const_ptr_restrict(output, input_const);
  output[0] += helper_restrict(input) + output_const[0] + (opaque != nullptr);
}
