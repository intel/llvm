// RUN: %clangxx -fsycl-device-only -S -Xclang -emit-llvm -Xclang -disable-llvm-passes %s -o - | FileCheck %s --check-prefix CHECK-IR

#include <sycl/ext/oneapi/experimental/free_function_traits.hpp>
#include <sycl/sycl.hpp>

namespace syclexp = sycl::ext::oneapi::experimental;

// CHECK-IR-DAG: define dso_local spir_func noundef i32 @_Z15helper_restrictPKi(ptr addrspace(4) noalias
SYCL_EXTERNAL int helper_restrict(const int *__restrict__ input) {
  return input[0];
}

// CHECK-IR-DAG: define dso_local spir_func void @_Z25helper_const_ptr_restrictPiPKi(ptr addrspace(4) noalias{{[^,]*}}, ptr addrspace(4) noalias
SYCL_EXTERNAL void
helper_const_ptr_restrict(int *const __restrict__ output,
                          const int *const __restrict__ input) {
  output[0] = input[0];
}

// CHECK-IR: define dso_local spir_kernel void @_Z43__sycl_kernel_free_function_kernel_restrictPiPKiS_S_S1_PKv(ptr addrspace(1) noalias{{[^,]*}}, ptr addrspace(1) noalias{{[^,]*}}, ptr addrspace(1) {{[^,]*}}, ptr addrspace(1) noalias{{[^,]*}}, ptr addrspace(1) noalias{{[^,]*}}, ptr addrspace(1) noalias
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(syclexp::single_task_kernel)
void free_function_kernel_restrict(int *__restrict__ output,
                                   const int *__restrict__ input, int *plain,
                                   int *const __restrict__ output_const,
                                   const int *const __restrict__ input_const,
                                   const void *const __restrict__ opaque) {
  output_const[0] = plain[0];
  helper_const_ptr_restrict(output, input_const);
  output[0] += helper_restrict(input) + output_const[0] + (opaque != nullptr);
}
