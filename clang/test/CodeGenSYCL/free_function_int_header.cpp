// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple spir64-unknown-unknown -sycl-std=2020 -fsycl-int-header=%t.h %s -emit-llvm -o %t.ll
// RUN: FileCheck -input-file=%t.h %s
// 
// This test checks integration header contents for free functions with scalar,
// pointer, simple struct parameters.

#include "mock_properties.hpp"
#include "sycl.hpp"

__attribute__((sycl_device))
[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel", 2)]]
void ff_2(int *ptr, int start, int end) {
  for (int i = start; i <= end; i++)
    ptr[i] = start + 66;
}
// CHECK:      const char* const kernel_names[] = {
// CHECK-NEXT:   "_Z18__sycl_kernel_ff_2Piii"
// CHECK-NEXT: };


// CHECK:      const kernel_param_desc_t kernel_signatures[] = {
// CHECK-NEXT:   //--- _Z18__sycl_kernel_ff_2Piii
// CHECK-NEXT:   { kernel_param_kind_t::kind_pointer, 8, 0 },
// CHECK-NEXT:   { kernel_param_kind_t::kind_std_layout, 4, 8 },
// CHECK-NEXT:   { kernel_param_kind_t::kind_std_layout, 4, 12 },

// CHECK:        { kernel_param_kind_t::kind_invalid, -987654321, -987654321 },
// CHECK-NEXT: };
