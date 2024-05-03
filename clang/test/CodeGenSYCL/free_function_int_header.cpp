// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple spir64-unknown-unknown -sycl-std=2020 -fsycl-int-header=%t.h %s
// RUN: FileCheck -input-file=%t.h %s
// 
// This test checks integration header contents for free functions with scalar
// and pointer parameters.

#include "mock_properties.hpp"
#include "sycl.hpp"

__attribute__((sycl_device))
[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel",
                                              2)]] void
ff_2(int *ptr, int start, int end) {
  for (int i = start; i <= end; i++)
    ptr[i] = start + 66;
}

template <typename T>
__attribute__((sycl_device))
[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 2)]] void
ff_3(T *ptr, T start, int end) {
  for (int i = start; i <= end; i++)
    ptr[i] = start;
}

// Explicit instantiation with "int*"
template void ff_3(int *ptr, int start, int end);

// CHECK:      const char* const kernel_names[] = {
// CHECK-NEXT:   {{.*}}__sycl_kernel_ff_2{{.*}}
// CHECK-NEXT:   {{.*}}__sycl_kernel_ff_3{{.*}}
// C HECK-NEXT:   "_Z18__sycl_kernel_ff_2Piii",
// C HECK-NEXT:   "_Z18__sycl_kernel_ff_3Piii"
// CHECK-NEXT: };

// CHECK:      const kernel_param_desc_t kernel_signatures[] = {
// CHECK-NEXT:   {{.*}}__sycl_kernel_ff_2{{.*}}
// CHECK-NEXT:   { kernel_param_kind_t::kind_pointer, 8, 0 },
// CHECK-NEXT:   { kernel_param_kind_t::kind_std_layout, 4, 8 },
// CHECK-NEXT:   { kernel_param_kind_t::kind_std_layout, 4, 12 },

// CHECK:        {{.*}}__sycl_kernel_ff_3{{.*}}
// CHECK-NEXT:   { kernel_param_kind_t::kind_pointer, 8, 0 },
// CHECK-NEXT:   { kernel_param_kind_t::kind_std_layout, 4, 8 },
// CHECK-NEXT:   { kernel_param_kind_t::kind_std_layout, 4, 12 },

// CHECK:        { kernel_param_kind_t::kind_invalid, -987654321, -987654321 },
// CHECK-NEXT: };