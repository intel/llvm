// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -fsycl-int-header=%t.h %s -o %t.out
// RUN: FileCheck -input-file=%t.h %s

// This test checks the integration header generated when:
// 1. Kernel argument is a pointer.
// 2. Kernel argument is a struct containing a pointer.

#include "Inputs/sycl.hpp"

struct decomposed_struct_with_pointer {
  int data_in_struct;
  int *ptr_in_struct;
  int *ptr_array_in_struct1[2];
  int *ptr_array_in_struct2[2][3];
  sycl::accessor<char, 1, sycl::access::mode::read> acc;
};

struct non_decomposed_struct_with_pointer {
  int data_in_struct;
  int *ptr_in_struct;
  int *ptr_array_in_struct1[2];
  int *ptr_array_in_struct2[2][3];
};

int main() {
  int *ptr;
  decomposed_struct_with_pointer obj1;
  non_decomposed_struct_with_pointer obj2;
  obj1.data_in_struct = 10;
  obj2.data_in_struct = 10;

  sycl::kernel_single_task<class test>([=]() {
    *ptr = 50;
    int local = obj1.data_in_struct + obj2.data_in_struct;
  });
}

// Integration header entries for pointer, scalar and wrapped pointer.
// CHECK:{ kernel_param_kind_t::kind_pointer, 8, 0 },
// CHECK:{ kernel_param_kind_t::kind_std_layout, 4, 8 },
// CHECK:{ kernel_param_kind_t::kind_std_layout, 8, 16 },
// CHECK:{ kernel_param_kind_t::kind_std_layout, 16, 24 },
// CHECK:{ kernel_param_kind_t::kind_std_layout, 48, 40 },
// CHECK:{ kernel_param_kind_t::kind_accessor, 4062, 88 },
// CHECK:{ kernel_param_kind_t::kind_std_layout, 80, 104 },
