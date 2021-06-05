// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown-sycldevice -fsycl-int-header=%t.h %s -o %t.out
// RUN: FileCheck -input-file=%t.h %s

// This test checks the integration header generated when:
// 1. Kernel argument is a pointer.
// 2. Kernel argument is a struct containing a pointer.

#include "Inputs/sycl.hpp"

struct struct_with_pointer {
  int data_in_struct;
  int *ptr_in_struct;
  int *ptr_array_in_struct1[2];
  int *ptr_array_in_struct2[2][3];
};

int main() {
  int *ptr;
  struct_with_pointer obj;
  obj.data_in_struct = 10;

  cl::sycl::kernel_single_task<class test>([=]() {
    *ptr = 50;
    int local = obj.data_in_struct;
  });
}

// Integration header entries for pointer, scalar and wrapped pointer.
// CHECK:{ kernel_param_kind_t::kind_pointer, 8, 0 },
// CHECK:{ kernel_param_kind_t::kind_std_layout, 4, 8 },
// CHECK:{ kernel_param_kind_t::kind_std_layout, 8, 16 },
// CHECK:{ kernel_param_kind_t::kind_std_layout, 8, 24 },
// CHECK:{ kernel_param_kind_t::kind_std_layout, 8, 32 },
// CHECK:{ kernel_param_kind_t::kind_std_layout, 8, 40 },
// CHECK:{ kernel_param_kind_t::kind_std_layout, 8, 48 },
// CHECK:{ kernel_param_kind_t::kind_std_layout, 8, 56 },
// CHECK:{ kernel_param_kind_t::kind_std_layout, 8, 64 },
// CHECK:{ kernel_param_kind_t::kind_std_layout, 8, 72 },
// CHECK:{ kernel_param_kind_t::kind_std_layout, 8, 80 },
