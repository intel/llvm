// RUN: %clang_cc1 -fsycl -fsycl-is-device -I %S/Inputs -triple spir64-unknown-unknown-sycldevice -fsycl-int-header=%t.h %s -o %t.out
// RUN: FileCheck -input-file=%t.h %s

// This test checks the integration header generated when:
// 1. Kernel argument is a pointer.
// 2. Kernel argument is a struct containing a pointer.

#include <sycl.hpp>

struct struct_with_pointer {
  int data_in_struct;
  int *ptr_in_struct;
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
