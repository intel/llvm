// RUN: %clang_cc1 -fsycl -fsycl-is-device -fsycl-int-header=%t.h %s
// RUN: FileCheck -input-file=%t.h %s

//==--pointer-int-header.cpp - USM kernel param aspace and int header test --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel(Func kernelFunc) {
  kernelFunc();
}

struct struct_with_pointer {
  int data_in_struct;
  int *ptr_in_struct;
};

int main() {
  int *ptr;
  struct_with_pointer obj;
  obj.data_in_struct = 10;

  kernel<class test>([=]() {
    *ptr = 50;
    int local = obj.data_in_struct;
  });
}

// Integration header entries for pointer, scalar and wrapped pointer.
// CHECK:{ kernel_param_kind_t::kind_pointer, 8, 0 },
// CHECK:{ kernel_param_kind_t::kind_std_layout, 4, 8 },
// CHECK:{ kernel_param_kind_t::kind_std_layout, 8, 16 },
