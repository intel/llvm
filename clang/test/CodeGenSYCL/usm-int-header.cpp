// RUN: %clang_cc1 -fsycl-is-device -ast-dump %s | FileCheck %s
// RUN: %clang_cc1 -fsycl-is-device -fsycl-int-header=%t.h %s -o %t.out
// RUN: FileCheck -input-file=%t.h %s --check-prefix=INT-HEADER

// INT-HEADER:{ kernel_param_kind_t::kind_pointer, 8, 0 },
// INT-HEADER:{ kernel_param_kind_t::kind_pointer, 8, 8 },

//==--usm-int-header.cpp - USM kernel param aspace and int header test -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Inputs/sycl.hpp"

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel(const Func &kernelFunc) {
  kernelFunc();
}

int main() {
  int* x;
  float* y;
  
  kernel<class usm_test>([=]() {
      *x = 42;
      *y = 3.14;
    });
}

// CHECK: FunctionDecl {{.*}}usm_test 'void (__global int *, __global float *)'

// TODO: SYCL specific fail - analyze and enable
// XFAIL: windows-msvc
