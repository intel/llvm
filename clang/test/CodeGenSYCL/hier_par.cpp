//==- hier_par.cpp --- hierarchical parallelism regression tests -----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: %clangxx -O2 -I %S/Inputs -fsycl -fsycl-device-only -c -Xclang -emit-llvm -o %t.ll %s
// RUN: cat %t.ll | FileCheck %s

// This test checks for bug fix regressions related to hierarchical parallelism.
// - bug1: private var's (cl::sycl::group argument) address shared locally
//   the test checks that a "shadow" local variable is generated for the group
//   argument
//
// This is compile-only test for now.
//
// XFAIL:* 
#include "sycl.hpp"

using namespace cl::sycl;

void foo() {
  int *ptr = nullptr;

  queue myQueue;
  buffer<int, 1> buf(ptr, range<1>(1));

  myQueue.submit([&](handler &cgh) {
    auto dev_ptr = buf.get_access<access::mode::read_write>(cgh);

    cgh.parallel_for_work_group<class hpar_hw>(
        range<1>(1), range<1>(1), [=](group<1> g) {
// CHECK: @[[SHADOW:[a-zA-Z0-9]+]] = internal unnamed_addr addrspace(3) global %[[GROUP_CLASS:"[^"]+"]] undef, align [[ALIGN:[0-9]+]]
// CHECK: define {{.*}} spir_func void @{{"[^"]+"}}({{[^,]+}}, %[[GROUP_CLASS]]* byval(%[[GROUP_CLASS]]) align {{[0-9]+}} %[[GROUP_OBJ:[A-Za-z_0-9]+]]) {{.*}}!work_group_scope{{.*}} {
// CHECK-NOT: {{^[ \t]*define}}
// CHECK: %[[TMP:[A-Za-z_0-9]+]] = bitcast %[[GROUP_CLASS]] addrspace(3)* @[[SHADOW]] to i8 addrspace(3)*
// CHECK:  %[[OBJ:[A-Za-z_0-9]+]] = bitcast %[[GROUP_CLASS]]* %[[GROUP_OBJ]] to i8*
// CHECK:  call void @llvm.memcpy.p3i8.p0i8.i64(i8 addrspace(3)* align [[ALIGN]] %[[TMP]], {{[^,]+}} %[[OBJ]], {{[^)]+}})
        });
  });
}
