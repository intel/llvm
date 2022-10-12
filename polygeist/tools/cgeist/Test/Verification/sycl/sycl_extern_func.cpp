// Copyright (C) Intel

//===--- sycl_extern_func.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// clang-format off
// RUN: clang++ -fsycl -fsycl-device-only -emit-mlir %s 2> /dev/null | FileCheck %s --check-prefix=CHECK-MLIR
// RUN: clang++ -fsycl -fsycl-device-only -S -emit-llvm -fsycl-targets=spir64-unknown-unknown-syclmlir %s | FileCheck %s --check-prefix=CHECK-LLVM

#include <sycl/sycl.hpp>

// CHECK-MLIR: gpu.module @device_functions

// CHECK-MLIR-LABEL: gpu.func @_ZTSZZ16host_single_taskvENKUlRN4sycl3_V17handlerEE_clES2_E18kernel_single_task
// CHECK-MLIR-SAME: kernel attributes {llvm.cconv = #llvm.cconv<spir_kernelcc>, llvm.linkage = #llvm.linkage<weak_odr>, passthrough = ["norecurse", "nounwind", "convergent", "mustprogress"]} {

// CHECK-MLIR-LABEL: func.func private @_ZZZ16host_single_taskvENKUlRN4sycl3_V17handlerEE_clES2_ENKUlvE_clEv
// CHECK-MLIR-SAME: attributes {llvm.cconv = #llvm.cconv<spir_funccc>, llvm.linkage = #llvm.linkage<internal>, passthrough = ["norecurse", "nounwind", "convergent", "mustprogress"]} {
// CHECK-MLIR: call @cons_5() : () -> ()

// CHECK-MLIR-LABEL: func.func @cons_5() attributes {llvm.cconv = #llvm.cconv<spir_funccc>, llvm.linkage = #llvm.linkage<external>, passthrough = ["norecurse", "nounwind", "convergent", "mustprogress"]} {
// CHECK-MLIR:           sycl.constructor(%{{.*}})
// CHECK-MLIR-NEXT:      return

// CHECK-LLMV-LABEL: define weak_odr spir_kernel void @_ZTSZZ16host_single_taskvENKUlRN4sycl3_V17handlerEE_clES2_E18kernel_single_task
// CHECK-SAME: #0

// CHECK-LLVM-LABEL: define internal spir_func void @_ZZZ16host_single_taskvENKUlRN4sycl3_V17handlerEE_clES2_ENKUlvE_clEv
// CHECK-LLVM-SAME: #0
// CHECK: call void @cons_5()

// CHECK-LLVM-LABEL: define spir_func void @cons_5() #0 {
// CHECK-LLVM-NEXT:  [[ACCESSOR:%.*]] = alloca %"class.sycl::_V1::accessor.1", align 8
// CHECK-LLVM-NEXT:  [[ACAST:%.*]] = addrspacecast %"class.sycl::_V1::accessor.1"* [[ACCESSOR]] to %"class.sycl::_V1::accessor.1" addrspace(4)*
// CHECK-LLVM-NEXT:  call void @_ZN4sycl3_V18accessorIiLi1ELNS0_6access4modeE1025ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEC1Ev(%"class.sycl::_V1::accessor.1" addrspace(4)* [[ACAST]])

extern "C" SYCL_EXTERNAL void cons_5() {
  sycl::accessor<sycl::cl_int, 1, sycl::access::mode::write> accessor;
}

void host_single_task() {
  auto q = sycl::queue{};
  auto range = sycl::range<1>{1};
  auto buf = sycl::buffer<int, 1>{nullptr, range};
  q.submit([&](sycl::handler &cgh) {
    auto A = buf.get_access<sycl::access::mode::write>(cgh);
    cgh.single_task<class kernel_single_task>([=]() {
      cons_5();
      A[0] = 42;
    });
  });
}

// Keep at the end of the file.
// CHECK-LLVM: attributes #0 = { convergent mustprogress norecurse nounwind }
