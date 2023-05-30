// RUN: %clangxx %s -fsycl-device-only -fintelfpga -S -o %t.ll
// RUN: FileCheck %s --input-file %t.ll

// This test checks that clang emits @llvm.ptr.annotation intrinsic correctly
// when an annotated field of a kernel functor is assigned.
// Note this is a temporary test for the FPGA-specific use model that will be
// replaced by kernel argument compile-time properties.

#include "Inputs/sycl.hpp"

// CHECK: @[[STR:.*]] = private unnamed_addr addrspace(1) constant [9 x i8] c"my_ann_1\00", section "llvm.metadata"

struct foo {
    [[clang::annotate("my_ann_1")]]
    int n;

// CHECK: define dso_local spir_kernel void @_ZTS3foo(i32 {{.*}}%[[ARG:.*]])
// CHECK: %[[ARG_ADDR:.*]] = alloca i32, align 4
// CHECK: store i32 %[[ARG]], i32* %[[ARG_ADDR]], align 4
// CHECK-DAG: %[[LOAD_ARG:.*]] = load i32, i32* %[[ARG_ADDR]], align 4
// CHECK-DAG: %[[GEP:.*]] = getelementptr inbounds
// CHECK-DAG: %[[ANNOTATED_PTR:.*]] = call i32 addrspace(4)* @llvm.ptr.annotation.p4i32.p1i8(i32 addrspace(4)* %[[GEP]], {{.*}} @[[STR]],
// CHECK: store i32 %[[LOAD_ARG]], i32 addrspace(4)* %[[ANNOTATED_PTR]], align 4
    foo(int _n) : n(_n) {}

    void operator()() const {}
};

int main() {
  sycl::handler h;
  h.single_task(foo{0});
  return 0;
}
