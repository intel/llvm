// RUN: %clang_cc1 -fsycl-is-device -triple spir64 -emit-llvm --disable-passes %s -o %t.ll
// RUN: opt < %t.ll -passes=deadargelim-sycl -S | FileCheck %s

// CHECK-NOT: !sycl_kernel_omit_args

__attribute__((sycl_device))
[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel", 0)]]
void func1(int a, int b) {
    a = 42;
}
// CHECK: define dso_local spir_kernel void @_Z19__sycl_kernel_func1ii(i32 noundef %__arg_a, i32 noundef %__arg_b)

__attribute__((sycl_device))
[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 0)]]
void func2(int a, int b) {
    b = 42;
}
// CHECK: define dso_local spir_kernel void @_Z19__sycl_kernel_func2ii(i32 noundef %__arg_a, i32 noundef %__arg_b)
