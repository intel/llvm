// RUN: %clang_cc1 -internal-isystem %S/Inputs -fsycl-is-device -triple spir64-unknown-unknown -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s

// This test tests the builtin __builtin_sycl_is_kernel

#include "sycl.hpp"

using namespace sycl;
queue q;

__attribute__((sycl_device))
[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel", 0)]]
void sstk_free_func() {
}

template <typename T>
[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel", 4)]]
void sstk_free_func_tmpl(T *ptr) {
  for (int i = 0; i <= 7; i++)
    ptr[i] = i + 11;
}

__attribute__((sycl_device))
[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel", 1)]]
void ovl_free_func(int *ptr) {
  for (int i = 0; i <= 7; i++)
    ptr[i] = i;
}

__attribute__((sycl_device))
[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 1)]]
void ovl_free_func(int *ptr, int val) {
  for (int i = 0; i <= 7; i++)
    ptr[i] = i + val;
}

__attribute__((sycl_device))
[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 4)]]
void sndrk_free_func() {
}

void func() {}

void foo() {
  bool b1 = __builtin_sycl_is_kernel(sstk_free_func);
  // CHECK: store i8 1, ptr addrspace(4) %b1{{.*}}, align 1
  bool b2 = __builtin_sycl_is_kernel(*sstk_free_func);
  // CHECK: store i8 1, ptr addrspace(4) %b2{{.*}}, align 1
  bool b3 = __builtin_sycl_is_kernel(&sstk_free_func);
  // CHECK: store i8 1, ptr addrspace(4) %b3{{.*}}, align 1
  bool b4 = __builtin_sycl_is_kernel(func);
  // CHECK: store i8 0, ptr addrspace(4) %b4{{.*}}, align 1
  bool b5 = __builtin_sycl_is_kernel(sndrk_free_func);
  // CHECK: store i8 1, ptr addrspace(4) %b5{{.*}}, align 1

  // Constexpr forms of the valid cases.
  constexpr bool b6 = __builtin_sycl_is_kernel(sstk_free_func);   // Okay and true.
  // CHECK: store i8 1, ptr addrspace(4) %b6{{.*}}, align 1
  constexpr bool b7 = __builtin_sycl_is_kernel(func);   // Okay, but false.
  // CHECK: store i8 0, ptr addrspace(4) %b7{{.*}}, align 1
  constexpr bool b8 = __builtin_sycl_is_kernel(sndrk_free_func);   // Okay, but false.
  // CHECK: store i8 1, ptr addrspace(4) %b8{{.*}}, align 1

  // Test function template.
  constexpr bool b9 = __builtin_sycl_is_kernel((void(*)(int *))sstk_free_func_tmpl<int>);  // Okay
  // CHECK: store i8 1, ptr addrspace(4) %b9{{.*}}, align 1

  // Test overloaded functions.
  constexpr bool b10 = __builtin_sycl_is_kernel((void(*)(int *))ovl_free_func);  // Okay
  // CHECK: store i8 1, ptr addrspace(4) %b10{{.*}}, align 1
  constexpr bool b11 = __builtin_sycl_is_kernel((void(*)(int *, int))ovl_free_func);  // Okay
  // CHECK: store i8 1, ptr addrspace(4) %b11{{.*}}, align 1
}

void f() {
  auto L = []() { foo(); };
  q.submit([&](handler &h) {
    h.single_task<class kernel_name_1>(L);
  });
}
