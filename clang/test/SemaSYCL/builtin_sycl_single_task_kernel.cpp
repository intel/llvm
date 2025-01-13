// RUN: %clang_cc1 -internal-isystem %S/Inputs -fsycl-is-device %s -verify

// This test tests the builtin __builtin_sycl_is_single_task_kernel

#include "sycl.hpp"

__attribute__((sycl_device))
[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel", 0)]]
void sstk_free_func1(int *ptr, int start, int end) {
  for (int i = start; i <= end; i++)
    ptr[i] = start;
}

__attribute__((sycl_device))
[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", "")]]
void sndrk_free_func1(int *ptr, int start, int end) {
  for (int i = start; i <= end; i++)
    ptr[i] = start;
}

template <typename T>
__attribute__((sycl_device))
[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel", 4)]]
void sstk_free_func_tmpl1(T *ptr) {
  for (int i = 0; i <= 7; i++)
    ptr[i] = i + 11;
}

__attribute__((sycl_device))
[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel", 1)]]
void ovl_free_func1(int *ptr) {
  for (int i = 0; i <= 7; i++)
    ptr[i] = i;
}

__attribute__((sycl_device))
[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 1)]]
void ovl_free_func1(int *ptr, int val) {
  for (int i = 0; i <= 7; i++)
    ptr[i] = i + val;
}

void func(int *ptr, int start, int end) {}

void foo() {
  // Check number of arguments.
  // expected-error@+1 {{builtin takes one argument}}
  bool b1 = __builtin_sycl_is_single_task_kernel();
  // expected-error@+1 {{builtin takes one argument}}
  bool b2 = __builtin_sycl_is_single_task_kernel(sstk_free_func1, 4);


  // The following four are okay.
  bool b3 = __builtin_sycl_is_single_task_kernel(sstk_free_func1);   // Okay
  bool b4 = __builtin_sycl_is_single_task_kernel(*sstk_free_func1);  // Okay
  bool b5 = __builtin_sycl_is_single_task_kernel(&sstk_free_func1);  // Okay
  bool b6 = __builtin_sycl_is_single_task_kernel(func);             // Okay

  // But the next two are not.
  // expected-error@+1 {{use of undeclared identifier 'undef'}}
  bool b7 = __builtin_sycl_is_single_task_kernel(undef);
  // expected-error@+1 {{use of undeclared identifier 'laterdef'}}
  bool b8 = __builtin_sycl_is_single_task_kernel(laterdef);

  // Constexpr forms of the valid cases.
  constexpr bool b9 = __builtin_sycl_is_single_task_kernel(sstk_free_func1);   // Okay and true.
  constexpr bool b10 = __builtin_sycl_is_single_task_kernel(func);   // Okay, but false.
  constexpr bool b11 = __builtin_sycl_is_single_task_kernel(sndrk_free_func1);   // Okay, but false.

  // expected-error@+1 {{constexpr variable 'b12' must be initialized by a constant expression}}
  constexpr bool b12 = __builtin_sycl_is_single_task_kernel(*sstk_free_func1);  // Okay
  // expected-error@+1 {{constexpr variable 'b13' must be initialized by a constant expression}}
  constexpr bool b13 = __builtin_sycl_is_single_task_kernel(&sstk_free_func1);  // Okay

  // Test with function templates.
  // expected-error@+1 {{1st argument must be a function pointer (was '<overloaded function type>')}}
  constexpr bool b14 = __builtin_sycl_is_single_task_kernel(&sstk_free_func_tmpl1<int>);
  // expected-error@+1 {{1st argument must be a function pointer (was '<overloaded function type>')}}
  constexpr bool b15 = __builtin_sycl_is_single_task_kernel(sstk_free_func_tmpl1<int>);
  constexpr bool b16 = __builtin_sycl_is_single_task_kernel((void(*)(int *))sstk_free_func_tmpl1<int>);  // Okay


  // Test with overloaded functions.
  // expected-error@+1 {{1st argument must be a function pointer (was '<overloaded function type>')}}
  constexpr bool b17 = __builtin_sycl_is_single_task_kernel(ovl_free_func1);
  constexpr bool b18 = __builtin_sycl_is_single_task_kernel((void(*)(int *))ovl_free_func1);  // Okay
  constexpr bool b19 = __builtin_sycl_is_single_task_kernel((void(*)(int *, int))ovl_free_func1);  // Okay
}

void laterdef();
