// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -sycl-std=2020 -fsycl-int-header=%t.h %s -o %t.out
// RUN: FileCheck -input-file=%t.h %s

#include "sycl.hpp"
// Test to ensure that the integration header properly prints class template specializations that contain
// function pointers to free function kernels as template arguments. 
// Also check that the forward declarations for free function kernels appear first in the integration header.

// CHECK: Forward declarations of kernel and its argument types:
// CHECK: void bar();

// CHECK: Forward declarations of kernel and its argument types:
// CHECK: template <typename T> void foo(T);
 
// CHECK: template <> struct KernelInfo<::Kernel<&foo<int>>> {

// CHECK: template <> struct KernelInfo<::Kernel<&bar>> {

template<typename T>
[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel", 2)]] 
void foo(T arg) {}


[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel", 2)]] 
void bar() {}

template<auto *Func>
struct Kernel;

void foo() {
  using kernel_name1 = Kernel<&foo<int>>;
  using kernel_name2 = Kernel<&bar>;
  sycl::kernel_single_task<kernel_name1>([](){});
  sycl::kernel_single_task<kernel_name2>([](){});
}
