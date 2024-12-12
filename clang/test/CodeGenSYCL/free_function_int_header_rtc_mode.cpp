// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple spir64-unknown-unknown -sycl-std=2020 -fsycl-rtc-mode -fsycl-int-header=%t.rtc.h %s
// RUN: FileCheck -input-file=%t.rtc.h --check-prefixes=CHECK,CHECK-RTC %s

// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple spir64-unknown-unknown -sycl-std=2020 -fno-sycl-rtc-mode -fsycl-int-header=%t.nortc.h %s
// RUN: FileCheck -input-file=%t.nortc.h --check-prefixes=CHECK,CHECK-NORTC %s

// This test checks that free-function kernel information is included or
// excluded from the integration header, depending on the '-fsycl-rtc-mode'
// flag.

#include "sycl.hpp"

[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel", 2)]]
void free_function_single(int* ptr, int start, int end){
  for(int i = start; i < end; ++i){
    ptr[i] = start + 66;
  }
}

[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 2)]]
void free_function_nd_range(int* ptr, int start, int end){
  for(int i = start; i < end; ++i){
    ptr[i] = start + 66;
  }
}

template<typename KernelName, typename KernelFunc>
__attribute__((sycl_kernel)) void kernel(const KernelFunc &kernelFunc){
  kernelFunc();
}

int main(){
  sycl::accessor<int, 1, sycl::access::mode::read_write> accessorA;
  kernel<class Kernel_Function>(
    [=]() {
      accessorA.use();
    });
  return 0;
}


// CHECK:       const char* const kernel_names[] = {
// CHECK-NEXT:    "{{.*}}__sycl_kernel_free_function_singlePiii",
// CHECK-NEXT:    "{{.*}}__sycl_kernel_free_function_nd_rangePiii",
// CHECK-NEXT:    "{{.*}}Kernel_Function",


// CHECK: static constexpr const char* getName() { return "{{.*}}__sycl_kernel_free_function_singlePiii"; }
// CHECK: static constexpr const char* getName() { return "{{.*}}__sycl_kernel_free_function_nd_rangePiii"; }
// CHECK: static constexpr const char* getName() { return "{{.*}}Kernel_Function"; }

// CHECK-RTC-NOT: free_function_single_kernel
// CHECK-RTC-NOT: free_function_nd_range

// CHECK-NORTC:       void free_function_single(int *ptr, int start, int end);
// CHECK-NORTC:       static constexpr auto __sycl_shim[[#FIRST:]]()
// CHECK-NORTC-NEXT:  return (void (*)(int *, int, int))free_function_single;

// CHECK-NORTC:       struct ext::oneapi::experimental::is_kernel<__sycl_shim[[#FIRST]]()> {
// CHECK-NORTC-NEXT:  static constexpr bool value = true;

// CHECK-NORTC:       struct ext::oneapi::experimental::is_single_task_kernel<__sycl_shim[[#FIRST]]()> {
// CHECK-NORTC-NEXT:  static constexpr bool value = true;


// CHECK-NORTC:       void free_function_nd_range(int *ptr, int start, int end);
// CHECK-NORTC:       static constexpr auto __sycl_shim[[#SECOND:]]() {
// CHECK-NORTC-NEXT:  return (void (*)(int *, int, int))free_function_nd_range;

// CHECK-NORTC:       struct ext::oneapi::experimental::is_kernel<__sycl_shim[[#SECOND]]()> {
// CHECK-NORTC-NEXT:  static constexpr bool value = true;

// CHECK-NORTC:       struct ext::oneapi::experimental::is_nd_range_kernel<__sycl_shim2(), 2> {
// CHECK-NORTC-NEXT:  static constexpr bool value = true;

// CHECK-NORTC:       kernel_id ext::oneapi::experimental::get_kernel_id<__sycl_shim[[#FIRST]]()>() {
// CHECK-NORTC-NEXT:  return sycl::detail::get_kernel_id_impl(std::string_view{"{{.*}}__sycl_kernel_free_function_singlePiii"});

// CHECK-NORTC:       kernel_id ext::oneapi::experimental::get_kernel_id<__sycl_shim[[#SECOND]]()>() {
// CHECK-NORTC-NEXT:  return sycl::detail::get_kernel_id_impl(std::string_view{"{{.*}}__sycl_kernel_free_function_nd_rangePiii"});
