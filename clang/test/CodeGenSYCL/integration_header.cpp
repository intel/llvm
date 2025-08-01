// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -sycl-std=2020 -fsycl-int-header=%t.h %s -emit-llvm -o %t.ll
// RUN: FileCheck -input-file=%t.h %s
//
// CHECK: #include <sycl/detail/kernel_desc.hpp>
//
// CHECK: // Forward declarations of templated kernel function types:
// CHECK: class first_kernel;
// CHECK-NEXT: namespace second_namespace {
// CHECK-NEXT: template <typename T> class second_kernel;
// CHECK-NEXT: }
// CHECK-NEXT: namespace template_arg_ns {
// CHECK-NEXT: template <int DimX> struct namespaced_arg;
// CHECK-NEXT: }
// CHECK-NEXT: template <typename ...Ts> class fourth_kernel;
// CHECK: class FinalClass final;
//
// CHECK: static constexpr
// CHECK-NEXT: const char* const kernel_names[] = {
// CHECK-NEXT:   "_ZTSZ4mainE12first_kernel",
// CHECK-NEXT:   "_ZTSN16second_namespace13second_kernelIcEE",
// CHECK-NEXT:   "_ZTS13fourth_kernelIJN15template_arg_ns14namespaced_argILi1EEEEE"
// CHECK-NEXT:   "_ZTSZ4mainE16accessor_in_base"
// CHECK-NEXT:   "_ZTSZ4mainE15annotated_types"
// CHECK-NEXT:   "_ZTS10FinalClass"
// CHECK-NEXT:   ""
// CHECK-NEXT: };
//
// CHECK: static constexpr unsigned kernel_args_sizes[] = {
//
// CHECK: static constexpr
// CHECK-NEXT: const kernel_param_desc_t kernel_signatures[] = {
// CHECK-NEXT:   //--- _ZTSZ4mainE12first_kernel
// CHECK-NEXT:   { kernel_param_kind_t::kind_std_layout, 4, 0 },
// CHECK-NEXT:   { kernel_param_kind_t::kind_std_layout, 8, 4 },
// CHECK-NEXT:   { kernel_param_kind_t::kind_accessor, 4062, 12 },
// CHECK-NEXT:   { kernel_param_kind_t::kind_accessor, 6112, 24 },
// CHECK-NEXT:   { kernel_param_kind_t::kind_sampler, 8, 40 },
// CHECK-EMPTY:
// CHECK-NEXT:   //--- _ZTSN16second_namespace13second_kernelIcEE
// CHECK-NEXT:   { kernel_param_kind_t::kind_std_layout, 4, 0 },
// CHECK-NEXT:   { kernel_param_kind_t::kind_accessor, 6112, 4 },
// CHECK-NEXT:   { kernel_param_kind_t::kind_sampler, 8, 16 },
// CHECK-EMPTY:
// CHECK-NEXT:   //--- _ZTS13fourth_kernelIJN15template_arg_ns14namespaced_argILi1EEEEE
// CHECK-NEXT:   { kernel_param_kind_t::kind_std_layout, 4, 0 },
// CHECK-NEXT:   { kernel_param_kind_t::kind_accessor, 6112, 4 },
// CHECK-EMPTY:
// CHECK-NEXT:  //--- _ZTSZ4mainE16accessor_in_base
// CHECK-NEXT:  { kernel_param_kind_t::kind_std_layout, 4, 0 },
// CHECK-NEXT:  { kernel_param_kind_t::kind_std_layout, 4, 4 },
// CHECK-NEXT:  { kernel_param_kind_t::kind_accessor, 4062, 8 },
// CHECK-NEXT:  { kernel_param_kind_t::kind_std_layout, 4, 20 },
// CHECK-NEXT:  { kernel_param_kind_t::kind_accessor, 4062, 24 },
// CHECK-NEXT:  { kernel_param_kind_t::kind_std_layout, 4, 36 },
// CHECK-NEXT:  { kernel_param_kind_t::kind_accessor, 4062, 40 },
// CHECK-NEXT:  { kernel_param_kind_t::kind_accessor, 4062, 52 },
// CHECK-EMPTY:
// CHECK-NEXT:  //--- _ZTSZ4mainE15annotated_types
// CHECK-NEXT:  { kernel_param_kind_t::kind_std_layout, 4, 0 },
// CHECK-NEXT:  { kernel_param_kind_t::kind_pointer, 8, 8 },
// CHECK-NEXT:  { kernel_param_kind_t::kind_pointer, 8, 16 },
// CHECK-EMPTY:
// CHECK-NEXT:  //--- _ZTS10FinalClass
// CHECK-EMPTY:
// CHECK-NEXT:  { kernel_param_kind_t::kind_invalid, -987654321, -987654321 },
// CHECK-NEXT: };
//
// CHECK: template <> struct KernelInfo<first_kernel> {
// CHECK: template <> struct KernelInfo<::second_namespace::second_kernel<char>> {
// CHECK: template <> struct KernelInfo<::fourth_kernel<::template_arg_ns::namespaced_arg<1>>> {

#include "Inputs/sycl.hpp"

template <typename KernelName, typename KernelType>
__attribute__((sycl_kernel)) void kernel_single_task(const KernelType &kernelFunc) {
  kernelFunc();
}
struct x {};
template <typename T>
struct point {};
namespace second_namespace {
template <typename T = int>
class second_kernel;
}

template <int a, typename T1, typename T2>
class third_kernel;

namespace template_arg_ns {
template <int DimX>
struct namespaced_arg {};
} // namespace template_arg_ns

template <typename... Ts>
class fourth_kernel;

namespace accessor_in_base {
struct other_base {
  int i;
};
struct base {
  int i, j;
  sycl::accessor<char, 1, sycl::access::mode::read> acc;
};

struct base2 : other_base,
               sycl::accessor<char, 1, sycl::access::mode::read> {
  int i;
  sycl::accessor<char, 1, sycl::access::mode::read> acc;
};

struct captured : base, base2 {
  sycl::accessor<char, 1, sycl::access::mode::read> acc;
  void use() const {}
};

}; // namespace accessor_in_base

struct MockProperty {};

class FinalClass final {};

int main() {

  sycl::accessor<char, 1, sycl::access::mode::read> acc1;
  sycl::accessor<float, 2, sycl::access::mode::write,
                     sycl::access::target::local,
                     sycl::access::placeholder::true_t>
      acc2;
  int i = 13;
  sycl::sampler smplr;
  struct {
    char c;
    int i;
  } test_s;
  test_s.c = 14;
  kernel_single_task<class first_kernel>([=]() {
    if (i == 13 && test_s.c == 14) {

      acc1.use();
      acc2.use();
      smplr.use();
    }
  });

  kernel_single_task<class second_namespace::second_kernel<char>>([=]() {
    if (i == 13) {
      acc2.use();
      smplr.use();
    }
  });

    kernel_single_task<class fourth_kernel<template_arg_ns::namespaced_arg<1>>>([=]() {
      if (i == 13) {
        acc2.use();
      }
  });

  accessor_in_base::captured c;
  kernel_single_task<class accessor_in_base>([=]() {
    c.use();
  });

  sycl::ext::oneapi::experimental::annotated_arg<int, MockProperty> AA1;
  sycl::ext::oneapi::experimental::annotated_ptr<int, MockProperty> AP2;
  sycl::ext::oneapi::experimental::annotated_arg<float*, MockProperty> AP3;
  kernel_single_task<class annotated_types>([=]() {
    (void)AA1;
    (void)AP2;
    (void)AP3;
  });

  kernel_single_task<class FinalClass>([=]() { });

  return 0;
}
