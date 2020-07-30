// RUN: %clang_cc1 -I %S/Inputs -fsycl -fsycl-is-device -triple spir64-unknown-unknown-sycldevice -fsycl-int-header=%t.h %s -emit-llvm -o %t.ll
// RUN: FileCheck -input-file=%t.h %s
//
// CHECK: #include <CL/sycl/detail/kernel_desc.hpp>
//
// CHECK: class first_kernel;
// CHECK-NEXT: namespace second_namespace {
// CHECK-NEXT: template <typename T> class second_kernel;
// CHECK-NEXT: }
// CHECK-NEXT: struct X;
// CHECK-NEXT: template <typename T> struct point;
// CHECK-NEXT: template <int a, typename T1, typename T2> class third_kernel;
// CHECK-NEXT: namespace template_arg_ns {
// CHECK-NEXT: template <int DimX> struct namespaced_arg;
// CHECK-NEXT: }
// CHECK-NEXT: template <typename ...Ts> class fourth_kernel;
//
// CHECK: static constexpr
// CHECK-NEXT: const char* const kernel_names[] = {
// CHECK-NEXT:   "_ZTSZ4mainE12first_kernel",
// CHECK-NEXT:   "_ZTSN16second_namespace13second_kernelIcEE",
// CHECK-NEXT:   "_ZTS12third_kernelILi1Ei5pointIZ4mainE1XEE"
// CHECK-NEXT:   "_ZTS13fourth_kernelIJN15template_arg_ns14namespaced_argILi1EEEEE"
// CHECK-NEXT:   "_ZTSZ4mainE16accessor_in_base"
// CHECK-NEXT: };
//
// CHECK: static constexpr
// CHECK-NEXT: const bool param_omit_table[] = {
// CHECK-NEXT:   // OMIT_TABLE_BEGIN
// CHECK-NEXT:   //--- _ZTSZ4mainE12first_kernel
// CHECK-NEXT:   false, false, false, false, false, false, false, false, false, false, false, false,
// CHECK-NEXT:   //--- _ZTSN16second_namespace13second_kernelIcEE
// CHECK-NEXT:   false, false, false, false, false, false,
// CHECK-NEXT:   //--- _ZTS12third_kernelILi1Ei5pointIZ4mainE1XEE
// CHECK-NEXT:   false, false, false, false, false, false,
// CHECK-NEXT:   //--- _ZTS13fourth_kernelIJN15template_arg_ns14namespaced_argILi1EEEEE
// CHECK-NEXT:   false, false, false, false, false,
// CHECK-NEXT:   //--- _ZTSZ4mainE16accessor_in_base
// CHECK-NEXT:   false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false,
// CHECK-NEXT:   // OMIT_TABLE_END
// CHECK-NEXT:   };
// CHECK: static constexpr
// CHECK-NEXT: const kernel_param_desc_t kernel_signatures[] = {
// CHECK-NEXT:   //--- _ZTSZ4mainE12first_kernel
// CHECK-NEXT:   { kernel_param_kind_t::kind_std_layout, 4, 0, param_omit_table[0]},
// CHECK-NEXT:   { kernel_param_kind_t::kind_std_layout, 1, 4, param_omit_table[1]},
// CHECK-NEXT:   { kernel_param_kind_t::kind_std_layout, 4, 8, param_omit_table[2]},
// CHECK-NEXT:   { kernel_param_kind_t::kind_accessor, 4062, 12, param_omit_table[3] | (param_omit_table[4] << 1) | (param_omit_table[5] << 2) | (param_omit_table[6] << 3)},
// CHECK-NEXT:   { kernel_param_kind_t::kind_accessor, 6112, 24, param_omit_table[7] | (param_omit_table[8] << 1) | (param_omit_table[9] << 2) | (param_omit_table[10] << 3)},
// CHECK-NEXT:   { kernel_param_kind_t::kind_sampler, 8, 40, param_omit_table[11]},
// CHECK-EMPTY:
// CHECK-NEXT:   //--- _ZTSN16second_namespace13second_kernelIcEE
// CHECK-NEXT:   { kernel_param_kind_t::kind_std_layout, 4, 0, param_omit_table[12]},
// CHECK-NEXT:   { kernel_param_kind_t::kind_accessor, 6112, 4, param_omit_table[13] | (param_omit_table[14] << 1) | (param_omit_table[15] << 2) | (param_omit_table[16] << 3)},
// CHECK-NEXT:   { kernel_param_kind_t::kind_sampler, 8, 16, param_omit_table[17]},
// CHECK-EMPTY:
// CHECK-NEXT:   //--- _ZTS12third_kernelILi1Ei5pointIZ4mainE1XEE
// CHECK-NEXT:   { kernel_param_kind_t::kind_std_layout, 4, 0, param_omit_table[18]},
// CHECK-NEXT:   { kernel_param_kind_t::kind_accessor, 6112, 4, param_omit_table[19] | (param_omit_table[20] << 1) | (param_omit_table[21] << 2) | (param_omit_table[22] << 3)},
// CHECK-NEXT:   { kernel_param_kind_t::kind_sampler, 8, 16, param_omit_table[23]},
// CHECK-EMPTY:
// CHECK-NEXT:   //--- _ZTS13fourth_kernelIJN15template_arg_ns14namespaced_argILi1EEEEE
// CHECK-NEXT:   { kernel_param_kind_t::kind_std_layout, 4, 0, param_omit_table[24]},
// CHECK-NEXT:   { kernel_param_kind_t::kind_accessor, 6112, 4, param_omit_table[25] | (param_omit_table[26] << 1) | (param_omit_table[27] << 2) | (param_omit_table[28] << 3)},
// CHECK-EMPTY:
// CHECK-NEXT:   //--- _ZTSZ4mainE16accessor_in_base
// CHECK-NEXT:   { kernel_param_kind_t::kind_std_layout, 4, 0, param_omit_table[29]},
// CHECK-NEXT:   { kernel_param_kind_t::kind_std_layout, 4, 4, param_omit_table[30]},
// CHECK-NEXT:   { kernel_param_kind_t::kind_accessor, 4062, 8, param_omit_table[31] | (param_omit_table[32] << 1) | (param_omit_table[33] << 2) | (param_omit_table[34] << 3)},
// CHECK-NEXT:   { kernel_param_kind_t::kind_std_layout, 4, 20, param_omit_table[35]},
// CHECK-NEXT:   { kernel_param_kind_t::kind_accessor, 4062, 24, param_omit_table[36] | (param_omit_table[37] << 1) | (param_omit_table[38] << 2) | (param_omit_table[39] << 3)},
// CHECK-NEXT:   { kernel_param_kind_t::kind_std_layout, 4, 36, param_omit_table[40]},
// CHECK-NEXT:   { kernel_param_kind_t::kind_accessor, 4062, 40, param_omit_table[41] | (param_omit_table[42] << 1) | (param_omit_table[43] << 2) | (param_omit_table[44] << 3)},
// CHECK-NEXT:   { kernel_param_kind_t::kind_accessor, 4062, 52, param_omit_table[45] | (param_omit_table[46] << 1) | (param_omit_table[47] << 2) | (param_omit_table[48] << 3)},
// CHECK-EMPTY:
// CHECK-NEXT: };

// CHECK: template <> struct KernelInfo<class first_kernel> {
// CHECK: template <> struct KernelInfo<::second_namespace::second_kernel<char>> {
// CHECK: template <> struct KernelInfo<::third_kernel<1, int, ::point<X>>> {
// CHECK: template <> struct KernelInfo<::fourth_kernel<::template_arg_ns::namespaced_arg<1>>> {

#include "sycl.hpp"

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
  cl::sycl::accessor<char, 1, cl::sycl::access::mode::read> acc;
};

struct base2 : other_base,
               cl::sycl::accessor<char, 1, cl::sycl::access::mode::read> {
  int i;
  cl::sycl::accessor<char, 1, cl::sycl::access::mode::read> acc;
};

struct captured : base, base2 {
  cl::sycl::accessor<char, 1, cl::sycl::access::mode::read> acc;
  void use() const {}
};

}; // namespace accessor_in_base

int main() {

  cl::sycl::accessor<char, 1, cl::sycl::access::mode::read> acc1;
  cl::sycl::accessor<float, 2, cl::sycl::access::mode::write,
                     cl::sycl::access::target::local,
                     cl::sycl::access::placeholder::true_t>
      acc2;
  int i = 13;
  cl::sycl::sampler smplr;
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
  kernel_single_task<class third_kernel<1, int,point<struct X>>>([=]() {
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

  return 0;
}
