// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown-sycldevice -fsycl-int-header=%t.h %s -fsyntax-only
// RUN: FileCheck -input-file=%t.h %s
// This test checks the integration header generated for a kernel
// with an argument that is a POD array.

// CHECK: #include <CL/sycl/detail/kernel_desc.hpp>

// CHECK: class kernel_B;

// CHECK: __SYCL_INLINE_NAMESPACE(cl) {
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT: namespace detail {

// CHECK: static constexpr
// CHECK-NEXT: const char* const kernel_names[] = {
// CHECK-NEXT:   "_ZTSZ4mainE8kernel_B",
// CHECK-NEXT:   "_ZTSZ4mainE8kernel_C",
// CHECK-NEXT:   "_ZTSZ4mainE8kernel_D"
// CHECK-NEXT: };

// CHECK: static constexpr
// CHECK-NEXT: const kernel_param_desc_t kernel_signatures[] = {
// CHECK-NEXT: //--- _ZTSZ4mainE8kernel_B
// CHECK-NEXT:   { kernel_param_kind_t::kind_std_layout, 20, 0 },
// CHECK-EMPTY:
// CHECK-NEXT: //--- _ZTSZ4mainE8kernel_C
// CHECK-NEXT:   { kernel_param_kind_t::kind_std_layout, 24, 0 },
// CHECK-EMPTY:
// CHECK-NEXT: //--- _ZTSZ4mainE8kernel_D
// CHECK-NEXT:   { kernel_param_kind_t::kind_std_layout, 48, 0 },
// CHECK-EMPTY:
// CHECK-NEXT: };

// CHECK: template <> struct KernelInfo<kernel_B> {
// CHECK: template <> struct KernelInfo<kernel_C> {
// CHECK: template <> struct KernelInfo<kernel_D> {

#include "Inputs/sycl.hpp"

using namespace cl::sycl;

template <typename name, typename Func>
__attribute__((sycl_kernel)) void a_kernel(const Func &kernelFunc) {
  kernelFunc();
}

int main() {

  int a[5];
  int b[2][3];
  int c[2][3][2];

  a_kernel<class kernel_B>(
      [=]() {
        int local = a[3];
      });

  a_kernel<class kernel_C>(
      [=]() {
        int local = b[0][1];
      });

  a_kernel<class kernel_D>(
      [=]() {
        int local = c[0][1][1];
      });
}
