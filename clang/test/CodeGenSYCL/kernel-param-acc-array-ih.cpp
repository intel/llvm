// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -fsycl-int-header=%t.h %s -fsyntax-only
// RUN: FileCheck -input-file=%t.h %s

// This test checks the integration header generated when
// the kernel argument is an Accessor array.

// CHECK: #include <sycl/detail/kernel_desc.hpp>

// CHECK: class kernel_A;

// CHECK: namespace sycl {
// CHECK-NEXT: inline namespace _V1 {
// CHECK-NEXT: namespace detail {

// CHECK: static constexpr
// CHECK-NEXT: const char* const kernel_names[] = {
// CHECK-NEXT:   "_ZTSZ4mainE8kernel_A"
// CHECK-NEXT:   ""
// CHECK-NEXT: };

// CHECK: static constexpr
// CHECK-NEXT: const kernel_param_desc_t kernel_signatures[] = {
// CHECK-NEXT:   //--- _ZTSZ4mainE8kernel_A
// CHECK-NEXT:   { kernel_param_kind_t::kind_accessor, 4062, 0 },
// CHECK-NEXT:   { kernel_param_kind_t::kind_accessor, 4062, 12 },
// CHECK-EMPTY:
// CHECK-NEXT:   { kernel_param_kind_t::kind_invalid, -987654321, -987654321 },
// CHECK-NEXT: };

// CHECK: template <> struct KernelInfo<kernel_A> {

#include "Inputs/sycl.hpp"

using namespace sycl;

template <typename name, typename Func>
__attribute__((sycl_kernel)) void a_kernel(const Func &kernelFunc) {
  kernelFunc();
}

int main() {

  using Accessor =
      accessor<int, 1, access::mode::read_write, access::target::global_buffer>;

  Accessor acc[2];

  a_kernel<class kernel_A>([=]() { acc[1].use(); });
}
