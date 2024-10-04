// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -fsycl-int-header=%t.h %s -o %t.out
// RUN: FileCheck -input-file=%t.h %s
//
// CHECK: #include <sycl/detail/kernel_desc.hpp>

// CHECK: class wrapped_access;

// CHECK: namespace sycl {
// CHECK-NEXT: inline namespace _V1 {
// CHECK-NEXT: namespace detail {

// CHECK: static constexpr
// CHECK-NEXT: const char* const kernel_names[] = {
// CHECK-NEXT:   "_ZTSZ4mainE14wrapped_access",
// CHECK-NEXT:   ""
// CHECK-NEXT: };

// CHECK: static constexpr
// CHECK-NEXT: const kernel_param_desc_t kernel_signatures[] = {
// CHECK-NEXT: //--- _ZTSZ4mainE14wrapped_access
// CHECK-NEXT:   { kernel_param_kind_t::kind_accessor, 4062, 0 },
// CHECK-EMPTY:
// CHECK-NEXT:   { kernel_param_kind_t::kind_invalid, -987654321, -987654321 },
// CHECK-NEXT: };

// CHECK: template <> struct KernelInfo<wrapped_access> {

#include "Inputs/sycl.hpp"

template <typename Acc>
struct AccWrapper { Acc accessor; };

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel(const Func &kernelFunc) {
  kernelFunc();
}

int main() {
  sycl::accessor<int, 1, sycl::access::mode::read_write> acc;
  auto acc_wrapped = AccWrapper<decltype(acc)>{acc};
  kernel<class wrapped_access>(
      [=]() {
        acc_wrapped.accessor.use();
      });
}
