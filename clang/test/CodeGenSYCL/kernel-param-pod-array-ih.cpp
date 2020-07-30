// RUN: %clang_cc1 -I %S/Inputs -fsycl -fsycl-is-device -triple spir64-unknown-unknown-sycldevice -fsycl-int-header=%t.h %s -fsyntax-only
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
// CHECK-NEXT:   "_ZTSZ4mainE8kernel_B"
// CHECK-NEXT: };

// CHECK: static constexpr
// CHECK-NEXT: const bool param_omit_table[] = {
// CHECK-NEXT:   // OMIT_TABLE_BEGIN
// CHECK-NEXT:   //--- _ZTSZ4mainE8kernel_B
// CHECK-NEXT:   false, false, false, false, false,
// CHECK-NEXT:   // OMIT_TABLE_END
// CHECK-NEXT:   };

// CHECK: static constexpr
// CHECK-NEXT: const kernel_param_desc_t kernel_signatures[] = {
// CHECK-NEXT:   //--- _ZTSZ4mainE8kernel_B
// CHECK-NEXT:   { kernel_param_kind_t::kind_std_layout, 4, 0, param_omit_table[0]},
// CHECK-NEXT:   { kernel_param_kind_t::kind_std_layout, 4, 4, param_omit_table[1]},
// CHECK-NEXT:   { kernel_param_kind_t::kind_std_layout, 4, 8, param_omit_table[2]},
// CHECK-NEXT:   { kernel_param_kind_t::kind_std_layout, 4, 12, param_omit_table[3]},
// CHECK-NEXT:   { kernel_param_kind_t::kind_std_layout, 4, 16, param_omit_table[4]},
// CHECK-EMPTY:
// CHECK-NEXT: };

// CHECK: static constexpr
// CHECK-NEXT: const unsigned kernel_signature_start[] = {
// CHECK-NEXT:  0 // _ZTSZ4mainE8kernel_B
// CHECK-NEXT: };

// CHECK: template <> struct KernelInfo<class kernel_B> {

#include <sycl.hpp>

using namespace cl::sycl;

template <typename name, typename Func>
__attribute__((sycl_kernel)) void a_kernel(const Func &kernelFunc) {
  kernelFunc();
}

int main() {

  int a[5];

  a_kernel<class kernel_B>(
      [=]() {
        int local = a[3];
      });
}
