// RUN: %clang_cc1 -I %S/Inputs -fsycl -fsycl-is-device -triple spir64-unknown-unknown-sycldevice -fsycl-int-header=%t.h %s -o %t.out
// RUN: FileCheck -input-file=%t.h %s

// This test checks the integration header generated when
// the kernel argument is union.

// CHECK: #include <CL/sycl/detail/kernel_desc.hpp>

// CHECK: class MyKernel;

// CHECK: __SYCL_INLINE_NAMESPACE(cl) {
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT: namespace detail {

// CHECK: static constexpr
// CHECK-NEXT: const char* const kernel_names[] = {
// CHECK-NEXT:   "_ZTSZZ5test0vENK3$_0clERN2cl4sycl7handlerEE8MyKernel"
// CHECK-NEXT: };

// CHECK: static constexpr
// CHECK-NEXT: const kernel_param_desc_t kernel_signatures[] = {
// CHECK-NEXT:   //--- _ZTSZZ5test0vENK3$_0clERN2cl4sycl7handlerEE8MyKernel
// CHECK-NEXT:   { kernel_param_kind_t::kind_accessor, 4062, 0 },
// CHECK-NEXT:   { kernel_param_kind_t::kind_std_layout, 12, 12 },
// CHECK-EMPTY:
// CHECK-NEXT: };

// CHECK: static constexpr
// CHECK-NEXT: const unsigned kernel_signature_start[] = {
// CHECK-NEXT:  0 // _ZTSZZ5test0vENK3$_0clERN2cl4sycl7handlerEE8MyKernel
// CHECK-NEXT: };

// CHECK: template <> struct KernelInfo<class MyKernel> {

#include "sycl.hpp"

using namespace cl::sycl;

union MyUnion {
  int FldInt;
  int FldArr[3];
};

MyUnion GlobS;

bool test0() {
  MyUnion S = GlobS;
  MyUnion S0 = {0};
  {
    buffer<MyUnion, 1> Buf(&S0, range<1>(1));
    queue myQueue;
    myQueue.submit([&](handler &cgh) {
      auto B = Buf.get_access<access::mode::write>(cgh);
      cgh.single_task<class MyKernel>([=] { B; S; });
    });
  }
}

