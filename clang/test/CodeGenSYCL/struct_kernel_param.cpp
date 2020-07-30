// RUN: %clang_cc1 -I %S/Inputs -fsycl -fsycl-is-device -triple spir64-unknown-unknown-sycldevice -fsycl-int-header=%t.h %s -o %t.out
// RUN: FileCheck -input-file=%t.h %s

// CHECK: static constexpr
// CHECK-NEXT: const char* const kernel_names[] = {
// CHECK-NEXT:   "_ZTSZZ5test0vENK3$_0clERN2cl4sycl7handlerEE8MyKernel"
// CHECK-NEXT: };

// CHECK: static constexpr
// CHECK-NEXT: const bool param_omit_table[] = {
// CHECK-NEXT:   // OMIT_TABLE_BEGIN
// CHECK-NEXT:   //--- _ZTSZZ5test0vENK3$_0clERN2cl4sycl7handlerEE8MyKernel
// CHECK-NEXT:   false, false, false, false, false, false, false, false, false, false,
// CHECK-NEXT:   // OMIT_TABLE_END
// CHECK-NEXT:   };

// CHECK: static constexpr
// CHECK-NEXT: const kernel_param_desc_t kernel_signatures[] = {
// CHECK-NEXT:   //--- _ZTSZZ5test0vENK3$_0clERN2cl4sycl7handlerEE8MyKernel
// CHECK-NEXT:   { kernel_param_kind_t::kind_accessor, 4062, 0, param_omit_table[0] | (param_omit_table[1] << 1) | (param_omit_table[2] << 2) | (param_omit_table[3] << 3)},
// CHECK-NEXT:   { kernel_param_kind_t::kind_std_layout, 4, 12, param_omit_table[4]},
// CHECK-NEXT:   { kernel_param_kind_t::kind_std_layout, 4, 16, param_omit_table[5]},
// CHECK-NEXT:   { kernel_param_kind_t::kind_std_layout, 4, 20, param_omit_table[6]},
// CHECK-NEXT:   { kernel_param_kind_t::kind_std_layout, 4, 24, param_omit_table[7]},
// CHECK-NEXT:   { kernel_param_kind_t::kind_std_layout, 4, 28, param_omit_table[8]},
// CHECK-NEXT:   { kernel_param_kind_t::kind_std_layout, 4, 32, param_omit_table[9]},
// CHECK-EMPTY:
// CHECK-NEXT: };

// This test checks if compiler accepts structures as kernel parameters.

#include "sycl.hpp"

using namespace cl::sycl;

struct MyNestedStruct {
  int FldArr[1];
  float FldFloat;
};

struct MyStruct {
  int FldInt;
  MyNestedStruct FldStruct;
  int FldArr[3];
};

MyStruct GlobS;

bool test0() {
  MyStruct S = GlobS;
  MyStruct S0 = { 0 };
  {
    buffer<MyStruct, 1> Buf(&S0, range<1>(1));
    queue myQueue;
    myQueue.submit([&](handler &cgh) {
      auto B = Buf.get_access<access::mode::write>(cgh);
      cgh.single_task<class MyKernel>([=] { B; S; });
    });
  }
}

