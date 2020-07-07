// RUN: %clang_cc1 -I %S/Inputs -fsycl -fsycl-is-device -fsycl-int-header=%t.h %s -o %t.out
// RUN: FileCheck -input-file=%t.h %s

// CHECK:     const kernel_param_desc_t kernel_signatures[] = {
// CHECK-NEXT:  //--- _ZTSZZ5test0vENK3$_0clERN2cl4sycl7handlerEE8MyKernel
// CHECK-NEXT:  { kernel_param_kind_t::kind_accessor, 4062, 0 },
// CHECK-NEXT:  { kernel_param_kind_t::kind_std_layout, 4, 12 },
// CHECK-NEXT:  { kernel_param_kind_t::kind_std_layout, 4, 16 },
// CHECK-NEXT:  { kernel_param_kind_t::kind_std_layout, 4, 20 },
// CHECK-NEXT:  { kernel_param_kind_t::kind_std_layout, 4, 24 },
// CHECK-NEXT:  { kernel_param_kind_t::kind_std_layout, 4, 28 },
// CHECK-NEXT:  { kernel_param_kind_t::kind_std_layout, 4, 32 },
// CHECK-EMPTY:
// CHECK-NEXT:};

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

