// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -fsycl-int-header=%t.h %s -o %t.out
// RUN: FileCheck -input-file=%t.h %s

// CHECK:     const kernel_param_desc_t kernel_signatures[] = {
// CHECK-NEXT:  //--- _ZTSZZ5test0vENKUlRN4sycl3_V17handlerEE_clES2_E8MyKernel
// Accessor
// CHECK-NEXT:  { kernel_param_kind_t::kind_accessor, 4062, 0 },
// FldInt, offset to 16 because the float* causes the alignment of the structs
// to change.
// MyStruct is not decomposed since it does not contain special types.
// CHECK-NEXT:  { kernel_param_kind_t::kind_std_layout, 40, 16 },
// CHECK-EMPTY:
// CHECK-NEXT:  { kernel_param_kind_t::kind_invalid, -987654321, -987654321 },
// CHECK-NEXT:};

// This test checks if compiler accepts structures as kernel parameters.

#include "Inputs/sycl.hpp"

using namespace sycl;

struct MyNestedStruct {
  int FldArr[1];
  float *FldFloat;
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

