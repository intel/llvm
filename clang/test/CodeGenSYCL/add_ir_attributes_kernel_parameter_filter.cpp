
// RUN: %clang_cc1 -internal-isystem %S/Inputs -disable-llvm-passes \
// RUN:    -triple spir64-unknown-unknown -fsycl-is-device -S \
// RUN:    -emit-llvm %s -o - | FileCheck %s

// Tests the optional filter parameter of
// __sycl_detail__::add_ir_attributes_kernel_parameter

#include "mock_properties.hpp"
#include "sycl.hpp"

template <typename... Properties> class __attribute__((sycl_special_class)) g {
public:
  int *x;

  g() : x(nullptr) {}
  g(int *_x) : x(_x) {}

#ifdef __SYCL_DEVICE_ONLY__
  void __init(
#ifdef __SYCL_DEVICE_ONLY__
      [[__sycl_detail__::add_ir_attributes_kernel_parameter(
          {"Prop1", "Prop6", "Prop4", "Prop2"},
          Properties::name..., Properties::value...)]]
#endif
      int *_x) {
    x = _x;
  }
#endif
};

int main() {
  sycl::queue q;
  g<prop2, prop3, prop1, prop6> a;
  q.submit([&](sycl::handler &h) {
    h.single_task<class test_kernel>(
        [=]() {
          (void)a;
        });
  });
}

// CHECK: "Prop1"="Property string"
// CHECK: "Prop2"="1"
// CHECK-NOT: "Prop3"="true"
// CHECK-NOT: "Prop4"="2"
// CHECK-NOT: "Prop5"
// CHECK: "Prop6"
