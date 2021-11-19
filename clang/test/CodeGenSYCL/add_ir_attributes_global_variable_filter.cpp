// RUN: %clang_cc1 -internal-isystem %S/Inputs -disable-llvm-passes \
// RUN:    -triple spir64-unknown-unknown -fsycl-is-device -S \
// RUN:    -emit-llvm %s -o - | FileCheck %s

// Tests the optional filter parameter of
// __sycl_detail__::add_ir_attributes_global_variable

#include "mock_properties.hpp"
#include "sycl.hpp"

template <typename... NameValues> struct
#ifdef __SYCL_DEVICE_ONLY__
    [[__sycl_detail__::add_ir_attributes_global_variable(
        {"Prop3", "Prop4", "Prop6"},
        NameValues::name..., NameValues::value...)]]
#endif
    g {
  int x;

  constexpr g() : x(1) {}
  constexpr g(int _x) : x(_x) {}
};

constexpr g<prop1, prop2, prop3, prop4> g_v;

int main() {
  sycl::queue q;
  g<prop1, prop2, prop3, prop4, prop5> a;
  q.submit([&](sycl::handler &h) {
    h.single_task<class test_kernel>(
        [=]() {
          (void)g_v.x;
        });
  });
}

// CHECK-NOT: "Prop1"="Property string"
// CHECK-NOT: "Prop2"="1"
// CHECK: "Prop3"="true"
// CHECK: "Prop4"="2"
// CHECK-NOT: "Prop5"
// CHECK-NOT: "Prop6"
