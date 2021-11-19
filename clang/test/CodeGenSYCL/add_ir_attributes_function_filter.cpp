// RUN: %clang_cc1 -internal-isystem %S/Inputs -disable-llvm-passes \
// RUN:    -triple spir64-unknown-unknown -fsycl-is-device -S \
// RUN:    -emit-llvm %s -o - | FileCheck %s

// Tests the optional filter parameter of
// __sycl_detail__::add_ir_attributes_function

#include "mock_properties.hpp"
#include "sycl.hpp"

template <typename... Properties>
#ifdef __SYCL_DEVICE_ONLY__
[[__sycl_detail__::add_ir_attributes_function(
    {"Prop4", "Prop3"},
    Properties::name..., Properties::value...)]]
#endif
void
free_func() {
}

template <typename... Properties>
class KernelFunctor {
public:
#ifdef __SYCL_DEVICE_ONLY__
  [[__sycl_detail__::add_ir_attributes_function(
      {"Prop3", "Prop5"},
      Properties::name..., Properties::value...)]]
#endif
  void
  operator()() const {
    free_func<Properties...>();
  }
};

int main() {
  sycl::queue q;
  auto f = [=]() {};
  q.submit([&](sycl::handler &h) {
    KernelFunctor<prop5, prop1, prop3, prop4> f{};
    h.single_task<class test_kernel>(f);
  });
}

// CHECK: define {{.*}}spir_kernel void @{{.*}}test_kernel() #[[KernFuncAttrs:[0-9]+]]
// CHECK: define {{.*}}spir_func void @{{.*}}free_func{{.*}}() #[[FuncAttrs:[0-9]+]]
// CHECK: attributes #[[KernFuncAttrs]] = {
// CHECK-NOT:  "Prop1"
// CHECK-NOT:  "Prop2"
// CHECK-SAME: "Prop3"
// CHECK-NOT:  "Prop4"
// CHECK-SAME: "Prop5"
// CHECK-NOT:  "Prop6"
// CHECK: attributes #[[FuncAttrs]] = {
// CHECK-NOT:  "Prop1"
// CHECK-NOT:  "Prop2"
// CHECK-SAME: "Prop3"
// CHECK-SAME: "Prop4"
// CHECK-NOT:  "Prop5"
// CHECK-NOT:  "Prop6"
