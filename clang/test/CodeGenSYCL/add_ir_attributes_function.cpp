// RUN: %clang_cc1 -internal-isystem %S/Inputs -disable-llvm-passes \
// RUN:    -triple spir64-unknown-unknown -fsycl-is-device -S \
// RUN:    -emit-llvm %s -o - | FileCheck %s

// Tests the generation of IR attributes when using
// __sycl_detail__::add_ir_attributes_function

#include "mock_properties.hpp"
#include "sycl.hpp"

template <typename... Properties>
#ifdef __SYCL_DEVICE_ONLY__
[[__sycl_detail__::add_ir_attributes_function(
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
    KernelFunctor<prop1, prop2, prop3, prop4, prop5, prop6> f{};
    h.single_task<class test_kernel>(f);
  });
}

// CHECK: define {{.*}}spir_kernel void @{{.*}}test_kernel() #[[KernFuncAttrs:[0-9]+]]
// CHECK: define {{.*}}spir_func void @{{.*}}free_func{{.*}}() #[[FuncAttrs:[0-9]+]]
// CHECK: attributes #[[KernFuncAttrs]] = { {{.*}}"Prop1"="Property string"{{.*}}"Prop2"="1"{{.*}}"Prop3"="true"{{.*}}"Prop4"="2"{{.*}}"Prop5"{{.*}}"Prop6"{{.*}} }
// CHECK: attributes #[[FuncAttrs]] = { {{.*}}"Prop1"="Property string"{{.*}}"Prop2"="1"{{.*}}"Prop3"="true"{{.*}}"Prop4"="2"{{.*}}"Prop5"{{.*}}"Prop6"{{.*}} }
