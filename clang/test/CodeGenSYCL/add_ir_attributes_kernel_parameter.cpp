
// RUN: %clang_cc1 -internal-isystem %S/Inputs -disable-llvm-passes \
// RUN:    -triple spir64-unknown-unknown -fsycl-is-device -S \
// RUN:    -emit-llvm %s -o - | FileCheck %s

// Tests the generation of IR attributes when using
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
      [[__sycl_detail__::add_ir_attributes_kernel_parameter(
          Properties::name..., Properties::value...)]] int *_x) {
    x = _x;
  }
#endif
};

int main() {
  sycl::queue q;
  g<prop1, prop2, prop3, prop4, prop5, prop6> a;
  q.submit([&](sycl::handler &h) {
    h.single_task<class test_kernel>(
        [=]() {
          (void)a;
        });
  });
}

// CHECK: define {{.*}}spir_kernel void @{{.*}}test_kernel({{.*}}i32 addrspace({{.*}})* {{.*}}"Prop1"="Property string"{{.*}}"Prop2"="1"{{.*}}"Prop3"="true"{{.*}}"Prop4"="2"{{.*}}"Prop5"{{.*}}"Prop6"{{.*}} %{{.*}})
