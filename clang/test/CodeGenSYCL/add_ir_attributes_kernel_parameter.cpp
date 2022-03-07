
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

class __attribute__((sycl_special_class)) h {
public:
  int *x;

  h() : x(nullptr) {}
  h(int *_x) : x(_x) {}

#ifdef __SYCL_DEVICE_ONLY__
  void __init(
      [[__sycl_detail__::add_ir_attributes_kernel_parameter(
          "Prop11", "Prop12", "Prop13", "Prop14", "Prop15", "Prop16",
          "Another property string", 2, false, TestEnum::Enum1, nullptr, nullptr)]] int *_x) {
    x = _x;
  }
#endif
};

template <typename... Properties> class __attribute__((sycl_special_class)) gh {
public:
  int *x;

  gh() : x(nullptr) {}
  gh(int *_x) : x(_x) {}

#ifdef __SYCL_DEVICE_ONLY__
  void __init(
      [[__sycl_detail__::add_ir_attributes_kernel_parameter(
          Properties::name..., "Prop11", "Prop12", "Prop13", "Prop14", "Prop15", "Prop16",
          Properties::value..., "Another property string", 2, false, TestEnum::Enum1, nullptr, nullptr)]] int *_x) {
    x = _x;
  }
#endif
};

template <typename... Properties> class __attribute__((sycl_special_class)) hg {
public:
  int *x;

  hg() : x(nullptr) {}
  hg(int *_x) : x(_x) {}

#ifdef __SYCL_DEVICE_ONLY__
  void __init(
      [[__sycl_detail__::add_ir_attributes_kernel_parameter(
          "Prop11", "Prop12", "Prop13", "Prop14", "Prop15", "Prop16", Properties::name...,
          "Another property string", 2, false, TestEnum::Enum1, nullptr, nullptr, Properties::value...)]] int *_x) {
    x = _x;
  }
#endif
};

int main() {
  sycl::queue q;
  g<prop1, prop2, prop3, prop4, prop5, prop6> a;
  q.submit([&](sycl::handler &h) {
    h.single_task<class test_kernel1>(
        [=]() {
          (void)a;
        });
  });
  h b;
  q.submit([&](sycl::handler &h) {
    h.single_task<class test_kernel2>(
        [=]() {
          (void)b;
        });
  });
  gh<prop1, prop2, prop3, prop4, prop5, prop6> c;
  q.submit([&](sycl::handler &h) {
    h.single_task<class test_kernel3>(
        [=]() {
          (void)c;
        });
  });
  hg<prop1, prop2, prop3, prop4, prop5, prop6> d;
  q.submit([&](sycl::handler &h) {
    h.single_task<class test_kernel4>(
        [=]() {
          (void)d;
        });
  });
}

// CHECK-DAG: define {{.*}}spir_kernel void @{{.*}}test_kernel1({{.*}}i32 addrspace({{.*}})* {{.*}}"Prop1"="Property string"{{.*}}"Prop2"="1"{{.*}}"Prop3"="true"{{.*}}"Prop4"="2"{{.*}}"Prop5"{{.*}}"Prop6"{{.*}} %{{.*}})
// CHECK-DAG: define {{.*}}spir_kernel void @{{.*}}test_kernel2({{.*}}i32 addrspace({{.*}})* {{.*}}"Prop11"="Another property string"{{.*}}"Prop12"="2"{{.*}}"Prop13"="false"{{.*}}"Prop14"="1"{{.*}}"Prop15"{{.*}}"Prop16"{{.*}} %{{.*}})
// CHECK-DAG: define {{.*}}spir_kernel void @{{.*}}test_kernel3({{.*}}i32 addrspace({{.*}})* {{.*}}"Prop1"="Property string"{{.*}}"Prop11"="Another property string"{{.*}}"Prop12"="2"{{.*}}"Prop13"="false"{{.*}}"Prop14"="1"{{.*}}"Prop15"{{.*}}"Prop16"{{.*}}"Prop2"="1"{{.*}}"Prop3"="true"{{.*}}"Prop4"="2"{{.*}}"Prop5"{{.*}}"Prop6"{{.*}} %{{.*}})
// CHECK-DAG: define {{.*}}spir_kernel void @{{.*}}test_kernel4({{.*}}i32 addrspace({{.*}})* {{.*}}"Prop1"="Property string"{{.*}}"Prop11"="Another property string"{{.*}}"Prop12"="2"{{.*}}"Prop13"="false"{{.*}}"Prop14"="1"{{.*}}"Prop15"{{.*}}"Prop16"{{.*}}"Prop2"="1"{{.*}}"Prop3"="true"{{.*}}"Prop4"="2"{{.*}}"Prop5"{{.*}}"Prop6"{{.*}} %{{.*}})
