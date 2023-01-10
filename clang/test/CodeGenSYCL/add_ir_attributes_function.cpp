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
free_func1() {
}

#ifdef __SYCL_DEVICE_ONLY__
[[__sycl_detail__::add_ir_attributes_function(
    "Prop11", "Prop12", "Prop13", "Prop14", "Prop15", "Prop16", "Prop17", "Prop18",
    "Another property string", 2, false, TestEnum::Enum1, nullptr, nullptr, ScopedTestEnum::ScopedEnum2, PropertyValue8)]]
#endif
void
free_func2() {
}

template <typename... Properties>
#ifdef __SYCL_DEVICE_ONLY__
[[__sycl_detail__::add_ir_attributes_function(
    Properties::name..., "Prop11", "Prop12", "Prop13", "Prop14", "Prop15", "Prop16", "Prop17", "Prop18",
    Properties::value..., "Another property string", 2, false, TestEnum::Enum1, nullptr, nullptr, ScopedTestEnum::ScopedEnum2, PropertyValue8)]]
#endif
void
free_func3() {
}

template <typename... Properties>
#ifdef __SYCL_DEVICE_ONLY__
[[__sycl_detail__::add_ir_attributes_function(
    "Prop11", "Prop12", "Prop13", "Prop14", "Prop15", "Prop16", "Prop17", "Prop18", Properties::name...,
    "Another property string", 2, false, TestEnum::Enum1, nullptr, nullptr, ScopedTestEnum::ScopedEnum2, PropertyValue8, Properties::value...)]]
#endif
void
free_func4() {
}

template <typename... Properties>
#ifdef __SYCL_DEVICE_ONLY__
[[__sycl_detail__::add_ir_attributes_function(
    "", "", "", "", "", "", "", "",
    "Another property string", 2, false, TestEnum::Enum1, nullptr, nullptr, ScopedTestEnum::ScopedEnum2, PropertyValue8)]]
#endif
void
free_func5() {
}

template <typename... Properties>
#ifdef __SYCL_DEVICE_ONLY__
[[__sycl_detail__::add_ir_attributes_function(
    "", "Prop12", "", "", "", "Prop16", "Prop17", "Prop18",
    "Another property string", 2, false, TestEnum::Enum1, nullptr, nullptr, ScopedTestEnum::ScopedEnum2, PropertyValue8)]]
#endif
void
free_func6() {
}

template <typename... Properties>
class KernelFunctor1 {
public:
#ifdef __SYCL_DEVICE_ONLY__
  [[__sycl_detail__::add_ir_attributes_function(
      Properties::name..., Properties::value...)]]
#endif
  void
  operator()() const {
    free_func1<Properties...>();
  }
};

class KernelFunctor2 {
public:
#ifdef __SYCL_DEVICE_ONLY__
  [[__sycl_detail__::add_ir_attributes_function(
      "Prop11", "Prop12", "Prop13", "Prop14", "Prop15", "Prop16", "Prop17", "Prop18",
      "Another property string", 2, false, TestEnum::Enum1, nullptr, nullptr, ScopedTestEnum::ScopedEnum2, PropertyValue8)]]
#endif
  void
  operator()() const {
    free_func2();
  }
};

template <typename... Properties>
class KernelFunctor3 {
public:
#ifdef __SYCL_DEVICE_ONLY__
  [[__sycl_detail__::add_ir_attributes_function(
      Properties::name..., "Prop11", "Prop12", "Prop13", "Prop14", "Prop15", "Prop16", "Prop17", "Prop18",
      Properties::value..., "Another property string", 2, false, TestEnum::Enum1, nullptr, nullptr, ScopedTestEnum::ScopedEnum2, PropertyValue8)]]
#endif
  void
  operator()() const {
    free_func3<Properties...>();
  }
};

template <typename... Properties>
class KernelFunctor4 {
public:
#ifdef __SYCL_DEVICE_ONLY__
  [[__sycl_detail__::add_ir_attributes_function(
      "Prop11", "Prop12", "Prop13", "Prop14", "Prop15", "Prop16", "Prop17", "Prop18", Properties::name...,
      "Another property string", 2, false, TestEnum::Enum1, nullptr, nullptr, ScopedTestEnum::ScopedEnum2, PropertyValue8, Properties::value...)]]
#endif
  void
  operator()() const {
    free_func4<Properties...>();
  }
};

class KernelFunctor5 {
public:
#ifdef __SYCL_DEVICE_ONLY__
  [[__sycl_detail__::add_ir_attributes_function(
      "", "", "", "", "", "", "", "",
      "Another property string", 2, false, TestEnum::Enum1, nullptr, nullptr, ScopedTestEnum::ScopedEnum2, PropertyValue8)]]
#endif
  void
  operator()() const {
    free_func5();
  }
};

class KernelFunctor6 {
public:
#ifdef __SYCL_DEVICE_ONLY__
  [[__sycl_detail__::add_ir_attributes_function(
      "", "Prop12", "", "", "", "Prop16", "Prop17", "Prop18",
      "Another property string", 2, false, TestEnum::Enum1, nullptr, nullptr, ScopedTestEnum::ScopedEnum2, PropertyValue8)]]
#endif
  void
  operator()() const {
    free_func6();
  }
};

int main() {
  sycl::queue q;
  q.submit([&](sycl::handler &h) {
    KernelFunctor1<prop1, prop2, prop3, prop4, prop5, prop6, prop7, prop8> f{};
    h.single_task<class test_kernel1>(f);
  });
  q.submit([&](sycl::handler &h) {
    KernelFunctor2 f{};
    h.single_task<class test_kernel2>(f);
  });
  q.submit([&](sycl::handler &h) {
    KernelFunctor3<prop1, prop2, prop3, prop4, prop5, prop6, prop7, prop8> f{};
    h.single_task<class test_kernel3>(f);
  });
  q.submit([&](sycl::handler &h) {
    KernelFunctor4<prop1, prop2, prop3, prop4, prop5, prop6, prop7, prop8> f{};
    h.single_task<class test_kernel4>(f);
  });
  q.submit([&](sycl::handler &h) {
    KernelFunctor5 f{};
    h.single_task<class test_kernel5>(f);
  });
  q.submit([&](sycl::handler &h) {
    KernelFunctor6 f{};
    h.single_task<class test_kernel6>(f);
  });
}

// CHECK-DAG: define {{.*}}spir_kernel void @{{.*}}test_kernel1() #[[KernFunc1Attrs:[0-9]+]]
// CHECK-DAG: define {{.*}}spir_kernel void @{{.*}}test_kernel2() #[[KernFunc2Attrs:[0-9]+]]
// CHECK-DAG: define {{.*}}spir_kernel void @{{.*}}test_kernel3() #[[KernFunc3And4Attrs:[0-9]+]]
// CHECK-DAG: define {{.*}}spir_kernel void @{{.*}}test_kernel4() #[[KernFunc3And4Attrs]]
// CHECK-DAG: define {{.*}}spir_kernel void @{{.*}}test_kernel5() {{.*}}
// CHECK-DAG: define {{.*}}spir_kernel void @{{.*}}test_kernel6() #[[KernFunc6Attrs:[0-9]+]]
// CHECK-DAG: define {{.*}}spir_func void @{{.*}}free_func1{{.*}}() #[[Func1Attrs:[0-9]+]]
// CHECK-DAG: define {{.*}}spir_func void @{{.*}}free_func2{{.*}}() #[[Func2Attrs:[0-9]+]]
// CHECK-DAG: define {{.*}}spir_func void @{{.*}}free_func3{{.*}}() #[[Func3and4Attrs:[0-9]+]]
// CHECK-DAG: define {{.*}}spir_func void @{{.*}}free_func4{{.*}}() #[[Func3and4Attrs]]
// CHECK-DAG: define {{.*}}spir_func void @{{.*}}free_func5{{.*}}() {{.*}}
// CHECK-DAG: define {{.*}}spir_func void @{{.*}}free_func6{{.*}}() #[[Func6Attrs:[0-9]+]]
// CHECK-DAG: attributes #[[KernFunc1Attrs]] = { {{.*}}"Prop1"="Property string"{{.*}}"Prop2"="1"{{.*}}"Prop3"="true"{{.*}}"Prop4"="2"{{.*}}"Prop5"{{.*}}"Prop6"{{.*}}"Prop7"="1"{{.*}}"Prop8"="Property"{{.*}} }
// CHECK-DAG: attributes #[[KernFunc2Attrs]] = { {{.*}}"Prop11"="Another property string"{{.*}}"Prop12"="2"{{.*}}"Prop13"="false"{{.*}}"Prop14"="1"{{.*}}"Prop15"{{.*}}"Prop16"{{.*}}"Prop17"="2"{{.*}}"Prop18"="Property"{{.*}} }
// CHECK-DAG: attributes #[[KernFunc3And4Attrs]] = { {{.*}}"Prop1"="Property string"{{.*}}"Prop11"="Another property string"{{.*}}"Prop12"="2"{{.*}}"Prop13"="false"{{.*}}"Prop14"="1"{{.*}}"Prop15"{{.*}}"Prop16"{{.*}}"Prop17"="2"{{.*}}"Prop18"="Property"{{.*}}"Prop2"="1"{{.*}}"Prop3"="true"{{.*}}"Prop4"="2"{{.*}}"Prop5"{{.*}}"Prop6"{{.*}}"Prop7"="1"{{.*}}"Prop8"="Property"{{.*}} }
// CHECK-DAG: attributes #[[KernFunc6Attrs]] = { {{.*}}"Prop12"="2"{{.*}}"Prop16"{{.*}}"Prop17"="2"{{.*}}"Prop18"="Property"{{.*}} }
// CHECK-DAG: attributes #[[Func1Attrs]] = { {{.*}}"Prop1"="Property string"{{.*}}"Prop2"="1"{{.*}}"Prop3"="true"{{.*}}"Prop4"="2"{{.*}}"Prop5"{{.*}}"Prop6"{{.*}}"Prop7"="1"{{.*}}"Prop8"="Property"{{.*}} }
// CHECK-DAG: attributes #[[Func2Attrs]] = { {{.*}}"Prop11"="Another property string"{{.*}}"Prop12"="2"{{.*}}"Prop13"="false"{{.*}}"Prop14"="1"{{.*}}"Prop15"{{.*}}"Prop16"{{.*}}"Prop17"="2"{{.*}}"Prop18"="Property"{{.*}} }
// CHECK-DAG: attributes #[[Func3and4Attrs]] = { {{.*}}"Prop1"="Property string"{{.*}}"Prop11"="Another property string"{{.*}}"Prop12"="2"{{.*}}"Prop13"="false"{{.*}}"Prop14"="1"{{.*}}"Prop15"{{.*}}"Prop16"{{.*}}"Prop17"="2"{{.*}}"Prop18"="Property"{{.*}}"Prop2"="1"{{.*}}"Prop3"="true"{{.*}}"Prop4"="2"{{.*}}"Prop5"{{.*}}"Prop6"{{.*}}"Prop7"="1"{{.*}}"Prop8"="Property"{{.*}} }
// CHECK-DAG: attributes #[[Func6Attrs]] = { {{.*}}"Prop12"="2"{{.*}}"Prop16"{{.*}}"Prop17"="2"{{.*}}"Prop18"="Property"{{.*}} }
// CHECK-NOT: ""="Another property string"
// CHECK-NOT: ""="1"
// CHECK-NOT: ""="2"
// CHECK-NOT: ""="false"
