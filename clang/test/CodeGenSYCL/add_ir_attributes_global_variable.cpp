// RUN: %clang_cc1 -internal-isystem %S/Inputs -disable-llvm-passes \
// RUN:    -triple spir64-unknown-unknown -fsycl-is-device -S \
// RUN:    -emit-llvm %s -o - | FileCheck %s

// Tests the generation of IR attributes when using
// __sycl_detail__::add_ir_attributes_global_variable

#include "mock_properties.hpp"
#include "sycl.hpp"

template <typename... NameValues> struct
#ifdef __SYCL_DEVICE_ONLY__
    [[__sycl_detail__::add_ir_attributes_global_variable(NameValues::name...,
                                                         NameValues::value...)]]
#endif
    g {
  int x;

  constexpr g() : x(1) {}
  constexpr g(int _x) : x(_x) {}
};

struct
#ifdef __SYCL_DEVICE_ONLY__
    [[__sycl_detail__::add_ir_attributes_global_variable(
        "Prop11", "Prop12", "Prop13", "Prop14", "Prop15", "Prop16",
        "Another property string", 2, false, TestEnum::Enum1, nullptr, nullptr)]]
#endif
    h {
  int x;

  constexpr h() : x(1) {}
  constexpr h(int _x) : x(_x) {}
};

template <typename... NameValues> struct
#ifdef __SYCL_DEVICE_ONLY__
    [[__sycl_detail__::add_ir_attributes_global_variable(
        NameValues::name..., "Prop11", "Prop12", "Prop13", "Prop14", "Prop15", "Prop16",
        NameValues::value..., "Another property string", 2, false, TestEnum::Enum1, nullptr, nullptr)]]
#endif
    gh {
  int x;

  constexpr gh() : x(1) {}
  constexpr gh(int _x) : x(_x) {}
};

template <typename... NameValues> struct
#ifdef __SYCL_DEVICE_ONLY__
    [[__sycl_detail__::add_ir_attributes_global_variable(
        "Prop11", "Prop12", "Prop13", "Prop14", "Prop15", "Prop16", NameValues::name...,
        "Another property string", 2, false, TestEnum::Enum1, nullptr, nullptr, NameValues::value...)]]
#endif
    hg {
  int x;

  constexpr hg() : x(1) {}
  constexpr hg(int _x) : x(_x) {}
};

constexpr g<prop1, prop2, prop3, prop4, prop5, prop6> g_v;
constexpr h h_v;
constexpr gh<prop1, prop2, prop3, prop4, prop5, prop6> gh_v;
constexpr hg<prop1, prop2, prop3, prop4, prop5, prop6> hg_v;

int main() {
  sycl::queue q;
  q.submit([&](sycl::handler &h) {
    h.single_task<class test_kernel>(
        [=]() {
          (void)g_v.x;
          (void)h_v.x;
          (void)gh_v.x;
          (void)hg_v.x;
        });
  });
}

// CHECK-DAG: @_ZL3g_v = internal addrspace(1) constant %struct.g { {{.*}} }, {{.*}} #[[GlobalVarGAttrs:[0-9]+]]
// CHECK-DAG: @_ZL3h_v = internal addrspace(1) constant %struct.h { {{.*}} }, {{.*}} #[[GlobalVarHAttrs:[0-9]+]]
// CHECK-DAG: @_ZL4gh_v = internal addrspace(1) constant %struct.gh { {{.*}} }, {{.*}} #[[GlobalVarHGAndGHAttrs:[0-9]+]]
// CHECK-DAG: @_ZL4hg_v = internal addrspace(1) constant %struct.hg { {{.*}} }, {{.*}} #[[GlobalVarHGAndGHAttrs]]
// CHECK-DAG: attributes #[[GlobalVarGAttrs]] = { {{.*}}"Prop1"="Property string"{{.*}}"Prop2"="1"{{.*}}"Prop3"="true"{{.*}}"Prop4"="2"{{.*}}"Prop5"{{.*}}"Prop6"{{.*}} }
// CHECK-DAG: attributes #[[GlobalVarHAttrs]] = { {{.*}}"Prop11"="Another property string"{{.*}}"Prop12"="2"{{.*}}"Prop13"="false"{{.*}}"Prop14"="1"{{.*}}"Prop15"{{.*}}"Prop16"{{.*}} }
// CHECK-DAG: attributes #[[GlobalVarHGAndGHAttrs]] = { {{.*}}"Prop1"="Property string"{{.*}}"Prop11"="Another property string"{{.*}}"Prop12"="2"{{.*}}"Prop13"="false"{{.*}}"Prop14"="1"{{.*}}"Prop15"{{.*}}"Prop16"{{.*}}"Prop2"="1"{{.*}}"Prop3"="true"{{.*}}"Prop4"="2"{{.*}}"Prop5"{{.*}}"Prop6"{{.*}} }
