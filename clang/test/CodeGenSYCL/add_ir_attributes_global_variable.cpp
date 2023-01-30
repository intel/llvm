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
        "Prop11", "Prop12", "Prop13", "Prop14", "Prop15", "Prop16", "Prop17", "Prop18",
        "Another property string", 2, false, TestEnum::Enum1, nullptr, nullptr, ScopedTestEnum::ScopedEnum2, PropertyValue8)]]
#endif
    h {
  int x;

  constexpr h() : x(1) {}
  constexpr h(int _x) : x(_x) {}
};

template <typename... NameValues> struct
#ifdef __SYCL_DEVICE_ONLY__
    [[__sycl_detail__::add_ir_attributes_global_variable(
        NameValues::name..., "Prop11", "Prop12", "Prop13", "Prop14", "Prop15", "Prop16", "Prop17", "Prop18",
        NameValues::value..., "Another property string", 2, false, TestEnum::Enum1, nullptr, nullptr, ScopedTestEnum::ScopedEnum2, PropertyValue8)]]
#endif
    gh {
  int x;

  constexpr gh() : x(1) {}
  constexpr gh(int _x) : x(_x) {}
};

template <typename... NameValues> struct
#ifdef __SYCL_DEVICE_ONLY__
    [[__sycl_detail__::add_ir_attributes_global_variable(
        "Prop11", "Prop12", "Prop13", "Prop14", "Prop15", "Prop16", "Prop17", "Prop18", NameValues::name...,
        "Another property string", 2, false, TestEnum::Enum1, nullptr, nullptr, ScopedTestEnum::ScopedEnum2, PropertyValue8, NameValues::value...)]]
#endif
    hg {
  int x;

  constexpr hg() : x(1) {}
  constexpr hg(int _x) : x(_x) {}
};

struct
#ifdef __SYCL_DEVICE_ONLY__
    [[__sycl_detail__::add_ir_attributes_global_variable(
        "", "", "", "", "", "", "", "",
        "Another property string", 2, false, TestEnum::Enum1, nullptr, nullptr, ScopedTestEnum::ScopedEnum2, PropertyValue8)]]
#endif
    np {
  int x;

  constexpr np() : x(1) {}
  constexpr np(int _x) : x(_x) {}
};

struct
#ifdef __SYCL_DEVICE_ONLY__
    [[__sycl_detail__::add_ir_attributes_global_variable(
        "", "Prop12", "", "", "", "Prop16", "Prop17", "Prop18",
        "Another property string", 2, false, TestEnum::Enum1, nullptr, nullptr, ScopedTestEnum::ScopedEnum2, PropertyValue8)]]
#endif
    mp {
  int x;

  constexpr mp() : x(1) {}
  constexpr mp(int _x) : x(_x) {}
};

template <typename... NameValues> struct ig : public g<NameValues...> {};
struct ih : public h {};
template <typename... NameValues> struct igh : public gh<NameValues...> {};
template <typename... NameValues> struct ihg : public hg<NameValues...> {};

constexpr g<prop1, prop2, prop3, prop4, prop5, prop6, prop7, prop8> g_v;
constexpr h h_v;
constexpr gh<prop1, prop2, prop3, prop4, prop5, prop6, prop7, prop8> gh_v;
constexpr hg<prop1, prop2, prop3, prop4, prop5, prop6, prop7, prop8> hg_v;

constexpr np np_v;
constexpr mp mp_v;

constexpr ig<prop1, prop2, prop3, prop4, prop5, prop6, prop7, prop8> ig_v;
constexpr ih ih_v;
constexpr igh<prop1, prop2, prop3, prop4, prop5, prop6, prop7, prop8> igh_v;
constexpr ihg<prop1, prop2, prop3, prop4, prop5, prop6, prop7, prop8> ihg_v;

int main() {
  sycl::queue q;
  q.submit([&](sycl::handler &h) {
    h.single_task<class test_kernel>(
        [=]() {
          (void)g_v.x;
          (void)h_v.x;
          (void)gh_v.x;
          (void)hg_v.x;
          (void)np_v.x;
          (void)mp_v.x;
          (void)ig_v.x;
          (void)ih_v.x;
          (void)igh_v.x;
          (void)ihg_v.x;
        });
  });
}

// CHECK-DAG: @_ZL3g_v = internal addrspace(1) constant %struct.g { {{.*}} }, {{.*}} #[[GlobalVarGAttrs:[0-9]+]]
// CHECK-DAG: @_ZL3h_v = internal addrspace(1) constant %struct.h { {{.*}} }, {{.*}} #[[GlobalVarHAttrs:[0-9]+]]
// CHECK-DAG: @_ZL4gh_v = internal addrspace(1) constant %struct.gh { {{.*}} }, {{.*}} #[[GlobalVarHGAndGHAttrs:[0-9]+]]
// CHECK-DAG: @_ZL4hg_v = internal addrspace(1) constant %struct.hg { {{.*}} }, {{.*}} #[[GlobalVarHGAndGHAttrs]]
// CHECK-DAG: @_ZL4np_v = internal addrspace(1) constant {{.*}}, align 4{{$}}
// CHECK-DAG: @_ZL4mp_v = internal addrspace(1) constant %struct.mp { {{.*}} }, {{.*}} #[[GlobalVarMPAttrs:[0-9]+]]
// CHECK-DAG: @_ZL4ig_v = internal addrspace(1) constant {{.*}}, align 4{{$}}
// CHECK-DAG: @_ZL4ih_v = internal addrspace(1) constant {{.*}}, align 4{{$}}
// CHECK-DAG: @_ZL5igh_v = internal addrspace(1) constant {{.*}}, align 4{{$}}
// CHECK-DAG: @_ZL5ihg_v = internal addrspace(1) constant {{.*}}, align 4{{$}}
// CHECK-DAG: attributes #[[GlobalVarGAttrs]] = { {{.*}}"Prop1"="Property string"{{.*}}"Prop2"="1"{{.*}}"Prop3"="true"{{.*}}"Prop4"="2"{{.*}}"Prop5"{{.*}}"Prop6"{{.*}}"Prop7"="1"{{.*}}"Prop8"="Property"{{.*}} }
// CHECK-DAG: attributes #[[GlobalVarHAttrs]] = { {{.*}}"Prop11"="Another property string"{{.*}}"Prop12"="2"{{.*}}"Prop13"="false"{{.*}}"Prop14"="1"{{.*}}"Prop15"{{.*}}"Prop16"{{.*}}"Prop17"="2"{{.*}}"Prop18"="Property"{{.*}} }
// CHECK-DAG: attributes #[[GlobalVarHGAndGHAttrs]] = { {{.*}}"Prop1"="Property string"{{.*}}"Prop11"="Another property string"{{.*}}"Prop12"="2"{{.*}}"Prop13"="false"{{.*}}"Prop14"="1"{{.*}}"Prop15"{{.*}}"Prop16"{{.*}}"Prop17"="2"{{.*}}"Prop18"="Property"{{.*}}"Prop2"="1"{{.*}}"Prop3"="true"{{.*}}"Prop4"="2"{{.*}}"Prop5"{{.*}}"Prop6"{{.*}}"Prop7"="1"{{.*}}"Prop8"="Property"{{.*}} }
// CHECK-DAG: attributes #[[GlobalVarMPAttrs]] = { {{.*}}"Prop12"="2"{{.*}}"Prop16"{{.*}}"Prop17"="2"{{.*}}"Prop18"="Property"{{.*}} }
