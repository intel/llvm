// RUN: %clang_cc1 -internal-isystem %S/Inputs -disable-llvm-passes \
// RUN:    -triple spir64-unknown-unknown -fsycl-is-device -S \
// RUN:    -opaque-pointers -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -internal-isystem %S/Inputs -disable-llvm-passes \
// RUN:    -triple spir64-unknown-unknown -fsycl-is-device -DTEST_SCALAR -S \
// RUN:    -opaque-pointers -emit-llvm %s -o - | FileCheck %s

// Tests the generation of IR annotation calls from
// __sycl_detail__::add_ir_annotations_member attributes.

#include "mock_properties.hpp"
#include "sycl.hpp"

#ifdef TEST_SCALAR
#define TEST_T char
#else
#define TEST_T int *
#endif

template <typename... Properties> class g {
public:
  TEST_T x
#ifdef __SYCL_DEVICE_ONLY__
      [[__sycl_detail__::add_ir_annotations_member(Properties::name..., Properties::value...)]]
#endif
      ;

  g() : x() {}
  g(TEST_T _x) : x(_x) {}
};

class h {
public:
  TEST_T x
#ifdef __SYCL_DEVICE_ONLY__
      [[__sycl_detail__::add_ir_annotations_member(
          "Prop11", "Prop12", "Prop13", "Prop14", "Prop15", "Prop16", "Prop17", "Prop18",
          "Another property string", 2, false, TestEnum::Enum1, nullptr, nullptr, ScopedTestEnum::ScopedEnum2, PropertyValue8)]]
#endif
      ;

  h() : x() {}
  h(TEST_T _x) : x(_x) {}
};

template <typename... Properties> class gh {
public:
  TEST_T x
#ifdef __SYCL_DEVICE_ONLY__
      [[__sycl_detail__::add_ir_annotations_member(
          Properties::name..., "Prop11", "Prop12", "Prop13", "Prop14", "Prop15", "Prop16", "Prop17", "Prop18",
          Properties::value..., "Another property string", 2, false, TestEnum::Enum1, nullptr, nullptr, ScopedTestEnum::ScopedEnum2, PropertyValue8)]]
#endif
      ;

  gh() : x() {}
  gh(TEST_T _x) : x(_x) {}
};

template <typename... Properties> class hg {
public:
  TEST_T x
#ifdef __SYCL_DEVICE_ONLY__
      [[__sycl_detail__::add_ir_annotations_member(
          "Prop11", "Prop12", "Prop13", "Prop14", "Prop15", "Prop16", "Prop17", "Prop18", Properties::name...,
          "Another property string", 2, false, TestEnum::Enum1, nullptr, nullptr, ScopedTestEnum::ScopedEnum2, PropertyValue8, Properties::value...)]]
#endif
      ;

  hg() : x() {}
  hg(TEST_T _x) : x(_x) {}
};

int main() {
  sycl::queue q;
  g<prop1, prop2, prop3, prop4, prop5, prop6, prop7, prop8> a;
  q.submit([&](sycl::handler &h) {
    h.single_task<class test_kernel1>(
        [=]() {
          (void)a.x;
        });
  });
  h b;
  q.submit([&](sycl::handler &h) {
    h.single_task<class test_kernel2>(
        [=]() {
          (void)b.x;
        });
  });
  gh<prop1, prop2, prop3, prop4, prop5, prop6, prop7, prop8> c;
  q.submit([&](sycl::handler &h) {
    h.single_task<class test_kernel3>(
        [=]() {
          (void)c.x;
        });
  });
  hg<prop1, prop2, prop3, prop4, prop5, prop6, prop7, prop8> d;
  q.submit([&](sycl::handler &h) {
    h.single_task<class test_kernel4>(
        [=]() {
          (void)d.x;
        });
  });
}

// CHECK-DAG: @[[AnnotName:.*]] = private unnamed_addr addrspace(1) constant [16 x i8] c"sycl-properties\00", section "llvm.metadata"

// CHECK-DAG: @[[Prop1Name:.*]] = private unnamed_addr addrspace(1) constant [6 x i8] c"Prop1\00", section "llvm.metadata"
// CHECK-DAG: @[[Prop2Name:.*]] = private unnamed_addr addrspace(1) constant [6 x i8] c"Prop2\00", section "llvm.metadata"
// CHECK-DAG: @[[Prop3Name:.*]] = private unnamed_addr addrspace(1) constant [6 x i8] c"Prop3\00", section "llvm.metadata"
// CHECK-DAG: @[[Prop4Name:.*]] = private unnamed_addr addrspace(1) constant [6 x i8] c"Prop4\00", section "llvm.metadata"
// CHECK-DAG: @[[Prop5Name:.*]] = private unnamed_addr addrspace(1) constant [6 x i8] c"Prop5\00", section "llvm.metadata"
// CHECK-DAG: @[[Prop6Name:.*]] = private unnamed_addr addrspace(1) constant [6 x i8] c"Prop6\00", section "llvm.metadata"
// CHECK-DAG: @[[Prop7Name:.*]] = private unnamed_addr addrspace(1) constant [6 x i8] c"Prop7\00", section "llvm.metadata"
// CHECK-DAG: @[[Prop8Name:.*]] = private unnamed_addr addrspace(1) constant [6 x i8] c"Prop8\00", section "llvm.metadata"
// CHECK-DAG: @[[Prop11Name:.*]] = private unnamed_addr addrspace(1) constant [7 x i8] c"Prop11\00", section "llvm.metadata"
// CHECK-DAG: @[[Prop12Name:.*]] = private unnamed_addr addrspace(1) constant [7 x i8] c"Prop12\00", section "llvm.metadata"
// CHECK-DAG: @[[Prop13Name:.*]] = private unnamed_addr addrspace(1) constant [7 x i8] c"Prop13\00", section "llvm.metadata"
// CHECK-DAG: @[[Prop14Name:.*]] = private unnamed_addr addrspace(1) constant [7 x i8] c"Prop14\00", section "llvm.metadata"
// CHECK-DAG: @[[Prop15Name:.*]] = private unnamed_addr addrspace(1) constant [7 x i8] c"Prop15\00", section "llvm.metadata"
// CHECK-DAG: @[[Prop16Name:.*]] = private unnamed_addr addrspace(1) constant [7 x i8] c"Prop16\00", section "llvm.metadata"
// CHECK-DAG: @[[Prop17Name:.*]] = private unnamed_addr addrspace(1) constant [7 x i8] c"Prop17\00", section "llvm.metadata"
// CHECK-DAG: @[[Prop18Name:.*]] = private unnamed_addr addrspace(1) constant [7 x i8] c"Prop18\00", section "llvm.metadata"

// CHECK-DAG: @[[Prop1Value:.*]] = private unnamed_addr addrspace(1) constant [16 x i8] c"Property string\00", section "llvm.metadata"
// CHECK-DAG: @[[Prop2_7_14Value:.*]] = private unnamed_addr addrspace(1) constant [2 x i8] c"1\00", section "llvm.metadata"
// CHECK-DAG: @[[Prop3Value:.*]] = private unnamed_addr addrspace(1) constant [5 x i8] c"true\00", section "llvm.metadata"
// CHECK-DAG: @[[Prop4_12_17Value:.*]] = private unnamed_addr addrspace(1) constant [2 x i8] c"2\00", section "llvm.metadata"
// CHECK-DAG: @[[Prop8_18Value:.*]] = private unnamed_addr addrspace(1) constant [9 x i8] c"Property\00", section "llvm.metadata"
// CHECK-DAG: @[[Prop11Value:.*]] = private unnamed_addr addrspace(1) constant [24 x i8] c"Another property string\00", section "llvm.metadata"
// CHECK-DAG: @[[Prop13Value:.*]] = private unnamed_addr addrspace(1) constant [6 x i8] c"false\00", section "llvm.metadata"

// CHECK-DAG: @[[GArgs:.*]] = private unnamed_addr addrspace(1) constant { ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1) } { ptr addrspace(1) @[[Prop1Name]], ptr addrspace(1) @[[Prop1Value]], ptr addrspace(1) @[[Prop2Name]], ptr addrspace(1) @[[Prop2_7_14Value]], ptr addrspace(1) @[[Prop3Name]], ptr addrspace(1) @[[Prop3Value]], ptr addrspace(1) @[[Prop4Name]], ptr addrspace(1) @[[Prop4_12_17Value]], ptr addrspace(1) @[[Prop5Name]], ptr addrspace(1) null, ptr addrspace(1) @[[Prop6Name]], ptr addrspace(1) null, ptr addrspace(1) @[[Prop7Name]], ptr addrspace(1) @[[Prop2_7_14Value]], ptr addrspace(1) @[[Prop8Name]], ptr addrspace(1) @[[Prop8_18Value]] }, section "llvm.metadata"
// CHECK-DAG: @[[HArgs:.*]] = private unnamed_addr addrspace(1) constant { ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1) } { ptr addrspace(1) @[[Prop11Name]], ptr addrspace(1) @[[Prop11Value]], ptr addrspace(1) @[[Prop12Name]], ptr addrspace(1) @[[Prop4_12_17Value]], ptr addrspace(1) @[[Prop13Name]], ptr addrspace(1) @[[Prop13Value]], ptr addrspace(1) @[[Prop14Name]], ptr addrspace(1) @[[Prop2_7_14Value]], ptr addrspace(1) @[[Prop15Name]], ptr addrspace(1) null, ptr addrspace(1) @[[Prop16Name]], ptr addrspace(1) null, ptr addrspace(1) @[[Prop17Name]], ptr addrspace(1) @[[Prop4_12_17Value]], ptr addrspace(1) @[[Prop18Name]], ptr addrspace(1) @[[Prop8_18Value]] }, section "llvm.metadata"
// CHECK-DAG: @[[GHArgs:.*]] = private unnamed_addr addrspace(1) constant { ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1) } { ptr addrspace(1) @[[Prop1Name]], ptr addrspace(1) @[[Prop1Value]], ptr addrspace(1) @[[Prop2Name]], ptr addrspace(1) @[[Prop2_7_14Value]], ptr addrspace(1) @[[Prop3Name]], ptr addrspace(1) @[[Prop3Value]], ptr addrspace(1) @[[Prop4Name]], ptr addrspace(1) @[[Prop4_12_17Value]], ptr addrspace(1) @[[Prop5Name]], ptr addrspace(1) null, ptr addrspace(1) @[[Prop6Name]], ptr addrspace(1) null, ptr addrspace(1) @[[Prop7Name]], ptr addrspace(1) @[[Prop2_7_14Value]], ptr addrspace(1) @[[Prop8Name]], ptr addrspace(1) @[[Prop8_18Value]], ptr addrspace(1) @[[Prop11Name]], ptr addrspace(1) @[[Prop11Value]], ptr addrspace(1) @[[Prop12Name]], ptr addrspace(1) @[[Prop4_12_17Value]], ptr addrspace(1) @[[Prop13Name]], ptr addrspace(1) @[[Prop13Value]], ptr addrspace(1) @[[Prop14Name]], ptr addrspace(1) @[[Prop2_7_14Value]], ptr addrspace(1) @[[Prop15Name]], ptr addrspace(1) null, ptr addrspace(1) @[[Prop16Name]], ptr addrspace(1) null, ptr addrspace(1) @[[Prop17Name]], ptr addrspace(1) @[[Prop4_12_17Value]], ptr addrspace(1) @[[Prop18Name]], ptr addrspace(1) @[[Prop8_18Value]] }, section "llvm.metadata"
// CHECK-DAG: @[[HGArgs:.*]] = private unnamed_addr addrspace(1) constant { ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1) } { ptr addrspace(1) @[[Prop11Name]], ptr addrspace(1) @[[Prop11Value]], ptr addrspace(1) @[[Prop12Name]], ptr addrspace(1) @[[Prop4_12_17Value]], ptr addrspace(1) @[[Prop13Name]], ptr addrspace(1) @[[Prop13Value]], ptr addrspace(1) @[[Prop14Name]], ptr addrspace(1) @[[Prop2_7_14Value]], ptr addrspace(1) @[[Prop15Name]], ptr addrspace(1) null, ptr addrspace(1) @[[Prop16Name]], ptr addrspace(1) null, ptr addrspace(1) @[[Prop17Name]], ptr addrspace(1) @[[Prop4_12_17Value]], ptr addrspace(1) @[[Prop18Name]], ptr addrspace(1) @[[Prop8_18Value]], ptr addrspace(1) @[[Prop1Name]], ptr addrspace(1) @[[Prop1Value]], ptr addrspace(1) @[[Prop2Name]], ptr addrspace(1) @[[Prop2_7_14Value]], ptr addrspace(1) @[[Prop3Name]], ptr addrspace(1) @[[Prop3Value]], ptr addrspace(1) @[[Prop4Name]], ptr addrspace(1) @[[Prop4_12_17Value]], ptr addrspace(1) @[[Prop5Name]], ptr addrspace(1) null, ptr addrspace(1) @[[Prop6Name]], ptr addrspace(1) null, ptr addrspace(1) @[[Prop7Name]], ptr addrspace(1) @[[Prop2_7_14Value]], ptr addrspace(1) @[[Prop8Name]], ptr addrspace(1) @[[Prop8_18Value]] }, section "llvm.metadata"

// CHECK-DAG: %{{.*}} = call ptr addrspace(4) @llvm.ptr.annotation.p4.p1(ptr {{.*}}, ptr addrspace(1) @[[AnnotName]], {{.*}}, i32 {{.*}}, ptr addrspace(1) @[[GArgs]])
// CHECK-DAG: %{{.*}} = call ptr addrspace(4) @llvm.ptr.annotation.p4.p1(ptr {{.*}}, ptr addrspace(1) @[[AnnotName]], {{.*}}, i32 {{.*}}, ptr addrspace(1) @[[HArgs]])
// CHECK-DAG: %{{.*}} = call ptr addrspace(4) @llvm.ptr.annotation.p4.p1(ptr {{.*}}, ptr addrspace(1) @[[AnnotName]], {{.*}}, i32 {{.*}}, ptr addrspace(1) @[[GHArgs]])
// CHECK-DAG: %{{.*}} = call ptr addrspace(4) @llvm.ptr.annotation.p4.p1(ptr {{.*}}, ptr addrspace(1) @[[AnnotName]], {{.*}}, i32 {{.*}}, ptr addrspace(1) @[[HGArgs]])
