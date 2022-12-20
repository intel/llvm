// RUN: %clang_cc1 -internal-isystem %S/Inputs -disable-llvm-passes \
// RUN:    -triple spir64-unknown-unknown -fsycl-is-device -S \
// RUN:    -opaque-pointers -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -internal-isystem %S/Inputs -disable-llvm-passes \
// RUN:    -triple spir64-unknown-unknown -fsycl-is-device -DTEST_SCALAR -S \
// RUN:    -opaque-pointers -emit-llvm %s -o - | FileCheck %s

// Tests the reuse of generated annotation value global variables for
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
      [[__sycl_detail__::add_ir_annotations_member(
          Properties::name..., Properties::value...)]]
#endif
      ;

  g() : x() {}
  g(TEST_T _x) : x(_x) {}
};

template <typename... Properties> class h {
public:
  TEST_T x
#ifdef __SYCL_DEVICE_ONLY__
      [[__sycl_detail__::add_ir_annotations_member(
          {"Prop1", "Prop2", "Prop3"},
          Properties::name..., Properties::value...)]]
#endif
      ;

  h() : x() {}
  h(TEST_T _x) : x(_x) {}
};

int main() {
  sycl::queue q;
  g<prop1, prop2, prop3> a;
  q.submit([&](sycl::handler &h) {
    h.single_task<class test_kernel1>(
        [=]() {
          (void)a.x;
        });
  });
  g<prop1, prop2, prop3> b;
  q.submit([&](sycl::handler &h) {
    h.single_task<class test_kernel2>(
        [=]() {
          (void)b.x;
        });
  });
  h<prop1, prop2, prop4, prop3> c;
  q.submit([&](sycl::handler &h) {
    h.single_task<class test_kernel3>(
        [=]() {
          (void)c.x;
        });
  });
  g<prop1, prop2> d;
  q.submit([&](sycl::handler &h) {
    h.single_task<class test_kernel4>(
        [=]() {
          (void)d.x;
        });
  });
  g<prop1, prop2, prop3, prop5> e;
  q.submit([&](sycl::handler &h) {
    h.single_task<class test_kernel5>(
        [=]() {
          (void)e.x;
        });
  });
  g<prop3, prop2, prop1> f;
  q.submit([&](sycl::handler &h) {
    h.single_task<class test_kernel6>(
        [=]() {
          (void)f.x;
        });
  });
}

// CHECK-DAG: @[[AnnotName:.*]] = private unnamed_addr addrspace(1) constant [16 x i8] c"sycl-properties\00", section "llvm.metadata"

// CHECK-DAG: @[[Prop1Name:.*]] = private unnamed_addr addrspace(1) constant [6 x i8] c"Prop1\00", section "llvm.metadata"
// CHECK-DAG: @[[Prop2Name:.*]] = private unnamed_addr addrspace(1) constant [6 x i8] c"Prop2\00", section "llvm.metadata"
// CHECK-DAG: @[[Prop3Name:.*]] = private unnamed_addr addrspace(1) constant [6 x i8] c"Prop3\00", section "llvm.metadata"
// CHECK-DAG: @[[Prop5Name:.*]] = private unnamed_addr addrspace(1) constant [6 x i8] c"Prop5\00", section "llvm.metadata"

// CHECK-DAG: @[[Prop1Value:.*]] = private unnamed_addr addrspace(1) constant [16 x i8] c"Property string\00", section "llvm.metadata"
// CHECK-DAG: @[[Prop2Value:.*]] = private unnamed_addr addrspace(1) constant [2 x i8] c"1\00", section "llvm.metadata"
// CHECK-DAG: @[[Prop3Value:.*]] = private unnamed_addr addrspace(1) constant [5 x i8] c"true\00", section "llvm.metadata"

// CHECK-DAG: @[[ReusedArgs:.*]] = private unnamed_addr addrspace(1) constant { ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1) } { ptr addrspace(1) @[[Prop1Name]], ptr addrspace(1) @[[Prop1Value]], ptr addrspace(1) @[[Prop2Name]], ptr addrspace(1) @[[Prop2Value]], ptr addrspace(1) @[[Prop3Name]], ptr addrspace(1) @[[Prop3Value]] }, section "llvm.metadata"
// CHECK-DAG: @[[DArgs:.*]] = private unnamed_addr addrspace(1) constant { ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1) } { ptr addrspace(1) @[[Prop1Name]], ptr addrspace(1) @[[Prop1Value]], ptr addrspace(1) @[[Prop2Name]], ptr addrspace(1) @[[Prop2Value]] }, section "llvm.metadata"
// CHECK-DAG: @[[EArgs:.*]] = private unnamed_addr addrspace(1) constant { ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1) } { ptr addrspace(1) @[[Prop1Name]], ptr addrspace(1) @[[Prop1Value]], ptr addrspace(1) @[[Prop2Name]], ptr addrspace(1) @[[Prop2Value]], ptr addrspace(1) @[[Prop3Name]], ptr addrspace(1) @[[Prop3Value]], ptr addrspace(1) @[[Prop5Name]], ptr addrspace(1) null }, section "llvm.metadata"
// CHECK-DAG: @[[FArgs:.*]] = private unnamed_addr addrspace(1) constant { ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1) } { ptr addrspace(1) @[[Prop3Name]], ptr addrspace(1) @[[Prop3Value]], ptr addrspace(1) @[[Prop2Name]], ptr addrspace(1) @[[Prop2Value]], ptr addrspace(1) @[[Prop1Name]], ptr addrspace(1) @[[Prop1Value]] }, section "llvm.metadata"

// CHECK-COUNT-3: %{{.*}} = call ptr addrspace(4) @llvm.ptr.annotation.p4.p1(ptr {{.*}}, ptr addrspace(1) @[[AnnotName]], {{.*}}, i32 {{.*}}, ptr addrspace(1) @[[ReusedArgs]])
// CHECK-DAG: %{{.*}} = call ptr addrspace(4) @llvm.ptr.annotation.p4.p1(ptr {{.*}}, ptr addrspace(1) @[[AnnotName]], {{.*}}, i32 {{.*}}, ptr addrspace(1) @[[DArgs]])
// CHECK-DAG: %{{.*}} = call ptr addrspace(4) @llvm.ptr.annotation.p4.p1(ptr {{.*}}, ptr addrspace(1) @[[AnnotName]], {{.*}}, i32 {{.*}}, ptr addrspace(1) @[[EArgs]])
// CHECK-DAG: %{{.*}} = call ptr addrspace(4) @llvm.ptr.annotation.p4.p1(ptr {{.*}}, ptr addrspace(1) @[[AnnotName]], {{.*}}, i32 {{.*}}, ptr addrspace(1) @[[FArgs]])
