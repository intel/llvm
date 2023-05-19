// RUN: %clang_cc1 -internal-isystem %S/Inputs -disable-llvm-passes \
// RUN:    -triple spir64-unknown-unknown -fsycl-is-device -S \
// RUN:    -opaque-pointers -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -internal-isystem %S/Inputs -disable-llvm-passes \
// RUN:    -triple spir64-unknown-unknown -fsycl-is-device -DTEST_SCALAR -S \
// RUN:    -opaque-pointers -emit-llvm %s -o - | FileCheck %s

// Tests the generation of IR annotation calls from
// __sycl_detail__::add_ir_annotations_member attributes.
// This test guarantees that the ptr.annotation is inserted at
// the correct position, which is before the ptr-to-ptr load

#include "mock_properties.hpp"
#include "sycl.hpp"

#define TEST_T int*

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

int main() {
  sycl::queue q;
  g<prop1, prop2, prop3, prop4, prop5, prop6, prop7, prop8> a;
  q.submit([&](sycl::handler &h) {
    h.single_task<class test_kernel1>(
        [=]() {
          *a.x = 1;
        });
  });
  return 0;
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

// CHECK-DAG: @[[Prop1Value:.*]] = private unnamed_addr addrspace(1) constant [16 x i8] c"Property string\00", section "llvm.metadata"
// CHECK-DAG: @[[Prop2_7_14Value:.*]] = private unnamed_addr addrspace(1) constant [2 x i8] c"1\00", section "llvm.metadata"
// CHECK-DAG: @[[Prop3Value:.*]] = private unnamed_addr addrspace(1) constant [5 x i8] c"true\00", section "llvm.metadata"
// CHECK-DAG: @[[Prop4_12_17Value:.*]] = private unnamed_addr addrspace(1) constant [2 x i8] c"2\00", section "llvm.metadata"
// CHECK-DAG: @[[Prop8_18Value:.*]] = private unnamed_addr addrspace(1) constant [9 x i8] c"Property\00", section "llvm.metadata"

// CHECK-DAG: @[[GArgs:.*]] = private unnamed_addr addrspace(1) constant { ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(1) } { ptr addrspace(1) @[[Prop1Name]], ptr addrspace(1) @[[Prop1Value]], ptr addrspace(1) @[[Prop2Name]], ptr addrspace(1) @[[Prop2_7_14Value]], ptr addrspace(1) @[[Prop3Name]], ptr addrspace(1) @[[Prop3Value]], ptr addrspace(1) @[[Prop4Name]], ptr addrspace(1) @[[Prop4_12_17Value]], ptr addrspace(1) @[[Prop5Name]], ptr addrspace(1) null, ptr addrspace(1) @[[Prop6Name]], ptr addrspace(1) null, ptr addrspace(1) @[[Prop7Name]], ptr addrspace(1) @[[Prop2_7_14Value]], ptr addrspace(1) @[[Prop8Name]], ptr addrspace(1) @[[Prop8_18Value]] }, section "llvm.metadata"

// CHECK-DAG: %[[PtrToPtr:.*]]   = call ptr addrspace(4) @llvm.ptr.annotation.p4.p1(ptr {{.*}}, ptr addrspace(1) @[[AnnotName]], {{.*}}, i32 {{.*}}, ptr addrspace(1) @[[GArgs]])
// CHECK-DAG: %[[Ptr:.*]]      = load ptr addrspace(4), ptr addrspace(4) %[[PtrToPtr]], {{.*}}
// CHECK-DAG: store i32 1, ptr addrspace(4) %[[Ptr]], {{.*}}
