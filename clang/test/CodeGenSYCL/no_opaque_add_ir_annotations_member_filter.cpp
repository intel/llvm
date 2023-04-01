// RUN: %clang_cc1 -internal-isystem %S/Inputs -disable-llvm-passes \
// RUN:    -triple spir64-unknown-unknown -fsycl-is-device -S \
// RUN:    -no-opaque-pointers -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -internal-isystem %S/Inputs -disable-llvm-passes \
// RUN:    -triple spir64-unknown-unknown -fsycl-is-device -DTEST_SCALAR -S \
// RUN:    -no-opaque-pointers -emit-llvm %s -o - | FileCheck %s

// Tests the optional filter parameter of
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
          {"Prop1", "Prop7", "Prop5"},
          Properties::name..., Properties::value...)]]
#endif
      ;

  g() : x() {}
  g(TEST_T _x) : x(_x) {}
};

int main() {
  sycl::queue q;
  g<prop1, prop2, prop3, prop4, prop5, prop6, prop7> a;
  q.submit([&](sycl::handler &h) {
    h.single_task<class test_kernel1>(
        [=]() {
          (void)a.x;
        });
  });
}

// CHECK-DAG: @[[AnnotName:.*]] = private unnamed_addr addrspace(1) constant [16 x i8] c"sycl-properties\00", section "llvm.metadata"

// CHECK-DAG: @[[Prop1Name:.*]] = private unnamed_addr addrspace(1) constant [6 x i8] c"Prop1\00", section "llvm.metadata"
// CHECK-DAG: @[[Prop5Name:.*]] = private unnamed_addr addrspace(1) constant [6 x i8] c"Prop5\00", section "llvm.metadata"
// CHECK-DAG: @[[Prop7Name:.*]] = private unnamed_addr addrspace(1) constant [6 x i8] c"Prop7\00", section "llvm.metadata"

// CHECK-DAG: @[[Prop1Value:.*]] = private unnamed_addr addrspace(1) constant [16 x i8] c"Property string\00", section "llvm.metadata"
// CHECK-DAG: @[[Prop2_7Value:.*]] = private unnamed_addr addrspace(1) constant [2 x i8] c"1\00", section "llvm.metadata"

// CHECK-DAG: @[[GArgs:.*]] = private unnamed_addr addrspace(1) constant { [6 x i8] addrspace(1)*, [16 x i8] addrspace(1)*, [6 x i8] addrspace(1)*, i8 addrspace(1)*, [6 x i8] addrspace(1)*, [2 x i8] addrspace(1)* } { [6 x i8] addrspace(1)* @[[Prop1Name]], [16 x i8] addrspace(1)* @[[Prop1Value]], [6 x i8] addrspace(1)* @[[Prop5Name]], i8 addrspace(1)* null, [6 x i8] addrspace(1)* @[[Prop7Name]], [2 x i8] addrspace(1)* @[[Prop2_7Value]] }, section "llvm.metadata"

// CHECK-DAG: %{{.*}} = call i8 addrspace(4)* @llvm.ptr.annotation.p4i8.p1i8(i8 {{.*}}, i8 addrspace(1)* getelementptr inbounds ([16 x i8], [16 x i8] addrspace(1)* @[[AnnotName]], i32 0, i32 0), i8 addrspace(1)* getelementptr inbounds {{.*}}, i32 {{.*}}, i8 addrspace(1)* bitcast ({ [6 x i8] addrspace(1)*, [16 x i8] addrspace(1)*, [6 x i8] addrspace(1)*, i8 addrspace(1)*, [6 x i8] addrspace(1)*, [2 x i8] addrspace(1)* } addrspace(1)* @[[GArgs]] to i8 addrspace(1)*))

// CHECK-NOT: @[[Prop2Name:.*]] = private unnamed_addr addrspace(1) constant [6 x i8] c"Prop2\00", section "llvm.metadata"
// CHECK-NOT: @[[Prop3Name:.*]] = private unnamed_addr addrspace(1) constant [6 x i8] c"Prop3\00", section "llvm.metadata"
// CHECK-NOT: @[[Prop4Name:.*]] = private unnamed_addr addrspace(1) constant [6 x i8] c"Prop4\00", section "llvm.metadata"
// CHECK-NOT: @[[Prop6Name:.*]] = private unnamed_addr addrspace(1) constant [6 x i8] c"Prop6\00", section "llvm.metadata"
// CHECK-NOT: @[{{.*}} = private unnamed_addr addrspace(1) constant [5 x i8] c"true\00", section "llvm.metadata"
// CHECK-NOT: @{{.*}} = private unnamed_addr addrspace(1) constant [2 x i8] c"2\00", section "llvm.metadata"
