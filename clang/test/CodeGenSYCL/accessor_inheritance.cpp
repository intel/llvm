// RUN:  %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s
#include "Inputs/sycl.hpp"

struct Base {
  int A, B;
  sycl::accessor<char, 1, sycl::access::mode::read> AccField;
};

struct Captured : Base,
                  sycl::accessor<char, 1, sycl::access::mode::read> {
  int C;
};

int main() {
  Captured Obj;
  sycl::kernel_single_task<class kernel>(
      [=]() {
        Obj.use();
      });
  return 0;
}

// Check kernel parameters
// CHECK: %[[RANGE_TYPE:"struct.*sycl::_V1::range"]]
// CHECK: %[[ID_TYPE:"struct.*sycl::_V1::id"]]
// CHECK: define {{.*}}spir_kernel void @_ZTSZ4mainE6kernel
// CHECK-SAME: i32 noundef [[ARG_A:%[a-zA-Z0-9_]+]],
// CHECK-SAME: i32 noundef [[ARG_B:%[a-zA-Z0-9_]+]],
// CHECK-SAME: ptr addrspace(1) noundef readonly align 1 [[ACC1_DATA:%[a-zA-Z0-9_]+]],
// CHECK-SAME: ptr noundef byval(%[[RANGE_TYPE]]) align 4 [[ACC1_RANGE1:%[a-zA-Z0-9_]+]],
// CHECK-SAME: ptr noundef byval(%[[RANGE_TYPE]]) align 4 [[ACC1_RANGE2:%[a-zA-Z0-9_]+]],
// CHECK-SAME: ptr noundef byval(%[[ID_TYPE]]) align 4 [[ACC1_ID:%[a-zA-Z0-9_]+]],
// CHECK-SAME: ptr addrspace(1) noundef readonly align 1 [[ACC2_DATA:%[a-zA-Z0-9_]+]],
// CHECK-SAME: ptr noundef byval(%[[RANGE_TYPE]]) align 4 [[ACC2_RANGE1:%[a-zA-Z0-9_]+]],
// CHECK-SAME: ptr noundef byval(%[[RANGE_TYPE]]) align 4 [[ACC2_RANGE2:%[a-zA-Z0-9_]+]],
// CHECK-SAME: ptr noundef byval(%[[ID_TYPE]]) align 4 [[ACC2_ID:%[a-zA-Z0-9_]+]],
// CHECK-SAME: i32 noundef [[ARG_C:%[a-zA-Z0-9_]+]])

// Allocas and addrspacecasts for kernel parameters
// CHECK: [[ARG_A]].addr = alloca i32
// CHECK: [[ARG_B]].addr = alloca i32
// CHECK: [[ACC1_DATA]].addr = alloca ptr addrspace(1)
// CHECK: [[ACC2_DATA]].addr = alloca ptr addrspace(1)
// CHECK: [[ARG_C]].addr = alloca i32
// CHECK: [[KERNEL:%[a-zA-Z0-9_]+]] = alloca %class{{.*}}.anon
// CHECK: [[ARG_A]].addr.ascast = addrspacecast ptr [[ARG_A]].addr to ptr addrspace(4)
// CHECK: [[ARG_B]].addr.ascast = addrspacecast ptr [[ARG_B]].addr to ptr addrspace(4)
// CHECK: [[ACC1_DATA]].addr.ascast = addrspacecast ptr [[ACC1_DATA]].addr to ptr addrspace(4)
// CHECK: [[ACC2_DATA]].addr.ascast = addrspacecast ptr [[ACC2_DATA]].addr to ptr addrspace(4)
// CHECK: [[ARG_C]].addr.ascast = addrspacecast ptr [[ARG_C]].addr to ptr addrspace(4)
//
// Lambda object alloca
// CHECK: [[KERNEL_OBJ:%[a-zA-Z0-9_.]+]] = addrspacecast ptr [[KERNEL]] to ptr addrspace(4)
//
// Kernel argument stores
// CHECK: store i32 [[ARG_A]], ptr addrspace(4) [[ARG_A]].addr.ascast
// CHECK: store i32 [[ARG_B]], ptr addrspace(4) [[ARG_B]].addr.ascast
// CHECK: store ptr addrspace(1) [[ACC1_DATA]], ptr addrspace(4) [[ACC1_DATA]].addr.ascast
// CHECK: store ptr addrspace(1) [[ACC2_DATA]], ptr addrspace(4) [[ACC2_DATA]].addr.ascast
// CHECK: store i32 [[ARG_C]], ptr addrspace(4) [[ARG_C]].addr.ascast
//
// Check A and B scalar fields initialization
// CHECK: [[GEP:%[a-zA-Z0-9_]+]] = getelementptr inbounds nuw %class{{.*}}.anon, ptr addrspace(4) [[KERNEL_OBJ]], i32 0, i32 0
// CHECK: [[FIELD_A:%[a-zA-Z0-9_]+]] = getelementptr inbounds nuw %struct{{.*}}Base, ptr addrspace(4) [[GEP]], i32 0, i32 0
// CHECK: [[ARG_A_LOAD:%[a-zA-Z0-9_]+]] = load i32, ptr addrspace(4) [[ARG_A]].addr.ascast
// CHECK: store i32 [[ARG_A_LOAD]], ptr addrspace(4) [[FIELD_A]]
// CHECK: [[FIELD_B:%[a-zA-Z0-9_]+]] = getelementptr inbounds nuw %struct{{.*}}Base, ptr addrspace(4) [[GEP]], i32 0, i32 1
// CHECK: [[ARG_B_LOAD:%[a-zA-Z0-9_]+]] = load i32, ptr addrspace(4) [[ARG_B]].addr.ascast
// CHECK: store i32 [[ARG_B_LOAD]], ptr addrspace(4) [[FIELD_B]]
//
// Check accessors initialization
// CHECK: [[ACC_FIELD:%[a-zA-Z0-9_]+]] = getelementptr inbounds nuw %struct{{.*}}Base, ptr addrspace(4) [[GEP]], i32 0, i32 2
// Default constructor call
// CHECK: call spir_func void @_ZN4sycl3_V18accessorIcLi1ELNS0_6access4modeE1024ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEC1Ev(ptr addrspace(4) {{[^,]*}} [[ACC_FIELD]])
// CHECK: [[GEP1:%[a-zA-Z0-9_]+]] = getelementptr inbounds i8, ptr addrspace(4) [[GEP]], i64 20
// Default constructor call
// CHECK: call spir_func void @_ZN4sycl3_V18accessorIcLi1ELNS0_6access4modeE1024ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEC2Ev(ptr addrspace(4) {{[^,]*}} [[GEP1]])

// CHECK C field initialization
// CHECK: [[FIELD_C:%[a-zA-Z0-9_]+]] = getelementptr inbounds nuw %struct{{.*}}Captured, ptr addrspace(4) [[GEP]], i32 0, i32 2
// CHECK: [[ARG_C_LOAD:%[a-zA-Z0-9_]+]] = load i32, ptr addrspace(4) [[ARG_C]].addr.ascast
// CHECK: store i32 [[ARG_C_LOAD]], ptr addrspace(4) [[FIELD_C]]
//
// Check __init method calls
// CHECK: [[GEP2:%[a-zA-Z0-9_]+]] = getelementptr inbounds nuw %class{{.*}}.anon, ptr addrspace(4) [[KERNEL_OBJ]], i32 0, i32 0
// CHECK: [[ACC1_FIELD:%[a-zA-Z0-9_]+]] = getelementptr inbounds nuw %struct{{.*}}Base, ptr addrspace(4) [[GEP2]], i32 0, i32 2
// CHECK: [[ACC1_DATA_LOAD:%[a-zA-Z0-9_]+]] = load ptr addrspace(1), ptr addrspace(4) [[ACC1_DATA]].addr.ascast
// CHECK: call spir_func void @{{.*}}__init{{.*}}(ptr addrspace(4) {{[^,]*}} [[ACC1_FIELD]], ptr addrspace(1) noundef [[ACC1_DATA_LOAD]]
//
// CHECK: [[GEP3:%[a-zA-Z0-9_]+]] = getelementptr inbounds nuw %class{{.*}}.anon, ptr addrspace(4) [[KERNEL_OBJ]], i32 0, i32 0
// CHECK: [[ACC2_DATA_LOAD:%[a-zA-Z0-9_]+]] = load ptr addrspace(1), ptr addrspace(4) [[ACC2_DATA]].addr.ascast
// CHECK: call spir_func void @{{.*}}__init{{.*}}(ptr addrspace(4) {{[^,]*}}, ptr addrspace(1) noundef [[ACC2_DATA_LOAD]]
