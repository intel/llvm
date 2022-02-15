// RUN:  %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s
#include "Inputs/sycl.hpp"

struct Base {
  int A, B;
  cl::sycl::accessor<char, 1, cl::sycl::access::mode::read> AccField;
};

struct Captured : Base,
                  cl::sycl::accessor<char, 1, cl::sycl::access::mode::read> {
  int C;
};

int main() {
  Captured Obj;
  cl::sycl::kernel_single_task<class kernel>(
      [=]() {
        Obj.use();
      });
  return 0;
}

// Check kernel parameters
// CHECK: %[[RANGE_TYPE:"struct.*cl::sycl::range"]]
// CHECK: %[[ID_TYPE:"struct.*cl::sycl::id"]]
// CHECK: define {{.*}}spir_kernel void @_ZTSZ4mainE6kernel
// CHECK-SAME: i32 noundef [[ARG_A:%[a-zA-Z0-9_]+]],
// CHECK-SAME: i32 noundef [[ARG_B:%[a-zA-Z0-9_]+]],
// CHECK-SAME: i8 addrspace(1)* noundef readonly align 1 [[ACC1_DATA:%[a-zA-Z0-9_]+]],
// CHECK-SAME: %[[RANGE_TYPE]]* noundef byval(%[[RANGE_TYPE]]) align 4 [[ACC1_RANGE1:%[a-zA-Z0-9_]+]],
// CHECK-SAME: %[[RANGE_TYPE]]* noundef byval(%[[RANGE_TYPE]]) align 4 [[ACC1_RANGE2:%[a-zA-Z0-9_]+]],
// CHECK-SAME: %[[ID_TYPE]]* noundef byval(%[[ID_TYPE]]) align 4 [[ACC1_ID:%[a-zA-Z0-9_]+]],
// CHECK-SAME: i8 addrspace(1)* noundef readonly align 1 [[ACC2_DATA:%[a-zA-Z0-9_]+]],
// CHECK-SAME: %[[RANGE_TYPE]]* noundef byval(%[[RANGE_TYPE]]) align 4 [[ACC2_RANGE1:%[a-zA-Z0-9_]+]],
// CHECK-SAME: %[[RANGE_TYPE]]* noundef byval(%[[RANGE_TYPE]]) align 4 [[ACC2_RANGE2:%[a-zA-Z0-9_]+]],
// CHECK-SAME: %[[ID_TYPE]]* noundef byval(%[[ID_TYPE]]) align 4 [[ACC2_ID:%[a-zA-Z0-9_]+]],
// CHECK-SAME: i32 noundef [[ARG_C:%[a-zA-Z0-9_]+]])

// Allocas and addrspacecasts for kernel parameters
// CHECK: [[ARG_A]].addr = alloca i32
// CHECK: [[ARG_B]].addr = alloca i32
// CHECK: [[ACC1_DATA]].addr = alloca i8 addrspace(1)
// CHECK: [[ACC2_DATA]].addr = alloca i8 addrspace(1)*
// CHECK: [[ARG_C]].addr = alloca i32
// CHECK: [[KERNEL:%[a-zA-Z0-9_]+]] = alloca %class{{.*}}.anon
// CHECK: [[ARG_A]].addr.ascast = addrspacecast i32* [[ARG_A]].addr to i32 addrspace(4)*
// CHECK: [[ARG_B]].addr.ascast = addrspacecast i32* [[ARG_B]].addr to i32 addrspace(4)*
// CHECK: [[ACC1_DATA]].addr.ascast = addrspacecast i8 addrspace(1)** [[ACC1_DATA]].addr to i8 addrspace(1)* addrspace(4)*
// CHECK: [[ACC2_DATA]].addr.ascast = addrspacecast i8 addrspace(1)** [[ACC2_DATA]].addr to i8 addrspace(1)* addrspace(4)*
// CHECK: [[ARG_C]].addr.ascast = addrspacecast i32* [[ARG_C]].addr to i32 addrspace(4)*
//
// Lambda object alloca
// CHECK: [[KERNEL_OBJ:%[a-zA-Z0-9_]+]] = addrspacecast %class{{.*}}.anon* [[KERNEL]] to %class{{.*}}.anon addrspace(4)*
//
// Kernel argument stores
// CHECK: store i32 [[ARG_A]], i32 addrspace(4)* [[ARG_A]].addr.ascast
// CHECK: store i32 [[ARG_B]], i32 addrspace(4)* [[ARG_B]].addr.ascast
// CHECK: store i8 addrspace(1)* [[ACC1_DATA]], i8 addrspace(1)* addrspace(4)* [[ACC1_DATA]].addr.ascast
// CHECK: store i8 addrspace(1)* [[ACC2_DATA]], i8 addrspace(1)* addrspace(4)* [[ACC2_DATA]].addr.ascast
// CHECK: store i32 [[ARG_C]], i32 addrspace(4)* [[ARG_C]].addr.ascast
//
// Check A and B scalar fields initialization
// CHECK: [[GEP:%[a-zA-Z0-9_]+]] = getelementptr inbounds %class{{.*}}.anon, %class{{.*}}.anon addrspace(4)* [[KERNEL_OBJ]], i32 0, i32 0
// CHECK: [[BITCAST:%[a-zA-Z0-9_]+]] = bitcast %struct{{.*}}Captured addrspace(4)* [[GEP]] to %struct{{.*}}Base addrspace(4)*
// CHECK: [[FIELD_A:%[a-zA-Z0-9_]+]] = getelementptr inbounds %struct{{.*}}Base, %struct{{.*}}Base addrspace(4)* [[BITCAST]], i32 0, i32 0
// CHECK: [[ARG_A_LOAD:%[a-zA-Z0-9_]+]] = load i32, i32 addrspace(4)* [[ARG_A]].addr.ascast
// CHECK: store i32 [[ARG_A_LOAD]], i32 addrspace(4)* [[FIELD_A]]
// CHECK: [[FIELD_B:%[a-zA-Z0-9_]+]] = getelementptr inbounds %struct{{.*}}Base, %struct{{.*}}Base addrspace(4)* [[BITCAST]], i32 0, i32 1
// CHECK: [[ARG_B_LOAD:%[a-zA-Z0-9_]+]] = load i32, i32 addrspace(4)* [[ARG_B]].addr.ascast
// CHECK: store i32 [[ARG_B_LOAD]], i32 addrspace(4)* [[FIELD_B]]
//
// Check accessors initialization
// CHECK: [[ACC_FIELD:%[a-zA-Z0-9_]+]] = getelementptr inbounds %struct{{.*}}Base, %struct{{.*}}Base addrspace(4)* [[BITCAST]], i32 0, i32 2
// Default constructor call
// CHECK: call spir_func void @_ZN2cl4sycl8accessorIcLi1ELNS0_6access4modeE1024ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEC1Ev(%"class{{.*}}cl::sycl::accessor" addrspace(4)* {{[^,]*}} [[ACC_FIELD]])
// CHECK: [[BITCAST1:%[a-zA-Z0-9_]+]] = bitcast %struct{{.*}}Captured addrspace(4)* [[GEP]] to i8 addrspace(4)*
// CHECK: [[GEP1:%[a-zA-Z0-9_]+]] = getelementptr inbounds i8, i8 addrspace(4)* [[BITCAST1]], i64 20
// CHECK: [[BITCAST2:%[a-zA-Z0-9_]+]] = bitcast i8 addrspace(4)* [[GEP1]] to %"class{{.*}}cl::sycl::accessor" addrspace(4)*
// Default constructor call
// CHECK: call spir_func void @_ZN2cl4sycl8accessorIcLi1ELNS0_6access4modeE1024ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEC2Ev(%"class{{.*}}cl::sycl::accessor" addrspace(4)* {{[^,]*}} [[BITCAST2]])

// CHECK C field initialization
// CHECK: [[FIELD_C:%[a-zA-Z0-9_]+]] = getelementptr inbounds %struct{{.*}}Captured, %struct{{.*}}Captured addrspace(4)* [[GEP]], i32 0, i32 2
// CHECK: [[ARG_C_LOAD:%[a-zA-Z0-9_]+]] = load i32, i32 addrspace(4)* [[ARG_C]].addr.ascast
// CHECK: store i32 [[ARG_C_LOAD]], i32 addrspace(4)* [[FIELD_C]]
//
// Check __init method calls
// CHECK: [[GEP2:%[a-zA-Z0-9_]+]] = getelementptr inbounds %class{{.*}}.anon, %class{{.*}}.anon addrspace(4)* [[KERNEL_OBJ]], i32 0, i32 0
// CHECK: [[BITCAST3:%[a-zA-Z0-9_]+]] = bitcast %struct{{.*}}Captured addrspace(4)* [[GEP2]] to %struct{{.*}}Base addrspace(4)*
// CHECK: [[ACC1_FIELD:%[a-zA-Z0-9_]+]] = getelementptr inbounds %struct{{.*}}Base, %struct{{.*}}Base addrspace(4)* [[BITCAST3]], i32 0, i32 2
// CHECK: [[ACC1_DATA_LOAD:%[a-zA-Z0-9_]+]] = load i8 addrspace(1)*, i8 addrspace(1)* addrspace(4)* [[ACC1_DATA]].addr.ascast
// CHECK: call spir_func void @{{.*}}__init{{.*}}(%"class{{.*}}cl::sycl::accessor" addrspace(4)* {{[^,]*}} [[ACC1_FIELD]], i8 addrspace(1)* noundef [[ACC1_DATA_LOAD]]
//
// CHECK: [[GEP3:%[a-zA-Z0-9_]+]] = getelementptr inbounds %class{{.*}}.anon, %class{{.*}}.anon addrspace(4)* [[KERNEL_OBJ]], i32 0, i32 0
// CHECK: [[ACC2_DATA_LOAD:%[a-zA-Z0-9_]+]] = load i8 addrspace(1)*, i8 addrspace(1)* addrspace(4)* [[ACC2_DATA]].addr.ascast
// CHECK: [[BITCAST4:%[a-zA-Z0-9_]+]] = bitcast %struct{{.*}}Captured addrspace(4)* [[GEP3]] to %"class{{.*}}cl::sycl::accessor" addrspace(4)*
// CHECK: call spir_func void @{{.*}}__init{{.*}}(%"class{{.*}}cl::sycl::accessor" addrspace(4)* {{[^,]*}} [[BITCAST4]], i8 addrspace(1)* noundef [[ACC2_DATA_LOAD]]
