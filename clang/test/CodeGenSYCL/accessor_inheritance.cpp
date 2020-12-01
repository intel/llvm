// RUN:  %clang_cc1 -fsycl -fsycl-is-device -triple spir64-unknown-unknown-sycldevice -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s
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
// CHECK: define spir_kernel void @_ZTSZ4mainE6kernel
// CHECK-SAME: i32 [[ARG_A:%[a-zA-Z0-9_]+]],
// CHECK-SAME: i32 [[ARG_B:%[a-zA-Z0-9_]+]],
// CHECK-SAME: i8 addrspace(1)* [[ACC1_DATA:%[a-zA-Z0-9_]+]],
// CHECK-SAME: %[[RANGE_TYPE]]* byval(%[[RANGE_TYPE]]) align 4 [[ACC1_RANGE1:%[a-zA-Z0-9_]+]],
// CHECK-SAME: %[[RANGE_TYPE]]* byval(%[[RANGE_TYPE]]) align 4 [[ACC1_RANGE2:%[a-zA-Z0-9_]+]],
// CHECK-SAME: %[[ID_TYPE]]* byval(%[[ID_TYPE]]) align 4 [[ACC1_ID:%[a-zA-Z0-9_]+]],
// CHECK-SAME: i8 addrspace(1)* [[ACC2_DATA:%[a-zA-Z0-9_]+]],
// CHECK-SAME: %[[RANGE_TYPE]]* byval(%[[RANGE_TYPE]]) align 4 [[ACC2_RANGE1:%[a-zA-Z0-9_]+]],
// CHECK-SAME: %[[RANGE_TYPE]]* byval(%[[RANGE_TYPE]]) align 4 [[ACC2_RANGE2:%[a-zA-Z0-9_]+]],
// CHECK-SAME: %[[ID_TYPE]]* byval(%[[ID_TYPE]]) align 4 [[ACC2_ID:%[a-zA-Z0-9_]+]],
// CHECK-SAME: i32 [[ARG_C:%[a-zA-Z0-9_]+]])

// Allocas for kernel parameters
// CHECK: [[ARG_A]].addr = alloca i32
// CHECK: [[ARG_B]].addr = alloca i32
// CHECK: [[ACC1_DATA]].addr = alloca i8 addrspace(1)*
// CHECK: [[ACC2_DATA]].addr = alloca i8 addrspace(1)*
// CHECK: [[ARG_C]].addr = alloca i32
//
// Lambda object alloca
// CHECK: [[KERNEL_OBJ:%[a-zA-Z0-9_]+]] = alloca %"class.{{.*}}.anon"
//
// Kernel argument stores
// CHECK: store i32 [[ARG_A]], i32* [[ARG_A]].addr
// CHECK: store i32 [[ARG_B]], i32* [[ARG_B]].addr
// CHECK: store i8 addrspace(1)* [[ACC1_DATA]], i8 addrspace(1)** [[ACC1_DATA]].addr
// CHECK: store i8 addrspace(1)* [[ACC2_DATA]], i8 addrspace(1)** [[ACC2_DATA]].addr
// CHECK: store i32 [[ARG_C]], i32* [[ARG_C]].addr
//
// Check A and B scalar fields initialization
// CHECK: [[GEP:%[a-zA-Z0-9_]+]] = getelementptr inbounds %"class._ZTSZ4mainE3$_0.anon", %"class._ZTSZ4mainE3$_0.anon"* [[KERNEL_OBJ]], i32 0, i32 0
// CHECK: [[BITCAST:%[a-zA-Z0-9_]+]] = bitcast %struct{{.*}}Captured* [[GEP]] to %struct{{.*}}Base*
// CHECK: [[FIELD_A:%[a-zA-Z0-9_]+]] = getelementptr inbounds %struct{{.*}}Base, %struct{{.*}}Base* [[BITCAST]], i32 0, i32 0
// CHECK: [[ARG_A_LOAD:%[a-zA-Z0-9_]+]] = load i32, i32* [[ARG_A]].addr
// CHECK: store i32 [[ARG_A_LOAD]], i32* [[FIELD_A]]
// CHECK: [[FIELD_B:%[a-zA-Z0-9_]+]] = getelementptr inbounds %struct{{.*}}Base, %struct{{.*}}Base* [[BITCAST]], i32 0, i32 1
// CHECK: [[ARG_B_LOAD:%[a-zA-Z0-9_]+]] = load i32, i32* [[ARG_B]].addr
// CHECK: store i32 [[ARG_B_LOAD]], i32* [[FIELD_B]]
//
// Check accessors initialization
// CHECK: [[ACC_FIELD:%[a-zA-Z0-9_]+]] = getelementptr inbounds %struct{{.*}}Base, %struct{{.*}}Base* [[BITCAST]], i32 0, i32 2
// CHECK: [[ACC1_AS_CAST:%[a-zA-Z0-9_]+]] = addrspacecast %"class{{.*}}cl::sycl::accessor"* [[ACC_FIELD]] to %"class{{.*}}cl::sycl::accessor" addrspace(4)*
// Default constructor call
// CHECK: call spir_func void @_ZN2cl4sycl8accessorIcLi1ELNS0_6access4modeE1024ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEC1Ev(%"class{{.*}}cl::sycl::accessor" addrspace(4)* {{[^,]*}} [[ACC1_AS_CAST]])
// CHECK: [[BITCAST1:%[a-zA-Z0-9_]+]] = bitcast %struct{{.*}}Captured* [[GEP]] to i8*
// CHECK: [[GEP1:%[a-zA-Z0-9_]+]] = getelementptr inbounds i8, i8* [[BITCAST1]], i64 20
// CHECK: [[BITCAST2:%[a-zA-Z0-9_]+]] = bitcast i8* [[GEP1]] to %"class{{.*}}cl::sycl::accessor"*
// CHECK: [[ACC2_AS_CAST:%[a-zA-Z0-9_]+]] = addrspacecast %"class{{.*}}cl::sycl::accessor"* [[BITCAST2]] to %"class{{.*}}cl::sycl::accessor" addrspace(4)*
// Default constructor call
// CHECK: call spir_func void @_ZN2cl4sycl8accessorIcLi1ELNS0_6access4modeE1024ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEC2Ev(%"class{{.*}}cl::sycl::accessor" addrspace(4)* {{[^,]*}} [[ACC2_AS_CAST]])

// CHECK C field initialization
// CHECK: [[FIELD_C:%[a-zA-Z0-9_]+]] = getelementptr inbounds %struct{{.*}}Captured, %struct{{.*}}Captured* [[GEP]], i32 0, i32 2
// CHECK: [[ARG_C_LOAD:%[a-zA-Z0-9_]+]] = load i32, i32* [[ARG_C]].addr
// CHECK: store i32 [[ARG_C_LOAD]], i32* [[FIELD_C]]
//
// Check __init method calls
// CHECK: [[GEP2:%[a-zA-Z0-9_]+]] = getelementptr inbounds %"class._ZTSZ4mainE3$_0.anon", %"class._ZTSZ4mainE3$_0.anon"* [[KERNEL_OBJ]], i32 0, i32 0
// CHECK: [[BITCAST3:%[a-zA-Z0-9_]+]] = bitcast %struct{{.*}}Captured* [[GEP2]] to %struct{{.*}}Base*
// CHECK: [[ACC1_FIELD:%[a-zA-Z0-9_]+]] = getelementptr inbounds %struct{{.*}}Base, %struct{{.*}}Base* [[BITCAST3]], i32 0, i32 2
// CHECK: [[ACC1_DATA_LOAD:%[a-zA-Z0-9_]+]] = load i8 addrspace(1)*, i8 addrspace(1)** [[ACC1_DATA]].addr
// CHECK: [[ACC1_AS_CAST1:%[a-zA-Z0-9_]+]] = addrspacecast %"class{{.*}}cl::sycl::accessor"* [[ACC1_FIELD]] to %"class{{.*}}cl::sycl::accessor" addrspace(4)*
// CHECK: call spir_func void @{{.*}}__init{{.*}}(%"class{{.*}}cl::sycl::accessor" addrspace(4)* {{[^,]*}} [[ACC1_AS_CAST1]], i8 addrspace(1)* [[ACC1_DATA_LOAD]]
//
// CHECK: [[GEP3:%[a-zA-Z0-9_]+]] = getelementptr inbounds %"class._ZTSZ4mainE3$_0.anon", %"class._ZTSZ4mainE3$_0.anon"* [[KERNEL_OBJ]], i32 0, i32 0
// CHECK: [[ACC2_DATA_LOAD:%[a-zA-Z0-9_]+]] = load i8 addrspace(1)*, i8 addrspace(1)** [[ACC2_DATA]].addr
// CHECK: [[AS_CAST_CAPTURED:%[a-zA-Z0-9_]+]] = addrspacecast %struct{{.*}}Captured* [[GEP3]] to %"class{{.*}}cl::sycl::accessor" addrspace(4)*
// CHECK: call spir_func void @{{.*}}__init{{.*}}(%"class{{.*}}cl::sycl::accessor" addrspace(4)* {{[^,]*}} [[AS_CAST_CAPTURED]], i8 addrspace(1)* [[ACC2_DATA_LOAD]]
