// RUN: %clang_cc1 -fsycl -fsycl-is-device -I %S/Inputs -triple spir64-unknown-unknown-sycldevice -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s

// This test checks if compiler accepts union as kernel parameters.

#include "sycl.hpp"

using namespace cl::sycl;

union MyUnion {
  int FldInt;
  int FldArr[3];
};

MyUnion GlobS;

bool test0() {
  MyUnion S = GlobS;
  MyUnion S0 = {0};
  {
    buffer<MyUnion, 1> Buf(&S0, range<1>(1));
    queue myQueue;
    myQueue.submit([&](handler &cgh) {
      auto B = Buf.get_access<access::mode::write>(cgh);
      cgh.single_task<class MyKernel>([=] { B; S; });
    });
  }
}

// CHECK MyKernel parameters
// CHECK: define spir_kernel void @{{.*}}MyKernel
// CHECK-SAME: %union.{{.*}}.MyUnion addrspace(1)* [[MEM_ARG1:%[a-zA-Z0-9_]+]],
// CHECK-SAME: %"struct.{{.*}}.cl::sycl::range"* byval({{.*}}) align 4 [[MEM_ARG2:%[a-zA-Z0-9_]+1]],
// CHECK-SAME: %"struct.{{.*}}.cl::sycl::range"* byval({{.*}}) align 4 [[MEM_ARG3:%[a-zA-Z0-9_]+2]],
// CHECK-SAME: %"struct.{{.*}}.cl::sycl::id"* byval({{.*}}) align 4 [[OFFSET1:%[a-zA-Z0-9_]+3]],
// CHECK-SAME: %union.{{.*}}.MyUnion* byval({{.*}}) align 4 [[MEM_ARG4:%[a-zA-Z0-9_]+4]])

// Check alloca for pointer arguments
// CHECK: [[MEM_ARG1]].addr{{[0-9]*}} = alloca %union._ZTS7MyUnion.MyUnion addrspace(1)*, align 8

// Check lambda object alloca
// CHECK: [[LOCAL_OBJECT:%0]] = alloca %"class.{{.*}}.anon", align 4

// Check allocas for ranges
// CHECK: [[ACC_RANGE1:%[a-zA-Z0-9_.]+]] = alloca %"struct.{{.*}}.cl::sycl::range"
// CHECK: [[ACC_RANGE2:%[a-zA-Z0-9_.]+]] = alloca %"struct.{{.*}}.cl::sycl::range"
// CHECK: [[OFFSET2:%[a-zA-Z0-9_.]+]] = alloca %"struct.{{.*}}.cl::sycl::id"

// CHECK: [[L_STRUCT_ADDR:%[a-zA-Z0-9_]+]] = getelementptr inbounds %"class.{{.*}}.anon", %"class.{{.*}}.anon"* [[LOCAL_OBJECT]], i32 0, i32 0
// CHECK: [[ACC_CAST1:%[0-9]+]] = addrspacecast %"class{{.*}}accessor"* [[L_STRUCT_ADDR]] to %"class{{.*}}accessor" addrspace(4)*
// CHECK: call spir_func void @{{.*}}MyUnion{{.*}}(%"class.{{.*}}.cl::sycl::accessor" addrspace(4)* [[ACC_CAST1]])
// CHECK: [[Z0:%[a-zA-Z0-9_]*]]  = getelementptr inbounds %"class.{{.*}}.anon", %"class.{{.*}}.anon"* [[LOCAL_OBJECT]], i32 0, i32 1
// CHECK: [[MEMCPY_DST:%[0-9a-zA-Z_]+]] = bitcast %union.{{.*}}MyUnion* [[Z0]] to i8*
// CHECK: [[MEMCPY_SRC:%[0-9a-zA-Z_]+]] = bitcast %union.{{.*}}MyUnion* [[MEM_ARG4]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 [[MEMCPY_DST]], i8* align 4 [[MEMCPY_SRC]], i64 12, i1 false)
// CHECK: [[Z1:%[a-zA-Z0-9_]*]]  = getelementptr inbounds %"class.{{.*}}.anon", %"class.{{.*}}.anon"* [[LOCAL_OBJECT]], i32 0, i32 0

// Check load from kernel pointer argument alloca
// CHECK: [[MEM_LOAD1:%[a-zA-Z0-9_]+]] = load %union._ZTS7MyUnion.MyUnion addrspace(1)*, %union._ZTS7MyUnion.MyUnion addrspace(1)** [[MEM_ARG1]].addr{{[0-9]*}}, align 8
// CHECK: [[MEMCPY_DST1:%[0-9a-zA-Z_]+]] = bitcast %"struct.{{.*}}.cl::sycl::range"* [[ACC_RANGE1]] to i8*
// CHECK: [[MEMCPY_SRC1:%[0-9a-zA-Z_]+]] = bitcast %"struct.{{.*}}.cl::sycl::range"* [[MEM_ARG2]] to i8*
// call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 [[MEMCPY_DST1]], i8* align 4 [[MEMCPY_SRC1]], i64 4, i1 false), !tbaa.struct [[ACC_CAST2:%[0-9]+]]
// CHECK: [[MEMCPY_DST2:%[0-9a-zA-Z_]+]] = bitcast %"struct.{{.*}}.cl::sycl::range"* [[ACC_RANGE2]] to i8*
// CHECK: [[MEMCPY_SRC2:%[0-9a-zA-Z_]+]] = bitcast %"struct.{{.*}}.cl::sycl::range"* [[MEM_ARG3]] to i8*
// call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 [[MEMCPY_DST2]], i8* align 4 [[MEMCPY_SRC2]], i64 4, i1 false), !tbaa.struct [[ACC_CAST2:%[0-9]+]]

// Check __init method call
// CHECK: [[ACC_CAST1:%[0-9]+]] = addrspacecast %"class{{.*}}accessor"* [[Z1]] to %"class{{.*}}accessor" addrspace(4)*
// CHECK: call spir_func void @{{.*}}__init{{.*}}(%"class.{{.*}}.cl::sycl::accessor" addrspace(4)* [[ACC_CAST1]], %union._ZTS7MyUnion.MyUnion addrspace(1)* [[MEM_LOAD1]], %"struct.{{.*}}.cl::sycl::range"* byval({{.*}}) align 4 [[ACC_RANGE1]], %"struct.{{.*}}.cl::sycl::range"* byval({{.*}}) align 4 [[ACC_RANGE2]], %"struct.{{.*}}.cl::sycl::id"* byval({{.*}}) align 4 [[OFFSET2]])
// CHECK: [[ACC_CAST2:%[0-9]+]] = addrspacecast %"class{{.*}}.anon"* [[LOCAL_OBJECT]] to %"class{{.*}}.anon" addrspace(4)*
// CHECK: call spir_func void @{{.*}}(%"class.{{.*}}.anon" addrspace(4)* [[ACC_CAST2]])
