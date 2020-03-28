// RUN: %clang_cc1 -fsycl -fsycl-is-device -I %S/Inputs -fsycl-int-header=%t.h -triple spir64-unknown-unknown-sycldevice -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s

// This test checks a kernel with struct parameter that contains an Accessor array.

#include <sycl.hpp>

using namespace cl::sycl;

template <typename name, typename Func>
__attribute__((sycl_kernel)) void a_kernel(Func kernelFunc) {
  kernelFunc();
}

int main() {

  using Accessor =
      accessor<int, 1, access::mode::read_write, access::target::global_buffer>;

  struct struct_acc_t {
    Accessor member_acc[2];
  } struct_acc;

  a_kernel<class kernel_C>(
      [=]() {
        struct_acc.member_acc[1].use();
      });
}

// CHECK kernel_C parameters
// CHECK: define spir_kernel void @{{.*}}kernel_C
// CHECK-SAME: %struct.{{.*}}.struct_acc_t* byval(%struct.{{.*}}.struct_acc_t) align 4 [[STRUCT:%[a-zA-Z0-9_]+]],
// CHECK-SAME: i32 addrspace(1)* [[MEM_ARG1:%[a-zA-Z0-9_]+]],
// CHECK-SAME: %"struct.{{.*}}.cl::sycl::range"* byval({{.*}}) align 4 [[ACC_RANGE1:%[a-zA-Z0-9_]+1]],
// CHECK-SAME: %"struct.{{.*}}.cl::sycl::range"* byval({{.*}}) align 4 [[MEM_RANGE1:%[a-zA-Z0-9_]+2]],
// CHECK-SAME: %"struct.{{.*}}.cl::sycl::id"* byval({{.*}}) align 4 [[OFFSET1:%[a-zA-Z0-9_]+3]],
// CHECK-SAME: i32 addrspace(1)* [[MEM_ARG2:%[a-zA-Z0-9_]+4]],
// CHECK-SAME: %"struct.{{.*}}.cl::sycl::range"* byval({{.*}}) align 4 [[ACC_RANGE2:%[a-zA-Z0-9_]+6]],
// CHECK-SAME: %"struct.{{.*}}.cl::sycl::range"* byval({{.*}}) align 4 [[MEM_RANGE2:%[a-zA-Z0-9_]+7]],
// CHECK-SAME: %"struct.{{.*}}.cl::sycl::id"* byval({{.*}}) align 4 [[OFFSET2:%[a-zA-Z0-9_]+8]])

// Check alloca for pointer arguments
// CHECK: [[MEM_ARG1]].addr{{[0-9]*}} = alloca i32 addrspace(1)*, align 8
// CHECK: [[MEM_ARG1]].addr{{[0-9]*}} = alloca i32 addrspace(1)*, align 8

// Check lambda object alloca
// CHECK: [[LOCAL_OBJECT:%0]] = alloca %"class.{{.*}}.anon", align 4

// Check allocas for ranges
// CHECK: [[ACC_RANGE1:%[a-zA-Z0-9_.]+]] = alloca %"struct.{{.*}}.cl::sycl::range"
// CHECK: [[MEM_RANGE1:%[a-zA-Z0-9_.]+]] = alloca %"struct.{{.*}}.cl::sycl::range"
// CHECK: [[OFFSET1:%[a-zA-Z0-9_.]+]] = alloca %"struct.{{.*}}.cl::sycl::id"
// CHECK: [[ACC_RANGE2:%[a-zA-Z0-9_.]+]] = alloca %"struct.{{.*}}.cl::sycl::range"
// CHECK: [[MEM_RANGE2:%[a-zA-Z0-9_.]+]] = alloca %"struct.{{.*}}.cl::sycl::range"
// CHECK: [[OFFSET2:%[a-zA-Z0-9_.]+]] = alloca %"struct.{{.*}}.cl::sycl::id"

// Check init of local struct
// CHECK: [[L_STRUCT_ADDR:%[a-zA-Z0-9_]+]] = getelementptr inbounds %"class.{{.*}}.anon", %"class.{{.*}}.anon"* [[LOCAL_OBJECT]], i32 0, i32 0
// CHECK: [[MEMCPY_DST:%[0-9a-zA-Z_]+]] = bitcast %struct.{{.*}}struct_acc_t* [[L_STRUCT_ADDR]] to i8*
// CHECK: [[MEMCPY_SRC:%[0-9a-zA-Z_]+]] = bitcast %struct.{{.*}}struct_acc_t* %{{[0-9a-zA-Z_]+}} to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 [[MEMCPY_DST]], i8* align 4 [[MEMCPY_SRC]], i64 24, i1 false)

// Check accessor array GEP for member_acc[0]
// CHECK: [[ACCESSOR_ARRAY1:%[a-zA-Z0-9_]+]] = getelementptr inbounds %"class.{{.*}}.anon", %"class.{{.*}}.anon"* [[LOCAL_OBJECT]], i32 0, i32 0
// CHECK: [[MEMBER1:%[a-zA-Z_]+]] = getelementptr inbounds %struct.{{.*}}.struct_acc_t, %struct.{{.*}}.struct_acc_t* [[ACCESSOR_ARRAY1]], i32 0, i32 0
// CHECK: [[Z0:%[a-zA-Z0-9_]*]] = getelementptr inbounds [2 x %"class.{{.*}}.cl::sycl::accessor"], [2 x %"class.{{.*}}.cl::sycl::accessor"]* [[MEMBER1]], i64 0, i64 0

// Check load from kernel pointer argument alloca
// CHECK: [[MEM_LOAD1:%[a-zA-Z0-9_]+]] = load i32 addrspace(1)*, i32 addrspace(1)** [[MEM_ARG1]].addr{{[0-9]*}}

// Check acc[0] __init method call
// CHECK: [[ACC_CAST1:%[0-9]+]] = addrspacecast %"class{{.*}}accessor"* [[Z0]] to %"class{{.*}}accessor" addrspace(4)*
// CHECK: call spir_func void @{{.*}}__init{{.*}}(%"class.{{.*}}.cl::sycl::accessor" addrspace(4)* [[ACC_CAST1]], i32 addrspace(1)* [[MEM_LOAD1]], %"struct.{{.*}}.cl::sycl::range"* byval({{.*}}) align 4 [[ACC_RANGE1]], %"struct.{{.*}}.cl::sycl::range"* byval({{.*}}) align 4 [[MEM_RANGE1]], %"struct.{{.*}}.cl::sycl::id"* byval({{.*}}) align 4 [[OFFSET1]])

// Check accessor array GEP for member_acc[1]
// CHECK: [[ACCESSOR_ARRAY2:%[a-zA-Z0-9_]+]] = getelementptr inbounds %"class.{{.*}}.anon", %"class.{{.*}}.anon"* [[LOCAL_OBJECT]], i32 0, i32 0
// CHECK: [[MEMBER2:%[a-zA-Z0-9_]+]] = getelementptr inbounds %struct.{{.*}}.struct_acc_t, %struct.{{.*}}.struct_acc_t* [[ACCESSOR_ARRAY2]], i32 0, i32 0
// CHECK: [[Z1:%[a-zA-Z0-9_]*]] = getelementptr inbounds [2 x %"class.{{.*}}.cl::sycl::accessor"], [2 x %"class.{{.*}}.cl::sycl::accessor"]* [[MEMBER2]], i64 0, i64 1

// Check load from kernel pointer argument alloca
// CHECK: [[MEM_LOAD2:%[a-zA-Z0-9_]+]] = load i32 addrspace(1)*, i32 addrspace(1)** [[MEM_ARG1]].addr{{[0-9]*}}

// Check acc[1] __init method call
// CHECK: [[ACC_CAST2:%[0-9]+]] = addrspacecast %"class{{.*}}accessor"* [[Z1]] to %"class{{.*}}accessor" addrspace(4)*
// CHECK: call spir_func void @{{.*}}__init{{.*}}(%"class.{{.*}}.cl::sycl::accessor" addrspace(4)* [[ACC_CAST2]], i32 addrspace(1)* [[MEM_LOAD2]], %"struct.{{.*}}.cl::sycl::range"* byval({{.*}}) align 4 [[ACC_RANGE2]], %"struct.{{.*}}.cl::sycl::range"* byval({{.*}}) align 4 [[MEM_RANGE2]], %"struct.{{.*}}.cl::sycl::id"* byval({{.*}}) align 4 [[OFFSET2]])
