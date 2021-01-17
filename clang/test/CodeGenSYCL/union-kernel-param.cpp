// RUN: %clang_cc1 -fsycl -fsycl-is-device -triple spir64-unknown-unknown-sycldevice -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s

// This test checks a kernel argument that is union with both array and non-array fields.

#include "Inputs/sycl.hpp"

using namespace cl::sycl;

union MyUnion {
  int FldInt;
  char FldChar;
  float FldArr[3];
};

template <typename name, typename Func>
__attribute__((sycl_kernel)) void a_kernel(Func kernelFunc) {
  kernelFunc();
}

int main() {

  MyUnion obj;

  a_kernel<class kernel_A>(
      [=]() {
        float local = obj.FldArr[2];
      });
}

// CHECK kernel_A parameters
// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_A(%union.{{.*}}.MyUnion* byval(%union.{{.*}}.MyUnion) align 4 [[MEM_ARG:%[a-zA-Z0-9_]+]])

// Check lambda object alloca
// CHECK: [[LOCAL_OBJECT:%0]] = alloca %"class.{{.*}}.anon", align 4

// CHECK: [[L_STRUCT_ADDR:%[a-zA-Z0-9_]+]] = getelementptr inbounds %"class.{{.*}}.anon", %"class.{{.*}}.anon"* [[LOCAL_OBJECT]], i32 0, i32 0
// CHECK: [[MEMCPY_DST:%[0-9a-zA-Z_]+]] = bitcast %union.{{.*}}MyUnion* [[L_STRUCT_ADDR]] to i8*
// CHECK: [[MEMCPY_SRC:%[0-9a-zA-Z_]+]] = bitcast %union.{{.*}}MyUnion* [[MEM_ARG]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 [[MEMCPY_DST]], i8* align 4 [[MEMCPY_SRC]], i64 12, i1 false)
// CHECK: [[ACC_CAST1:%[0-9]+]] = addrspacecast %"class.{{.*}}.anon"* [[LOCAL_OBJECT]] to %"class.{{.*}}.anon" addrspace(4)*
// CHECK: call spir_func void @{{.*}}(%"class.{{.*}}.anon" addrspace(4)* {{[^,]*}} [[ACC_CAST1]])
