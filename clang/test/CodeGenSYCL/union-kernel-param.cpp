// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s

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
// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_A(%union.MyUnion* noundef byval(%union.MyUnion) align 4 [[MEM_ARG:%[a-zA-Z0-9_]+]])

// Check lambda object alloca
// CHECK: [[LOCAL_OBJECT:%0]] = alloca %class.anon, align 4

// CHECK: [[LOCAL_OBJECTAS:%.*]] = addrspacecast %class.anon* [[LOCAL_OBJECT]] to %class.anon addrspace(4)*
// CHECK: [[MEM_ARGAS:%.*]] = addrspacecast %union.MyUnion* [[MEM_ARG]] to %union.MyUnion addrspace(4)*
// CHECK: [[L_STRUCT_ADDR:%[a-zA-Z0-9_]+]] = getelementptr inbounds %class.anon, %class.anon addrspace(4)* [[LOCAL_OBJECTAS]], i32 0, i32 0
// CHECK: [[MEMCPY_DST:%[0-9a-zA-Z_]+]] = bitcast %union.{{.*}}MyUnion addrspace(4)* [[L_STRUCT_ADDR]] to i8 addrspace(4)*
// CHECK: [[MEMCPY_SRC:%[0-9a-zA-Z_]+]] = bitcast %union.{{.*}}MyUnion addrspace(4)* [[MEM_ARGAS]] to i8 addrspace(4)*
// CHECK: call void @llvm.memcpy.p4i8.p4i8.i64(i8 addrspace(4)* align 4 [[MEMCPY_DST]], i8 addrspace(4)* align 4 [[MEMCPY_SRC]], i64 12, i1 false)
// CHECK: call spir_func void @{{.*}}(%class.anon addrspace(4)* {{[^,]*}} [[LOCAL_OBJECTAS]])
