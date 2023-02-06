// RUN: %clang_cc1 -fno-sycl-force-inline-kernel-lambda -fsycl-is-device -triple spir64-unknown-unknown -disable-llvm-passes -no-opaque-pointers -emit-llvm %s -o - | FileCheck %s

// This test checks a kernel argument that is union with both array and non-array fields.

#include "Inputs/sycl.hpp"

using namespace sycl;

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
// CHECK: [[LOCAL_OBJECT:%__wrapper_union]] = alloca %union.__wrapper_union, align 4
// CHECK: [[FUNCTOR_PTR:%.*]] = alloca %class.anon addrspace(4)*

// CHECK: [[LOCAL_OBJECTAS:%.*]] = addrspacecast %union.__wrapper_union* [[LOCAL_OBJECT]] to %union.__wrapper_union addrspace(4)*
// CHECK: [[FUNCTOR_PTRAS:%.*]] = addrspacecast %class.anon addrspace(4)** [[FUNCTOR_PTR]] to %class.anon addrspace(4)* addrspace(4)*
// CHECK: [[MEM_ARGAS:%.*]] = addrspacecast %union.MyUnion* [[MEM_ARG]] to %union.MyUnion addrspace(4)*
// CHECK: [[ANON_OBJECTAS:%.*]] = bitcast %union.__wrapper_union addrspace(4)* [[LOCAL_OBJECTAS]] to %class.anon addrspace(4)*
// CHECK: [[L_STRUCT_ADDR:%[a-zA-Z0-9_]+]] = getelementptr inbounds %class.anon, %class.anon addrspace(4)* [[ANON_OBJECTAS]], i32 0, i32 0
// CHECK: [[MEMCPY_DST:%[0-9a-zA-Z_]+]] = bitcast %union.{{.*}}MyUnion addrspace(4)* [[L_STRUCT_ADDR]] to i8 addrspace(4)*
// CHECK: [[MEMCPY_SRC:%[0-9a-zA-Z_]+]] = bitcast %union.{{.*}}MyUnion addrspace(4)* [[MEM_ARGAS]] to i8 addrspace(4)*
// CHECK: call void @llvm.memcpy.p4i8.p4i8.i64(i8 addrspace(4)* align 4 [[MEMCPY_DST]], i8 addrspace(4)* align 4 [[MEMCPY_SRC]], i64 12, i1 false)
// CHECK: [[FUNCTOR:%[0-9a-zA-Z_]+]] = load %class.anon addrspace(4)*, %class.anon addrspace(4)* addrspace(4)* [[FUNCTOR_PTRAS]]
// CHECK: call spir_func void @{{.*}}(%class.anon addrspace(4)* {{[^,]*}} [[FUNCTOR]])
