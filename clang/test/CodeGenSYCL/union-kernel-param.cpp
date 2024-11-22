// RUN: %clang_cc1 -fno-sycl-force-inline-kernel-lambda -fsycl-is-device -triple spir64-unknown-unknown -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s

// This test checks a kernel argument that is union with both array and non-array fields.

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
// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_A(ptr noundef byval(%union.MyUnion) align 4 [[MEM_ARG:%[a-zA-Z0-9_]+]])

// Check lambda object alloca
// CHECK: [[LOCAL_OBJECT:%__SYCLKernel]] = alloca %class.anon, align 4

// CHECK: [[LOCAL_OBJECTAS:%.*]] = addrspacecast ptr [[LOCAL_OBJECT]] to ptr addrspace(4)
// CHECK: [[MEM_ARGAS:%.*]] = addrspacecast ptr [[MEM_ARG]] to ptr addrspace(4)
// CHECK: [[L_STRUCT_ADDR:%[a-zA-Z0-9_]+]] = getelementptr inbounds nuw %class.anon, ptr addrspace(4) [[LOCAL_OBJECTAS]], i32 0, i32 0
// CHECK: call void @llvm.memcpy.p4.p4.i64(ptr addrspace(4) align 4 [[L_STRUCT_ADDR]], ptr addrspace(4) align 4 [[MEM_ARGAS]], i64 12, i1 false)
// CHECK: call spir_func void @{{.*}}(ptr addrspace(4) {{[^,]*}} [[LOCAL_OBJECTAS]])
