// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-linux -std=c++11 -disable-llvm-passes -S -no-opaque-pointers -emit-llvm -x c++ %s -o - | FileCheck %s

// CHECK: [[STRUCT:%.*]] = type { i32, float }
struct State {
  int x;
  float y;
};

// CHECK-DAG: [[ANN1:@.str[\.]*[0-9]*]] = {{.*}}"testA\00"
// CHECK-DAG: [[ANN2:@.str[\.]*[0-9]*]] = {{.*}}"testB\00"
// CHECK-DAG: [[ANN3:@.str[\.]*[0-9]*]] = {{.*}}"0\00"
// CHECK-DAG: [[ANN4:@.str[\.]*[0-9]*]] = {{.*}}"127\00"
// CHECK-DAG: [[ARG1:@.args[\.]*[0-9]*]] = {{.*}}[[ANN1]]{{.*}}[[ANN3]]
// CHECK-DAG: [[ARG2:@.args[\.]*[0-9]*]] = {{.*}}[[ANN2]]{{.*}}[[ANN4]]

// CHECK: define {{.*}}spir_func void @{{.*}}(float addrspace(4)* noundef %A, i32 addrspace(4)* noundef %B, [[STRUCT]] addrspace(4)* noundef %C, [[STRUCT]] addrspace(4)*{{.*}}%D)
void foo(float *A, int *B, State *C, State &D) {
  float *x;
  int *y;
  State *z;
  double *f;

  // CHECK-DAG: [[Aaddr:%.*]] = alloca float addrspace(4)*
  // CHECK-DAG: [[Baddr:%.*]] = alloca i32 addrspace(4)*
  // CHECK-DAG: [[Caddr:%.*]] = alloca [[STRUCT]] addrspace(4)*
  // CHECK-DAG: [[Daddr:%.*]] = alloca [[STRUCT]] addrspace(4)*

  // CHECK-DAG: [[A:%[0-9]+]] = load float addrspace(4)*, float addrspace(4)* addrspace(4)* [[Aaddr]]
  // CHECK-DAG: [[PTR1:%[0-9]+]] = call float addrspace(4)* @llvm.ptr.annotation{{.*}}[[A]]{{.*}}[[ARG1]]{{.*}}
  // CHECK-DAG: store float addrspace(4)* [[PTR1]], float addrspace(4)* addrspace(4)* %x
  x = __builtin_intel_fpga_sycl_ptr_annotation(A, "testA", 0);

  // CHECK-DAG: [[B:%[0-9]+]] = load i32 addrspace(4)*, i32 addrspace(4)* addrspace(4)* [[Baddr]]
  // CHECK-DAG: [[PTR2:%[0-9]+]] = call i32 addrspace(4)* @llvm.ptr.annotation{{.*}}[[B]]{{.*}}[[ARG1]]{{.*}}
  // CHECK-DAG: store i32 addrspace(4)* [[PTR2]], i32 addrspace(4)* addrspace(4)* %y
  y = __builtin_intel_fpga_sycl_ptr_annotation(B, "testA", 0);

  // CHECK-DAG: [[C:%[0-9]+]] = load [[STRUCT]] addrspace(4)*, [[STRUCT]] addrspace(4)* addrspace(4)* [[Caddr]]
  // CHECK-DAG: [[PTR3:%[0-9]+]] = call [[STRUCT]] addrspace(4)* @llvm.ptr.annotation{{.*}}[[C]]{{.*}}[[ARG1]]{{.*}}
  // CHECK-DAG: store [[STRUCT]] addrspace(4)* [[PTR3]], [[STRUCT]] addrspace(4)* addrspace(4)* %z
  z = __builtin_intel_fpga_sycl_ptr_annotation(C, "testA", 0);

  // CHECK-DAG: [[A2:%[0-9]+]] = load float addrspace(4)*, float addrspace(4)* addrspace(4)* [[Aaddr]]
  // CHECK-DAG: [[PTR4:%[0-9]+]] = call float addrspace(4)* @llvm.ptr.annotation{{.*}}[[A2]]{{.*}}[[ARG2]]{{.*}}
  // CHECK-DAG: store float addrspace(4)* [[PTR4]], float addrspace(4)* addrspace(4)* %x
  x = __builtin_intel_fpga_sycl_ptr_annotation(A, "testB", 127);

  // CHECK-DAG: [[B2:%[0-9]+]] = load i32 addrspace(4)*, i32 addrspace(4)* addrspace(4)* [[Baddr]]
  // CHECK-DAG: [[PTR5:%[0-9]+]] = call i32 addrspace(4)* @llvm.ptr.annotation{{.*}}[[B2]]{{.*}}[[ARG2]]{{.*}}
  // CHECK-DAG: store i32 addrspace(4)* [[PTR5]], i32 addrspace(4)* addrspace(4)* %y
  y = __builtin_intel_fpga_sycl_ptr_annotation(B, "testB", 127);

  // CHECK-DAG: [[C2:%[0-9]+]] = load [[STRUCT]] addrspace(4)*, [[STRUCT]] addrspace(4)* addrspace(4)* [[Caddr]]
  // CHECK-DAG: [[PTR6:%[0-9]+]] = call [[STRUCT]] addrspace(4)* @llvm.ptr.annotation{{.*}}[[C2]]{{.*}}[[ARG2]]{{.*}}
  // CHECK-DAG: store [[STRUCT]] addrspace(4)* [[PTR6]], [[STRUCT]] addrspace(4)* addrspace(4)* %z
  z = __builtin_intel_fpga_sycl_ptr_annotation(C, "testB", 127);

  // CHECK-DAG: [[D:%[0-9]+]] = load [[STRUCT]] addrspace(4)*, [[STRUCT]] addrspace(4)* addrspace(4)* [[Daddr]]
  // CHECK-DAG: [[PTR7:%[0-9]+]] = call [[STRUCT]] addrspace(4)* @llvm.ptr.annotation{{.*}}[[D]]{{.*}}[[ARG2]]{{.*}}
  // CHECK-DAG: store [[STRUCT]] addrspace(4)* [[PTR7]], [[STRUCT]] addrspace(4)* addrspace(4)* %z
  z = __builtin_intel_fpga_sycl_ptr_annotation(&D, "testB", 127);
}

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(const Func &kernelFunc) {
  kernelFunc();
}

int main() {
  kernel_single_task<class fake_kernel>([]() {
    float *A;
    int *B;
    State *C;
    State D;
    foo(A, B, C, D); });
  return 0;
}
