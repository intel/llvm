// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-linux -disable-llvm-passes -no-opaque-pointers -emit-llvm -internal-isystem %S/Inputs %s -o - | FileCheck %s

#include "sycl.hpp"
class kernel;
// CHECK: [[STRUCT:%.*]] = type { i32, float }
struct State {
  int x;
  float y;
};

// CHECK-DAG: [[ANN1:@.str[\.]*[0-9]*]] = {{.*}}"testA\00"
// CHECK-DAG: [[ANN2:@.str[\.]*[0-9]*]] = {{.*}}"testB\00"
// CHECK-DAG: [[ANN3:@.str[\.]*[0-9]*]] = {{.*}}"0\00"
// CHECK-DAG: [[ANN4:@.str[\.]*[0-9]*]] = {{.*}}"127\00"
// CHECK-DAG: [[ANN5:@.str[\.]*[0-9]*]] = {{.*}}"testG\00"
// CHECK-DAG: [[ANN6:@.str[\.]*[0-9]*]] = {{.*}}"testH\00"
// CHECK-DAG: [[ARG1:@.args[\.]*[0-9]*]] = {{.*}}[[ANN1]]{{.*}}[[ANN3]]
// CHECK-DAG: [[ARG2:@.args[\.]*[0-9]*]] = {{.*}}[[ANN2]]{{.*}}[[ANN4]]
// CHECK-DAG: [[ARG3:@.args[\.]*[0-9]*]] = {{.*}}[[ANN5]]{{.*}}[[ANN3]]
// CHECK-DAG: [[ARG4:@.args[\.]*[0-9]*]] = {{.*}}[[ANN6]]{{.*}}[[ANN3]]

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
  x = __builtin_intel_sycl_ptr_annotation(A, "testA", 0);

  // CHECK-DAG: [[B:%[0-9]+]] = load i32 addrspace(4)*, i32 addrspace(4)* addrspace(4)* [[Baddr]]
  // CHECK-DAG: [[PTR2:%[0-9]+]] = call i32 addrspace(4)* @llvm.ptr.annotation{{.*}}[[B]]{{.*}}[[ARG1]]{{.*}}
  // CHECK-DAG: store i32 addrspace(4)* [[PTR2]], i32 addrspace(4)* addrspace(4)* %y
  y = __builtin_intel_sycl_ptr_annotation(B, "testA", 0);

  // CHECK-DAG: [[C:%[0-9]+]] = load [[STRUCT]] addrspace(4)*, [[STRUCT]] addrspace(4)* addrspace(4)* [[Caddr]]
  // CHECK-DAG: [[PTR3:%[0-9]+]] = call [[STRUCT]] addrspace(4)* @llvm.ptr.annotation{{.*}}[[C]]{{.*}}[[ARG1]]{{.*}}
  // CHECK-DAG: store [[STRUCT]] addrspace(4)* [[PTR3]], [[STRUCT]] addrspace(4)* addrspace(4)* %z
  z = __builtin_intel_sycl_ptr_annotation(C, "testA", 0);

  // CHECK-DAG: [[A2:%[0-9]+]] = load float addrspace(4)*, float addrspace(4)* addrspace(4)* [[Aaddr]]
  // CHECK-DAG: [[PTR4:%[0-9]+]] = call float addrspace(4)* @llvm.ptr.annotation{{.*}}[[A2]]{{.*}}[[ARG2]]{{.*}}
  // CHECK-DAG: store float addrspace(4)* [[PTR4]], float addrspace(4)* addrspace(4)* %x
  x = __builtin_intel_sycl_ptr_annotation(A, "testB", 127);

  // CHECK-DAG: [[B2:%[0-9]+]] = load i32 addrspace(4)*, i32 addrspace(4)* addrspace(4)* [[Baddr]]
  // CHECK-DAG: [[PTR5:%[0-9]+]] = call i32 addrspace(4)* @llvm.ptr.annotation{{.*}}[[B2]]{{.*}}[[ARG2]]{{.*}}
  // CHECK-DAG: store i32 addrspace(4)* [[PTR5]], i32 addrspace(4)* addrspace(4)* %y
  y = __builtin_intel_sycl_ptr_annotation(B, "testB", 127);

  // CHECK-DAG: [[C2:%[0-9]+]] = load [[STRUCT]] addrspace(4)*, [[STRUCT]] addrspace(4)* addrspace(4)* [[Caddr]]
  // CHECK-DAG: [[PTR6:%[0-9]+]] = call [[STRUCT]] addrspace(4)* @llvm.ptr.annotation{{.*}}[[C2]]{{.*}}[[ARG2]]{{.*}}
  // CHECK-DAG: store [[STRUCT]] addrspace(4)* [[PTR6]], [[STRUCT]] addrspace(4)* addrspace(4)* %z
  z = __builtin_intel_sycl_ptr_annotation(C, "testB", 127);

  // CHECK-DAG: [[D:%[0-9]+]] = load [[STRUCT]] addrspace(4)*, [[STRUCT]] addrspace(4)* addrspace(4)* [[Daddr]]
  // CHECK-DAG: [[PTR7:%[0-9]+]] = call [[STRUCT]] addrspace(4)* @llvm.ptr.annotation{{.*}}[[D]]{{.*}}[[ARG2]]{{.*}}
  // CHECK-DAG: store [[STRUCT]] addrspace(4)* [[PTR7]], [[STRUCT]] addrspace(4)* addrspace(4)* %z
  z = __builtin_intel_sycl_ptr_annotation(&D, "testB", 127);
}

// This check makes sure the generated LoadInst consumes the annotated ptr directly
// CHECK: define {{.*}}spir_func noundef i32 @{{.*}}(i32 addrspace(4)* noundef %g)
int annotation_with_load(int* g) {
  // CHECK: [[PTR13:%[0-9]+]] = call i32 addrspace(4)* @llvm.ptr.annotation{{.*}}[[ARG3]]{{.*}}
  // CHECK: load i32, i32 addrspace(4)* [[PTR13]]
  return *__builtin_intel_sycl_ptr_annotation(g, "testG", 0);
}

// This check makes sure the generated StoreInst consumes the annotated ptr directly
// CHECK: define {{.*}}spir_func void @{{.*}}(i32 addrspace(4)* noundef %h)
void annotation_with_store(int* h) {
  // CHECK: [[PTR14:%[0-9]+]] = call i32 addrspace(4)* @llvm.ptr.annotation{{.*}}[[ARG4]]{{.*}}
  // CHECK: store i32 1, i32 addrspace(4)* [[PTR14]]
  *__builtin_intel_sycl_ptr_annotation(h, "testH", 0) = 1;
}

int main() {
  sycl::queue q;
  q.submit([&](sycl::handler &h) {
    h.single_task<class kernel>([=](){
        float *A;
        int *B;
        State *C;
        State D;
        foo(A, B, C, D);

        int *a;
        annotation_with_load(a);
        annotation_with_store(a);
    });
  });
  return 0;
}

