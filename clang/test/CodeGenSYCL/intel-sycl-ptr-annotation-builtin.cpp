// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-linux -disable-llvm-passes -emit-llvm %s -internal-isystem %S/Inputs -o - | FileCheck %s

// This test checks that using of __builtin_intel_sycl_ptr_annotation results in correct
// generation of annotations in LLVM IR.
#include "sycl.hpp"
class kernel;
// CHECK: [[STRUCT:%.*]] = type { i32, float }
struct State {
  int x;
  float y;
};

// CHECK-DAG: [[ANN1:@.str[\.]*[0-9]*]] = {{.*}}"testA\00"
// CHECK-DAG: [[ANN2:@.str[\.]*[0-9]*]] = {{.*}}"testB\00"
// CHECK-DAG: [[ANN3:@.str[\.]*[0-9]*]] = {{.*}}"testC\00"
// CHECK-DAG: [[ANN4:@.str[\.]*[0-9]*]] = {{.*}}"testD\00"
// CHECK-DAG: [[ANN5:@.str[\.]*[0-9]*]] = {{.*}}"testE\00"
// CHECK-DAG: [[ANN6:@.str[\.]*[0-9]*]] = {{.*}}"testF\00"
// CHECK-DAG: [[ANN7:@.str[\.]*[0-9]*]] = {{.*}}"0\00"
// CHECK-DAG: [[ANN8:@.str[\.]*[0-9]*]] = {{.*}}"127\00"
// CHECK-DAG: [[ANN9:@.str[\.]*[0-9]*]] = {{.*}}"7\00"
// CHECK-DAG: [[ANN10:@.str[\.]*[0-9]*]] = {{.*}}"8\00"
// CHECK-DAG: [[ANN11:@.str[\.]*[0-9]*]] = {{.*}}"testG\00"
// CHECK-DAG: [[ANN12:@.str[\.]*[0-9]*]] = {{.*}}"testH\00"
// CHECK-DAG: [[ARG1:@.args[\.]*[0-9]*]] = {{.*}}[[ANN1]]{{.*}}[[ANN7]]
// CHECK-DAG: [[ARG2:@.args[\.]*[0-9]*]] = {{.*}}[[ANN2]]{{.*}}[[ANN8]]
// CHECK-DAG: [[ARG3:@.args[\.]*[0-9]*]] = {{.*}}[[ANN3]]{{.*}}[[ANN7]]
// CHECK-DAG: [[ARG4:@.args[\.]*[0-9]*]] = {{.*}}[[ANN4]]{{.*}}[[ANN7]]
// CHECK-DAG: [[ARG5:@.args[\.]*[0-9]*]] = {{.*}}[[ANN5]]{{.*}}[[ANN7]]
// CHECK-DAG: [[ARG6:@.args[\.]*[0-9]*]] = {{.*}}[[ANN6]]{{.*}}[[ANN7]]{{.*}}[[ANN9]]{{.*}}[[ANN10]]
// CHECK-DAG: [[ARG7:@.args[\.]*[0-9]*]] = {{.*}}[[ANN11]]{{.*}}[[ANN7]]
// CHECK-DAG: [[ARG8:@.args[\.]*[0-9]*]] = {{.*}}[[ANN12]]{{.*}}[[ANN7]]


// CHECK: define {{.*}}spir_func void @{{.*}}(ptr addrspace(4) noundef %A, ptr addrspace(4) noundef %B, ptr addrspace(4) noundef %C, ptr addrspace(4){{.*}}%D)
void foo(float *A, int *B, State *C, State &D) {
  float *x;
  int *y;
  State *z;
  double *f;

  // CHECK-DAG: [[Aaddr:%.*]] = alloca ptr addrspace(4)
  // CHECK-DAG: [[Baddr:%.*]] = alloca ptr addrspace(4)
  // CHECK-DAG: [[Caddr:%.*]] = alloca ptr addrspace(4)
  // CHECK-DAG: [[Daddr:%.*]] = alloca ptr addrspace(4)

  // CHECK-DAG: [[A:%[0-9]+]] = load ptr addrspace(4), ptr addrspace(4) [[Aaddr]]
  // CHECK-DAG: [[PTR1:%[0-9]+]] = call ptr addrspace(4) @llvm.ptr.annotation{{.*}}[[A]]{{.*}}[[ARG1]]{{.*}}
  // CHECK-DAG: store ptr addrspace(4) [[PTR1]], ptr addrspace(4) %x
  x = __builtin_intel_sycl_ptr_annotation(A, "testA", 0);

  // CHECK-DAG: [[B:%[0-9]+]] = load ptr addrspace(4), ptr addrspace(4) [[Baddr]]
  // CHECK-DAG: [[PTR2:%[0-9]+]] = call ptr addrspace(4) @llvm.ptr.annotation{{.*}}[[B]]{{.*}}[[ARG1]]{{.*}}
  // CHECK-DAG: store ptr addrspace(4) [[PTR2]], ptr addrspace(4) %y
  y = __builtin_intel_sycl_ptr_annotation(B, "testA", 0);

  // CHECK-DAG: [[C:%[0-9]+]] = load ptr addrspace(4), ptr addrspace(4) [[Caddr]]
  // CHECK-DAG: [[PTR3:%[0-9]+]] = call ptr addrspace(4) @llvm.ptr.annotation{{.*}}[[C]]{{.*}}[[ARG1]]{{.*}}
  // CHECK-DAG: store ptr addrspace(4) [[PTR3]], ptr addrspace(4) %z
  z = __builtin_intel_sycl_ptr_annotation(C, "testA", 0);

  // CHECK-DAG: [[A2:%[0-9]+]] = load ptr addrspace(4), ptr addrspace(4) [[Aaddr]]
  // CHECK-DAG: [[PTR4:%[0-9]+]] = call ptr addrspace(4) @llvm.ptr.annotation{{.*}}[[A2]]{{.*}}[[ARG2]]{{.*}}
  // CHECK-DAG: store ptr addrspace(4) [[PTR4]], ptr addrspace(4) %x
  x = __builtin_intel_sycl_ptr_annotation(A, "testB", 127);

  // CHECK-DAG: [[B2:%[0-9]+]] = load ptr addrspace(4), ptr addrspace(4) [[Baddr]]
  // CHECK-DAG: [[PTR5:%[0-9]+]] = call ptr addrspace(4) @llvm.ptr.annotation{{.*}}[[B2]]{{.*}}[[ARG2]]{{.*}}
  // CHECK-DAG: store ptr addrspace(4) [[PTR5]], ptr addrspace(4) %y
  y = __builtin_intel_sycl_ptr_annotation(B, "testB", 127);

  // CHECK-DAG: [[C2:%[0-9]+]] = load ptr addrspace(4), ptr addrspace(4) [[Caddr]]
  // CHECK-DAG: [[PTR6:%[0-9]+]] = call ptr addrspace(4) @llvm.ptr.annotation{{.*}}[[C2]]{{.*}}[[ARG2]]{{.*}}
  // CHECK-DAG: store ptr addrspace(4) [[PTR6]], ptr addrspace(4) %z
  z = __builtin_intel_sycl_ptr_annotation(C, "testB", 127);

  // CHECK-DAG: [[D:%[0-9]+]] = load ptr addrspace(4), ptr addrspace(4) [[Daddr]]
  // CHECK-DAG: [[PTR7:%[0-9]+]] = call ptr addrspace(4) @llvm.ptr.annotation{{.*}}[[D]]{{.*}}[[ARG2]]{{.*}}
  // CHECK-DAG: store ptr addrspace(4) [[PTR7]], ptr addrspace(4) %z
  z = __builtin_intel_sycl_ptr_annotation(&D, "testB", 127);

  // CHECK-DAG: [[A3:%[0-9]+]] = load ptr addrspace(4), ptr addrspace(4) [[Aaddr]]
  // CHECK-DAG: [[PTR9:%[0-9]+]] = call ptr addrspace(4) @llvm.ptr.annotation{{.*}}[[A3]]{{.*}}[[ARG3]]{{.*}}
  // CHECK-DAG: store ptr addrspace(4) [[PTR9]], ptr addrspace(4) %x
  x = __builtin_intel_sycl_ptr_annotation(A, "testC", "testC", "testC", 0, 0, 0);

  // CHECK-DAG: [[A4:%[0-9]+]] = load ptr addrspace(4), ptr addrspace(4) [[Aaddr]]
  // CHECK-DAG: [[PTR10:%[0-9]+]] = call ptr addrspace(4) @llvm.ptr.annotation{{.*}}[[A4]]{{.*}}[[ARG4]]{{.*}}
  // CHECK-DAG: store ptr addrspace(4) [[PTR10]], ptr addrspace(4) %x
  x = __builtin_intel_sycl_ptr_annotation(A, "testD", "testD", "testD", 0, 0, 0);

  // CHECK-DAG: [[B3:%[0-9]+]] = load ptr addrspace(4), ptr addrspace(4) [[Baddr]]
  // CHECK-DAG: [[PTR11:%[0-9]+]] = call ptr addrspace(4) @llvm.ptr.annotation{{.*}}[[B3]]{{.*}}[[ARG5]]{{.*}}
  // CHECK-DAG: store ptr addrspace(4) [[PTR11]], ptr addrspace(4) %y
  y = __builtin_intel_sycl_ptr_annotation(B, "testE", "testE", 0, 0);

  constexpr int TestVal1 = 7;
  constexpr int TestVal2 = 8;

  // CHECK-DAG: [[D1:%[0-9]+]] = load ptr addrspace(4), ptr addrspace(4) [[Daddr]]
  // CHECK-DAG: [[PTR12:%[0-9]+]] = call ptr addrspace(4) @llvm.ptr.annotation{{.*}}[[D1]]{{.*}}[[ARG6]]{{.*}}
  // CHECK-DAG: store ptr addrspace(4) [[PTR12]], ptr addrspace(4) %z
  z = __builtin_intel_sycl_ptr_annotation(&D, "testF", "testF", "testF", "testF", 0, 0, TestVal1, TestVal2);
}

// This check makes sure the generated LoadInst consumes the annotated ptr directly
// CHECK: define {{.*}}spir_func noundef i32 @{{.*}}(ptr addrspace(4) noundef %g)
int annotation_with_load(int* g) {
  // CHECK: [[PTR13:%[0-9]+]] = call ptr addrspace(4) @llvm.ptr.annotation{{.*}}[[ARG7]]{{.*}}
  // CHECK: load i32, ptr addrspace(4) [[PTR13]]
  return *__builtin_intel_sycl_ptr_annotation(g, "testG", 0);
}

// This check makes sure the generated StoreInst consumes the annotated ptr directly
// CHECK: define {{.*}}spir_func void @{{.*}}(ptr addrspace(4) noundef %h)
void annotation_with_store(int* h) {
  // CHECK: [[PTR14:%[0-9]+]] = call ptr addrspace(4) @llvm.ptr.annotation{{.*}}[[ARG8]]{{.*}}
  // CHECK: store i32 1, ptr addrspace(4) [[PTR14]]
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
