// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-linux -std=c++11 -disable-llvm-passes -S -opaque-pointers -emit-llvm -x c++ %s -o - | FileCheck %s

#define PARAM_1 1U << 7
#define PARAM_2 1U << 8

// This test checks that using of __builtin_intel_fpga_mem results in correct
// generation of annotations in LLVM IR.

// CHECK: [[STRUCT:%.*]] = type { i32, float }
struct State {
  int x;
  float y;
};

// CHECK: [[ANN1:@.str[\.]*[0-9]*]] = {{.*}}{params:384}{cache-size:0}{anchor-id:-1}{target-anchor:0}{type:0}{cycle:0}
// CHECK: [[ANN2:@.str[\.]*[0-9]*]] = {{.*}}{params:384}{cache-size:127}{anchor-id:-1}{target-anchor:0}{type:0}{cycle:0}
// CHECK: [[ANN3:@.str[\.]*[0-9]*]] = {{.*}}{params:384}{cache-size:127}{anchor-id:10}{target-anchor:20}{type:30}{cycle:40}
// CHECK: [[ANN4:@.str[\.]*[0-9]*]] = {{.*}}{params:384}{cache-size:127}{anchor-id:11}{target-anchor:12}{type:0}{cycle:0}
// CHECK: [[ANN5:@.str[\.]*[0-9]*]] = {{.*}}{params:384}{cache-size:127}{anchor-id:100}{target-anchor:0}{type:0}{cycle:0}
// CHECK: [[ANN6:@.str[\.]*[0-9]*]] = {{.*}}{params:384}{cache-size:128}{anchor-id:4}{target-anchor:7}{type:8}{cycle:0}

// CHECK: define {{.*}}spir_func void @{{.*}}(ptr addrspace(4) noundef %A, ptr addrspace(4) noundef %B, ptr addrspace(4) noundef %C, ptr addrspace(4){{.*}}%D)
void foo(float *A, int *B, State *C, State &D) {
  float *x;
  int *y;
  State *z;
  double F = 0.0;
  double *f;

  // CHECK-DAG: [[Aaddr:%.*]] = alloca ptr addrspace(4)
  // CHECK-DAG: [[Baddr:%.*]] = alloca ptr addrspace(4)
  // CHECK-DAG: [[Caddr:%.*]] = alloca ptr addrspace(4)
  // CHECK-DAG: [[Daddr:%.*]] = alloca ptr addrspace(4)
  // CHECK-DAG: [[F:%.*]] = alloca double
  // CHECK-DAG: [[f:%.*]] = alloca ptr addrspace(4)

  // CHECK-DAG: [[A:%[0-9]+]] = load ptr addrspace(4), ptr addrspace(4) [[Aaddr]]
  // CHECK-DAG: [[PTR1:%[0-9]+]] = call ptr addrspace(4) @llvm.ptr.annotation{{.*}}[[A]]{{.*}}[[ANN1]]{{.*}}[[ATT:#[0-9]+]]
  // CHECK-DAG: store ptr addrspace(4) [[PTR1]], ptr addrspace(4) %x
  x = __builtin_intel_fpga_mem(A, PARAM_1 | PARAM_2, 0);

  // CHECK-DAG: [[B:%[0-9]+]] = load ptr addrspace(4), ptr addrspace(4) [[Baddr]]
  // CHECK-DAG: [[PTR2:%[0-9]+]] = call ptr addrspace(4) @llvm.ptr.annotation{{.*}}[[B]]{{.*}}[[ANN1]]{{.*}}[[ATT:#[0-9]+]]
  // CHECK-DAG: store ptr addrspace(4) [[PTR2]], ptr addrspace(4) %y
  y = __builtin_intel_fpga_mem(B, PARAM_1 | PARAM_2, 0);

  // CHECK-DAG: [[C:%[0-9]+]] = load ptr addrspace(4), ptr addrspace(4) [[Caddr]]
  // CHECK-DAG: [[PTR3:%[0-9]+]] = call ptr addrspace(4) @llvm.ptr.annotation{{.*}}[[C]]{{.*}}[[ANN1]]{{.*}}[[ATT:#[0-9]+]]
  // CHECK-DAG: store ptr addrspace(4) [[PTR3]], ptr addrspace(4) %z
  z = __builtin_intel_fpga_mem(C, PARAM_1 | PARAM_2, 0);

  // CHECK-DAG: [[A2:%[0-9]+]] = load ptr addrspace(4), ptr addrspace(4) [[Aaddr]]
  // CHECK-DAG: [[PTR4:%[0-9]+]] = call ptr addrspace(4) @llvm.ptr.annotation{{.*}}[[A2]]{{.*}}[[ANN2]]{{.*}}[[ATT:#[0-9]+]]
  // CHECK-DAG: store ptr addrspace(4) [[PTR4]], ptr addrspace(4) %x
  x = __builtin_intel_fpga_mem(A, PARAM_1 | PARAM_2, 127);

  // CHECK-DAG: [[B2:%[0-9]+]] = load ptr addrspace(4), ptr addrspace(4) [[Baddr]]
  // CHECK-DAG: [[PTR5:%[0-9]+]] = call ptr addrspace(4) @llvm.ptr.annotation{{.*}}[[B2]]{{.*}}[[ANN2]]{{.*}}[[ATT:#[0-9]+]]
  // CHECK-DAG: store ptr addrspace(4) [[PTR5]], ptr addrspace(4) %y
  y = __builtin_intel_fpga_mem(B, PARAM_1 | PARAM_2, 127);

  // CHECK-DAG: [[C2:%[0-9]+]] = load ptr addrspace(4), ptr addrspace(4) [[Caddr]]
  // CHECK-DAG: [[PTR6:%[0-9]+]] = call ptr addrspace(4) @llvm.ptr.annotation{{.*}}[[C2]]{{.*}}[[ANN2]]{{.*}}[[ATT:#[0-9]+]]
  // CHECK-DAG: store ptr addrspace(4) [[PTR6]], ptr addrspace(4) %z
  z = __builtin_intel_fpga_mem(C, PARAM_1 | PARAM_2, 127);

  // CHECK-DAG: [[D:%[0-9]+]] = load ptr addrspace(4), ptr addrspace(4) [[Daddr]]
  // CHECK-DAG: [[PTR7:%[0-9]+]] = call ptr addrspace(4) @llvm.ptr.annotation{{.*}}[[D]]{{.*}}[[ANN2]]{{.*}}[[ATT:#[0-9]+]]
  // CHECK-DAG: store ptr addrspace(4) [[PTR7]], ptr addrspace(4) %z
  z = __builtin_intel_fpga_mem(&D, PARAM_1 | PARAM_2, 127);

  // CHECK-DAG: [[PTR8:%[0-9]+]] = call ptr addrspace(4) @llvm.ptr.annotation{{.*}}[[F]]{{.*}}[[ANN2]]{{.*}}[[ATT:#[0-9]+]]
  // CHECK-DAG: store ptr addrspace(4) [[PTR8]], ptr addrspace(4) %f
  f = __builtin_intel_fpga_mem(&F, PARAM_1 | PARAM_2, 127);

  // CHECK-DAG: [[A3:%[0-9]+]] = load ptr addrspace(4), ptr addrspace(4) [[Aaddr]]
  // CHECK-DAG: [[PTR9:%[0-9]+]] = call ptr addrspace(4) @llvm.ptr.annotation{{.*}}[[A3]]{{.*}}[[ANN3]]{{.*}}[[ATT:#[0-9]+]]
  // CHECK-DAG: store ptr addrspace(4) [[PTR9]], ptr addrspace(4) %x
  x = __builtin_intel_fpga_mem(A, PARAM_1 | PARAM_2, 127, 10, 20, 30, 40);

  // CHECK-DAG: [[A4:%[0-9]+]] = load ptr addrspace(4), ptr addrspace(4) [[Aaddr]]
  // CHECK-DAG: [[PTR10:%[0-9]+]] = call ptr addrspace(4) @llvm.ptr.annotation{{.*}}[[A4]]{{.*}}[[ANN4]]{{.*}}[[ATT:#[0-9]+]]
  // CHECK-DAG: store ptr addrspace(4) [[PTR10]], ptr addrspace(4) %x
  x = __builtin_intel_fpga_mem(A, PARAM_1 | PARAM_2, 127, 11, 12);

  // CHECK-DAG: [[B3:%[0-9]+]] = load ptr addrspace(4), ptr addrspace(4) [[Baddr]]
  // CHECK-DAG: [[PTR11:%[0-9]+]] = call ptr addrspace(4) @llvm.ptr.annotation{{.*}}[[B3]]{{.*}}[[ANN5]]{{.*}}[[ATT:#[0-9]+]]
  // CHECK-DAG: store ptr addrspace(4) [[PTR11]], ptr addrspace(4) %y
  y = __builtin_intel_fpga_mem(B, PARAM_1 | PARAM_2, 127, 100);

  constexpr int TestVal1 = 7;
  constexpr int TestVal2 = 8;

  // CHECK-DAG: [[D1:%[0-9]+]] = load ptr addrspace(4), ptr addrspace(4) [[Daddr]]
  // CHECK-DAG: [[PTR12:%[0-9]+]] = call ptr addrspace(4) @llvm.ptr.annotation{{.*}}[[D1]]{{.*}}[[ANN6]]{{.*}}[[ATT:#[0-9]+]]
  // CHECK-DAG: store ptr addrspace(4) [[PTR12]], ptr addrspace(4) %z
  z = __builtin_intel_fpga_mem(&D, PARAM_1 | PARAM_2, 128, 4, TestVal1, TestVal2);
}

// CHECK-DAG: attributes [[ATT]] = { memory(none) }

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
