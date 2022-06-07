// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-linux -std=c++11 -disable-llvm-passes -S -opaque-pointers -emit-llvm -x c++ %s -o - | FileCheck %s

#define PARAM_1 1U << 7
#define PARAM_2 1U << 8

// CHECK: [[STRUCT:%.*]] = type { i32, float }
struct State {
  int x;
  float y;
};

// CHECK: [[ANN1:@.str[\.]*[0-9]*]] = {{.*}}{params:384}{cache-size:0}
// CHECK: [[ANN2:@.str[\.]*[0-9]*]] = {{.*}}{params:384}{cache-size:127}

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
}

// CHECK-DAG: attributes [[ATT]] = { readnone }

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
