// RUN: %clang_cc1 -internal-isystem %S/Inputs -fsycl-is-device -triple spir64 \
// RUN: -emit-llvm %s -o - | FileCheck %s
// This test checks parameter IR generation for free functions with parameters
// of non-decomposed struct type.

#include "sycl.hpp"

struct NoPointers {
  int f;
};

struct Pointers {
  int * a;
  float * b;
};

struct Agg {
  NoPointers F1;
  int F2;
  int *F3;
  Pointers F4;
};

__attribute__((sycl_device))
[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel", 0)]]
void ff_4(NoPointers S1, Pointers S2, Agg S3) {
}

// CHECK: %struct.NoPointers = type { i32 }
// CHECK: %struct.Pointers = type { ptr addrspace(4), ptr addrspace(4) }
// CHECK: %struct.Agg = type { %struct.NoPointers, i32, ptr addrspace(4), %struct.Pointers }
// CHECK: %struct.__generated_Pointers = type { ptr addrspace(1), ptr addrspace(1) }
// CHECK: %struct.__generated_Agg = type { %struct.NoPointers, i32, ptr addrspace(1), %struct.__generated_Pointers.0 }
// CHECK: %struct.__generated_Pointers.0 = type { ptr addrspace(1), ptr addrspace(1) }
// CHECK: define dso_local spir_kernel void @{{.*}}__sycl_kernel{{.*}}(ptr noundef byval(%struct.NoPointers) align 4 %__arg_S1, ptr noundef byval(%struct.__generated_Pointers) align 8 %__arg_S2, ptr noundef byval(%struct.__generated_Agg) align 8 %__arg_S3)
