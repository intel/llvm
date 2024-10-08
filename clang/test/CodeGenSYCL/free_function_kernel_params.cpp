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

constexpr int TestArrSize = 3;

template <int ArrSize>
struct KArgWithPtrArray {
  int *data[ArrSize];
  int start[ArrSize];
  int end[ArrSize];
  constexpr int getArrSize() { return ArrSize; }
};

template <int ArrSize>
[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel", 0)]]
void ff_6(KArgWithPtrArray<ArrSize> KArg) {
  for (int j = 0; j < ArrSize; j++)
    for (int i = KArg.start[j]; i <= KArg.end[j]; i++)
      KArg.data[j][i] = KArg.start[j] + KArg.end[j];
}

template void ff_6(KArgWithPtrArray<TestArrSize> KArg);

// CHECK: %struct.NoPointers = type { i32 }
// CHECK: %struct.Pointers = type { ptr addrspace(4), ptr addrspace(4) }
// CHECK: %struct.Agg = type { %struct.NoPointers, i32, ptr addrspace(4), %struct.Pointers }
// CHECK: %struct.__generated_Pointers = type { ptr addrspace(1), ptr addrspace(1) }
// CHECK: %struct.__generated_Agg = type { %struct.NoPointers, i32, ptr addrspace(1), %struct.__generated_Pointers.0 }
// CHECK: %struct.__generated_Pointers.0 = type { ptr addrspace(1), ptr addrspace(1) }
// CHECK: %struct.__generated_KArgWithPtrArray = type { [3 x ptr addrspace(1)], [3 x i32], [3 x i32] }
// CHECK: %struct.KArgWithPtrArray = type { [3 x ptr addrspace(4)], [3 x i32], [3 x i32] }
// CHECK: define dso_local spir_kernel void @{{.*}}__sycl_kernel{{.*}}(ptr noundef byval(%struct.NoPointers) align 4 %__arg_S1, ptr noundef byval(%struct.__generated_Pointers) align 8 %__arg_S2, ptr noundef byval(%struct.__generated_Agg) align 8 %__arg_S3)
// CHECK: define dso_local spir_kernel void @{{.*}}__sycl_kernel_ff_6{{.*}}(ptr noundef byval(%struct.__generated_KArgWithPtrArray) align 8 %__arg_KArg)
