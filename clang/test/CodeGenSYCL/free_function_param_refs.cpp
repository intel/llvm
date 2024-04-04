// RUN: %clang_cc1 -internal-isystem %S/Inputs -disable-llvm-passes \
// RUN:    -triple spir64-unknown-unknown -fsycl-is-device -S \
// RUN:    -emit-llvm %s -o - | FileCheck %s
// 
// This test checks device code generation for free functions with scalar,
// pointer, simple struct and struct with pointer parameters.

#include "mock_properties.hpp"
#include "sycl.hpp"

struct Simple {
  int x;
  char c[100];
  float f;
};
struct WithPointer {
  int x;
  float* fp;
  float f;
};

// CHECK: %struct.Simple = type { i32, [100 x i8], float }
// CHECK: %struct.WithPointer = type { i32, ptr addrspace(4), float }
// CHECK: %struct.__generated_WithPointer = type { i32, ptr addrspace(1), float }

__attribute__((sycl_device))
[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel", 2)]]
void ff_2(int *ptr, int start, int end, struct Simple S, struct WithPointer P) {
  for (int i = start; i <= end; i++)
    ptr[i] = start + S.x + S.f + S.c[2]+ P.x + 66;
}
// CHECK: spir_kernel void @__free_function_ff_2(ptr addrspace(1) {{.*}} %_arg_ptr, i32 noundef %_arg_start, i32 noundef %_arg_end, ptr noundef byval(%struct.Simple) align 4 %_arg_S, ptr noundef byval(%struct.__generated_WithPointer) align 8 %_arg_P)
// CHECK: store ptr addrspace(1) %_arg_ptr, ptr addrspace(4) %_arg_ptr.{{.*}}
// CHECK: store i32 %_arg_start, ptr addrspace(4) %_arg_start.{{.*}}
// CHECK: store i32 %_arg_end, ptr addrspace(4) %_arg_end.{{.*}}
// CHECK: %x = getelementptr inbounds %struct.Simple
// CHECK: %f = getelementptr inbounds %struct.Simple
// CHECK: %c = getelementptr inbounds %struct.Simple
// CHECK: %__generated_x = getelementptr inbounds %struct.__generated_WithPointer, ptr addrspace(4) %_arg_P.{{.*}}
