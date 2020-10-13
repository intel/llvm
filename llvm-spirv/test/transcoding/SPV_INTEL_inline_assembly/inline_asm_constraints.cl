// RUN: %clang_cc1 -triple spir64-unknown-unknown -x cl -cl-std=CL2.0 -O0 -emit-llvm-bc %s -o %t.bc
// RUN: llvm-spirv -spirv-ext=+SPV_INTEL_inline_assembly %t.bc -o %t.spv
// RUN: llvm-spirv %t.spv -to-text -o %t.spt
// RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV
// RUN: llvm-spirv -r %t.spv -o %t.bc
// RUN: llvm-dis < %t.bc | FileCheck %s --check-prefix=CHECK-LLVM

// Excerpt from opencl-c-base.h
typedef __SIZE_TYPE__ size_t;
typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned int uint;
typedef unsigned long ulong;

// Excerpt from opencl-c.h to speed up compilation.
#define __ovld __attribute__((overloadable))
#define __cnfn __attribute__((const))
size_t __ovld __cnfn get_global_id(unsigned int dimindx);

// CHECK-SPIRV: {{[0-9]+}} Capability AsmINTEL
// CHECK-SPIRV: {{[0-9]+}} Extension "SPV_INTEL_inline_assembly"
// CHECK-SPIRV-COUNT-1: {{[0-9]+}} AsmTargetINTEL

// CHECK-LLVM: [[STRUCTYPE:%[a-z]+]] = type { i32, i8, float }

// CHECK-LLVM-LABEL: define spir_kernel void @test_int
// CHECK-SPIRV: {{[0-9]+}} AsmINTEL {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} "intcommand $0 $1""=r,r"
// CHECK-LLVM: [[VALUE:%[0-9]+]] = call i32 asm sideeffect "intcommand $0 $1", "=r,r"(i32 %{{[0-9]+}})
// CHECK-LLVM-NEXT: store i32 [[VALUE]], i32 addrspace(1)*

kernel void test_int(global int *in, global int *out) {
  int i = get_global_id(0);
  __asm__ volatile ("intcommand %0 %1" : "=r"(out[i]) : "r"(in[i]));
}

// CHECK-LLVM-LABEL: define spir_kernel void @test_float
// CHECK-SPIRV: {{[0-9]+}} AsmINTEL {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} "floatcommand $0 $1""=r,r"
// CHECK-LLVM: [[VALUE:%[0-9]+]] = call float asm sideeffect "floatcommand $0 $1", "=r,r"(float %{{[0-9]+}})
// CHECK-LLVM-NEXT: store float [[VALUE]], float addrspace(1)*

kernel void test_float(global float *in, global float *out) {
  int i = get_global_id(0);
  __asm__ volatile ("floatcommand %0 %1" : "=r"(out[i]) : "r"(in[i]));
}

// CHECK-LLVM-LABEL: define spir_kernel void @test_mixed_integral
// CHECK-SPIRV: {{[0-9]+}} AsmINTEL {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} "mixed_integral_command $0 $3 $1 $2""=r,r,r,r"
// CHECK-LLVM: [[VALUE:%[0-9]+]] = call i64 asm sideeffect "mixed_integral_command $0 $3 $1 $2", "=r,r,r,r"(i16 %{{[0-9]+}}, i32 %{{[0-9]+}}, i8 %{{[0-9]+}})
// CHECK-LLVM-NEXT: store i64 [[VALUE]], i64 addrspace(1)*

kernel void test_mixed_integral(global uchar *A, global ushort *B, global uint *C, global ulong *D) {
  int wiId = get_global_id(0);
  __asm__ volatile ("mixed_integral_command %0 %3 %1 %2"
          : "=r"(D[wiId]) : "r"(B[wiId]), "r"(C[wiId]), "r"(A[wiId]));
}

// CHECK-LLVM-LABEL: define spir_kernel void @test_mixed_floating
// CHECK-SPIRV: {{[0-9]+}} AsmINTEL {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} "mixed_floating_command $0 $1 $2""=r,r,r"
// CHECK-LLVM: [[VALUE:%[0-9]+]] = call half asm sideeffect "mixed_floating_command $0 $1 $2", "=r,r,r"(double %{{[0-9]+}}, float %{{[0-9]+}})
// CHECK-LLVM-NEXT: store half [[VALUE]], half addrspace(1)*

kernel void test_mixed_floating(global float *A, global half *B, global double *C) {
  int wiId = get_global_id(0);
  __asm__ volatile ("mixed_floating_command %0 %1 %2"
          : "=r"(B[wiId]) : "r"(C[wiId]), "r"(A[wiId]));
}

// CHECK-LLVM-LABEL: define spir_kernel void @test_mixed_all
// CHECK-SPIRV: {{[0-9]+}} AsmINTEL {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} "mixed_all_command $0 $3 $1 $2""=r,r,r,r"
// CHECK-LLVM: [[VALUE:%[0-9]+]] = call i8 asm sideeffect "mixed_all_command $0 $3 $1 $2", "=r,r,r,r"(float %{{[0-9]+}}, i32 %{{[0-9]+}}, i8 %{{[0-9]+}})
// CHECK-LLVM-NEXT: store i8 [[VALUE]], i8 addrspace(1)*

kernel void test_mixed_all(global uchar *A, global float *B, global uint *C, global bool *D) {
  int wiId = get_global_id(0);
  __asm__ volatile ("mixed_all_command %0 %3 %1 %2"
          : "=r"(D[wiId]) : "r"(B[wiId]), "r"(C[wiId]), "r"(A[wiId]));
}

// CHECK-LLVM-LABEL: define spir_kernel void @test_multiple
// CHECK-SPIRV: {{[0-9]+}} AsmINTEL {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} "multiple_command $0 $0 $1 $1 $2 $2""=r,=r,=r,0,1,2"
// CHECK-LLVM: [[VALUE:%[0-9]+]] = call [[STRUCTYPE]] asm sideeffect "multiple_command $0 $0 $1 $1 $2 $2", "=r,=r,=r,0,1,2"(i32 %{{[0-9]+}}, i8 %{{[0-9]+}}, float %{{[0-9]+}})
// CHECK-LLVM-NEXT: extractvalue [[STRUCTYPE]] [[VALUE]], 0
// CHECK-LLVM-NEXT: extractvalue [[STRUCTYPE]] [[VALUE]], 1
// CHECK-LLVM-NEXT: extractvalue [[STRUCTYPE]] [[VALUE]], 2

kernel void test_multiple(global uchar *A, global float *B, global uint *C) {
  int wiId = get_global_id(0);
  __asm__ volatile ("multiple_command %0 %0 %1 %1 %2 %2"
          : "+r"(C[wiId]), "+r"(A[wiId]), "+r"(B[wiId]));
}

// CHECK-LLVM-LABEL: define spir_kernel void @test_constants
// CHECK-SPIRV: {{[0-9]+}} AsmINTEL {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} "constcommand $0 $1""i,i"
// CHECK-LLVM: call void asm sideeffect "constcommand $0 $1", "i,i"(i32 1, double 2.000000e+00)

kernel void test_constants() {
  int i = get_global_id(0);
  __asm__ volatile ("constcommand %0 %1" : : "i"(1), "i"(2.0));
}

