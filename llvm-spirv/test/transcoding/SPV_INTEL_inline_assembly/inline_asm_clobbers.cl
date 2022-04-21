// RUN: %clang_cc1 -triple spir64-unknown-unknown -x cl -cl-std=CL2.0 -O0 -emit-llvm-bc %s -o %t.bc
// RUN: llvm-spirv -spirv-ext=+SPV_INTEL_inline_assembly %t.bc -o %t.spv
// RUN: llvm-spirv %t.spv -to-text -o %t.spt
// RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV
// RUN: llvm-spirv -r %t.spv -o %t.bc
// RUN: llvm-dis < %t.bc | FileCheck %s --check-prefix=CHECK-LLVM

// Excerpt from opencl-c-base.h
typedef __SIZE_TYPE__ size_t;

// Excerpt from opencl-c.h to speed up compilation.
#define __ovld __attribute__((overloadable))
#define __cnfn __attribute__((const))
size_t __ovld __cnfn get_global_id(unsigned int dimindx);

// CHECK-SPIRV: {{[0-9]+}} Capability AsmINTEL
// CHECK-SPIRV: {{[0-9]+}} Extension "SPV_INTEL_inline_assembly"
// CHECK-SPIRV: {{[0-9]+}} AsmTargetINTEL

// XCHECK-LLVM: [[STRUCTYPE:%[a-z0-9]+]] = type { i32, i32 }

// CHECK-LLVM-LABEL: define spir_kernel void @mem_clobber
// CHECK-SPIRV: {{[0-9]+}} AsmINTEL {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} """~{cc},~{memory}"
// CHECK-LLVM: [[VALUE:%[0-9]+]] = load i32 addrspace(1)*, i32 addrspace(1)**
// CHECK-LLVM-NEXT: getelementptr inbounds i32, i32 addrspace(1)* [[VALUE]], i64 0
// CHECK-LLVM-NEXT: store i32 1, i32 addrspace(1)*
// CHECK-LLVM-NEXT: call void asm sideeffect "", "~{cc},~{memory}"()
// CHECK-LLVM-NEXT: load i32 addrspace(1)*, i32 addrspace(1)**

kernel void mem_clobber(global int *x) {
  x[0] = 1;
  __asm__ ("":::"cc","memory");
  x[0] += 1;
}

// CHECK-LLVM-LABEL: define spir_kernel void @out_clobber
// CHECK-SPIRV: {{[0-9]+}} AsmINTEL {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} "earlyclobber_instruction_out $0""=&r"
// CHECK-LLVM: barrier
// CHECK-LLVM: store i32 %{{[a-z0-9]+}}, i32* [[VALUE:%[a-z0-9]+]], align 4
// CHECK-LLVM-NEXT: [[STOREVAL:%[a-z0-9]+]] = call i32 asm "earlyclobber_instruction_out $0", "=&r"()
// CHECK-LLVM: store i32 [[STOREVAL]], i32* [[VALUE]], align 4

kernel void out_clobber(global int *x) {
  int i = get_global_id(0);
  __asm__ ("barrier");
  int a = x[i];
  __asm__ ("earlyclobber_instruction_out %0":"=&r"(a));
  a += 1;
  x[i] = a;
}

// TODO: This fails on debug build with assert "function type not legal for constraints"
//       Probably I am not completely understand what happens
//       Or bug in clang FE. To investigate later, change xchecks to checks and enable

// XCHECK-LLVM-LABEL: define spir_kernel void @in_clobber
// XCHECK-SPIRV: {{[0-9]+}} AsmINTEL {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} "earlyclobber_instruction_in $0""&r"
// XCHECK-LLVM: barrier
// XCHECK-LLVM: getelementptr
// XCHECK-LLVM: store i32  %{{[a-z0-9]+}}, i32* [[LOADVAL:%[a-z0-9]+]], align 4
// XCHECK-LLVM-NEXT: [[VALUE:%[a-z0-9]+]] = load i32, i32* [[LOADVAL]], align 4
// XCHECK-LLVM-NEXT: call void asm sideeffect "earlyclobber_instruction_in $0", "&r"(i32 [[VALUE]])
// XCHECK-LLVM: %{{[a-z0-9]+}} = load i32, i32* [[LOADVAL]], align 4

#if 0
kernel void in_clobber(global int *x) {
  int i = get_global_id(0);
  __asm__ ("barrier");
  int a = x[i];
  __asm__ ("earlyclobber_instruction_in %0"::"&r"(a));
  a += 1;
  x[i] = a;
}
#endif

// XCHECK-LLVM-LABEL: define spir_kernel void @mixed_clobber
// XCHECK-SPIRV: {{[0-9]+}} AsmINTEL {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} "mixedclobber_instruction $0 $1 $2""=&r,=&r,&r,1,~{cc},~{memory}"

#if 0
kernel void mixed_clobber(global int *x, global int *y, global int *z) {
  int i = get_global_id(0);
  int a = x[i];
  int b = y[i];
  int c = z[i];
  __asm__ ("mixedclobber_instruction %0 %1 %2":"=&r"(a),"+&r"(b):"&r"(c):"cc","memory");
  a += 1;
  b += 1;
  c += 1;
  x[i] = c;
  y[i] = a;
  z[i] = b;
}
#endif

