kernel void test(int global *in, int global *in2) {
  if (!in) 
    return;
  if (in == 1)
    return; 
  if (in > in2)   
    return;
  if (in < in2)
    return;
}
// RUN: %clang_cc1 -triple spir64 -x cl -cl-std=CL2.0 -O0 -emit-llvm-bc %s -o %t.bc
// RUN: llvm-spirv %t.bc -spirv-text -o %t.spt
// RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV
// RUN: llvm-spirv %t.bc -o %t.spv
// RUN: spirv-val %t.spv

// CHECK-SPIRV:ConvertPtrToU 
// CHECK-SPIRV:ConvertPtrToU
// CHECK-SPIRV:INotEqual
// CHECK-SPIRV:ConvertPtrToU
// CHECK-SPIRV:ConvertPtrToU
// CHECK-SPIRV:IEqual
// CHECK-SPIRV:ConvertPtrToU
// CHECK-SPIRV:ConvertPtrToU
// CHECK-SPIRV:UGreaterThan
// CHECK-SPIRV:ConvertPtrToU
// CHECK-SPIRV:ConvertPtrToU
// CHECK-SPIRV:ULessThan
