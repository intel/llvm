// RUN: %clang_cc1 -triple spir64 -fdeclare-opencl-builtins -finclude-default-header -emit-llvm-bc %s -o %t.bc
// RUN: llvm-spirv %t.bc -o %t.spv
// RUN: spirv-val %t.spv
// RUN: llvm-spirv %t.bc -spirv-text -o - | FileCheck %s

// Check that out of range dimension index values are handled according to the
// OpenCL C specification.

kernel void ggs(global size_t *out, uint x) {
  // CHECK-DAG: Constant [[#]] [[#CONST64_1:]] 1 0
  // CHECK-DAG: Constant [[#]] [[#CONST3:]] 3
  // CHECK-DAG: Constant [[#]] [[#CONST0:]] 0
  // CHECK-DAG: ConstantTrue [[#]] [[#CONSTTRUE:]]
  // CHECK-DAG: ConstantFalse [[#]] [[#CONSTFALSE:]]

  // CHECK: FunctionParameter [[#]] [[#PARAMOUT:]]
  // CHECK: FunctionParameter [[#]] [[#PARAMX:]]

  // CHECK: Load [[#]] [[#LD0:]]
  // CHECK: CompositeExtract [[#]] [[#SCAL0:]] [[#LD0]] 0
  // CHECK: Select [[#]] [[#RES0:]] [[#CONSTTRUE]] [[#SCAL0]] [[#CONST64_1]]
  // CHECK: Store [[#]] [[#RES0]]
  out[0] = get_global_size(0);

  // CHECK: Load [[#]] [[#LD1:]]
  // CHECK: CompositeExtract [[#]] [[#SCAL1:]] [[#LD1]] 0
  // CHECK: Select [[#]] [[#RES1:]] [[#CONSTFALSE]] [[#SCAL1]] [[#CONST64_1]]
  // CHECK: Store [[#]] [[#RES1]]
  out[1] = get_global_size(3);

  // CHECK: Load [[#]] [[#LD2:]]
  // CHECK: ULessThan [[#]] [[#CMP:]] [[#PARAMX]] [[#CONST3]]
  // CHECK: Select [[#]] [[#SEL:]] [[#CMP]] [[#PARAMX]] [[#CONST0]]
  // CHECK: VectorExtractDynamic 2 [[#SCAL2:]] [[#LD2:]] [[#SEL]]
  // CHECK: Select [[#]] [[#RES2:]] [[#CMP]] [[#SCAL2]] [[#CONST64_1]]
  // CHECK: Store [[#]] [[#RES2]]
  out[2] = get_global_size(x);
}
