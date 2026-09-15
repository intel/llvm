; Multiplication of two boolean values must be translated to OpLogicalAnd,
; since OpIMul requires integer operands and OpTypeBool is not an integer type.
; RUN: llvm-spirv %s -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv %t.spv -to-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

target triple = "spir64-unknown-unknown"

; CHECK-SPIRV-NOT: IMul

define spir_func i1 @bool_mul(i1 %a, i1 %b) {
  %c = mul i1 %a, %b
  ret i1 %c
}
; CHECK-SPIRV: FunctionParameter {{[0-9]+}} [[A1:[0-9]+]]
; CHECK-SPIRV: FunctionParameter {{[0-9]+}} [[B1:[0-9]+]]
; CHECK-SPIRV: LogicalAnd {{[0-9]+}} {{[0-9]+}} [[A1]] [[B1]]
; CHECK-LLVM: define spir_func i1 @bool_mul(i1 [[A1:%[a-zA-Z0-9_.]+]], i1 [[B1:%[a-zA-Z0-9_.]+]])
; CHECK-LLVM: and i1 [[A1]], [[B1]]

define spir_func i1 @bool_mul_nsw(i1 %a, i1 %b) {
  %c = mul nsw i1 %a, %b
  ret i1 %c
}
; CHECK-SPIRV: FunctionParameter {{[0-9]+}} [[A2:[0-9]+]]
; CHECK-SPIRV: FunctionParameter {{[0-9]+}} [[B2:[0-9]+]]
; CHECK-SPIRV: LogicalAnd {{[0-9]+}} {{[0-9]+}} [[A2]] [[B2]]
; CHECK-LLVM: define spir_func i1 @bool_mul_nsw(i1 [[A2:%[a-zA-Z0-9_.]+]], i1 [[B2:%[a-zA-Z0-9_.]+]])
; CHECK-LLVM: and i1 [[A2]], [[B2]]

define spir_func i1 @bool_mul_nuw(i1 %a, i1 %b) {
  %c = mul nuw i1 %a, %b
  ret i1 %c
}
; CHECK-SPIRV: FunctionParameter {{[0-9]+}} [[A3:[0-9]+]]
; CHECK-SPIRV: FunctionParameter {{[0-9]+}} [[B3:[0-9]+]]
; CHECK-SPIRV: LogicalAnd {{[0-9]+}} {{[0-9]+}} [[A3]] [[B3]]
; CHECK-LLVM: define spir_func i1 @bool_mul_nuw(i1 [[A3:%[a-zA-Z0-9_.]+]], i1 [[B3:%[a-zA-Z0-9_.]+]])
; CHECK-LLVM: and i1 [[A3]], [[B3]]

define spir_func <4 x i1> @vec_bool_mul(<4 x i1> %a, <4 x i1> %b) {
  %c = mul <4 x i1> %a, %b
  ret <4 x i1> %c
}
; CHECK-SPIRV: FunctionParameter {{[0-9]+}} [[A4:[0-9]+]]
; CHECK-SPIRV: FunctionParameter {{[0-9]+}} [[B4:[0-9]+]]
; CHECK-SPIRV: LogicalAnd {{[0-9]+}} {{[0-9]+}} [[A4]] [[B4]]
; CHECK-LLVM: define spir_func <4 x i1> @vec_bool_mul(<4 x i1> [[A4:%[a-zA-Z0-9_.]+]], <4 x i1> [[B4:%[a-zA-Z0-9_.]+]])
; CHECK-LLVM: and <4 x i1> [[A4]], [[B4]]
