; Binary operations on boolean values must be translated to the corresponding
; logical opcodes, since OpIMul and OpIAdd require integer operands and
; OpTypeBool is not an integer type.
; RUN: llvm-spirv %s -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv %t.spv -to-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

target triple = "spir64-unknown-unknown"

; CHECK-SPIRV-NOT: IMul
; CHECK-SPIRV-NOT: IAdd

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

define spir_func i1 @bool_add(i1 %a, i1 %b) {
  %c = add i1 %a, %b
  ret i1 %c
}
; CHECK-SPIRV: FunctionParameter {{[0-9]+}} [[A5:[0-9]+]]
; CHECK-SPIRV: FunctionParameter {{[0-9]+}} [[B5:[0-9]+]]
; CHECK-SPIRV: LogicalNotEqual {{[0-9]+}} {{[0-9]+}} [[A5]] [[B5]]
; CHECK-LLVM: define spir_func i1 @bool_add(i1 [[A5:%[a-zA-Z0-9_.]+]], i1 [[B5:%[a-zA-Z0-9_.]+]])
; CHECK-LLVM: icmp ne i1 [[A5]], [[B5]]

define spir_func i1 @bool_add_nsw(i1 %a, i1 %b) {
  %c = add nsw i1 %a, %b
  ret i1 %c
}
; CHECK-SPIRV: FunctionParameter {{[0-9]+}} [[A6:[0-9]+]]
; CHECK-SPIRV: FunctionParameter {{[0-9]+}} [[B6:[0-9]+]]
; CHECK-SPIRV: LogicalNotEqual {{[0-9]+}} {{[0-9]+}} [[A6]] [[B6]]
; CHECK-LLVM: define spir_func i1 @bool_add_nsw(i1 [[A6:%[a-zA-Z0-9_.]+]], i1 [[B6:%[a-zA-Z0-9_.]+]])
; CHECK-LLVM: icmp ne i1 [[A6]], [[B6]]

define spir_func i1 @bool_add_nuw(i1 %a, i1 %b) {
  %c = add nuw i1 %a, %b
  ret i1 %c
}
; CHECK-SPIRV: FunctionParameter {{[0-9]+}} [[A7:[0-9]+]]
; CHECK-SPIRV: FunctionParameter {{[0-9]+}} [[B7:[0-9]+]]
; CHECK-SPIRV: LogicalNotEqual {{[0-9]+}} {{[0-9]+}} [[A7]] [[B7]]
; CHECK-LLVM: define spir_func i1 @bool_add_nuw(i1 [[A7:%[a-zA-Z0-9_.]+]], i1 [[B7:%[a-zA-Z0-9_.]+]])
; CHECK-LLVM: icmp ne i1 [[A7]], [[B7]]

define spir_func <4 x i1> @vec_bool_add(<4 x i1> %a, <4 x i1> %b) {
  %c = add <4 x i1> %a, %b
  ret <4 x i1> %c
}
; CHECK-SPIRV: FunctionParameter {{[0-9]+}} [[A8:[0-9]+]]
; CHECK-SPIRV: FunctionParameter {{[0-9]+}} [[B8:[0-9]+]]
; CHECK-SPIRV: LogicalNotEqual {{[0-9]+}} {{[0-9]+}} [[A8]] [[B8]]
; CHECK-LLVM: define spir_func <4 x i1> @vec_bool_add(<4 x i1> [[A8:%[a-zA-Z0-9_.]+]], <4 x i1> [[B8:%[a-zA-Z0-9_.]+]])
; CHECK-LLVM: icmp ne <4 x i1> [[A8]], [[B8]]
