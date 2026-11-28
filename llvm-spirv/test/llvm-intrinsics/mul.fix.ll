; RUN: llvm-spirv %s -o %t.spv
; RUN: llvm-spirv %t.spv -to-text -o - | FileCheck %s
; RUN: spirv-val %t.spv

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-unknown"

; CHECK-DAG: Capability Int16

; CHECK: Name [[test_smul:[0-9]+]] "test_smul"
; CHECK: Name [[test_umul_scale0:[0-9]+]] "test_umul_scale0"
; CHECK: Name [[test_umul_full:[0-9]+]] "test_umul_full"
; CHECK: Name [[test_smul_vec:[0-9]+]] "test_smul_vec"
; CHECK: Name [[test_umul_i16:[0-9]+]] "test_umul_i16"
; CHECK: Name [[test_smul_i1:[0-9]+]] "test_smul_i1"

; CHECK-DAG: TypeBool [[bool:[0-9]+]]
; CHECK-DAG: TypeInt [[i16:[0-9]+]] 16 0
; CHECK-DAG: TypeInt [[i32:[0-9]+]] 32 0
; CHECK-DAG: TypeInt [[i64:[0-9]+]] 64 0
; CHECK-DAG: Constant [[i32]] [[three:[0-9]+]] 3
; CHECK-DAG: Constant [[i32]] [[twentynine:[0-9]+]] 29
; CHECK-DAG: Constant [[i16]] [[five16:[0-9]+]] 5
; CHECK-DAG: Constant [[i16]] [[eleven16:[0-9]+]] 11
; CHECK-DAG: TypeStruct [[i16x2:[0-9]+]] [[i16]] [[i16]]
; CHECK-DAG: TypeStruct [[i32x2:[0-9]+]] [[i32]] [[i32]]
; CHECK-DAG: TypeVector [[v4i64:[0-9]+]] [[i64]] 4
; CHECK-DAG: TypeStruct [[v4i64x2:[0-9]+]] [[v4i64]] [[v4i64]]
; CHECK-DAG: Constant [[i64]] [[five_s:[0-9]+]] 5 0
; CHECK-DAG: Constant [[i64]] [[fiftynine_s:[0-9]+]] 59 0
; CHECK-DAG: ConstantComposite [[v4i64]] [[five:[0-9]+]] [[five_s]] [[five_s]] [[five_s]] [[five_s]]
; CHECK-DAG: ConstantComposite [[v4i64]] [[fiftynine:[0-9]+]] [[fiftynine_s]] [[fiftynine_s]] [[fiftynine_s]] [[fiftynine_s]]

define spir_func i32 @test_smul(i32 %a, i32 %b) {
  %r = call i32 @llvm.smul.fix.i32(i32 %a, i32 %b, i32 3)
  ret i32 %r
}

; CHECK: Function [[i32]] [[test_smul]]
; CHECK-NEXT: FunctionParameter [[i32]] [[a:[0-9]+]]
; CHECK-NEXT: FunctionParameter [[i32]] [[b:[0-9]+]]
; CHECK-EMPTY:
; CHECK-NEXT: Label
; CHECK-NEXT: SMulExtended [[i32x2]] [[mul:[0-9]+]] [[a]] [[b]]
; CHECK-NEXT: CompositeExtract [[i32]] [[hi:[0-9]+]] [[mul]] 1
; CHECK-NEXT: CompositeExtract [[i32]] [[lo:[0-9]+]] [[mul]] 0
; CHECK-NEXT: ShiftRightLogical [[i32]] [[lobits:[0-9]+]] [[lo]] [[three]]
; CHECK-NEXT: ShiftLeftLogical [[i32]] [[hibits:[0-9]+]] [[hi]] [[twentynine]]
; CHECK-NEXT: BitwiseOr [[i32]] [[res:[0-9]+]] [[hibits]] [[lobits]]
; CHECK-NEXT: ReturnValue [[res]]

define spir_func i32 @test_umul_scale0(i32 %a, i32 %b) {
  %r = call i32 @llvm.umul.fix.i32(i32 %a, i32 %b, i32 0)
  ret i32 %r
}

; CHECK: Function [[i32]] [[test_umul_scale0]]
; CHECK-NEXT: FunctionParameter [[i32]] [[a:[0-9]+]]
; CHECK-NEXT: FunctionParameter [[i32]] [[b:[0-9]+]]
; CHECK-EMPTY:
; CHECK-NEXT: Label
; CHECK-NEXT: IMul [[i32]] [[res:[0-9]+]] [[a]] [[b]]
; CHECK-NEXT: ReturnValue [[res]]

define spir_func i32 @test_umul_full(i32 %a, i32 %b) {
  %r = call i32 @llvm.umul.fix.i32(i32 %a, i32 %b, i32 32)
  ret i32 %r
}

; CHECK: Function [[i32]] [[test_umul_full]]
; CHECK-NEXT: FunctionParameter [[i32]] [[a:[0-9]+]]
; CHECK-NEXT: FunctionParameter [[i32]] [[b:[0-9]+]]
; CHECK-EMPTY:
; CHECK-NEXT: Label
; CHECK-NEXT: UMulExtended [[i32x2]] [[mul:[0-9]+]] [[a]] [[b]]
; CHECK-NEXT: CompositeExtract [[i32]] [[hi:[0-9]+]] [[mul]] 1
; CHECK-NEXT: ReturnValue [[hi]]

define spir_func <4 x i64> @test_smul_vec(<4 x i64> %a, <4 x i64> %b) {
  %r = call <4 x i64> @llvm.smul.fix.v4i64(<4 x i64> %a, <4 x i64> %b, i32 5)
  ret <4 x i64> %r
}

; CHECK: Function [[v4i64]] [[test_smul_vec]]
; CHECK-NEXT: FunctionParameter [[v4i64]] [[a:[0-9]+]]
; CHECK-NEXT: FunctionParameter [[v4i64]] [[b:[0-9]+]]
; CHECK-EMPTY:
; CHECK-NEXT: Label
; CHECK-NEXT: SMulExtended [[v4i64x2]] [[mul:[0-9]+]] [[a]] [[b]]
; CHECK-NEXT: CompositeExtract [[v4i64]] [[hi:[0-9]+]] [[mul]] 1
; CHECK-NEXT: CompositeExtract [[v4i64]] [[lo:[0-9]+]] [[mul]] 0
; CHECK-NEXT: ShiftRightLogical [[v4i64]] [[lobits:[0-9]+]] [[lo]] [[five]]
; CHECK-NEXT: ShiftLeftLogical [[v4i64]] [[hibits:[0-9]+]] [[hi]] [[fiftynine]]
; CHECK-NEXT: BitwiseOr [[v4i64]] [[res:[0-9]+]] [[hibits]] [[lobits]]
; CHECK-NEXT: ReturnValue [[res]]

define spir_func i16 @test_umul_i16(i16 %a, i16 %b) {
  %r = call i16 @llvm.umul.fix.i16(i16 %a, i16 %b, i32 5)
  ret i16 %r
}

; CHECK: Function [[i16]] [[test_umul_i16]]
; CHECK-NEXT: FunctionParameter [[i16]] [[a:[0-9]+]]
; CHECK-NEXT: FunctionParameter [[i16]] [[b:[0-9]+]]
; CHECK-EMPTY:
; CHECK-NEXT: Label
; CHECK-NEXT: UMulExtended [[i16x2]] [[mul:[0-9]+]] [[a]] [[b]]
; CHECK-NEXT: CompositeExtract [[i16]] [[hi:[0-9]+]] [[mul]] 1
; CHECK-NEXT: CompositeExtract [[i16]] [[lo:[0-9]+]] [[mul]] 0
; CHECK-NEXT: ShiftRightLogical [[i16]] [[lobits:[0-9]+]] [[lo]] [[five16]]
; CHECK-NEXT: ShiftLeftLogical [[i16]] [[hibits:[0-9]+]] [[hi]] [[eleven16]]
; CHECK-NEXT: BitwiseOr [[i16]] [[res:[0-9]+]] [[hibits]] [[lobits]]
; CHECK-NEXT: ReturnValue [[res]]

define spir_func i1 @test_smul_i1(i1 %a, i1 %b) {
  %r = call i1 @llvm.smul.fix.i1(i1 %a, i1 %b, i32 0)
  ret i1 %r
}

; CHECK: Function [[bool]] [[test_smul_i1]]
; CHECK-NEXT: FunctionParameter [[bool]] [[a:[0-9]+]]
; CHECK-NEXT: FunctionParameter [[bool]] [[b:[0-9]+]]
; CHECK-EMPTY:
; CHECK-NEXT: Label
; CHECK-NEXT: LogicalAnd [[bool]] [[res:[0-9]+]] [[a]] [[b]]
; CHECK-NEXT: ReturnValue [[res]]
