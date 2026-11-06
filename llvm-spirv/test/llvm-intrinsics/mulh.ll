; RUN: llvm-spirv %s -o %t.spv
; RUN: llvm-spirv %t.spv -to-text -o - | FileCheck %s
; RUN: spirv-val %t.spv

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-unknown"

; CHECK-DAG: Capability Int8

; CHECK: Name [[test_smulh:[0-9]+]] "test_smulh"
; CHECK: Name [[test_umulh_i8:[0-9]+]] "test_umulh_i8"
; CHECK: Name [[test_umulh_v2i1:[0-9]+]] "test_umulh_v2i1"

; CHECK-DAG: TypeBool [[bool:[0-9]+]]
; CHECK-DAG: TypeVector [[v2bool:[0-9]+]] [[bool]] 2
; CHECK-DAG: ConstantNull [[v2bool]] [[v2false:[0-9]+]]
; CHECK-DAG: TypeInt [[i8:[0-9]+]] 8 0
; CHECK-DAG: TypeInt [[i32:[0-9]+]] 32 0
; CHECK-DAG: TypeStruct [[i8x2:[0-9]+]] [[i8]] [[i8]]
; CHECK-DAG: TypeStruct [[i32x2:[0-9]+]] [[i32]] [[i32]]

define spir_func i32 @test_smulh(i32 %a, i32 %b) {
  %r = call i32 @llvm.smulh.i32(i32 %a, i32 %b)
  ret i32 %r
}

; CHECK: Function [[i32]] [[test_smulh]]
; CHECK-NEXT: FunctionParameter [[i32]] [[a:[0-9]+]]
; CHECK-NEXT: FunctionParameter [[i32]] [[b:[0-9]+]]
; CHECK-EMPTY:
; CHECK-NEXT: Label
; CHECK-NEXT: SMulExtended [[i32x2]] [[mul:[0-9]+]] [[a]] [[b]]
; CHECK-NEXT: CompositeExtract [[i32]] [[hi:[0-9]+]] [[mul]] 1
; CHECK-NEXT: ReturnValue [[hi]]

define spir_func i8 @test_umulh_i8(i8 %a, i8 %b) {
  %r = call i8 @llvm.umulh.i8(i8 %a, i8 %b)
  ret i8 %r
}

; CHECK: Function [[i8]] [[test_umulh_i8]]
; CHECK-NEXT: FunctionParameter [[i8]] [[a:[0-9]+]]
; CHECK-NEXT: FunctionParameter [[i8]] [[b:[0-9]+]]
; CHECK-EMPTY:
; CHECK-NEXT: Label
; CHECK-NEXT: UMulExtended [[i8x2]] [[mul:[0-9]+]] [[a]] [[b]]
; CHECK-NEXT: CompositeExtract [[i8]] [[hi:[0-9]+]] [[mul]] 1
; CHECK-NEXT: ReturnValue [[hi]]

define spir_func <2 x i1> @test_umulh_v2i1(<2 x i1> %a, <2 x i1> %b) {
  %r = call <2 x i1> @llvm.umulh.v2i1(<2 x i1> %a, <2 x i1> %b)
  ret <2 x i1> %r
}

; CHECK: Function [[v2bool]] [[test_umulh_v2i1]]
; CHECK-NEXT: FunctionParameter [[v2bool]] [[a:[0-9]+]]
; CHECK-NEXT: FunctionParameter [[v2bool]] [[b:[0-9]+]]
; CHECK-EMPTY:
; CHECK-NEXT: Label
; CHECK-NEXT: ReturnValue [[v2false]]
