; RUN: llvm-spirv %s -o %t.spv
; RUN: llvm-spirv --to-text %t.spv -o - | FileCheck %s
; RUN: spirv-val %t.spv

; CHECK: ExtInstImport [[#Ext:]] "OpenCL.std"
; CHECK-DAG: TypeInt [[#I32:]] 32 0
; CHECK-DAG: TypeInt [[#I64:]] 64 0
; CHECK-DAG: TypeFloat [[#F32:]] 32
; CHECK-DAG: TypeFloat [[#F64:]] 64
; CHECK-DAG: TypeVector [[#V4F32:]] [[#F32]] 4
; CHECK-DAG: TypeVector [[#V4I32:]] [[#I32]] 4
; CHECK: ExtInst [[#F32]] [[#R0:]] [[#Ext]] rint
; CHECK: ConvertFToS [[#I32]] [[#]] [[#R0]]
; CHECK: ExtInst [[#F64]] [[#R1:]] [[#Ext]] rint
; CHECK: ConvertFToS [[#I64]] [[#]] [[#R1]]
; CHECK: ExtInst [[#F32]] [[#R2:]] [[#Ext]] rint
; CHECK: ConvertFToS [[#I64]] [[#]] [[#R2]]
; CHECK: ExtInst [[#V4F32]] [[#R3:]] [[#Ext]] rint
; CHECK: ConvertFToS [[#V4I32]] [[#]] [[#R3]]

target triple = "spir64-unknown-unknown"

define spir_func i32 @lrint_f32(float %x) {
  %r = call i32 @llvm.lrint.i32.f32(float %x)
  ret i32 %r
}

define spir_func i64 @lrint_f64(double %x) {
  %r = call i64 @llvm.lrint.i64.f64(double %x)
  ret i64 %r
}

define spir_func i64 @llrint_f32(float %x) {
  %r = call i64 @llvm.llrint.i64.f32(float %x)
  ret i64 %r
}

define spir_func <4 x i32> @lrint_v4f32(<4 x float> %x) {
  %r = call <4 x i32> @llvm.lrint.v4i32.v4f32(<4 x float> %x)
  ret <4 x i32> %r
}
