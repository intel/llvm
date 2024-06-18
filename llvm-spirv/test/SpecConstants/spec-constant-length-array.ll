; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv -spirv-text -o - %t.bc | FileCheck %s --check-prefix CHECK-SPV
; RUN: llvm-spirv -o %t.spv %t.bc
; RUN: spirv-val %t.spv
; RUN: llvm-spirv -r -o - %t.spv | llvm-dis | FileCheck %s --check-prefix CHECK-LLVM

; CHECK-SPV-DAG: Decorate [[#I64_CONST:]] SpecId [[#]]
; CHECK-SPV-DAG: Decorate [[#I32_CONST:]] SpecId [[#]]
; CHECK-SPV-DAG: Decorate [[#I8_CONST:]] SpecId [[#]]
; CHECK-SPV-DAG: Decorate [[#SCLA_0:]] Alignment 4
; CHECK-SPV-DAG: Decorate [[#SCLA_1:]] Alignment 2
; CHECK-SPV-DAG: Decorate [[#SCLA_2:]] Alignment 16

; CHECK-SPV-DAG: TypeInt [[#I64_TY:]] 64
; CHECK-SPV-DAG: TypeInt [[#I32_TY:]] 32
; CHECK-SPV-DAG: TypeInt [[#I8_TY:]] 8

; CHECK-SPV-DAG: SpecConstant [[#I64_TY]] [[#LENGTH_0:]]
; CHECK-SPV-DAG: SpecConstant [[#I32_TY]] [[#LENGTH_1:]]
; CHECK-SPV-DAG: SpecConstant [[#I8_TY]] [[#LENGTH_2:]]

; CHECK-SPV-DAG: TypeFloat [[#FLOAT_TY:]] 32
; CHECK-SPV-DAG: TypePointer [[#FLOAT_PTR_TY:]] [[#FUNCTION_SC:]] [[#FLOAT_TY]]
; CHECK-SPV-DAG: TypeArray [[#ARR_TY_0:]] [[#FLOAT_TY]] [[#LENGTH_0]]
; CHECK-SPV-DAG: TypePointer [[#ARR_PTR_TY_0:]] [[#FUNCTION_SC]] [[#ARR_TY_0]]
; CHECK-SPV-DAG: TypePointer [[#I8_PTR_TY:]] [[#FUNCTION_SC]] [[#I8_TY]]
; CHECK-SPV-DAG: TypeArray [[#ARR_TY_1:]] [[#I8_TY]] [[#LENGTH_1]]
; CHECK-SPV-DAG: TypePointer [[#ARR_PTR_TY_1:]] [[#FUNCTION_SC]] [[#ARR_TY_1]]
; CHECK-SPV-DAG: TypeFloat [[#DOUBLE_TY:]] 64
; CHECK-SPV-DAG: TypeStruct [[#STR_TY:]] [[#DOUBLE_TY]] [[#DOUBLE_TY]]
; CHECK-SPV-DAG: TypePointer [[#STR_PTR_TY:]] [[#FUNCTION_SC]] [[#STR_TY]]
; CHECK-SPV-DAG: TypeArray [[#ARR_TY_2:]] [[#STR_TY]] [[#LENGTH_2]]
; CHECK-SPV-DAG: TypePointer [[#ARR_PTR_TY_2:]] [[#FUNCTION_SC]] [[#ARR_TY_2:]]

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

%struct_type = type { double, double }

define spir_kernel void @test() {
 entry:
  %length0 = call i64 @_Z20__spirv_SpecConstantix(i32 0, i64 1), !SYCL_SPEC_CONST_SYM_ID !0
  %length1 = call i32 @_Z20__spirv_SpecConstantii(i32 1, i32 2), !SYCL_SPEC_CONST_SYM_ID !1
  %length2 = call i8 @_Z20__spirv_SpecConstantic(i32 2, i8 4), !SYCL_SPEC_CONST_SYM_ID !2

  ; CHECK-SPV: Variable [[#ARR_PTR_TY_0]] [[#SCLA_0]] [[#FUNCTION_SC]]
  ; CHECK-SPV: Variable [[#ARR_PTR_TY_1]] [[#SCLA_1]] [[#FUNCTION_SC]]
  ; CHECK-SPV: Variable [[#ARR_PTR_TY_2]] [[#SCLA_2]] [[#FUNCTION_SC]]

  ; CHECK-LLVM: %[[ALLOCA0:.*]] = alloca [1 x float], align 4
  ; CHECK-LLVM: %[[ALLOCA1:.*]] = alloca [2 x i8], align 2
  ; CHECK-LLVM: %[[ALLOCA2:.*]] = alloca [4 x %struct_type], align 16

  ; CHECK-SPV: Bitcast [[#FLOAT_PTR_TY]] [[#]] [[#SCLA_0]]

  ; CHECK-LLVM: %[[VAR0:.*]] = bitcast ptr %[[ALLOCA0]] to ptr
  %scla0 = alloca float, i64 %length0, align 4

  ; CHECK-SPV: Bitcast [[#I8_PTR_TY]] [[#]] [[#SCLA_1]]

  ; CHECK-LLVM: %[[VAR1:.*]] = bitcast ptr %[[ALLOCA1]] to ptr
  %scla1 = alloca i8, i32 %length1, align 2

  ; CHECK-SPV: Bitcast [[#STR_PTR_TY]] [[#]] [[#SCLA_2]]

  ; CHECK-LLVM: %[[VAR2:.*]] = bitcast ptr %[[ALLOCA2]] to ptr
  %scla2 = alloca %struct_type, i8 %length2, align 16
  ret void
}

declare i8 @_Z20__spirv_SpecConstantic(i32, i8)
declare i32 @_Z20__spirv_SpecConstantii(i32, i32)
declare i64 @_Z20__spirv_SpecConstantix(i32, i64)

!0 = !{!"i64_spec_const", i32 0}
!1 = !{!"i32_spec_const", i32 1}
!2 = !{!"i8_spec_const", i32 2}
