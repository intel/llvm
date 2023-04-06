; RUN: llvm-as < %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv %t.spv -to-text -o - | FileCheck %s
; RUN: spirv-val %t.spv

; Test checks that saturation addition and substraction llvm intrinsics
; are translated into instruction from OpenCL Extended Instruction Set.

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-unknown"

; CHECK: ExtInstImport [[ext:[0-9]+]] "OpenCL.std"

; CHECK: Name [[test_uadd:[0-9]+]] "test_uadd"
; CHECK: Name [[test_usub:[0-9]+]] "test_usub"
; CHECK: Name [[test_sadd:[0-9]+]] "test_sadd"
; CHECK: Name [[test_ssub:[0-9]+]] "test_ssub"
; CHECK: Name [[test_vectors:[0-9]+]] "test_vectors"

; CHECK-DAG: TypeInt [[int:[0-9]+]] 32 0
; CHECK-DAG: TypeVoid [[void:[0-9]+]]
; CHECK: TypeVector [[vector:[0-9]+]] [[int]] 4

define spir_func void @test_uadd(i32 %a, i32 %b) {
entry:
  %0 = call i32 @llvm.uadd.sat.i32(i32 %a, i32 %b)
  ret void
}

; CHECK: Function [[void]] [[test_uadd]]
; CHECK-NEXT: FunctionParameter [[int]] [[lhs:[0-9]+]]
; CHECK-NEXT: FunctionParameter [[int]] [[rhs:[0-9]+]]
; CHECK-EMPTY:
; CHECK-NEXT: Label
; CHECK-NEXT: ExtInst [[int]] {{[0-9]+}} [[ext]] u_add_sat [[lhs]] [[rhs]]
; CHECK-NEXT: Return

define spir_func void @test_usub(i32 %a, i32 %b) {
entry:
  %0 = call i32 @llvm.usub.sat.i32(i32 %a, i32 %b)
  ret void
}

; CHECK: Function [[void]] [[test_usub]]
; CHECK-NEXT: FunctionParameter [[int]] [[lhs:[0-9]+]]
; CHECK-NEXT: FunctionParameter [[int]] [[rhs:[0-9]+]]
; CHECK-EMPTY:
; CHECK-NEXT: Label
; CHECK-NEXT: ExtInst [[int]] {{[0-9]+}} [[ext]] u_sub_sat [[lhs]] [[rhs]]
; CHECK-NEXT: Return

define spir_func void @test_sadd(i32 %a, i32 %b) {
entry:
  %0 = call i32 @llvm.sadd.sat.i32(i32 %a, i32 %b)
  ret void
}

; CHECK: Function [[void]] [[test_sadd]]
; CHECK-NEXT: FunctionParameter [[int]] [[lhs:[0-9]+]]
; CHECK-NEXT: FunctionParameter [[int]] [[rhs:[0-9]+]]
; CHECK-EMPTY:
; CHECK-NEXT: Label
; CHECK-NEXT: ExtInst [[int]] {{[0-9]+}} [[ext]] s_add_sat [[lhs]] [[rhs]]
; CHECK-NEXT: Return

define spir_func void @test_ssub(i32 %a, i32 %b) {
entry:
  %0 = call i32 @llvm.ssub.sat.i32(i32 %a, i32 %b)
  ret void
}

; CHECK: Function [[void]] [[test_ssub]]
; CHECK-NEXT: FunctionParameter [[int]] [[lhs:[0-9]+]]
; CHECK-NEXT: FunctionParameter [[int]] [[rhs:[0-9]+]]
; CHECK-EMPTY:
; CHECK-NEXT: Label
; CHECK-NEXT: ExtInst [[int]] {{[0-9]+}} [[ext]] s_sub_sat [[lhs]] [[rhs]]
; CHECK-NEXT: Return

define spir_func void @test_vectors(<4 x i32> %a, <4 x i32> %b) {
entry:
  %0 = call <4 x i32> @llvm.uadd.sat.v4i32(<4 x i32> %a, <4 x i32> %b)
  ret void
}

; CHECK: Function [[void]] [[test_vectors]]
; CHECK-NEXT: FunctionParameter [[vector]] [[lhs:[0-9]+]]
; CHECK-NEXT: FunctionParameter [[vector]] [[rhs:[0-9]+]]
; CHECK-EMPTY:
; CHECK-NEXT: Label
; CHECK-NEXT: ExtInst [[vector]] {{[0-9]+}} [[ext]] u_add_sat [[lhs]] [[rhs]]
; CHECK-NEXT: Return

declare i32 @llvm.uadd.sat.i32(i32, i32);
declare i32 @llvm.usub.sat.i32(i32, i32);
declare i32 @llvm.sadd.sat.i32(i32, i32);
declare i32 @llvm.ssub.sat.i32(i32, i32);
declare <4 x i32> @llvm.uadd.sat.v4i32(<4 x i32>, <4 x i32>);

!llvm.module.flags = !{!0}
!opencl.ocl.version = !{!1}
!opencl.spir.version = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, i32 0}
!2 = !{i32 1, i32 2}
