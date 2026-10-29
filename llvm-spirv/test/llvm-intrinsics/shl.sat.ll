; RUN: llvm-spirv %s -o %t.spv
; RUN: llvm-spirv %t.spv -to-text -o - | FileCheck %s
; RUN: spirv-val %t.spv

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-unknown"

; CHECK: Name [[test_ushl:[0-9]+]] "test_ushl"
; CHECK: Name [[test_sshl:[0-9]+]] "test_sshl"
; CHECK: Name [[test_ushl_vec:[0-9]+]] "test_ushl_vec"

; CHECK-DAG: TypeInt [[i32:[0-9]+]] 32 0
; CHECK-DAG: TypeVoid [[void:[0-9]+]]
; CHECK-DAG: TypeBool [[bool:[0-9]+]]
; CHECK-DAG: Constant [[i32]] [[allones:[0-9]+]] 4294967295
; CHECK: TypeVector [[v4i32:[0-9]+]] [[i32]] 4

define spir_func void @test_ushl(i32 %a, i32 %b) {
entry:
  %0 = call i32 @llvm.ushl.sat.i32(i32 %a, i32 %b)
  ret void
}

; CHECK: Function [[void]] [[test_ushl]]
; CHECK-NEXT: FunctionParameter [[i32]] [[a:[0-9]+]]
; CHECK-NEXT: FunctionParameter [[i32]] [[b:[0-9]+]]
; CHECK-EMPTY:
; CHECK-NEXT: Label
; CHECK-NEXT: ShiftLeftLogical [[i32]] [[shifted:[0-9]+]] [[a]] [[b]]
; CHECK-NEXT: ShiftRightLogical [[i32]] [[shiftedback:[0-9]+]] [[shifted]] [[b]]
; CHECK-NEXT: INotEqual [[bool]] [[overflow:[0-9]+]] [[shiftedback]] [[a]]
; CHECK-NEXT: Select [[i32]] {{[0-9]+}} [[overflow]] [[allones]] [[shifted]]
; CHECK-NEXT: Return

define spir_func void @test_sshl(i32 %a, i32 %b) {
entry:
  %0 = call i32 @llvm.sshl.sat.i32(i32 %a, i32 %b)
  ret void
}

; CHECK: Function [[void]] [[test_sshl]]
; CHECK-NEXT: FunctionParameter [[i32]] [[a2:[0-9]+]]
; CHECK-NEXT: FunctionParameter [[i32]] [[b2:[0-9]+]]
; CHECK-EMPTY:
; CHECK-NEXT: Label
; CHECK-NEXT: ShiftLeftLogical [[i32]] [[shifted2:[0-9]+]] [[a2]] [[b2]]
; CHECK-NEXT: ShiftRightArithmetic [[i32]] [[shiftedback2:[0-9]+]] [[shifted2]] [[b2]]
; CHECK-NEXT: INotEqual [[bool]] [[overflow2:[0-9]+]] [[shiftedback2]] [[a2]]
; CHECK-NEXT: SLessThan [[bool]] [[isneg:[0-9]+]] [[a2]] {{[0-9]+}}
; CHECK-NEXT: Select [[i32]] [[satval:[0-9]+]] [[isneg]] {{[0-9]+}} {{[0-9]+}}
; CHECK-NEXT: Select [[i32]] {{[0-9]+}} [[overflow2]] [[satval]] [[shifted2]]
; CHECK-NEXT: Return

define spir_func void @test_ushl_vec(<4 x i32> %a, <4 x i32> %b) {
entry:
  %0 = call <4 x i32> @llvm.ushl.sat.v4i32(<4 x i32> %a, <4 x i32> %b)
  ret void
}

; CHECK: Function [[void]] [[test_ushl_vec]]
; CHECK-NEXT: FunctionParameter [[v4i32]] [[va:[0-9]+]]
; CHECK-NEXT: FunctionParameter [[v4i32]] [[vb:[0-9]+]]
; CHECK-EMPTY:
; CHECK-NEXT: Label
; CHECK-NEXT: ShiftLeftLogical [[v4i32]] [[vshifted:[0-9]+]] [[va]] [[vb]]
; CHECK-NEXT: ShiftRightLogical [[v4i32]] [[vshiftedback:[0-9]+]] [[vshifted]] [[vb]]
; CHECK-NEXT: INotEqual {{[0-9]+}} [[voverflow:[0-9]+]] [[vshiftedback]] [[va]]
; CHECK-NEXT: Select [[v4i32]] {{[0-9]+}} [[voverflow]] {{[0-9]+}} [[vshifted]]
; CHECK-NEXT: Return

declare i32 @llvm.ushl.sat.i32(i32, i32)
declare i32 @llvm.sshl.sat.i32(i32, i32)
declare <4 x i32> @llvm.ushl.sat.v4i32(<4 x i32>, <4 x i32>)

!llvm.module.flags = !{!0}
!opencl.ocl.version = !{!1}
!opencl.spir.version = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, i32 0}
!2 = !{i32 1, i32 2}
