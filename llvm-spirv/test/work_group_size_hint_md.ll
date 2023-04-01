; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv -to-text %t.spv -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV
;
; The purpose of this test is to check that the work_group_size_hint metadata
; is correctly converted to the LocalSizeHint execution mode for the kernels it
; is applied to.
;
; CHECK-SPIRV: EntryPoint 6 [[TEST1:[0-9]+]] "test1"
; CHECK-SPIRV: EntryPoint 6 [[TEST2:[0-9]+]] "test2"
; CHECK-SPIRV: EntryPoint 6 [[TEST3:[0-9]+]] "test3"
; CHECK-SPIRV: ExecutionMode [[TEST1]] 18 1 2 3
; CHECK-SPIRV: ExecutionMode [[TEST2]] 18 2 3 1
; CHECK-SPIRV: ExecutionMode [[TEST3]] 18 3 1 1

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-unknown"

define spir_kernel void @test1() !work_group_size_hint !1 {
entry:
  ret void
}

define spir_kernel void @test2() !work_group_size_hint !2 {
entry:
  ret void
}

define spir_kernel void @test3() !work_group_size_hint !3 {
entry:
  ret void
}

!1 = !{i32 1, i32 2, i32 3}
!2 = !{i32 2, i32 3}
!3 = !{i32 3}
