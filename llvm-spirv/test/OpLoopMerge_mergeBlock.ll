; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv --to-text %t.spv -o - | FileCheck %s

; CHECK: LoopMerge
; CHECK-NEXT: BranchConditional

target triple = "spir64"

define spir_kernel void @test(i32 %n) {
entry:
  %cmp0 = icmp sgt i32 %n, 0
  br i1 %cmp0, label %outer, label %exit
outer:
  %i = phi i32 [ 0, %entry ], [ %i.next, %latch ]
  %cmp1 = icmp eq i32 %i, 0
  br i1 %cmp1, label %latch, label %inner
inner:
  %cmp2 = icmp eq i32 %i, 1
  br i1 %cmp2, label %latch, label %inner, !llvm.loop !1
latch:
  %i.next = add i32 %i, 1
  %cmp3 = icmp slt i32 %i.next, %n
  br i1 %cmp3, label %outer, label %exit, !llvm.loop !0
exit:
  ret void
}

!0 = distinct !{!0, !2}
!1 = distinct !{!1, !2}
!2 = !{!"llvm.loop.unroll.disable"}
