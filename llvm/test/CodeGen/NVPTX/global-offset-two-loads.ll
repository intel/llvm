; RUN: opt -bugpoint-enable-legacy-pm -globaloffset %s -S -o - | FileCheck %s

target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

declare ptr @llvm.nvvm.implicit.offset()

define i32 @test_two_loads() {
; CHECK-LABEL: define i32 @test_two_loads() {
; CHECK-NEXT:  [[ENTRY:.*:]]
; CHECK-NEXT:    [[RES:%.*]] = add i32 0, 0
; CHECK-NEXT:    ret i32 [[RES]]
;
entry:
  %offset = tail call ptr @llvm.nvvm.implicit.offset()
  %gep = getelementptr inbounds nuw i8, ptr %offset, i64 4
  %load1 = load i32, ptr %gep, align 4
  %load2 = load i32, ptr %offset, align 4
  %res = add i32 %load1, %load2
  ret i32 %res
}

; CHECK-LABEL: define i32 @test_two_loads_with_offset(
; CHECK-SAME: ptr [[PTR:%.*]]) {
; CHECK-NEXT:  [[ENTRY:.*:]]
; CHECK-NEXT:    [[GEP:%.*]] = getelementptr inbounds nuw i8, ptr [[PTR]], i64 4
; CHECK-NEXT:    [[LOAD1:%.*]] = load i32, ptr [[GEP]], align 4
; CHECK-NEXT:    [[LOAD2:%.*]] = load i32, ptr [[PTR]], align 4
; CHECK-NEXT:    [[RES:%.*]] = add i32 [[LOAD1]], [[LOAD2]]
; CHECK-NEXT:    ret i32 [[RES]]

!llvm.module.flags = !{!0}

!0 = !{i32 1, !"sycl-device", i32 1}
