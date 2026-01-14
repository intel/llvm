; RUN: opt -bugpoint-enable-legacy-pm -globaloffset %s -S -o - | FileCheck %s

; Check that phi is correctly handled in load's defs collection.

target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

define i64 @test_phi(i32 %x) {
; CHECK-LABEL: define i64 @test_phi(
; CHECK-SAME: i32 [[X:%.*]]) {
; CHECK-NEXT:  [[ENTRY:.*:]]
; CHECK-NEXT:    switch i32 [[X]], label %[[B5:.*]] [
; CHECK-NEXT:      i32 0, label %[[B1:.*]]
; CHECK-NEXT:      i32 1, label %[[B2:.*]]
; CHECK-NEXT:      i32 2, label %[[B3:.*]]
; CHECK-NEXT:    ]
; CHECK:       [[B1]]:
; CHECK-NEXT:    br label %[[B4:.*]]
; CHECK:       [[B2]]:
; CHECK-NEXT:    br label %[[B4]]
; CHECK:       [[B3]]:
; CHECK-NEXT:    br label %[[B4]]
; CHECK:       [[B4]]:
; CHECK-NEXT:    [[EXT1:%.*]] = zext i32 0 to i64
; CHECK-NEXT:    [[EXT2:%.*]] = zext i32 0 to i64
; CHECK-NEXT:    [[RES:%.*]] = add nuw nsw i64 [[EXT1]], [[EXT2]]
; CHECK-NEXT:    ret i64 [[RES]]
; CHECK:       [[B5]]:
; CHECK-NEXT:    unreachable
;
entry:
  switch i32 %x, label %b5 [
  i32 0, label %b1
  i32 1, label %b2
  i32 2, label %b3
  ]

b1:                                                ; preds = %entry
  %offset0 = tail call ptr addrspace(5) @llvm.amdgcn.implicit.offset()
  br label %b4

b2:                                                ; preds = %entry
  %offset1 = tail call ptr addrspace(5) @llvm.amdgcn.implicit.offset()
  %gep1 = getelementptr inbounds nuw i8, ptr addrspace(5) %offset1, i32 4
  br label %b4

b3:                                                ; preds = %entry
  %offset2 = tail call ptr addrspace(5) @llvm.amdgcn.implicit.offset()
  %gep2 = getelementptr inbounds nuw i8, ptr addrspace(5) %offset2, i32 8
  br label %b4

b4:                                               ; preds = %b3, %b2, %b1
  %p = phi ptr addrspace(5) [ %offset0, %b1 ], [ %gep1, %b2 ], [ %gep2, %b3 ]
  %load1 = load i32, ptr addrspace(5) %p, align 4
  %load2 = load i32, ptr addrspace(5) %p, align 4
  %ext1 = zext i32 %load1 to i64
  %ext2 = zext i32 %load2 to i64
  %res = add nuw nsw i64 %ext1, %ext2
  ret i64 %res

b5:                                               ; preds = %entry
  unreachable
}

; CHECK-LABEL: define i64 @test_phi_with_offset(
; CHECK-SAME: i32 [[X:%.*]], ptr addrspace(5) [[PTR:%.*]]) {
; CHECK-NEXT:  [[ENTRY:.*:]]
; CHECK-NEXT:    switch i32 [[X]], label %[[B5:.*]] [
; CHECK-NEXT:      i32 0, label %[[B1:.*]]
; CHECK-NEXT:      i32 1, label %[[B2:.*]]
; CHECK-NEXT:      i32 2, label %[[B3:.*]]
; CHECK-NEXT:    ]
; CHECK:       [[B1]]:
; CHECK-NEXT:    br label %[[B4:.*]]
; CHECK:       [[B2]]:
; CHECK-NEXT:    [[GEP1:%.*]] = getelementptr inbounds nuw i8, ptr addrspace(5) [[PTR]], i32 4
; CHECK-NEXT:    br label %[[B4]]
; CHECK:       [[B3]]:
; CHECK-NEXT:    [[GEP2:%.*]] = getelementptr inbounds nuw i8, ptr addrspace(5) [[PTR]], i32 8
; CHECK-NEXT:    br label %[[B4]]
; CHECK:       [[B4]]:
; CHECK-NEXT:    [[P:%.*]] = phi ptr addrspace(5) [ [[PTR]], %[[B1]] ], [ [[GEP1]], %[[B2]] ], [ [[GEP2]], %[[B3]] ]
; CHECK-NEXT:    [[LOAD1:%.*]] = load i32, ptr addrspace(5) [[P]], align 4
; CHECK-NEXT:    [[LOAD2:%.*]] = load i32, ptr addrspace(5) [[P]], align 4
; CHECK-NEXT:    [[EXT1:%.*]] = zext i32 [[LOAD1]] to i64
; CHECK-NEXT:    [[EXT2:%.*]] = zext i32 [[LOAD2]] to i64
; CHECK-NEXT:    [[RES:%.*]] = add nuw nsw i64 [[EXT1]], [[EXT2]]
; CHECK-NEXT:    ret i64 [[RES]]
; CHECK:       [[B5]]:
; CHECK-NEXT:    unreachable

declare ptr addrspace(5) @llvm.amdgcn.implicit.offset()

!llvm.module.flags = !{!0}

!0 = !{i32 1, !"sycl-device", i32 1}
