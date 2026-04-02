; RUN: opt < %s -passes=instcombine -S | FileCheck %s

; GEP canonicalization must skip element types that wrap spirv.CooperativeMatrixKHR
; because DL.getTypeAllocSize() is not meaningful for this target extension type.
; Verify that visiting chained GEPs on [N x [M x JointMatrix]] arrays does NOT
; rewrite the resulting single-index GEP to use an [sizeof x i8] stride.

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spirv64-unknown-unknown"

%matrix_acc_t = type { target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 2) }
%matrix_a_t   = type { target("spirv.CooperativeMatrixKHR", i16,   3, 16, 32, 0) }
%matrix_b_t   = type { target("spirv.CooperativeMatrixKHR", i16,   3, 32, 16, 1) }

; InstCombine folds [2 x [2 x %matrix_acc_t]], ptr, 0, %i -> [2 x %matrix_acc_t], ptr, %i
; then [2 x %matrix_acc_t], ptr, 0, %j -> %matrix_acc_t, ptr, %j.
; The final single-index GEP on %matrix_acc_t must NOT be canonicalized to [N x i8].
define ptr addrspace(4) @test_acc_matrix_2d_array(ptr addrspace(4) %p, i64 %i, i64 %j) {
; CHECK-LABEL: define ptr addrspace(4) @test_acc_matrix_2d_array(
; CHECK-SAME: ptr addrspace(4) [[P:%.*]], i64 [[I:%.*]], i64 [[J:%.*]]) {
; CHECK-NEXT:    [[ARRAYIDX:%.*]] = getelementptr inbounds nuw [2 x [[MATRIX_ACC_T:%.*]]], ptr addrspace(4) [[P]], i64 [[I]]
; CHECK-NEXT:    [[ARRAYIDX2:%.*]] = getelementptr inbounds nuw [[MATRIX_ACC_T]], ptr addrspace(4) [[ARRAYIDX]], i64 [[J]]
; CHECK-NEXT:    ret ptr addrspace(4) [[ARRAYIDX2]]
;
  %arrayidx = getelementptr inbounds nuw [2 x [2 x %matrix_acc_t]], ptr addrspace(4) %p, i64 0, i64 %i
  %arrayidx2 = getelementptr inbounds nuw [2 x %matrix_acc_t], ptr addrspace(4) %arrayidx, i64 0, i64 %j
  ret ptr addrspace(4) %arrayidx2
}

; Same pattern for the A matrix: [2 x [1 x %matrix_a_t]]
define ptr addrspace(4) @test_a_matrix_2d_array(ptr addrspace(4) %p, i64 %i) {
; CHECK-LABEL: define ptr addrspace(4) @test_a_matrix_2d_array(
; CHECK-SAME: ptr addrspace(4) [[P:%.*]], i64 [[I:%.*]]) {
; CHECK-NEXT:    [[ARRAYIDX:%.*]] = getelementptr inbounds nuw [1 x [[MATRIX_A_T:%.*]]], ptr addrspace(4) [[P]], i64 [[I]]
; CHECK-NEXT:    ret ptr addrspace(4) [[ARRAYIDX]]
;
  %arrayidx = getelementptr inbounds nuw [2 x [1 x %matrix_a_t]], ptr addrspace(4) %p, i64 0, i64 %i
  %arrayidx2 = getelementptr inbounds nuw [1 x %matrix_a_t], ptr addrspace(4) %arrayidx, i64 0, i64 0
  ret ptr addrspace(4) %arrayidx2
}

; Same pattern for the B matrix: [2 x [1 x %matrix_b_t]]
define ptr addrspace(4) @test_b_matrix_2d_array(ptr addrspace(4) %p, i64 %i) {
; CHECK-LABEL: define ptr addrspace(4) @test_b_matrix_2d_array(
; CHECK-SAME: ptr addrspace(4) [[P:%.*]], i64 [[I:%.*]]) {
; CHECK-NEXT:    [[ARRAYIDX:%.*]] = getelementptr inbounds nuw [1 x [[MATRIX_B_T:%.*]]], ptr addrspace(4) [[P]], i64 [[I]]
; CHECK-NEXT:    ret ptr addrspace(4) [[ARRAYIDX]]
;
  %arrayidx = getelementptr inbounds nuw [2 x [1 x %matrix_b_t]], ptr addrspace(4) %p, i64 0, i64 %i
  %arrayidx2 = getelementptr inbounds nuw [1 x %matrix_b_t], ptr addrspace(4) %arrayidx, i64 0, i64 0
  ret ptr addrspace(4) %arrayidx2
}
