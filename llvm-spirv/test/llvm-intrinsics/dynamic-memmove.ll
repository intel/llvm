; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o %t.txt
; RUN: FileCheck < %t.txt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; CHECK-SPIRV-NOT: llvm.memmove

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

declare void @llvm.memmove.p1i8.p1i8.i64(i8 addrspace(1)* nocapture, i8 addrspace(1)* nocapture readonly, i64, i1)

define spir_func void @memmove_caller(i8 addrspace(1)* %dst, i8 addrspace(1)* %src, i64 %n) {
entry:
  call void @llvm.memmove.p1i8.p1i8.i64(i8 addrspace(1)* %dst, i8 addrspace(1)* %src, i64 %n, i1 false)
  call void @llvm.memmove.p1i8.p1i8.i64(i8 addrspace(1)* %dst, i8 addrspace(1)* %src, i64 %n, i1 false)
  call void @llvm.memmove.p1i8.p1i8.i64(i8 addrspace(1)* %dst, i8 addrspace(1)* %src, i64 %n, i1 false)
  ret void

; CHECK-LLVM: @memmove_caller(i8 addrspace(1)* [[DST:%.*]], i8 addrspace(1)* [[SRC:%.*]], i64 [[N:%.*]])
; CHECK-LLVM:  [[SRC_CMP:%.*]] = ptrtoint i8 addrspace(1)* [[SRC]] to i64
; CHECK-LLVM:  [[DST_CMP:%.*]] = ptrtoint i8 addrspace(1)* [[DST]] to i64
; CHECK-LLVM: [[COMPARE_SRC_DST:%.*]] = icmp ult i64 [[SRC_CMP]], [[DST_CMP]]
; CHECK-LLVM-NEXT: [[COMPARE_N_TO_0:%.*]] = icmp eq i64 [[N]], 0
; CHECK-LLVM-NEXT: br i1 [[COMPARE_SRC_DST]], label %[[COPY_BACKWARDS:.*]], label %[[COPY_FORWARD:.*]]
; CHECK-LLVM: [[COPY_BACKWARDS]]:
}
