; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.bc
; RUN: llvm-dis < %t.bc | FileCheck %s --check-prefix=CHECK-LLVM

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-unknown"

; CHECK-SPIRV: BranchConditional [[#]] [[#if_then:]] [[#if_end:]]
; CHECK-SPIRV: Label [[#if_then]]
; CHECK-SPIRV: Branch [[#if_end]]
; CHECK-SPIRV: Label [[#if_end]]
; CHECK-SPIRV: Phi [[#]] [[#Var:]]
; CHECK-SPIRV: Switch [[#Var]]
; CHECK-SPIRV-COUNT-10: Label
; CHECK-SPIRV: Label [[#epilog:]]
; CHECK-SPIRV: Branch [[#exit:]]
; CHECK-SPIRV: Label [[#exit]] 
; CHECK-SPIRV: Return
; CHECK-SPIRV-NOT: Label
; CHECK-SPIRV: FunctionEnd

; CHECK-LLVM: br i1 %{{.*}}, label %[[IF_THEN:if\.then]], label %[[IF_END:if\.end]]
; CHECK-LLVM: [[IF_THEN]]:
; CHECK-LLVM: br label %[[IF_END]]
; CHECK-LLVM: [[IF_END]]:
; CHECK-LLVM: %[[SWVAL:.*]] = phi i8 [ %{{.*}}, %[[IF_THEN]] ], [ %{{.*}}, %[[ENTRY:entry]] ]
; CHECK-LLVM: switch i8 %[[SWVAL]], label %[[DEFAULT:.*]]
; CHECK-LLVM-COUNT-10: br label %[[SW_EPILOG:.*]]

define spir_func void @foo(i64 noundef %addr, i64 noundef %as) {
entry:
  %src = inttoptr i64 %as to ptr addrspace(4)
  %val = load i8, ptr addrspace(4) %src
  %cmp = icmp sgt i8 %val, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:
  %add.ptr = getelementptr inbounds i8, ptr addrspace(4) %src, i64 1
  %cond = load i8, ptr addrspace(4) %add.ptr
  br label %if.end

if.end:
  %swval = phi i8 [ %cond, %if.then ], [ %val, %entry ]
  switch i8 %swval, label %sw.default [
    i8 -127, label %sw.epilog
    i8 -126, label %sw.bb3
    i8 -125, label %sw.bb4
    i8 -111, label %sw.bb5
    i8 -110, label %sw.bb6
    i8 -109, label %sw.bb7
    i8 -15, label %sw.bb8
    i8 -14, label %sw.bb8
    i8 -13, label %sw.bb8
    i8 -124, label %sw.bb9
    i8 -95, label %sw.bb10
    i8 -123, label %sw.bb11
  ]

sw.bb3:
  br label %sw.epilog

sw.bb4:
  br label %sw.epilog

sw.bb5:
  br label %sw.epilog

sw.bb6:
  br label %sw.epilog

sw.bb7:
  br label %sw.epilog

sw.bb8:
  br label %sw.epilog

sw.bb9:
  br label %sw.epilog

sw.bb10:
  br label %sw.epilog

sw.bb11:
  br label %sw.epilog

sw.default:
  br label %sw.epilog

sw.epilog:
  br label %exit

exit:
  ret void
}
