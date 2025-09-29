; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc -o - | FileCheck %s --check-prefix=CHECK-LLVM

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64"

; CHECK-SPIRV-DAG: Name [[#EQ:]] "test_eq"
; CHECK-SPIRV-DAG: Name [[#NE:]] "test_ne"
; CHECK-SPIRV-DAG: Name [[#ULT:]] "test_ult"
; CHECK-SPIRV-DAG: Name [[#SLT:]] "test_slt"
; CHECK-SPIRV-DAG: Name [[#ULE:]] "test_ule"
; CHECK-SPIRV-DAG: Name [[#SLE:]] "test_sle"
; CHECK-SPIRV-DAG: Name [[#UGT:]] "test_ugt"
; CHECK-SPIRV-DAG: Name [[#SGT:]] "test_sgt"
; CHECK-SPIRV-DAG: Name [[#UGE:]] "test_uge"
; CHECK-SPIRV-DAG: Name [[#SGE:]] "test_sge"
; CHECK-SPIRV-DAG: Name [[#v3EQ:]] "test_v3_eq"
; CHECK-SPIRV-DAG: Name [[#v3NE:]] "test_v3_ne"
; CHECK-SPIRV-DAG: Name [[#v3ULT:]] "test_v3_ult"
; CHECK-SPIRV-DAG: Name [[#v3SLT:]] "test_v3_slt"
; CHECK-SPIRV-DAG: Name [[#v3ULE:]] "test_v3_ule"
; CHECK-SPIRV-DAG: Name [[#v3SLE:]] "test_v3_sle"
; CHECK-SPIRV-DAG: Name [[#v3UGT:]] "test_v3_ugt"
; CHECK-SPIRV-DAG: Name [[#v3SGT:]] "test_v3_sgt"
; CHECK-SPIRV-DAG: Name [[#v3UGE:]] "test_v3_uge"
; CHECK-SPIRV-DAG: Name [[#v3SGE:]] "test_v3_sge"

; CHECK-SPIRV-DAG: Function [[#]] [[#EQ]] [[#]] [[#]]
; CHECK-SPIRV-DAG: FunctionParameter [[#]] [[#A:]]
; CHECK-SPIRV-DAG: FunctionParameter [[#]] [[#B:]]
; CHECK-SPIRV-DAG: Label [[#]]
; CHECK-SPIRV-DAG: IEqual [[#]] [[#R:]] [[#A]] [[#B]]
; CHECK-SPIRV-DAG: ReturnValue [[#R]]
; CHECK-SPIRV-DAG: FunctionEnd
; CHECK-LLVM-DAG: icmp eq 

define i1 @test_eq(i32 %a, i32 %b) {
  %r = icmp eq i32 %a, %b
  ret i1 %r
}

; CHECK-SPIRV-DAG: Function [[#]] [[#NE]] [[#]] [[#]]
; CHECK-SPIRV-DAG: FunctionParameter [[#]] [[#A:]]
; CHECK-SPIRV-DAG: FunctionParameter [[#]] [[#B:]]
; CHECK-SPIRV-DAG: Label [[#]]
; CHECK-SPIRV-DAG: INotEqual [[#]] [[#R:]] [[#A]] [[#B]]
; CHECK-SPIRV-DAG: ReturnValue [[#R]]
; CHECK-SPIRV-DAG: FunctionEnd
; CHECK-LLVM-DAG: icmp ne

define i1 @test_ne(i32 %a, i32 %b) {
  %r = icmp ne i32 %a, %b
  ret i1 %r
}

; CHECK-SPIRV-DAG: Function [[#]] [[#SLT]] [[#]] [[#]]
; CHECK-SPIRV-DAG: FunctionParameter [[#]] [[#A:]]
; CHECK-SPIRV-DAG: FunctionParameter [[#]] [[#B:]]
; CHECK-SPIRV-DAG: Label [[#]]
; CHECK-SPIRV-DAG: SLessThan [[#]] [[#R:]] [[#A]] [[#B]]
; CHECK-SPIRV-DAG: ReturnValue [[#R]]
; CHECK-SPIRV-DAG: FunctionEnd
; CHECK-LLVM-DAG: icmp slt

define i1 @test_slt(i32 %a, i32 %b) {
  %r = icmp slt i32 %a, %b
  ret i1 %r
}

; CHECK-SPIRV-DAG: Function [[#]] [[#ULT]] [[#]] [[#]]
; CHECK-SPIRV-DAG: FunctionParameter [[#]] [[#A:]]
; CHECK-SPIRV-DAG: FunctionParameter [[#]] [[#B:]]
; CHECK-SPIRV-DAG: Label [[#]]
; CHECK-SPIRV-DAG: ULessThan [[#]] [[#R:]] [[#A]] [[#B]]
; CHECK-SPIRV-DAG: ReturnValue [[#R]]
; CHECK-SPIRV-DAG: FunctionEnd
; CHECK-LLVM-DAG: icmp ult

define i1 @test_ult(i32 %a, i32 %b) {
  %r = icmp ult i32 %a, %b
  ret i1 %r
}

; CHECK-SPIRV-DAG: Function [[#]] [[#ULE]] [[#]] [[#]]
; CHECK-SPIRV-DAG: FunctionParameter [[#]] [[#A:]]
; CHECK-SPIRV-DAG: FunctionParameter [[#]] [[#B:]]
; CHECK-SPIRV-DAG: Label [[#]]
; CHECK-SPIRV-DAG: ULessThanEqual [[#]] [[#R:]] [[#A]] [[#B]]
; CHECK-SPIRV-DAG: ReturnValue [[#R]]
; CHECK-SPIRV-DAG: FunctionEnd
; CHECK-LLVM-DAG: icmp ule

define i1 @test_ule(i32 %a, i32 %b) {
  %r = icmp ule i32 %a, %b
  ret i1 %r
}

; CHECK-SPIRV-DAG: Function [[#]] [[#SLE]] [[#]] [[#]]
; CHECK-SPIRV-DAG: FunctionParameter [[#]] [[#A:]]
; CHECK-SPIRV-DAG: FunctionParameter [[#]] [[#B:]]
; CHECK-SPIRV-DAG: Label [[#]]
; CHECK-SPIRV-DAG: SLessThanEqual [[#]] [[#R:]] [[#A]] [[#B]]
; CHECK-SPIRV-DAG: ReturnValue [[#R]]
; CHECK-SPIRV-DAG: FunctionEnd
; CHECK-LLVM-DAG: icmp sle

define i1 @test_sle(i32 %a, i32 %b) {
  %r = icmp sle i32 %a, %b
  ret i1 %r
}

; CHECK-SPIRV-DAG: Function [[#]] [[#UGT]] [[#]] [[#]]
; CHECK-SPIRV-DAG: FunctionParameter [[#]] [[#A:]]
; CHECK-SPIRV-DAG: FunctionParameter [[#]] [[#B:]]
; CHECK-SPIRV-DAG: Label [[#]]
; CHECK-SPIRV-DAG: UGreaterThan [[#]] [[#R:]] [[#A]] [[#B]]
; CHECK-SPIRV-DAG: ReturnValue [[#R]]
; CHECK-SPIRV-DAG: FunctionEnd
; CHECK-LLVM-DAG: icmp ugt

define i1 @test_ugt(i32 %a, i32 %b) {
  %r = icmp ugt i32 %a, %b
  ret i1 %r
}

; CHECK-SPIRV-DAG: Function [[#]] [[#SGT]] [[#]] [[#]]
; CHECK-SPIRV-DAG: FunctionParameter [[#]] [[#A:]]
; CHECK-SPIRV-DAG: FunctionParameter [[#]] [[#B:]]
; CHECK-SPIRV-DAG: Label [[#]]
; CHECK-SPIRV-DAG: SGreaterThan [[#]] [[#R:]] [[#A]] [[#B]]
; CHECK-SPIRV-DAG: ReturnValue [[#R]]
; CHECK-SPIRV-DAG: FunctionEnd
; CHECK-LLVM-DAG: icmp sgt

define i1 @test_sgt(i32 %a, i32 %b) {
  %r = icmp sgt i32 %a, %b
  ret i1 %r
}

; CHECK-SPIRV-DAG: Function [[#]] [[#UGE]] [[#]] [[#]]
; CHECK-SPIRV-DAG: FunctionParameter [[#]] [[#A:]]
; CHECK-SPIRV-DAG: FunctionParameter [[#]] [[#B:]]
; CHECK-SPIRV-DAG: Label [[#]]
; CHECK-SPIRV-DAG: UGreaterThanEqual [[#]] [[#R:]] [[#A]] [[#B]]
; CHECK-SPIRV-DAG: ReturnValue [[#R]]
; CHECK-SPIRV-DAG: FunctionEnd
; CHECK-LLVM-DAG: icmp uge

define i1 @test_uge(i32 %a, i32 %b) {
  %r = icmp uge i32 %a, %b
  ret i1 %r
}

; CHECK-SPIRV-DAG: Function [[#]] [[#SGE]] [[#]] [[#]]
; CHECK-SPIRV-DAG: FunctionParameter [[#]] [[#A:]]
; CHECK-SPIRV-DAG: FunctionParameter [[#]] [[#B:]]
; CHECK-SPIRV-DAG: Label [[#]]
; CHECK-SPIRV-DAG: SGreaterThanEqual [[#]] [[#R:]] [[#A]] [[#B]]
; CHECK-SPIRV-DAG: ReturnValue [[#R]]
; CHECK-SPIRV-DAG: FunctionEnd
; CHECK-LLVM-DAG: icmp sge

define i1 @test_sge(i32 %a, i32 %b) {
  %r = icmp sge i32 %a, %b
  ret i1 %r
}

; CHECK-SPIRV-DAG: Function [[#]] [[#v3EQ]] [[#]] [[#]]
; CHECK-SPIRV-DAG: FunctionParameter [[#]] [[#A:]]
; CHECK-SPIRV-DAG: FunctionParameter [[#]] [[#B:]]
; CHECK-SPIRV-DAG: Label [[#]]
; CHECK-SPIRV-DAG: IEqual [[#]] [[#R:]] [[#A]] [[#B]]
; CHECK-SPIRV-DAG: ReturnValue [[#R]]
; CHECK-SPIRV-DAG: FunctionEnd
; CHECK-LLVM-DAG: icmp eq

define <3 x i1> @test_v3_eq(<3 x i32> %a, <3 x i32> %b) {
  %r = icmp eq <3 x i32> %a, %b
  ret <3 x i1> %r
}

; CHECK-SPIRV-DAG: Function [[#]] [[#v3NE]] [[#]] [[#]]
; CHECK-SPIRV-DAG: FunctionParameter [[#]] [[#A:]]
; CHECK-SPIRV-DAG: FunctionParameter [[#]] [[#B:]]
; CHECK-SPIRV-DAG: Label [[#]]
; CHECK-SPIRV-DAG: INotEqual [[#]] [[#R:]] [[#A]] [[#B]]
; CHECK-SPIRV-DAG: ReturnValue [[#R]]
; CHECK-SPIRV-DAG: FunctionEnd
; CHECK-LLVM-DAG: icmp ne

define <3 x i1> @test_v3_ne(<3 x i32> %a, <3 x i32> %b) {
  %r = icmp ne <3 x i32> %a, %b
  ret <3 x i1> %r
}

; CHECK-SPIRV-DAG: Function [[#]] [[#v3SLT]] [[#]] [[#]]
; CHECK-SPIRV-DAG: FunctionParameter [[#]] [[#A:]]
; CHECK-SPIRV-DAG: FunctionParameter [[#]] [[#B:]]
; CHECK-SPIRV-DAG: Label [[#]]
; CHECK-SPIRV-DAG: SLessThan [[#]] [[#R:]] [[#A]] [[#B]]
; CHECK-SPIRV-DAG: ReturnValue [[#R]]
; CHECK-SPIRV-DAG: FunctionEnd
; CHECK-LLVM-DAG: icmp slt

define <3 x i1> @test_v3_slt(<3 x i32> %a, <3 x i32> %b) {
  %r = icmp slt <3 x i32> %a, %b
  ret <3 x i1> %r
}

; CHECK-SPIRV-DAG: Function [[#]] [[#v3ULT]] [[#]] [[#]]
; CHECK-SPIRV-DAG: FunctionParameter [[#]] [[#A:]]
; CHECK-SPIRV-DAG: FunctionParameter [[#]] [[#B:]]
; CHECK-SPIRV-DAG: Label [[#]]
; CHECK-SPIRV-DAG: ULessThan [[#]] [[#R:]] [[#A]] [[#B]]
; CHECK-SPIRV-DAG: ReturnValue [[#R]]
; CHECK-SPIRV-DAG: FunctionEnd
; CHECK-LLVM-DAG: icmp ult

define <3 x i1> @test_v3_ult(<3 x i32> %a, <3 x i32> %b) {
  %r = icmp ult <3 x i32> %a, %b
  ret <3 x i1> %r
}

; CHECK-SPIRV-DAG: Function [[#]] [[#v3ULE]] [[#]] [[#]]
; CHECK-SPIRV-DAG: FunctionParameter [[#]] [[#A:]]
; CHECK-SPIRV-DAG: FunctionParameter [[#]] [[#B:]]
; CHECK-SPIRV-DAG: Label [[#]]
; CHECK-SPIRV-DAG: ULessThanEqual [[#]] [[#R:]] [[#A]] [[#B]]
; CHECK-SPIRV-DAG: ReturnValue [[#R]]
; CHECK-SPIRV-DAG: FunctionEnd
; CHECK-LLVM-DAG: icmp ule

define <3 x i1> @test_v3_ule(<3 x i32> %a, <3 x i32> %b) {
  %r = icmp ule <3 x i32> %a, %b
  ret <3 x i1> %r
}

; CHECK-SPIRV-DAG: Function [[#]] [[#v3SLE]] [[#]] [[#]]
; CHECK-SPIRV-DAG: FunctionParameter [[#]] [[#A:]]
; CHECK-SPIRV-DAG: FunctionParameter [[#]] [[#B:]]
; CHECK-SPIRV-DAG: Label [[#]]
; CHECK-SPIRV-DAG: SLessThanEqual [[#]] [[#R:]] [[#A]] [[#B]]
; CHECK-SPIRV-DAG: ReturnValue [[#R]]
; CHECK-SPIRV-DAG: FunctionEnd
; CHECK-LLVM-DAG: icmp sle

define <3 x i1> @test_v3_sle(<3 x i32> %a, <3 x i32> %b) {
  %r = icmp sle <3 x i32> %a, %b
  ret <3 x i1> %r
}

; CHECK-SPIRV-DAG: Function [[#]] [[#v3UGT]] [[#]] [[#]]
; CHECK-SPIRV-DAG: FunctionParameter [[#]] [[#A:]]
; CHECK-SPIRV-DAG: FunctionParameter [[#]] [[#B:]]
; CHECK-SPIRV-DAG: Label [[#]]
; CHECK-SPIRV-DAG: UGreaterThan [[#]] [[#R:]] [[#A]] [[#B]]
; CHECK-SPIRV-DAG: ReturnValue [[#R]]
; CHECK-SPIRV-DAG: FunctionEnd
; CHECK-LLVM-DAG: icmp ugt

define <3 x i1> @test_v3_ugt(<3 x i32> %a, <3 x i32> %b) {
  %r = icmp ugt <3 x i32> %a, %b
  ret <3 x i1> %r
}

; CHECK-SPIRV-DAG: Function [[#]] [[#v3SGT]] [[#]] [[#]]
; CHECK-SPIRV-DAG: FunctionParameter [[#]] [[#A:]]
; CHECK-SPIRV-DAG: FunctionParameter [[#]] [[#B:]]
; CHECK-SPIRV-DAG: Label [[#]]
; CHECK-SPIRV-DAG: SGreaterThan [[#]] [[#R:]] [[#A]] [[#B]]
; CHECK-SPIRV-DAG: ReturnValue [[#R]]
; CHECK-SPIRV-DAG: FunctionEnd
; CHECK-LLVM-DAG: icmp sgt

define <3 x i1> @test_v3_sgt(<3 x i32> %a, <3 x i32> %b) {
  %r = icmp sgt <3 x i32> %a, %b
  ret <3 x i1> %r
}

; CHECK-SPIRV-DAG: Function [[#]] [[#v3UGE]] [[#]] [[#]]
; CHECK-SPIRV-DAG: FunctionParameter [[#]] [[#A:]]
; CHECK-SPIRV-DAG: FunctionParameter [[#]] [[#B:]]
; CHECK-SPIRV-DAG: Label [[#]]
; CHECK-SPIRV-DAG: UGreaterThanEqual [[#]] [[#R:]] [[#A]] [[#B]]
; CHECK-SPIRV-DAG: ReturnValue [[#R]]
; CHECK-SPIRV-DAG: FunctionEnd
; CHECK-LLVM-DAG: icmp uge

define <3 x i1> @test_v3_uge(<3 x i32> %a, <3 x i32> %b) {
  %r = icmp uge <3 x i32> %a, %b
  ret <3 x i1> %r
}

; CHECK-SPIRV-DAG: Function [[#]] [[#v3SGE]] [[#]] [[#]]
; CHECK-SPIRV-DAG: FunctionParameter [[#]] [[#A:]]
; CHECK-SPIRV-DAG: FunctionParameter [[#]] [[#B:]]
; CHECK-SPIRV-DAG: Label [[#]]
; CHECK-SPIRV-DAG: SGreaterThanEqual [[#]] [[#R:]] [[#A]] [[#B]]
; CHECK-SPIRV-DAG: ReturnValue [[#R]]
; CHECK-SPIRV-DAG: FunctionEnd
; CHECK-LLVM-DAG: icmp sge

define <3 x i1> @test_v3_sge(<3 x i32> %a, <3 x i32> %b) {
  %r = icmp sge <3 x i32> %a, %b
  ret <3 x i1> %r
}
