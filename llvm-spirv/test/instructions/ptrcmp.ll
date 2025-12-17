; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -o - -spirv-text | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM
; XFAIL: *

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir"

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

; CHECK-SPIRV:      Function [[#]] [[#EQ]]
; CHECK-SPIRV: FunctionParameter [[#]] [[#A:]]
; CHECK-SPIRV: FunctionParameter [[#]] [[#B:]]
; CHECK-SPIRV: PtrEqual [[#]] [[#R:]] [[#A]] [[#B]]
; CHECK-SPIRV: ReturnValue [[#R]]
; CHECK-SPIRV: FunctionEnd

; CHECK-LLVM: define spir_func i1 @test_eq
; CHECK-LLVM: %[[R:.*]] = icmp eq
; CHECK-LLVM: ret i1 %[[R]]

define i1 @test_eq(ptr %a, ptr %b) {
  %r = icmp eq ptr %a, %b
  ret i1 %r
}

; CHECK-SPIRV:      Function [[#]] [[#NE]]
; CHECK-SPIRV: FunctionParameter [[#]] [[#A:]]
; CHECK-SPIRV: FunctionParameter [[#]] [[#B:]]
; CHECK-SPIRV: PtrNotEqual [[#]] [[#R:]] [[#A]] [[#B]]
; CHECK-SPIRV: ReturnValue [[#R]]
; CHECK-SPIRV: FunctionEnd

; CHECK-LLVM: define spir_func i1 @test_ne
; CHECK-LLVM: %[[R:.*]] = icmp ne
; CHECK-LLVM: ret i1 %[[R]]

define i1 @test_ne(ptr %a, ptr %b) {
  %r = icmp ne ptr %a, %b
  ret i1 %r
}

; CHECK-SPIRV:      Function [[#]] [[#SLT]]
; CHECK-SPIRV: FunctionParameter [[#]] [[#A:]]
; CHECK-SPIRV: FunctionParameter [[#]] [[#B:]]
; CHECK-SPIRV: ConvertPtrToU [[#]] [[#AI:]] [[#A]]
; CHECK-SPIRV: ConvertPtrToU [[#]] [[#BI:]] [[#B]]
; CHECK-SPIRV:      SLessThan [[#]] [[#R:]] [[#AI]] [[#BI]]
; CHECK-SPIRV: ReturnValue [[#R]]
; CHECK-SPIRV: FunctionEnd

; CHECK-LLVM: define spir_func i1 @test_slt
; CHECK-LLVM: %[[R:.*]] = icmp slt
; CHECK-LLVM: ret i1 %[[R]]

define i1 @test_slt(ptr %a, ptr %b) {
  %r = icmp slt ptr %a, %b
  ret i1 %r
}

; CHECK-SPIRV:      Function [[#]] [[#ULT]]
; CHECK-SPIRV: FunctionParameter [[#]] [[#A:]]
; CHECK-SPIRV: FunctionParameter [[#]] [[#B:]]
; CHECK-SPIRV: ConvertPtrToU [[#]] [[#AI:]] [[#A]]
; CHECK-SPIRV: ConvertPtrToU [[#]] [[#BI:]] [[#B]]
; CHECK-SPIRV:      ULessThan [[#]] [[#R:]] [[#AI]] [[#BI]]
; CHECK-SPIRV: ReturnValue [[#R]]
; CHECK-SPIRV: FunctionEnd

; CHECK-LLVM: define spir_func i1 @test_ult
; CHECK-LLVM: %[[R:.*]] = icmp ult
; CHECK-LLVM: ret i1 %[[R]]

define i1 @test_ult(ptr %a, ptr %b) {
  %r = icmp ult ptr %a, %b
  ret i1 %r
}

; CHECK-SPIRV:      Function [[#]] [[#ULE]]
; CHECK-SPIRV: FunctionParameter [[#]] [[#A:]]
; CHECK-SPIRV: FunctionParameter [[#]] [[#B:]]
; CHECK-SPIRV: ConvertPtrToU [[#]] [[#AI:]] [[#A]]
; CHECK-SPIRV: ConvertPtrToU [[#]] [[#BI:]] [[#B]]
; CHECK-SPIRV:      ULessThanEqual [[#]] [[#R:]] [[#AI]] [[#BI]]
; CHECK-SPIRV: ReturnValue [[#R]]
; CHECK-SPIRV: FunctionEnd

; CHECK-LLVM: define spir_func i1 @test_ule
; CHECK-LLVM: %[[R:.*]] = icmp ule
; CHECK-LLVM: ret i1 %[[R]]

define i1 @test_ule(ptr %a, ptr %b) {
  %r = icmp ule ptr %a, %b
  ret i1 %r
}

; CHECK-SPIRV:      Function [[#]] [[#SLE]]
; CHECK-SPIRV: FunctionParameter [[#]] [[#A:]]
; CHECK-SPIRV: FunctionParameter [[#]] [[#B:]]
; CHECK-SPIRV: ConvertPtrToU [[#]] [[#AI:]] [[#A]]
; CHECK-SPIRV: ConvertPtrToU [[#]] [[#BI:]] [[#B]]
; CHECK-SPIRV:      SLessThanEqual [[#]] [[#R:]] [[#AI]] [[#BI]]
; CHECK-SPIRV: ReturnValue [[#R]]
; CHECK-SPIRV: FunctionEnd

; CHECK-LLVM: define spir_func i1 @test_sle
; CHECK-LLVM: %[[R:.*]] = icmp sle
; CHECK-LLVM: ret i1 %[[R]]

define i1 @test_sle(ptr %a, ptr %b) {
  %r = icmp sle ptr %a, %b
  ret i1 %r
}

; CHECK-SPIRV:      Function [[#]] [[#UGT]]
; CHECK-SPIRV: FunctionParameter [[#]] [[#A:]]
; CHECK-SPIRV: FunctionParameter [[#]] [[#B:]]
; CHECK-SPIRV: ConvertPtrToU [[#]] [[#AI:]] [[#A]]
; CHECK-SPIRV: ConvertPtrToU [[#]] [[#BI:]] [[#B]]
; CHECK-SPIRV:      UGreaterThan [[#]] [[#R:]] [[#AI]] [[#BI]]
; CHECK-SPIRV: ReturnValue [[#R]]
; CHECK-SPIRV: FunctionEnd

; CHECK-LLVM: define spir_func i1 @test_ugt
; CHECK-LLVM: %[[R:.*]] = icmp ugt
; CHECK-LLVM: ret i1 %[[R]]

define i1 @test_ugt(ptr %a, ptr %b) {
  %r = icmp ugt ptr %a, %b
  ret i1 %r
}

; CHECK-SPIRV:      Function [[#]] [[#SGT]]
; CHECK-SPIRV: FunctionParameter [[#]] [[#A:]]
; CHECK-SPIRV: FunctionParameter [[#]] [[#B:]]
; CHECK-SPIRV: ConvertPtrToU [[#]] [[#AI:]] [[#A]]
; CHECK-SPIRV: ConvertPtrToU [[#]] [[#BI:]] [[#B]]
; CHECK-SPIRV:      SGreaterThan [[#]] [[#R:]] [[#AI]] [[#BI]]
; CHECK-SPIRV: ReturnValue [[#R]]
; CHECK-SPIRV: FunctionEnd

; CHECK-LLVM: define spir_func i1 @test_sgt
; CHECK-LLVM: %[[R:.*]] = icmp sgt
; CHECK-LLVM: ret i1 %[[R]]

define i1 @test_sgt(ptr %a, ptr %b) {
  %r = icmp sgt ptr %a, %b
  ret i1 %r
}

; CHECK-SPIRV:      Function [[#]] [[#UGE]]
; CHECK-SPIRV: FunctionParameter [[#]] [[#A:]]
; CHECK-SPIRV: FunctionParameter [[#]] [[#B:]]
; CHECK-SPIRV: ConvertPtrToU [[#]] [[#AI:]] [[#A]]
; CHECK-SPIRV: ConvertPtrToU [[#]] [[#BI:]] [[#B]]
; CHECK-SPIRV:      UGreaterThanEqual [[#]] [[#R:]] [[#AI]] [[#BI]]
; CHECK-SPIRV: ReturnValue [[#R]]
; CHECK-SPIRV: FunctionEnd

; CHECK-LLVM: define spir_func i1 @test_uge
; CHECK-LLVM: %[[R:.*]] = icmp uge
; CHECK-LLVM: ret i1 %[[R]]

define i1 @test_uge(ptr %a, ptr %b) {
  %r = icmp uge ptr %a, %b
  ret i1 %r
}

; CHECK-SPIRV:      Function [[#]] [[#SGE]]
; CHECK-SPIRV: FunctionParameter [[#]] [[#A:]]
; CHECK-SPIRV: FunctionParameter [[#]] [[#B:]]
; CHECK-SPIRV: ConvertPtrToU [[#]] [[#AI:]] [[#A]]
; CHECK-SPIRV: ConvertPtrToU [[#]] [[#BI:]] [[#B]]
; CHECK-SPIRV:      SGreaterThanEqual [[#]] [[#R:]] [[#AI]] [[#BI]]
; CHECK-SPIRV: ReturnValue [[#R]]
; CHECK-SPIRV: FunctionEnd

; CHECK-LLVM: define spir_func i1 @test_sge
; CHECK-LLVM: %[[R:.*]] = icmp sge
; CHECK-LLVM: ret i1 %[[R]]

define i1 @test_sge(ptr %a, ptr %b) {
  %r = icmp sge ptr %a, %b
  ret i1 %r
}
