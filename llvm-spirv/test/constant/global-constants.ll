; RUN: llvm-as < %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv -to-text %t.spv -o - | FileCheck %s
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

@global   = addrspace(1) constant i32 1 ; OpenCL global memory
@constant = addrspace(2) constant i32 2 ; OpenCL constant memory
@local    = addrspace(3) constant i32 3 ; OpenCL local memory

define i32 @getGlobal1() {
  %g = load i32, ptr addrspace(1) @global
  ret i32 %g
}

define i32 @getGlobal2() {
  %g = load i32, ptr addrspace(2) @constant
  ret i32 %g
}

define i32 @getGlobal3() {
  %g = load i32, ptr addrspace(3) @local
  ret i32 %g
}

; CHECK:    TypeInt [[#INT:]] 32 0
; CHECK-DAG: Constant [[#INT]] [[#CST_AS1:]] 1
; CHECK-DAG: Constant [[#INT]] [[#CST_AS2:]] 2
; CHECK-DAG: Constant [[#INT]] [[#CST_AS3:]] 3
; CHECK-DAG: TypePointer [[#PTR_TO_INT_AS1:]] 5 [[#INT]]
; CHECK-DAG: TypePointer [[#PTR_TO_INT_AS2:]] 0 [[#INT]]
; CHECK-DAG: TypePointer [[#PTR_TO_INT_AS3:]] 4 [[#INT]]
; CHECK-DAG: Variable [[#PTR_TO_INT_AS1]] [[#GV1:]] 5 [[#CST_AS1]]
; CHECK-DAG: Variable [[#PTR_TO_INT_AS2]] [[#GV2:]] 0 [[#CST_AS2]]
; CHECK-DAG: Variable [[#PTR_TO_INT_AS3]] [[#GV3:]] 4 [[#CST_AS3]]

; CHECK: Load [[#INT]] [[#]] [[#GV1]]
; CHECK: Load [[#INT]] [[#]] [[#GV2]]
; CHECK: Load [[#INT]] [[#]] [[#GV3]]

; CHECK-LLVM: @{{.*}} = addrspace(1) constant i32 1
; CHECK-LLVM: @{{.*}} = addrspace(2) constant i32 2
; CHECK-LLVM: @{{.*}} = addrspace(3) constant i32 3
