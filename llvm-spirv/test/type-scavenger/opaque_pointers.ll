; RUN: llvm-as < %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv -to-text %t.spv -o - | FileCheck %s
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

; CHECK: TypeInt [[#Int64Ty:]] 64 0
; CHECK: TypeInt [[#Int32Ty:]] 32 0
; CHECK: Constant [[#Int32Ty]] [[#ZeroConst:]] 0
; CHECK: TypePointer [[#PtrInt32Ty:]] 7 [[#Int32Ty]]
; CHECK: TypeFunction [[#FTy:]] [[#Int64Ty]] [[#PtrInt32Ty]]
; CHECK: TypePointer [[#PtrInt64Ty:]] 7 [[#Int64Ty]]

; CHECK: Function [[#Int64Ty]] [[#FuncId:]] 0 [[#FTy]]
; CHECK: FunctionParameter [[#PtrInt32Ty]] [[#ParamId:]]
; CHECK: Store [[#ParamId]] [[#ZeroConst]] 2 4
; CHECK: Bitcast [[#PtrInt64Ty]] [[#CastId:]] [[#ParamId]]
; CHECK: Load [[#Int64Ty]] [[#ResultId:]] [[#CastId]] 2 4
; CHECK: ReturnValue [[#ResultId]]

; CHECK-LLVM: define spir_func i64 @test(ptr [[p:%.*]])
; CHECK-LLVM:   store i32 0, ptr [[p]], align 4
; CHECK-LLVM:   [[bc:%.*]] = bitcast ptr [[p]] to ptr
; CHECK-LLVM:   [[v:%.*]] = load i64, ptr [[bc]], align 4

define i64 @test(ptr %p) {
  store i32 0, ptr %p, align 4
  %v = load i64, ptr %p, align 4
  ret i64 %v
}
