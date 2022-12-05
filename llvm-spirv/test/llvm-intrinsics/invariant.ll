; Make sure the translator doesn't crash if the input LLVM IR contains llvm.invariant.* intrinsics
; RUN: llvm-as < %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv -to-text %t.spv -o - | FileCheck %s

; CHECK-NOT: FunctionParameter
; CHECK-NOT: FunctionCall

source_filename = "<stdin>"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-linux"

@WGSharedVar = internal addrspace(3) constant i64 0, align 8

; Function Attrs: argmemonly nounwind
declare {}* @llvm.invariant.start.p3i8(i64 immarg, i8 addrspace(3)* nocapture) #0

; Function Attrs: argmemonly nounwind
declare void @llvm.invariant.end.p3i8({}*, i64 immarg, i8 addrspace(3)* nocapture) #0

define linkonce_odr dso_local spir_func void @func() {
  store i64 2, i64 addrspace(3)* @WGSharedVar
  %1 = bitcast i64 addrspace(3)* @WGSharedVar to i8 addrspace(3)*
  %2 = call {}* @llvm.invariant.start.p3i8(i64 8, i8 addrspace(3)* %1)
  call void @llvm.invariant.end.p3i8({}* %2, i64 8, i8 addrspace(3)* %1)
  ret void
}

attributes #0 = { argmemonly nounwind }

!spirv.ExecutionMode = !{}
