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
declare ptr @llvm.invariant.start.p3(i64 immarg, ptr addrspace(3) nocapture) #0

; Function Attrs: argmemonly nounwind
declare void @llvm.invariant.end.p3(ptr, i64 immarg, ptr addrspace(3) nocapture) #0

define linkonce_odr dso_local spir_func void @func() {
  store i64 2, ptr addrspace(3) @WGSharedVar
  %1 = call ptr @llvm.invariant.start.p3(i64 8, ptr addrspace(3) @WGSharedVar)
  call void @llvm.invariant.end.p3(ptr %1, i64 8, ptr addrspace(3) @WGSharedVar)
  ret void
}

attributes #0 = { argmemonly nounwind }

!spirv.ExecutionMode = !{}
