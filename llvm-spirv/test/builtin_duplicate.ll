;; This test checks if we generate a single builtin variable for the following
;; LLVM IR.
;; @__spirv_BuiltInLocalInvocationId - A global variable
;; %3 = tail call i64 @_Z12get_local_idj(i32 0) - A function call

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv %t.bc -spirv-text -o %t.txt
; RUN: FileCheck < %t.txt %s --check-prefix=CHECK-SPIRV
; RUN: spirv-val %t.spv

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

%opencl.event_t = type opaque

; CHECK-SPIRV: {{[0-9]+}} Name {{[0-9]+}} "__spirv_BuiltInLocalInvocationId"
; CHECK-SPIRV-NOT: {{[0-9]+}} Name {{[0-9]+}} "__spirv_BuiltInLocalInvocationId.1"

@__spirv_BuiltInLocalInvocationId = external dso_local local_unnamed_addr addrspace(1) constant <3 x i64>, align 32

declare spir_func i64 @_Z12get_local_idj(i32) local_unnamed_addr

; Function Attrs: nounwind
define spir_kernel void @test_fn(i32 %a) {
entry:
  %3 = tail call i64 @_Z12get_local_idj(i32 0)
  ret void
}

!spirv.Source = !{!0}

!0 = !{i32 6, i32 100000}
