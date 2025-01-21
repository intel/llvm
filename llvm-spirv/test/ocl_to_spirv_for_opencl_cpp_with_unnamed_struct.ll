;; This test checks if functions in LLVM IR generated from OpenCL_CPP sources
;; including unnamed structs are correctly translated to SPIR-V.

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

%opencl.event_t = type opaque

declare spir_func void @_ZNKSt7complexIfE5__repEv(ptr addrspace(4) dead_on_unwind noalias writable sret({ float, float }) align 4)

!spirv.Source = !{!0}

!0 = !{i32 4, i32 100000}
