;; This test checks if builtins in LLVM IR generated from CPP_for_OpenCL sources
;; are correctly translated to SPIR-V builtin variables.

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv %t.bc -spirv-text -o %t.txt
; RUN: FileCheck < %t.txt %s --check-prefix=CHECK-SPIRV
; RUN: spirv-val %t.spv

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

%opencl.event_t = type opaque

; CHECK-SPIRV: {{[0-9]+}} Name {{[0-9]+}} "__spirv_BuiltInWorkgroupId"
; Function Attrs: nounwind
define spir_kernel void @test_fn(i32 %a) {
entry:
  %call15 = call spir_func i32 @_Z12get_group_idj(i32 0)
  ret void
}

declare spir_func i32 @_Z12get_group_idj(i32)

!spirv.Source = !{!0}

!0 = !{i32 6, i32 100000}
