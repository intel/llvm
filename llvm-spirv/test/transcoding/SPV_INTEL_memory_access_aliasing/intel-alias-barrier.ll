; The test checks if the translator won't crash

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_INTEL_memory_access_aliasing -o %t.spv

; ModuleID = 'main'
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-unknown"

%class.anon = type { i8 }

; Function Attrs: nounwind
define spir_kernel void @barrier_simple()
{
  tail call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 2, i32 2, i32 272), !noalias !1
  ret void
}

declare dso_local spir_func void @_Z22__spirv_ControlBarrierjjj(i32, i32, i32)

!1 = !{!2}
!2 = distinct !{!2, !3}
!3 = distinct !{!3}
