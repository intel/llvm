; The test checks if the translator won't crash
; TODO need to figure out how to translate the alias.scope/noalias metadata
; in a case when it attached to a call to lifetime intrinsic. Now just skip it.

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_INTEL_memory_access_aliasing -o %t.spv

; ModuleID = 'main'
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-unknown"

%class.anon = type { i8 }

; Function Attrs: nounwind
define spir_kernel void @lifetime_simple()
{
  %1 = alloca i32
  %2 = bitcast i32* %1 to i8*
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %2), !noalias !1
  ret void
}

; Function Attrs: nounwind
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #0

attributes #0 = { nounwind }

!1 = !{!2}
!2 = distinct !{!2, !3}
!3 = distinct !{!3}
