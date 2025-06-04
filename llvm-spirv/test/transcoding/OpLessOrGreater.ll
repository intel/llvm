; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv %t.spv -to-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV
; RUN: spirv-val %t.spv

; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc
; RUN: FileCheck < %t.rev.ll %s --check-prefix=CHECK-LLVM

; SPIR-V 1.5
; CHECK-SPIRV: 66816

; CHECK-SPIRV-NOT: LessOrGreater
; CHECK-SPIRV: FOrdNotEqual

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-unknown"

; Function Attrs: nounwind
define spir_kernel void @test(float %a) local_unnamed_addr #0 {
entry:
; CHECK-LLVM: fcmp one float %a, %a
  %call = tail call spir_func i1 @_Z21__spirv_LessOrGreater(float %a, float %a)
  ret void
}

; This is needed to check that 1.5 is enabled
define dso_local spir_kernel void @test2(i16 noundef signext %a, i32 noundef %id) local_unnamed_addr #0  {
entry:
  %call = tail call spir_func signext i16 @_Z31sub_group_non_uniform_broadcastsj(i16 noundef signext %a, i32 noundef %id)
  ret void
}

declare spir_func noundef i1 @_Z21__spirv_LessOrGreater(float, float)
declare spir_func signext i16 @_Z31sub_group_non_uniform_broadcastsj(i16 noundef signext, i32 noundef)

attributes #0 = { convergent nounwind writeonly }

!llvm.module.flags = !{!0}
!opencl.ocl.version = !{!1}
!opencl.spir.version = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 2, i32 0}
