; The test checks command-line option for the translator
; which will allow to represent unknown llvm intrinsics as external function call in SPIR-V.
; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv -spirv-allow-unknown-intrinsics %t.bc -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv %t.spv -to-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; CHECK-LLVM: declare float @llvm.fma.f32(float, float, float)
; CHECK-SPIRV: LinkageAttributes "llvm.fma.f32" Import
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64"

; Function Attrs: nounwind
define spir_func void @foo(float %a, float %b, float %c) #0 {
entry:
  %0 = call float @llvm.fma.f32(float %a, float %b, float %c)
  ret void
}

; Function Attrs: nounwind readnone
declare float @llvm.fma.f32(float, float, float) #1

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!opencl.enable.FP_CONTRACT = !{}
!opencl.spir.version = !{!0}
!opencl.ocl.version = !{!1}
!opencl.used.extensions = !{!2}
!opencl.used.optional.core.features = !{!3}
!opencl.compiler.options = !{!2}

!0 = !{i32 1, i32 2}
!1 = !{i32 2, i32 0}
!2 = !{}
!3 = !{!"cl_doubles"}
