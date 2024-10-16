; RUN: llvm-as %s -o %t.bc
; RUN: not llvm-spirv %t.bc --spirv-ext=+SPV_INTEL_bindless_images 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR-1
; CHECK-ERROR-1: InvalidInstruction: Can't translate llvm instruction:
; CHECK-ERROR-1-NEXT: ConvertHandleToImageINTEL
; CHECK-ERROR-1-NEXT: Parameter value must be a 32-bit scalar in case of Physical32 addressing model or a 64-bit scalar in case of Physical64 addressing model
; CHECK-ERROR-1-NEXT: Type size: 32
; CHECK-ERROR-1-NEXT: Addressing model: Physical64

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

define spir_func void @foo(i32 %in) {
  %img = call spir_func target("spirv.Image", i32, 2, 0, 0, 0, 0, 0, 0) @_Z33__spirv_ConvertHandleToImageINTELi(i32 %in)
  %samp = call spir_func target("spirv.Sampler") @_Z35__spirv_ConvertHandleToSamplerINTELl(i64 42)
  %sampImage = call spir_func target("spirv.SampledImage", i64, 1, 0, 0, 0, 0, 0, 0) @_Z40__spirv_ConvertHandleToSampledImageINTELl(i64 43)
  ret void
}

declare spir_func target("spirv.Image", i32, 2, 0, 0, 0, 0, 0, 0) @_Z33__spirv_ConvertHandleToImageINTELi(i32)

declare spir_func target("spirv.Sampler") @_Z35__spirv_ConvertHandleToSamplerINTELl(i64)

declare spir_func target("spirv.SampledImage", i64, 1, 0, 0, 0, 0, 0, 0) @_Z40__spirv_ConvertHandleToSampledImageINTELl(i64)

!opencl.spir.version = !{!0}
!spirv.Source = !{!1}
!llvm.ident = !{!2}

!0 = !{i32 1, i32 2}
!1 = !{i32 4, i32 100000}
!2 = !{!"clang version 17.0.0"}
