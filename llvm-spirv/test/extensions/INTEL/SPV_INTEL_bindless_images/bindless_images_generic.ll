; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv --spirv-ext=+SPV_INTEL_bindless_images
; RUN: llvm-spirv %t.spv -o %t.spt --to-text
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.spv -o %t.rev.bc -r --spirv-target-env=SPV-IR
; RUN: llvm-dis %t.rev.bc -o %t.rev.ll
; RUN: FileCheck < %t.rev.ll %s --check-prefix=CHECK-LLVM

; RUN: not llvm-spirv %t.bc 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR
; CHECK-ERROR: RequiresExtension: Feature requires the following SPIR-V extension:
; CHECK-ERROR-NEXT: SPV_INTEL_bindless_images

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

; CHECK-SPIRV: Capability BindlessImagesINTEL
; CHECK-SPIRV: Extension "SPV_INTEL_bindless_images"
; CHECK-SPIRV-DAG: TypeVoid [[#VoidTy:]]
; CHECK-SPIRV-DAG: TypeInt [[#Int64Ty:]] 64
; CHECK-SPIRV-DAG: Constant [[#Int64Ty]] [[#Const42:]] 42 0
; CHECK-SPIRV-DAG: Constant [[#Int64Ty]] [[#Const43:]] 43 0
; CHECK-SPIRV-DAG: TypeImage [[#IntImgTy:]] [[#Int64Ty]]
; CHECK-SPIRV-DAG: TypeSampler [[#SamplerTy:]]
; CHECK-SPIRV-DAG: TypeImage [[#IntSmpImgTy:]] [[#Int64Ty]]
; CHECK-SPIRV-DAG: TypeSampledImage [[#SampImageTy:]] [[#IntSmpImgTy]]
; CHECK-SPIRV: FunctionParameter [[#Int64Ty]] [[#Input:]]
; CHECK-SPIRV: ConvertHandleToImageINTEL [[#IntImgTy]] [[#]] [[#Input]]
; CHECK-SPIRV: ConvertHandleToSamplerINTEL [[#SamplerTy]] [[#]] [[#Const42]]
; CHECK-SPIRV: ConvertHandleToSampledImageINTEL [[#SampImageTy]] [[#]] [[#Const43]]

; CHECK-LLVM: call spir_func target("spirv.Image", i64, 2, 0, 0, 0, 0, 0, 0) @_Z76__spirv_ConvertHandleToImageINTEL_RPU3AS133__spirv_Image__long_2_0_0_0_0_0_0m(i64 %{{.*}})
; CHECK-LLVM: call spir_func target("spirv.Sampler") @_Z35__spirv_ConvertHandleToSamplerINTELm(i64 42)
; CHECK-LLVM: call spir_func target("spirv.SampledImage", i64, 1, 0, 0, 0, 0, 0, 0) @_Z90__spirv_ConvertHandleToSampledImageINTEL_RPU3AS140__spirv_SampledImage__long_1_0_0_0_0_0_0m(i64 43)

define spir_func void @foo(i64 %in) {
  %img = call spir_func target("spirv.Image", i64, 2, 0, 0, 0, 0, 0, 0) @_Z33__spirv_ConvertHandleToImageINTELl(i64 %in)
  %samp = call spir_func target("spirv.Sampler") @_Z35__spirv_ConvertHandleToSamplerINTELl(i64 42)
  %sampImage = call spir_func target("spirv.SampledImage", i64, 1, 0, 0, 0, 0, 0, 0) @_Z40__spirv_ConvertHandleToSampledImageINTELl(i64 43)
  ret void
}

declare spir_func target("spirv.Image", i64, 2, 0, 0, 0, 0, 0, 0) @_Z33__spirv_ConvertHandleToImageINTELl(i64)

declare spir_func target("spirv.Sampler") @_Z35__spirv_ConvertHandleToSamplerINTELl(i64)

declare spir_func target("spirv.SampledImage", i64, 1, 0, 0, 0, 0, 0, 0) @_Z40__spirv_ConvertHandleToSampledImageINTELl(i64)

!opencl.spir.version = !{!0}
!spirv.Source = !{!1}
!llvm.ident = !{!2}

!0 = !{i32 1, i32 2}
!1 = !{i32 4, i32 100000}
!2 = !{!"clang version 17.0.0"}
