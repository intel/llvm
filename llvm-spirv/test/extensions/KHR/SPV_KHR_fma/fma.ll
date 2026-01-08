; RUN: llvm-spirv %s --spirv-ext=+SPV_KHR_fma -o %t.spv
; TODO: enable once spirv-val supports the extension.
; RUNx: spirv-val %t.spv
; RUN: llvm-spirv %t.spv -to-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv -r %t.spv -o - | llvm-dis -o %t.rev.ll
; RUN: FileCheck < %t.rev.ll %s --check-prefix=CHECK-LLVM

; RUN: llvm-spirv %s -spirv-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV-NO-EXT
; RUN: llvm-spirv %s -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv -r %t.spv -o - | llvm-dis -o %t.rev.ll
; RUN: FileCheck < %t.rev.ll %s --check-prefix=CHECK-LLVM

; CHECK-SPIRV: Capability FMAKHR
; CHECK-SPIRV: Extension "SPV_KHR_fma"
; CHECK-SPIRV: TypeFloat [[#TYPE_FLOAT:]] 32
; CHECK-SPIRV: TypeVector [[#TYPE_VEC:]] [[#TYPE_FLOAT]] 4
; CHECK-SPIRV: FmaKHR [[#TYPE_FLOAT]] [[#]]
; CHECK-SPIRV: FmaKHR [[#TYPE_VEC]] [[#]]
; CHECK-SPIRV: FmaKHR [[#TYPE_FLOAT]] [[#]]

; CHECK-SPIRV-NO-EXT-NOT: Capability FMAKHR
; CHECK-SPIRV-NO-EXT-NOT: Extension "SPV_KHR_fma"
; CHECK-SPIRV-NO-EXT: TypeFloat [[#TYPE_FLOAT:]] 32
; CHECK-SPIRV-NO-EXT: TypeVector [[#TYPE_VEC:]] [[#TYPE_FLOAT]] 4
; CHECK-SPIRV-NO-EXT: ExtInst [[#TYPE_FLOAT]] [[#]] [[#]] fma
; CHECK-SPIRV-NO-EXT: ExtInst [[#TYPE_VEC]] [[#]] [[#]] fma

; CHECK-LLVM: %{{.*}} = call spir_func float @_Z3fmafff(float %{{.*}}, float %{{.*}}, float %{{.*}})
; CHECK-LLVM: %{{.*}} = call spir_func <4 x float> @_Z3fmaDv4_fS_S_(<4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}})
; CHECK-LLVM: %{{.*}} = call spir_func float @_Z3fmafff(float %{{.*}}, float %{{.*}}, float %{{.*}})

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-unknown"

define spir_func float @test_fma_scalar(float %a, float %b, float %c) {
entry:
  %result = call float @llvm.fma.f32(float %a, float %b, float %c)
  ret float %result
}

define spir_func <4 x float> @test_fma_vector(<4 x float> %a, <4 x float> %b, <4 x float> %c) {
entry:
  %result = call <4 x float> @llvm.fma.v4f32(<4 x float> %a, <4 x float> %b, <4 x float> %c)
  ret <4 x float> %result
}

; Case to test fma translation via OCL builtins.
define spir_func float @test_fma_ocl_scalar(float %a, float %b, float %c) {
entry:
  %result = call spir_func float @_Z15__spirv_ocl_fmafff(float %a, float %b, float %c)
  ret float %result
}

declare float @llvm.fma.f32(float, float, float)
declare <4 x float> @llvm.fma.v4f32(<4 x float>, <4 x float>, <4 x float>)
declare spir_func float @_Z15__spirv_ocl_fmafff(float, float, float)
