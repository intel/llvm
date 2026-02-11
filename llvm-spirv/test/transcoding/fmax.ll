; RUN: llvm-spirv %s --spirv-ext=+SPV_KHR_fma -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv %t.spv -to-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv -r %t.spv -o - | llvm-dis -o %t.rev.ll
; RUN: FileCheck < %t.rev.ll %s --check-prefix=CHECK-LLVM

; Check enabling SPV_KHR_fma does not translate fmax to fma.

; CHECK-SPIRV: ExtInst [[#]] [[#]] [[#]] fmax [[#]] [[#]]

; CHECK-LLVM: %{{.*}} = call spir_func float @_Z4fmaxff(float %{{.*}}, float %{{.*}})

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-unknown"

; Case to test fmax translation via OCL builtins.
define spir_func float @test_fmax_ocl_scalar(float %a, float %b) {
entry:
  %result = call spir_func float @_Z16__spirv_ocl_fmaxff(float %a, float %b)
  ret float %result
}

declare spir_func float @_Z16__spirv_ocl_fmaxff(float, float)
