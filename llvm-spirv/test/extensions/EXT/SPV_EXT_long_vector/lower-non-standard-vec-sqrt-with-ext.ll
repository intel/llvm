; RUN: llvm-as %s -o %t.bc
; RUN: not llvm-spirv -s %t.bc 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR
; RUN: llvm-spirv --spirv-ext=+SPV_EXT_long_vector %t.bc
; RUN: llvm-spirv %t.spv -to-text -o - | FileCheck %s

; CHECK-ERROR: Unsupported vector type with 5 elements

; CHECK: ExtInstImport [[ExtInstSetId:[0-9]+]] "OpenCL.std"
; CHECK: TypeFloat [[Float:[0-9]+]] 32
; CHECK: TypeVector [[Float5:[0-9]+]] [[Float]] 5

; ModuleID = 'lower-non-standard-vec-with-ext'
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

declare <5 x float> @llvm.sqrt.f32(<5 x float> %x)

; Function Attrs: convergent norecurse
define dso_local spir_func <5 x float> @test_sqrt(<5 x float> %src) local_unnamed_addr {
entry:
  %res = call <5 x float> @llvm.sqrt.f32(<5 x float> %src)
; CHECK: ExtInst [[Float5]] {{[0-9]+}} [[ExtInstSetId]] sqrt
  ret <5 x float> %res
}
