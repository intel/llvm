; RUN: llvm-as %s -o %t.bc
; RUN: not llvm-spirv -s %t.bc 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR
; RUN: llvm-spirv --spirv-ext=+SPV_EXT_long_vector %t.bc
; RUN: llvm-spirv %t.spv -to-text -o - | FileCheck %s

; CHECK-ERROR: Unsupported vector type with 5 elements

; CHECK: ExtInstImport [[ExtInstSetId:[0-9]+]] "OpenCL.std"
; CHECK: TypeInt [[Int:[0-9]+]] 32
; CHECK: TypeVector [[Int5:[0-9]+]] [[Int]] 5

; ModuleID = 'lower-non-standard-vec-with-ext'
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

declare <5 x i32> @llvm.abs.v5i32(<5 x i32> %x, i1 %is_int_min_poison)

; Function Attrs: convergent norecurse
define dso_local spir_func <5 x i32> @test_abs(<5 x i32> %src) local_unnamed_addr {
entry:
  %res = call <5 x i32> @llvm.abs.v5i32(<5 x i32> %src, i1 false)
; CHECK: ExtInst [[Int5]] {{[0-9]+}} [[ExtInstSetId]] s_abs
  ret <5 x i32> %res
}
