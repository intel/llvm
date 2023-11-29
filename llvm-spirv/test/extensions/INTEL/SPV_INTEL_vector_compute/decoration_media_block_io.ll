; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv --spirv-ext=+SPV_INTEL_vector_compute --spirv-allow-unknown-intrinsics=llvm.genx
; RUN: llvm-spirv %t.spv -o %t.spt --to-text
; RUN: llvm-spirv -r %t.spv -o %t.bc --spirv-target-env=SPV-IR
; RUN: llvm-dis %t.bc -o %t.ll
; RUN: FileCheck %s --input-file %t.spt -check-prefix=SPV
; RUN: FileCheck %s --input-file %t.ll  -check-prefix=LLVM

target datalayout = "e-p:64:64-i64:64-n8:16:32"
target triple = "spir"


; SPV-DAG: 4 Name [[IM2D:[0-9]+]] "im2d"
; SPV-DAG: 4 Name [[IM3D:[0-9]+]] "im3d"
; SPV-DAG: 3 Decorate [[IM3D]] MediaBlockIOINTEL
; SPV-DAG: 3 Decorate [[IM2D]] MediaBlockIOINTEL

; LLVM: @test
; LLVM-SAME: target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 2)
; LLVM-SAME: "VCMediaBlockIO"
; LLVM-SAME: %im2d,
; LLVM-SAME: target("spirv.Image", void, 2, 0, 0, 0, 0, 0, 2)
; LLVM-SAME: "VCMediaBlockIO"
; LLVM-SAME: %im3d,

define spir_kernel void @test(target("spirv.BufferSurfaceINTEL", 2) %buf, target("spirv.Image", void, 0, 0, 0, 0, 0, 0, 2) %im1d, target("spirv.Image", void, 5, 0, 0, 0, 0, 0, 2) %im1db, target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 2) "VCMediaBlockIO" %im2d, target("spirv.Image", void, 2, 0, 0, 0, 0, 0, 2) "VCMediaBlockIO" %im3d, target("spirv.Sampler") %samp, ptr addrspace(1) %ptr, <4 x i32> %gen) #0 {
entry:
  ret void
}

attributes #0 = { "VCFunction" }

