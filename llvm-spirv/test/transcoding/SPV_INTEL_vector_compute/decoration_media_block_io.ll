; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv --spirv-ext=+SPV_INTEL_vector_compute --spirv-allow-unknown-intrinsics=llvm.genx
; RUN: llvm-spirv %t.spv -o %t.spt --to-text
; RUN: llvm-spirv -r %t.spv -o %t.bc
; RUN: llvm-dis %t.bc -o %t.ll
; RUN: FileCheck %s --input-file %t.spt -check-prefix=SPV
; RUN: FileCheck %s --input-file %t.ll  -check-prefix=LLVM

target datalayout = "e-p:64:64-i64:64-n8:16:32"
target triple = "spir"


%intel.buffer_rw_t = type opaque
%opencl.image1d_rw_t = type opaque
%opencl.image1d_buffer_rw_t = type opaque
%opencl.image2d_rw_t = type opaque
%opencl.image3d_rw_t = type opaque
%opencl.sampler_t = type opaque

; SPV-DAG: 4 Name [[IM2D:[0-9]+]] "im2d"
; SPV-DAG: 4 Name [[IM3D:[0-9]+]] "im3d"
; SPV-DAG: 3 Decorate [[IM3D]] MediaBlockIOINTEL
; SPV-DAG: 3 Decorate [[IM2D]] MediaBlockIOINTEL

; LLVM: @test
; LLVM-SAME: %opencl.image2d_rw_t
; LLVM-SAME: "VCMediaBlockIO"
; LLVM-SAME: %im2d,
; LLVM-SAME: %opencl.image3d_rw_t
; LLVM-SAME: "VCMediaBlockIO"
; LLVM-SAME: %im3d,

define spir_kernel void @test(%intel.buffer_rw_t addrspace(1)* %buf, %opencl.image1d_rw_t addrspace(1)* %im1d, %opencl.image1d_buffer_rw_t addrspace(1)* %im1db, %opencl.image2d_rw_t addrspace(1)* "VCMediaBlockIO" %im2d, %opencl.image3d_rw_t addrspace(1)* "VCMediaBlockIO" %im3d, %opencl.sampler_t addrspace(2)* %samp, i8 addrspace(1)* %ptr, <4 x i32> %gen) #0 {
entry:
  ret void
}

attributes #0 = { "VCFunction" }

