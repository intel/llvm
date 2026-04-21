; RUN: llvm-spirv %s --spirv-ext=+SPV_KHR_bfloat16 -o %t.spv
; RUN: llvm-spirv %t.spv -to-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV

; RUN: llvm-spirv -r %t.spv --spirv-target-env=SPV-IR -o - | llvm-dis -o %t.rev.ll
; RUN: FileCheck < %t.rev.ll %s --check-prefixes=CHECK-SPV-IR

; Check OpenCL built-in nan translation.
; Verify it's possible to distinguish between bfloat and half versions.

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spir64"

; CHECK-SPIRV: TypeFloat [[#BFLOAT:]] 16 0 {{$}}
; CHECK-SPIRV: TypeFloat [[#HALF:]] 16 {{$}}
; CHECK-SPIRV: ExtInst [[#BFLOAT]] [[#]] [[#]] nan
; CHECK-SPIRV: ExtInst [[#HALF]] [[#]] [[#]] nan

; CHECK-SPV-IR: call spir_func bfloat @_Z22__spirv_ocl_nan_RDF16bt(
; CHECK-SPV-IR: call spir_func half @_Z15__spirv_ocl_nant(

define dso_local spir_kernel void @test_bfloat(ptr addrspace(1) align 2 %a, i16 %b) {
entry:
  %call = tail call spir_func bfloat @_Z23__spirv_ocl_nan__RDF16bt(i16 %b)
  %call2 = tail call spir_func half @_Z22__spirv_ocl_nan__Rhalft(i16 %b)
  ret void
}

declare spir_func bfloat @_Z23__spirv_ocl_nan__RDF16bt(i16)
declare spir_func half @_Z22__spirv_ocl_nan__Rhalft(i16)


!opencl.ocl.version = !{!0}

!0 = !{i32 3, i32 0}
