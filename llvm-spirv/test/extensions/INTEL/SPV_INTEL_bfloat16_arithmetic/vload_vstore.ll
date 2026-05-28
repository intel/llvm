; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o %t.spt --spirv-ext=+SPV_KHR_bfloat16,+SPV_INTEL_bfloat16_arithmetic
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv -to-binary %t.spt -o %t.spv
; TODO: reenable the validation once the BFloat16 type is supported in ExtInst.
; Currently fails with: ExtInst doesn't support BFloat16 type.
; RUNx: spirv-val %t.spv
; RUN: llvm-spirv -r %t.spv -o - | llvm-dis -o %t.rev.ll
; RUN: FileCheck < %t.rev.ll %s --check-prefix=CHECK-LLVM
; RUN: llvm-spirv -r %t.spv --spirv-target-env=SPV-IR -o - | llvm-dis -o %t.rev.ll
; RUN: FileCheck < %t.rev.ll %s --check-prefix=CHECK-SPV-IR


; CHECK-SPIRV: Capability BFloat16TypeKHR
; CHECK-SPIRV: Extension "SPV_KHR_bfloat16"
; CHECK-SPIRV: TypeFloat [[#BFLOAT:]] 16 0
; CHECK-SPIRV: TypeVector [[#VEC:]] [[#BFLOAT]] 2
; CHECK-SPIRV: TypePointer [[#PTR:]] [[#]] [[#BFLOAT]]

; CHECK-LABEL: Function
; CHECK-SPIRV: FunctionParameter [[#PTR]] [[#PTR_ARG:]]
; CHECK-SPIRV: ExtInst [[#VEC]] [[#]] [[#]] vloadn [[#]] [[#PTR_ARG]] 2

; CHECK-LABEL: Function
; CHECK-SPIRV: FunctionParameter [[#VEC]] [[#DATA_ARG:]]
; CHECK-SPIRV: FunctionParameter [[#PTR]] [[#PTR_ARG2:]]
; CHECK-SPIRV: ExtInst [[#]] [[#]] [[#]] vstoren [[#DATA_ARG]] [[#]] [[#PTR_ARG2]]

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

; CHECK-LLVM: call spir_func <2 x bfloat> @_Z6vload2mPU3AS1KDF16b(i64 %offset, ptr addrspace(1) %ptr)
; CHECK-LLVM: call spir_func void @_Z7vstore2Dv2_DF16bmPU3AS1DF16b(<2 x bfloat> %data, i64 %offset, ptr addrspace(1) %ptr)

; CHECK-SPV-IR: call spir_func <2 x bfloat> @_Z26__spirv_ocl_vloadn_RDF16b2mPU3AS1KDF16bi(i64 %offset, ptr addrspace(1) %ptr, i32 2)
; CHECK-SPV-IR: call spir_func void @_Z19__spirv_ocl_vstorenDv2_DF16bmPU3AS1DF16b(<2 x bfloat> %data, i64 %offset, ptr addrspace(1) %ptr)

define spir_func <2 x bfloat> @test_spirv_ocl_vload2(i64 %offset, ptr addrspace(1) %ptr) {
  %result = call spir_func <2 x bfloat> @_Z26__spirv_ocl_vloadn__RDF16blPU3AS1DF16bi(i64 %offset, ptr addrspace(1) %ptr, i32 2)
  ret <2 x bfloat> %result
}

define spir_func void @test_spirv_ocl_vstore2(<2 x bfloat> %data, i64 %offset, ptr addrspace(1) %ptr) {
  call spir_func void @_Z19__spirv_ocl_vstorenDv2_DF16blPU3AS1DF16b(<2 x bfloat> %data, i64 %offset, ptr addrspace(1) %ptr)
  ret void
}

declare spir_func <2 x bfloat> @_Z26__spirv_ocl_vloadn__RDF16blPU3AS1DF16bi(i64, bfloat addrspace(1)*, i32)
declare spir_func void @_Z19__spirv_ocl_vstorenDv2_DF16blPU3AS1DF16b(<2 x bfloat>, i64, bfloat addrspace(1)*)

!opencl.enable.FP_CONTRACT = !{}
!opencl.spir.version = !{!0}
!opencl.ocl.version = !{!1}
!opencl.used.extensions = !{!2}
!opencl.used.optional.core.features = !{!3}
!opencl.compiler.options = !{!3}

!0 = !{i32 1, i32 2}
!1 = !{i32 2, i32 0}
!2 = !{!"cl_khr_fp16"}
!3 = !{}
