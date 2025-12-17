; This test checks the translation of SPV_INTEL_subgroups extension with the SPV_KHR_untyped_pointers extension.
; It verifies instructions validity and mangling even if the pointer is fortified with other instructions like bitcast and addrspacecast.

; RUN: llvm-spirv %s -o %t.spt -spirv-text --spirv-ext=+SPV_INTEL_subgroups,+SPV_KHR_untyped_pointers
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %s -o %t.spv --spirv-ext=+SPV_INTEL_subgroups,+SPV_KHR_untyped_pointers
; RUN: spirv-val %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc --spirv-target-env=SPV-IR
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM-SPIRV

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64"

; CHECK-SPIRV: Capability SubgroupBufferBlockIOINTEL
; CHECK-SPIRV: Extension "SPV_INTEL_subgroups"

; CHECK-SPIRV: TypeInt [[#Int32:]] 32 0
; CHECK-SPIRV-DAG: TypeVector [[#IntVec2:]] [[#Int32]] 2
; CHECK-SPIRV-DAG: TypeUntypedPointerKHR [[#PtrTy:]] 5
; CHECK-SPIRV-DAG: TypePointer [[#Int32Ptr:]] 5 [[#Int32]]

; CHECK-SPIRV: Function
; CHECK-SPIRV: FunctionParameter [[#PtrTy]] [[#ParamPtr:]]

; CHECK-SPIRV: Bitcast [[#Int32Ptr]] [[#TypedPtr:]] [[#]]
; CHECK-SPIRV: SubgroupBlockReadINTEL [[#IntVec2]] [[#Res:]] [[#TypedPtr]]
; CHECK-SPIRV: Bitcast [[#Int32Ptr]] [[#TypedPtr:]] [[#ParamPtr]]
; CHECK-SPIRV: SubgroupBlockWriteINTEL [[#TypedPtr]] [[#Res]]

; CHECK-SPIRV: SubgroupBlockReadINTEL
; CHECK-SPIRV: SubgroupBlockWriteINTEL

; CHECK-LLVM:    %[[#IntVec:]] = call spir_func <2 x i32> @_Z27intel_sub_group_block_read2PU3AS1Kj(ptr addrspace(1) %[[#]])
; CHECK-LLVM:    %[[#P:]] = bitcast ptr addrspace(1) %p0 to ptr addrspace(1)
; CHECK-LLVM:    call spir_func void @_Z28intel_sub_group_block_write2PU3AS1jDv2_j(ptr addrspace(1) %[[#P]], <2 x i32> %[[#IntVec]])

; CHECK-LLVM:    %[[#CP:]] = bitcast ptr addrspace(1) %p1 to ptr addrspace(1)
; CHECK-LLVM:    %[[#CharVec:]] = call spir_func <16 x i8> @_Z31intel_sub_group_block_read_uc16PU3AS1Kh(ptr addrspace(1) %[[#CP]])
; CHECK-LLVM:    %[[#CP1:]] = bitcast ptr addrspace(1) %p1 to ptr addrspace(1)
; CHECK-LLVM:    call spir_func void @_Z32intel_sub_group_block_write_uc16PU3AS1hDv16_h(ptr addrspace(1) %[[#CP1]], <16 x i8> %[[#CharVec]])

; CHECK-LLVM-SPIRV: call spir_func <2 x i32> @_Z36__spirv_SubgroupBlockReadINTEL_Rint2PU3AS1Kj(
; CHECK-LLVM-SPIRV: call spir_func void @_Z31__spirv_SubgroupBlockWriteINTELPU3AS1jDv2_j(
; CHECK-LLVM-SPIRV: call spir_func <16 x i8> @_Z38__spirv_SubgroupBlockReadINTEL_Rchar16PU3AS1Kh(
; CHECK-LLVM-SPIRV: call spir_func void @_Z31__spirv_SubgroupBlockWriteINTELPU3AS1hDv16_h(

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64"

define spir_kernel void @test_subgroup_block_read_write(ptr addrspace(1) %p0, ptr addrspace(1) %p1) {

entry:
  %0 = addrspacecast ptr addrspace(1) %p0 to ptr addrspace(4)
  %1 = call spir_func ptr addrspace(1) @__to_global(ptr addrspace(4) %0)
  %2 = tail call spir_func <2 x i32> @_Z27intel_sub_group_block_read2PU3AS1Kj(ptr addrspace(1) %1)
  tail call spir_func void @_Z28intel_sub_group_block_write2PU3AS1jDv2_j(ptr addrspace(1) %p0, <2 x i32> %2)

  %3 = tail call spir_func <16 x i8> @_Z31intel_sub_group_block_read_uc16PU3AS1Kh(ptr addrspace(1) %p1)
  tail call spir_func void @_Z32intel_sub_group_block_write_uc16PU3AS1hDv16_h(ptr addrspace(1) %p1, <16 x i8> %3)

  ret void
}

declare spir_func <2 x i32> @_Z27intel_sub_group_block_read2PU3AS1Kj(ptr addrspace(1))
declare spir_func <16 x i8> @_Z31intel_sub_group_block_read_uc16PU3AS1Kh(ptr addrspace(1))

declare spir_func void @_Z28intel_sub_group_block_write2PU3AS1jDv2_j(ptr addrspace(1), <2 x i32>)
declare spir_func void @_Z32intel_sub_group_block_write_uc16PU3AS1hDv16_h(ptr addrspace(1), <16 x i8>)

declare spir_func ptr addrspace(1) @__to_global(ptr addrspace(4))

!opencl.ocl.version = !{!0}
!opencl.spir.version = !{!0}

!0 = !{i32 1, i32 2}
