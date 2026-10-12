; RUN: llvm-spirv %s -o %t.spv --spirv-ext=+SPV_INTEL_subgroup_buffer_prefetch
; RUN: spirv-val %t.spv
; RUN: llvm-spirv %t.spv --to-text -o - | FileCheck %s --check-prefixes=CHECK-SPIRV,CHECK-SPIRV-TYPED-PTRS
; RUN: llvm-spirv %t.spv -r --spirv-target-env=SPV-IR -o - | llvm-dis | FileCheck %s --check-prefix=CHECK-LLVM

; RUN: llvm-spirv %s -o %t.spv --spirv-ext=+SPV_INTEL_subgroup_buffer_prefetch,+SPV_KHR_untyped_pointers
; RUN: spirv-val %t.spv
; RUN: llvm-spirv %t.spv --to-text -o -| FileCheck %s --check-prefixes=CHECK-SPIRV,CHECK-SPIRV-UNTYPED-PTRS
; RUN: llvm-spirv %t.spv -r --spirv-target-env=SPV-IR -o - | llvm-dis | FileCheck %s --check-prefix=CHECK-LLVM

; RUN: not llvm-spirv %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR

; CHECK-ERROR: RequiresExtension: Feature requires the following SPIR-V extension:
; CHECK-ERROR-NEXT: SPV_INTEL_subgroup_buffer_prefetch

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-unknown"

; CHECK-SPIRV-DAG: Capability SubgroupBufferPrefetchINTEL
; CHECK-SPIRV-DAG: Extension "SPV_INTEL_subgroup_buffer_prefetch"
; CHECK-SPIRV-UNTYPED-PTRS-DAG: Capability UntypedPointersKHR
; CHECK-SPIRV-UNTYPED-PTRS-DAG: Extension "SPV_KHR_untyped_pointers"
; CHECK-SPIRV-DAG: TypeInt [[#Int8Ty:]] 8 0
; CHECK-SPIRV-DAG: TypeInt [[#Int32Ty:]] 32 0
; CHECK-SPIRV-TYPED-PTRS-DAG: TypePointer [[#GlbPtrTy:]] 5 [[#Int8Ty]]
; CHECK-SPIRV-UNTYPED-PTRS-DAG: TypeUntypedPointerKHR [[#GlbPtrTy:]] 5

; CHECK-SPIRV: FunctionParameter [[#GlbPtrTy]] [[#Ptr:]]
; CHECK-SPIRV: FunctionParameter [[#Int32Ty]] [[#NumBytes:]]
; CHECK-SPIRV: SubgroupBlockPrefetchINTEL [[#Ptr]] [[#NumBytes]]

; CHECK-LLVM: call spir_func void @_Z34__spirv_SubgroupBlockPrefetchINTELPU3AS1Khj(ptr addrspace(1) %{{.*}}, i32 %{{.*}})

define spir_kernel void @test(ptr addrspace(1) %ptr, i32 %num_bytes) {
entry:
  call spir_func void @_Z34__spirv_SubgroupBlockPrefetchINTELPU3AS1Khj(ptr addrspace(1) %ptr, i32 %num_bytes)
  ret void
}

declare spir_func void @_Z34__spirv_SubgroupBlockPrefetchINTELPU3AS1Khj(ptr addrspace(1), i32)

!opencl.spir.version = !{!0}
!spirv.Source = !{!1}
!0 = !{i32 1, i32 0}
!1 = !{i32 4, i32 100000}
