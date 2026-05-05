; RUN: llvm-spirv %s -o %t.spv --spirv-ext=+SPV_INTEL_subgroup_buffer_prefetch,+SPV_INTEL_cache_controls
; RUN: llvm-spirv %t.spv --to-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.spv -r --spirv-target-env=SPV-IR -o - | llvm-dis | FileCheck %s --check-prefix=CHECK-LLVM

; SPV_INTEL_subgroup_buffer_prefetch interaction with SPV_INTEL_cache_controls:
; CacheControlLoadINTEL decoration may be used to control which cache levels
; the data will be prefetched into.

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-unknown"

@.str.1 = private unnamed_addr addrspace(1) constant [7 x i8] c"file.h\00", section "llvm.metadata"
; {6442:"0,1"} = {CacheControlLoadINTEL_Token:"CacheLevel,CacheControl"} = L1 Cached
@.str.cc = private unnamed_addr addrspace(1) constant [13 x i8] c"{6442:\220,1\22}\00", section "llvm.metadata"

; CHECK-SPIRV-DAG: Capability SubgroupBufferPrefetchINTEL
; CHECK-SPIRV-DAG: Capability CacheControlsINTEL
; CHECK-SPIRV-DAG: Extension "SPV_INTEL_subgroup_buffer_prefetch"
; CHECK-SPIRV-DAG: Extension "SPV_INTEL_cache_controls"
; CHECK-SPIRV-DAG: TypeInt [[#Int8Ty:]] 8 0
; CHECK-SPIRV-DAG: TypeInt [[#Int32Ty:]] 32 0
; CHECK-SPIRV-DAG: TypePointer [[#GlbPtrTy:]] 5 [[#Int8Ty]]
; CHECK-SPIRV-DAG: Decorate [[#Ptr:]] CacheControlLoadINTEL 0 1

; CHECK-SPIRV: FunctionParameter [[#GlbPtrTy]] [[#Ptr]]
; CHECK-SPIRV: FunctionParameter [[#Int32Ty]] [[#NumBytes:]]
; CHECK-SPIRV: SubgroupBlockPrefetchINTEL [[#Ptr]] [[#NumBytes]]

; CHECK-LLVM: spirv.ParameterDecorations ![[#ParamDecs:]]
; CHECK-LLVM: call spir_func void @_Z34__spirv_SubgroupBlockPrefetchINTELPU3AS1Khj
; CHECK-LLVM: ![[#ParamDecs]] = !{![[#FirstParam:]], ![[#]]}
; CHECK-LLVM: ![[#FirstParam]] = !{![[#DecoNode:]]}
; CHECK-LLVM: ![[#DecoNode]] = !{i32 6442, i32 0, i32 1}

define spir_kernel void @test(ptr addrspace(1) %ptr, i32 %num_bytes) {
entry:
  %annotated = call ptr addrspace(1) @llvm.ptr.annotation.p1.p1(ptr addrspace(1) %ptr, ptr addrspace(1) @.str.cc, ptr addrspace(1) @.str.1, i32 0, ptr addrspace(1) null)
  call spir_func void @_Z34__spirv_SubgroupBlockPrefetchINTELPU3AS1Khj(ptr addrspace(1) %annotated, i32 %num_bytes)
  ret void
}

declare ptr addrspace(1) @llvm.ptr.annotation.p1.p1(ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), i32, ptr addrspace(1))
declare spir_func void @_Z34__spirv_SubgroupBlockPrefetchINTELPU3AS1Khj(ptr addrspace(1), i32)

!opencl.spir.version = !{!0}
!spirv.Source = !{!1}
!0 = !{i32 1, i32 0}
!1 = !{i32 4, i32 100000}
