; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_INTEL_masked_gather_scatter -o %t.spv
; RUN: llvm-spirv %t.spv --to-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV

; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM-OPAQUE

; RUN: not llvm-spirv %t.bc 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR
; CHECK-ERROR: RequiresExtension: Feature requires the following SPIR-V extension:
; CHECK-ERROR-NEXT: SPV_INTEL_masked_gather_scatter
; CHECK-ERROR-NEXT: NOTE: LLVM module contains vector of pointers, translation of which requires this extension


; CHECK-SPIRV-DAG: Capability MaskedGatherScatterINTEL
; CHECK-SPIRV-DAG: Extension "SPV_INTEL_masked_gather_scatter"

; CHECK-SPIRV-DAG: TypeInt [[#TYPEINT1:]] 8 0
; CHECK-SPIRV-DAG: TypeInt [[#TYPEINT2:]] 32 0
; CHECK-SPIRV-DAG: TypePointer [[#TYPEPTR1:]] 5 [[#TYPEINT1]]
; CHECK-SPIRV-DAG: TypeVector [[#TYPEVEC1:]] [[#TYPEPTR1]] 4
; CHECK-SPIRV-DAG: TypeVoid [[#TYPEVOID:]]
; CHECK-SPIRV-DAG: TypePointer [[#TYPEPTR2:]] 8 [[#TYPEINT1]]
; CHECK-SPIRV-DAG: TypeVector [[#TYPEVEC2:]] [[#TYPEPTR2]] 4
; CHECK-SPIRV-DAG: TypePointer [[#PTRTOVECTYPE:]] 7 [[#TYPEVEC2]]
; CHECK-SPIRV-DAG: TypePointer [[#TYPEPTR4:]] 5 [[#TYPEINT2]]
; CHECK-SPIRV-DAG: TypeVector [[#TYPEVEC3:]] [[#TYPEPTR4]] 4

; CHECK-SPIRV: Variable [[#PTRTOVECTYPE]]
; CHECK-SPIRV: Variable [[#PTRTOVECTYPE]]
; CHECK-SPIRV: Load [[#TYPEVEC2]]
; CHECK-SPIRV: Store
; CHECK-SPIRV: GenericCastToPtr [[#TYPEVEC1]]
; CHECK-SPIRV: FunctionCall [[#TYPEVEC3]]
; CHECK-SPIRV: InBoundsPtrAccessChain [[#TYPEVEC3]]

; CHECK-LLVM-OPAQUE: alloca <4 x ptr addrspace(4)>
; CHECK-LLVM-OPAQUE: alloca <4 x ptr addrspace(4)>
; CHECK-LLVM-OPAQUE: load <4 x ptr addrspace(4)>, ptr
; CHECK-LLVM-OPAQUE: store <4 x ptr addrspace(4)> %[[#]], ptr
; CHECK-LLVM-OPAQUE: addrspacecast <4 x ptr addrspace(4)> %{{.*}} to <4 x ptr addrspace(1)>
; CHECK-LLVM-OPAQUE: call spir_func <4 x ptr addrspace(1)> @boo(<4 x ptr addrspace(1)>
; CHECK-LLVM-OPAQUE: getelementptr inbounds i32, <4 x ptr addrspace(1)> %{{.*}}, i32 1

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir"

; Function Attrs: nounwind readnone
define spir_kernel void @foo() {
entry:
  %arg1 = alloca <4 x ptr addrspace(4)>
  %arg2 = alloca <4 x ptr addrspace(4)>
  %0 = load <4 x ptr addrspace(4)>, ptr %arg1
  store <4 x ptr addrspace(4)> %0, ptr %arg2
  %tmp1 = addrspacecast <4 x ptr addrspace(4)> %0 to  <4 x ptr addrspace(1)>
  %tmp2 = call <4 x ptr addrspace(1)> @boo(<4 x ptr addrspace(1)> %tmp1)
  %tmp3 = getelementptr inbounds i32, <4 x ptr addrspace(1)> %tmp2, i32 1
  ret void
}

declare <4 x ptr addrspace(1)> @boo(<4 x ptr addrspace(1)> %a)

!llvm.module.flags = !{!0}
!opencl.spir.version = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, i32 2}
