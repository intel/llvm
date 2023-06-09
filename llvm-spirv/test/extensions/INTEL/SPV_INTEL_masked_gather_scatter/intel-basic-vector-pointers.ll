; RUN: llvm-as -opaque-pointers=0 %s -o %t.bc
; RUN: llvm-spirv %t.bc -opaque-pointers=0 --spirv-ext=+SPV_INTEL_masked_gather_scatter -o %t.spv
; RUN: llvm-spirv %t.spv --to-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV

; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis -opaque-pointers=0 < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; RUN: not llvm-spirv %t.bc -opaque-pointers=0 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR
; CHECK-ERROR: RequiresExtension: Feature requires the following SPIR-V extension:
; CHECK-ERROR-NEXT: SPV_INTEL_masked_gather_scatter
; CHECK-ERROR-NEXT: NOTE: LLVM module contains vector of pointers, translation of which requires this extension


; CHECK-SPIRV-DAG: Capability MaskedGatherScatterINTEL
; CHECK-SPIRV-DAG: Extension "SPV_INTEL_masked_gather_scatter"

; CHECK-SPIRV-DAG: TypeInt [[#INTTYPE1:]] 32 0
; CHECK-SPIRV-DAG: TypeInt [[#INTTYPE2:]] 8 0
; CHECK-SPIRV-DAG: TypePointer [[#PTRTYPE1:]] 5 [[#INTTYPE1]]
; CHECK-SPIRV-DAG: TypeVector [[#VECTYPE1:]] [[#PTRTYPE1]] 4
; CHECK-SPIRV-DAG: TypePointer [[#PTRTYPE2:]] 8 [[#INTTYPE2]]
; CHECK-SPIRV-DAG: TypeVector [[#VECTYPE2:]] [[#PTRTYPE2]] 4
; CHECK-SPIRV-DAG: TypePointer [[#PTRTOVECTYPE:]] 7 [[#VECTYPE2]]
; CHECK-SPIRV-DAG: TypePointer [[#PTRTYPE3:]] 8 [[#INTTYPE1]]
; CHECK-SPIRV-DAG: TypeVector [[#VECTYPE3:]] [[#PTRTYPE3]] 4

; CHECK-SPIRV: Variable [[#PTRTOVECTYPE]]
; CHECK-SPIRV: Variable [[#PTRTOVECTYPE]]
; CHECK-SPIRV: Load [[#VECTYPE2]]
; CHECK-SPIRV: Store
; CHECK-SPIRV: Bitcast [[#VECTYPE3]]
; CHECK-SPIRV: GenericCastToPtr [[#VECTYPE1]]
; CHECK-SPIRV: FunctionCall [[#VECTYPE1]]
; CHECK-SPIRV: InBoundsPtrAccessChain [[#VECTYPE1]]

; CHECK-LLVM: alloca <4 x i8 addrspace(4)*>
; CHECK-LLVM-NEXT: alloca <4 x i8 addrspace(4)*>
; CHECK-LLVM-NEXT: load <4 x i8 addrspace(4)*>, <4 x i8 addrspace(4)*>*
; CHECK-LLVM-NEXT: store <4 x i8 addrspace(4)*> %[[#]], <4 x i8 addrspace(4)*>*
; CHECK-LLVM-NEXT: bitcast <4 x i8 addrspace(4)*> %[[#]] to <4 x i32 addrspace(4)*>
; CHECK-LLVM-NEXT: addrspacecast <4 x i32 addrspace(4)*> %{{.*}} to <4 x i32 addrspace(1)*>
; CHECK-LLVM-NEXT: call spir_func <4 x i32 addrspace(1)*> @boo(<4 x i32 addrspace(1)*>
; CHECK-LLVM-NEXT: getelementptr inbounds i32, <4 x i32 addrspace(1)*> %{{.*}}, i32 1

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir"

; Function Attrs: nounwind readnone
define spir_kernel void @foo() {
entry:
  %arg1 = alloca <4 x i8 addrspace(4)*>
  %arg2 = alloca <4 x i8 addrspace(4)*>
  %0 = load <4 x i8 addrspace(4)*>, <4 x i8 addrspace(4)*>* %arg1
  store <4 x i8 addrspace(4)*> %0, <4 x i8 addrspace(4)*>* %arg2
  %tmp1 = bitcast <4 x i8 addrspace(4)*> %0 to <4 x i32 addrspace(4)*>
  %tmp2 = addrspacecast <4 x i32 addrspace(4)*> %tmp1 to  <4 x i32 addrspace(1)*>
  %tmp3 = call <4 x i32 addrspace(1)*> @boo(<4 x i32 addrspace(1)*> %tmp2)
  %tmp4 = getelementptr inbounds i32, <4 x i32 addrspace(1)*> %tmp3, i32 1
  %tmp5 = addrspacecast <4 x i32 addrspace(4)*> %tmp1 to <4 x i8 addrspace(1)*>
  ret void
}

declare <4 x i32 addrspace(1)*> @boo(<4 x i32 addrspace(1)*> %a)

!llvm.module.flags = !{!0}
!opencl.spir.version = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, i32 2}
