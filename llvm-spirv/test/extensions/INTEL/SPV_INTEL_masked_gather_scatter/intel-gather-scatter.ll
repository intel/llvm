; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_INTEL_masked_gather_scatter -o %t.spv
; RUN: llvm-spirv %t.spv --to-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV

; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_INTEL_masked_gather_scatter -o %t.spv -spirv-allow-unknown-intrinsics
; RUN: llvm-spirv %t.spv --to-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV

; RUN: not llvm-spirv %t.bc 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR
; CHECK-ERROR: RequiresExtension: Feature requires the following SPIR-V extension:
; CHECK-ERROR-NEXT: SPV_INTEL_masked_gather_scatter
; CHECK-ERROR-NEXT: NOTE: LLVM module contains vector of pointers, translation of which requires this extension

; CHECK-SPIRV-NOT: Name [[#]] "llvm.masked.gather.v4i32.v4p4"
; CHECK-SPIRV-NOT: Name [[#]] "llvm.masked.scatter.v4i32.v4p4"

; CHECK-SPIRV-DAG: Capability MaskedGatherScatterINTEL
; CHECK-SPIRV-DAG: Extension "SPV_INTEL_masked_gather_scatter"

; CHECK-SPIRV-DAG: TypeInt [[#TYPEINT:]] 32 0
; CHECK-SPIRV-DAG: TypePointer [[#TYPEPTRINT:]] [[#]] [[#TYPEINT]]
; CHECK-SPIRV-DAG: TypeVector [[#TYPEVECPTR:]] [[#TYPEPTRINT]] 4
; CHECK-SPIRV-DAG: TypeVector [[#TYPEVECINT:]] [[#TYPEINT]] 4

; CHECK-SPIRV-DAG: Constant [[#TYPEINT]] [[#CONST4:]] 4
; CHECK-SPIRV-DAG: Constant [[#TYPEINT]] [[#CONST0:]] 0
; CHECK-SPIRV-DAG: Constant [[#TYPEINT]] [[#CONST1:]] 1
; CHECK-SPIRV-DAG: ConstantTrue [[#]] [[#TRUE:]]
; CHECK-SPIRV-DAG: ConstantFalse [[#]] [[#FALSE:]]
; CHECK-SPIRV-DAG: ConstantComposite [[#]] [[#MASK1:]] [[#TRUE]] [[#FALSE]] [[#TRUE]] [[#TRUE]]
; CHECK-SPIRV-DAG: ConstantComposite [[#]] [[#FILL:]] [[#CONST4]] [[#CONST0]] [[#CONST1]] [[#CONST0]]
; CHECK-SPIRV-DAG: ConstantComposite [[#]] [[#MASK2:]] [[#TRUE]] [[#TRUE]] [[#TRUE]] [[#TRUE]]

; CHECK-SPIRV: Load [[#TYPEVECPTR]] [[#VECGATHER:]]
; CHECK-SPIRV: Load [[#TYPEVECPTR]] [[#VECSCATTER:]]
; CHECK-SPIRV: MaskedGatherINTEL [[#TYPEVECINT]] [[#GATHER:]] [[#VECGATHER]] 4 [[#MASK1]] 23
; CHECK-SPIRV: MaskedScatterINTEL [[#GATHER]] [[#VECSCATTER]] 4 [[#MASK2]]

; CHECK-LLVM: %[[#VECGATHER:]] = load <4 x ptr addrspace(4)>, ptr
; CHECK-LLVM: %[[#VECSCATTER:]] = load <4 x ptr addrspace(4)>, ptr
; CHECK-LLVM: %[[GATHER:[a-z0-9]+]] = call <4 x i32> @llvm.masked.gather.v4i32.v4p4(<4 x ptr addrspace(4)> %[[#VECGATHER]], i32 4, <4 x i1> <i1 true, i1 false, i1 true, i1 true>, <4 x i32> <i32 4, i32 0, i32 1, i32 0>)
; CHECK-LLVM: call void @llvm.masked.scatter.v4i32.v4p4(<4 x i32> %[[GATHER]], <4 x ptr addrspace(4)> %[[#VECSCATTER]], i32 4, <4 x i1> <i1 true, i1 true, i1 true, i1 true>)

; CHECK-LLVM-DAG: declare <4 x i32> @llvm.masked.gather.v4i32.v4p4(<4 x ptr addrspace(4)>, i32 immarg, <4 x i1>, <4 x i32>)
; CHECK-LLVM-DAG: declare void @llvm.masked.scatter.v4i32.v4p4(<4 x i32>, <4 x ptr addrspace(4)>, i32 immarg, <4 x i1>)

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir"

; Function Attrs: nounwind readnone
define spir_kernel void @foo() {
entry:
  %arg0 = alloca <4 x ptr addrspace(4)>
  %arg1 = alloca <4 x ptr addrspace(4)>
  %0 = load <4 x ptr addrspace(4)>, ptr %arg0
  %1 = load <4 x ptr addrspace(4)>, ptr %arg1
  %res = call <4 x i32> @llvm.masked.gather.v4i32.v4p4(<4 x ptr addrspace(4)> %0, i32 4, <4 x i1> <i1 true, i1 false, i1 true, i1 true>, <4 x i32> <i32 4, i32 0, i32 1, i32 0>)
  call void @llvm.masked.scatter.v4i32.v4p4(<4 x i32> %res, <4 x ptr addrspace(4)> %1, i32 4, <4 x i1> <i1 true, i1 true, i1 true, i1 true>)
  ret void
}

declare <4 x i32> @llvm.masked.gather.v4i32.v4p4(<4 x ptr addrspace(4)>, i32, <4 x i1>, <4 x i32>)

declare void @llvm.masked.scatter.v4i32.v4p4(<4 x i32>, <4 x ptr addrspace(4)>, i32, <4 x i1>)

!llvm.module.flags = !{!0}
!opencl.spir.version = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, i32 2}
