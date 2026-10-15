; Test memset with non-zero fill, and a constant length too large for
; a single OpConstantComposite that exceeds maximum WordCount.

; RUN: llvm-as %s -o %t.bc
; RUN: not llvm-spirv %t.bc -o %t.spv 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR

; RUN: llvm-spirv --spirv-ext=+SPV_INTEL_long_composites %t.bc -o %t.spv
; RUN: llvm-spirv %t.spv --to-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV --implicit-check-not=Variable --implicit-check-not=CopyMemorySized
; RUN: llvm-spirv -r %t.spv -o - | llvm-dis | FileCheck %s --check-prefix=CHECK-LLVM
; TODO: currently spirv-val falsely rejects OpConstantCompositeContinuedINTEL in the module.
; Re-enable spirv-val once it's fixed. Tracker: https://github.com/KhronosGroup/SPIRV-Tools/issues/6954
; RUNx: spirv-val %t.spv

; CHECK-ERROR: InvalidWordCount: Can't encode instruction with word count greater than 65535:
; CHECK-ERROR-NEXT: Id: [[#]], OpCode: ConstantComposite

; CHECK-SPIRV:      Capability LongCompositesINTEL
; CHECK-SPIRV:      Extension "SPV_INTEL_long_composites"
; CHECK-SPIRV:      TypeInt [[#ByteTy:]] 8 0
; CHECK-SPIRV:      Constant [[#]] [[#Length:]] 70000
; CHECK-SPIRV:      Constant [[#ByteTy]] [[#Byte:]] 3
; CHECK-SPIRV:      TypeArray [[#ArrayTy:]] [[#ByteTy]] [[#Length]]
; CHECK-SPIRV:      TypePointer [[#StorePtrTy:]] 5 [[#ArrayTy]]
; CHECK-SPIRV:      65535 ConstantComposite [[#ArrayTy]] [[#Init:]] [[#Byte]]{{( [0-9]+)+ ?$}}
; CHECK-SPIRV-NEXT: 4469 ConstantCompositeContinuedINTEL [[#Byte]]{{( [0-9]+)+ ?$}}
; CHECK-SPIRV:      Bitcast [[#StorePtrTy]] [[#ArrayPtr:]] [[#]]
; CHECK-SPIRV:      Store [[#ArrayPtr]] [[#Init]] 2 4

; CHECK-LLVM: store [70000 x i8] c"{{(\\03)+}}", ptr addrspace(1) %[[#]], align 4

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-unknown"

define spir_kernel void @fill(ptr addrspace(1) %dst) {
entry:
  call void @llvm.memset.p1.i64(ptr addrspace(1) align 4 %dst, i8 3, i64 70000, i1 false)
  ret void
}

declare void @llvm.memset.p1.i64(ptr addrspace(1) noalias writeonly, i8, i64, i1 immarg)
