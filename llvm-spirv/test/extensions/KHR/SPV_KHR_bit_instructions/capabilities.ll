; RUN: llvm-as %s -o %t.bc

; RUN: not llvm-spirv %t.bc -spirv-text --spirv-ext=-SPV_KHR_bit_instructions -o - 2>&1 | FileCheck %s --check-prefixes=CHECK-WITHOUT-EXT

; RUN: llvm-spirv %t.bc -spirv-text --spirv-ext=+SPV_KHR_bit_instructions -o %t.txt
; RUN: FileCheck < %t.txt %s --check-prefixes=CHECK-WITH-EXT

; CHECK-WITHOUT-EXT: Unexpected llvm intrinsic:
; CHECK-WITHOUT-EXT: Translation of llvm.bitreverse intrinsic requires SPV_KHR_bit_instructions extension.

; CHECK-WITH-EXT: Capability BitInstructions
; CHECK-WITH-EXT: Extension "SPV_KHR_bit_instructions"

; CHECK-WITH-EXT: 4 BitReverse

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

; Function Attrs: nounwind
define spir_kernel void @TestSatPacked(i32 %0, i32 %1) #0 {
  %3 = call i32 @llvm.bitreverse.i32(i32 %0)
  ret void
}

declare i32 @llvm.bitreverse.i32(i32) #1

attributes #0 = { nounwind }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!spirv.MemoryModel = !{!0}
!spirv.Source = !{!1}
!opencl.spir.version = !{!2}
!opencl.ocl.version = !{!2}
!opencl.used.extensions = !{!3}
!opencl.used.optional.core.features = !{!3}

!0 = !{i32 2, i32 2}
!1 = !{i32 3, i32 102000}
!2 = !{i32 1, i32 2}
!3 = !{}
