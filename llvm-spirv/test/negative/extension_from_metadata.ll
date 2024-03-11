; Check whether the translator reports an error for a module with !spirv.Extension
; metadata containing an extension that is disabled by --spirv-ext

; RUN: llvm-as < %s -o %t.bc
; RUN: not llvm-spirv --spirv-ext=-SPV_KHR_bit_instructions %t.bc 2>&1 | FileCheck %s

; CHECK: RequiresExtension: Feature requires the following SPIR-V extension:
; CHECK-NEXT: SPV_KHR_bit_instructions

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

; Function Attrs: nounwind
define spir_kernel void @foo() #0 {
entry:
  ret void
}

attributes #0 = { nounwind }

!spirv.MemoryModel = !{!0}
!spirv.Source = !{!1}
!spirv.Extension = !{!3}
!opencl.spir.version = !{!0}
!opencl.ocl.version = !{!0}
!opencl.used.extensions = !{!2}
!opencl.used.optional.core.features = !{!2}

!0 = !{i32 1, i32 2}
!1 = !{i32 3, i32 102000}
!2 = !{}
!3 = !{!"SPV_KHR_bit_instructions"}
