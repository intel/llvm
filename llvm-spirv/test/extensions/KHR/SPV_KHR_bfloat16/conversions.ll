; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_KHR_bfloat16 -o %t.spv
; RUN: llvm-spirv %t.spv -to-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV

; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc -o - | FileCheck %s --check-prefix=CHECK-LLVM

; RUN: not llvm-spirv %t.bc 2>&1 >/dev/null | FileCheck %s --check-prefix=CHECK-ERROR
; RUN: not llvm-spirv %t.bc --spirv-ext=+SPV_INTEL_bfloat16_arithmetic 2>&1 >/dev/null | FileCheck %s --check-prefix=CHECK-ERROR
; CHECK-ERROR: RequiresExtension: Feature requires the following SPIR-V extension:

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spirv64-unknown-unknown"

; CHECK-SPIRV: Capability BFloat16TypeKHR
; CHECK-SPIRV-NOT: Capability BFloat16ArithmeticINTEL
; CHECK-SPIRV-NOT: Extension "SPV_INTEL_bfloat16_arithmetic"
; CHECK-SPIRV: Extension "SPV_KHR_bfloat16"
; CHECK-SPIRV: TypeFloat [[BFLOAT:[0-9]+]] 16 0
; CHECK-SPIRV: Variable [[#]] [[ADDR1:[0-9]+]]
; CHECK-SPIRV: Load [[BFLOAT]] [[DATA1:[0-9]+]] [[ADDR1]]
; CHECK-SPIRV: ConvertFToU [[#]] [[#]] [[DATA1]]
; CHECK-SPIRV: ConvertFToS [[#]] [[#]] [[DATA1]]
; CHECK-SPIRV: ConvertSToF [[BFLOAT]] [[#]] [[#]]
; CHECK-SPIRV: ConvertUToF [[BFLOAT]] [[#]] [[#]]


; CHECK-LLVM: [[ADDR1:[%a-z0-9]+]] = alloca bfloat
; CHECK-LLVM: [[DATA1:[%a-z0-9]+]] = load bfloat, ptr [[ADDR1]]
; CHECK-LLVM: %OpConvertFToU = fptoui bfloat [[DATA1]] to i32
; CHECK-LLVM: %OpConvertFToS = fptosi bfloat [[DATA1]] to i32
; CHECK-LLVM: %OpConvertSToF = sitofp i32 0 to bfloat
; CHECK-LLVM: %OpConvertUToF = uitofp i32 0 to bfloat

define spir_kernel void @testConversions() {
entry:
  %addr1 = alloca bfloat
  %data1 = load bfloat, ptr %addr1
  %OpConvertFToU = fptoui bfloat %data1 to i32
  %OpConvertFToS = fptosi bfloat %data1 to i32
  %OpConvertSToF = sitofp i32 0 to bfloat
  %OpConvertUToF = uitofp i32 0 to bfloat
  ret void
}

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
