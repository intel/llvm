; TODO: add spirv-val once SPV_INTEL_subgroup_scaled_matrix_multiply_accumulate
; is registered with Khronos and SPIRV-Tools recognizes its capability/opcode.

; RUN: llvm-spirv %s -o %t.spv --spirv-ext=+SPV_INTEL_subgroup_matrix_multiply_accumulate,+SPV_INTEL_subgroup_matrix_multiply_accumulate_float4,+SPV_INTEL_subgroup_scaled_matrix_multiply_accumulate
; RUN: llvm-spirv %t.spv --to-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV

; RUN: llvm-spirv %t.spv -r --spirv-target-env=SPV-IR -o - | llvm-dis | FileCheck %s --check-prefix=CHECK-LLVM

; RUN: not llvm-spirv %s --spirv-ext=+SPV_INTEL_subgroup_matrix_multiply_accumulate,+SPV_INTEL_subgroup_scaled_matrix_multiply_accumulate -o %t.fail.spv 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR-FP4
; CHECK-ERROR-FP4: RequiresExtension: Feature requires the following SPIR-V extension:
; CHECK-ERROR-FP4: SPV_INTEL_subgroup_matrix_multiply_accumulate_float4
; CHECK-ERROR-FP4: SubgroupScaledMatrixMultiplyAccumulateINTEL with FP4 operand flags

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-unknown"

; CHECK-SPIRV-DAG: Capability SubgroupMatrixMultiplyAccumulateINTEL
; CHECK-SPIRV-DAG: Capability SubgroupScaledMatrixMultiplyAccumulateINTEL
; CHECK-SPIRV-DAG: Extension "SPV_INTEL_subgroup_matrix_multiply_accumulate"
; CHECK-SPIRV-DAG: Extension "SPV_INTEL_subgroup_matrix_multiply_accumulate_float4"
; CHECK-SPIRV-DAG: Extension "SPV_INTEL_subgroup_scaled_matrix_multiply_accumulate"

; CHECK-SPIRV: SubgroupScaledMatrixMultiplyAccumulateINTEL

; CHECK-LLVM: %{{.*}} = call spir_func <8 x float> @_Z51__spirv_SubgroupScaledMatrixMultiplyAccumulateINTELiDv8_sDv8_iDv8_fDv2_cS2_i(i32 64, <8 x i16> %{{.*}}, <8 x i32> %{{.*}}, <8 x float> %{{.*}}, <2 x i8> %{{.*}}, <2 x i8> %{{.*}}, i32 3932160)

define spir_func void @foo(<8 x i16> %sM8, <8 x i32> %iM8, <8 x float> %fM8,
                           <2 x i8> %sA, <2 x i8> %sB) {
entry:
  ; Operands: ScaleA/BFloat8E8M0INTEL (0x300000) | MatrixAPackedFloat4E2M1INTEL (0x40000) | MatrixBPackedFloat4E2M1INTEL (0x80000)
  ;         = 0x3C0000 = 3932160
  %call = call spir_func <8 x float> @_Z51__spirv_SubgroupScaledMatrixMultiplyAccumulateINTELiDv8_sDv8_iDv8_fDv2_cS2_i(i32 64, <8 x i16> %sM8, <8 x i32> %iM8, <8 x float> %fM8, <2 x i8> %sA, <2 x i8> %sB, i32 3932160)
  ret void
}

declare spir_func <8 x float> @_Z51__spirv_SubgroupScaledMatrixMultiplyAccumulateINTELiDv8_sDv8_iDv8_fDv2_cS2_i(i32, <8 x i16>, <8 x i32>, <8 x float>, <2 x i8>, <2 x i8>, i32)

!opencl.spir.version = !{!0}
!spirv.Source = !{!1}

!0 = !{i32 1, i32 0}
!1 = !{i32 4, i32 100000}
