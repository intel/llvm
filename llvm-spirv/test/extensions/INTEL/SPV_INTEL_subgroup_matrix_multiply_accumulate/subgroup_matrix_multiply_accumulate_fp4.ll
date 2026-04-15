; This test checks that SubgroupMatrixMultiplyAccumulateINTEL with FP4 operand flags
; requires the SPV_INTEL_subgroup_matrix_multiply_accumulate_float4 extension.

; RUN: llvm-spirv %s -o %t.spv --spirv-ext=+SPV_INTEL_subgroup_matrix_multiply_accumulate,+SPV_INTEL_subgroup_matrix_multiply_accumulate_float4
; RUN: llvm-spirv %t.spv -o %t.spt --to-text
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV

; RUN: not llvm-spirv %s -o %t.spv --spirv-ext=+SPV_INTEL_subgroup_matrix_multiply_accumulate 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR

; CHECK-ERROR: RequiresExtension: Feature requires the following SPIR-V extension:
; CHECK-ERROR: SPV_INTEL_subgroup_matrix_multiply_accumulate_float4

; CHECK-SPIRV-DAG: Capability SubgroupMatrixMultiplyAccumulateINTEL
; CHECK-SPIRV-DAG: Extension "SPV_INTEL_subgroup_matrix_multiply_accumulate"
; CHECK-SPIRV-DAG: Extension "SPV_INTEL_subgroup_matrix_multiply_accumulate_float4"
; CHECK-SPIRV-DAG: SubgroupMatrixMultiplyAccumulateINTEL {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} 262144
; CHECK-SPIRV-DAG: SubgroupMatrixMultiplyAccumulateINTEL {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} 524288
; CHECK-SPIRV-DAG: SubgroupMatrixMultiplyAccumulateINTEL {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} 786432

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-unknown"

; Test MatrixAPackedFloat4E2M1INTEL operand (0x40000 = 262144)
define spir_func <4 x float> @test_fp4_matrix_a(<4 x float> %c, <4 x i8> %a, <8 x i8> %b) {
entry:
  %result = call spir_func <4 x float> @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv4_hDv8_hDv4_fi(i32 8, <4 x i8> %a, <8 x i8> %b, <4 x float> %c, i32 262144)
  ret <4 x float> %result
}

; Test MatrixBPackedFloat4E2M1INTEL operand (0x80000 = 524288)
define spir_func <4 x float> @test_fp4_matrix_b(<4 x float> %c, <4 x i8> %a, <8 x i8> %b) {
entry:
  %result = call spir_func <4 x float> @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv4_hDv8_hDv4_fi(i32 8, <4 x i8> %a, <8 x i8> %b, <4 x float> %c, i32 524288)
  ret <4 x float> %result
}

; Test both FP4 operands (0xC0000 = 786432)
define spir_func <4 x float> @test_fp4_matrix_both(<4 x float> %c, <4 x i8> %a, <8 x i8> %b) {
entry:
  %result = call spir_func <4 x float> @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv4_hDv8_hDv4_fi(i32 8, <4 x i8> %a, <8 x i8> %b, <4 x float> %c, i32 786432)
  ret <4 x float> %result
}

declare spir_func <4 x float> @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv4_hDv8_hDv4_fi(i32, <4 x i8>, <8 x i8>, <4 x float>, i32)

!opencl.spir.version = !{!0}
!spirv.Source = !{!1}
!llvm.ident = !{!2}

!0 = !{i32 1, i32 0}
!1 = !{i32 4, i32 100000}
!2 = !{!"clang version 17.0.0"}
