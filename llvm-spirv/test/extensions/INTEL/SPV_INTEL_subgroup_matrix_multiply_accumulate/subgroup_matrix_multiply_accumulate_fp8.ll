; This test checks that SubgroupMatrixMultiplyAccumulateINTEL with FP8 operand flags
; requires the SPV_INTEL_subgroup_matrix_multiply_accumulate_float8 extension.

; RUN: llvm-spirv %s -o %t.spv --spirv-ext=+SPV_INTEL_subgroup_matrix_multiply_accumulate,+SPV_INTEL_subgroup_matrix_multiply_accumulate_float8
; RUN: llvm-spirv %t.spv -o %t.spt --to-text
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV

; RUN: not llvm-spirv %s -o %t.spv --spirv-ext=+SPV_INTEL_subgroup_matrix_multiply_accumulate 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR

; CHECK-ERROR: RequiresExtension: Feature requires the following SPIR-V extension:
; CHECK-ERROR: SPV_INTEL_subgroup_matrix_multiply_accumulate_float8

; CHECK-SPIRV-DAG: Capability SubgroupMatrixMultiplyAccumulateINTEL
; CHECK-SPIRV-DAG: Extension "SPV_INTEL_subgroup_matrix_multiply_accumulate"
; CHECK-SPIRV-DAG: Extension "SPV_INTEL_subgroup_matrix_multiply_accumulate_float8"
; CHECK-SPIRV-DAG: SubgroupMatrixMultiplyAccumulateINTEL {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} 16384
; CHECK-SPIRV-DAG: SubgroupMatrixMultiplyAccumulateINTEL {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} 32768
; CHECK-SPIRV-DAG: SubgroupMatrixMultiplyAccumulateINTEL {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} 65536
; CHECK-SPIRV-DAG: SubgroupMatrixMultiplyAccumulateINTEL {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} 131072

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-unknown"

; Test MatrixAPackedFloat8E4M3INTEL operand (0x4000 = 16384)
define spir_func <4 x float> @test_fp8_e4m3_matrix_a(<4 x float> %c, <4 x i8> %a, <8 x i8> %b) {
entry:
  %result = call spir_func <4 x float> @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv4_hDv8_hDv4_fi(i32 8, <4 x i8> %a, <8 x i8> %b, <4 x float> %c, i32 16384)
  ret <4 x float> %result
}

; Test MatrixBPackedFloat8E4M3INTEL operand (0x8000 = 32768)
define spir_func <4 x float> @test_fp8_e4m3_matrix_b(<4 x float> %c, <4 x i8> %a, <8 x i8> %b) {
entry:
  %result = call spir_func <4 x float> @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv4_hDv8_hDv4_fi(i32 8, <4 x i8> %a, <8 x i8> %b, <4 x float> %c, i32 32768)
  ret <4 x float> %result
}

; Test MatrixAPackedFloat8E5M2INTEL operand (0x10000 = 65536)
define spir_func <4 x float> @test_fp8_e5m2_matrix_a(<4 x float> %c, <4 x i8> %a, <8 x i8> %b) {
entry:
  %result = call spir_func <4 x float> @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv4_hDv8_hDv4_fi(i32 8, <4 x i8> %a, <8 x i8> %b, <4 x float> %c, i32 65536)
  ret <4 x float> %result
}

; Test MatrixBPackedFloat8E5M2INTEL operand (0x20000 = 131072)
define spir_func <4 x float> @test_fp8_e5m2_matrix_b(<4 x float> %c, <4 x i8> %a, <8 x i8> %b) {
entry:
  %result = call spir_func <4 x float> @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv4_hDv8_hDv4_fi(i32 8, <4 x i8> %a, <8 x i8> %b, <4 x float> %c, i32 131072)
  ret <4 x float> %result
}

declare spir_func <4 x float> @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv4_hDv8_hDv4_fi(i32, <4 x i8>, <8 x i8>, <4 x float>, i32)

!opencl.spir.version = !{!0}
!spirv.Source = !{!1}
!llvm.ident = !{!2}

!0 = !{i32 1, i32 0}
!1 = !{i32 4, i32 100000}
!2 = !{!"clang version 17.0.0"}
