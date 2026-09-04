; Source (OpenCL C, conceptual):
; float8 __spirv_SubgroupScaledMatrixMultiplyAccumulateINTEL(
;     int K_Dim, short8 Matrix_A, int8 Matrix_B, float8 Matrix_C,
;     uchar Scale_A, uchar Scale_B, int Operands);
; int __spirv_SubgroupScaledMatrixMultiplyAccumulateINTEL(
;     int K_Dim, int Matrix_A, int8 Matrix_B, int Matrix_C,
;     uchar Scale_A, uchar Scale_B);

; TODO: add spirv-val once SPV_INTEL_subgroup_scaled_matrix_multiply_accumulate
; is registered with Khronos and SPIRV-Tools recognizes its capability/opcode.

; RUN: llvm-spirv %s -o %t.spv --spirv-ext=+SPV_INTEL_subgroup_matrix_multiply_accumulate,+SPV_INTEL_subgroup_scaled_matrix_multiply_accumulate
; RUN: llvm-spirv %t.spv --to-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV

; RUN: llvm-spirv %t.spv -r --spirv-target-env=SPV-IR -o - | llvm-dis | FileCheck %s --check-prefix=CHECK-LLVM

; RUN: not llvm-spirv %s -o %t.fail.spv 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR
; CHECK-ERROR: RequiresExtension: Feature requires the following SPIR-V extension:
; CHECK-ERROR: SPV_INTEL_subgroup_matrix_multiply_accumulate

; RUN: not llvm-spirv %s --spirv-ext=+SPV_INTEL_subgroup_matrix_multiply_accumulate -o %t.fail.spv 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR-NEW
; CHECK-ERROR-NEW: RequiresExtension: Feature requires the following SPIR-V extension:
; CHECK-ERROR-NEW: SPV_INTEL_subgroup_scaled_matrix_multiply_accumulate

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-unknown"

; CHECK-SPIRV-DAG: Capability SubgroupMatrixMultiplyAccumulateINTEL
; CHECK-SPIRV-DAG: Capability SubgroupScaledMatrixMultiplyAccumulateINTEL
; CHECK-SPIRV-DAG: Extension "SPV_INTEL_subgroup_matrix_multiply_accumulate"
; CHECK-SPIRV-DAG: Extension "SPV_INTEL_subgroup_scaled_matrix_multiply_accumulate"

; CHECK-SPIRV-DAG: TypeInt [[#Int32Ty:]] 32 0
; CHECK-SPIRV-DAG: Constant [[#Int32Ty]] [[#Const16:]] 16
; CHECK-SPIRV-DAG: TypeFloat [[#FloatTy:]] 32
; CHECK-SPIRV-DAG: TypeVector [[#Vec8FloatTy:]] [[#FloatTy]] 8

; fp16 x fp16 + fp32 with e8m0 scales (operands mask = 0x30000A: MatrixA/BPackedFloat16 (0xA) + ScaleA/BFloat8E8M0 (0x300000)).
; CHECK-SPIRV: SubgroupScaledMatrixMultiplyAccumulateINTEL [[#Vec8FloatTy]] [[#]] [[#Const16]] [[#]] [[#iM8:]] [[#]] [[#sA:]] [[#sB:]] {{[0-9]+}}
; CHECK-SPIRV: SubgroupScaledMatrixMultiplyAccumulateINTEL [[#Int32Ty]] [[#]] [[#Const16]] [[#]] [[#iM8]] [[#]] [[#sA]] [[#sB]] {{$}}

; CHECK-LLVM: %{{.*}} = call spir_func <8 x float> @_Z51__spirv_SubgroupScaledMatrixMultiplyAccumulateINTELiDv8_sDv8_iDv8_fcci(i32 16, <8 x i16> %{{.*}}, <8 x i32> %{{.*}}, <8 x float> %{{.*}}, i8 %{{.*}}, i8 %{{.*}}, i32 3145738)
; CHECK-LLVM: %{{.*}} = call spir_func i32 @_Z51__spirv_SubgroupScaledMatrixMultiplyAccumulateINTELiiDv8_iicc(i32 16, i32 %{{.*}}, <8 x i32> %{{.*}}, i32 %{{.*}}, i8 %{{.*}}, i8 %{{.*}})

define spir_func void @foo(<8 x i16> %sM8, <8 x i32> %iM8, <8 x float> %fM8,
                           i8 %sA, i8 %sB, i32 %iA, i32 %iC) {
entry:
  ; Operands: ScaleAFloat8E8M0INTEL (0x100000) | ScaleBFloat8E8M0INTEL (0x200000)
  ;         | MatrixAPackedFloat16INTEL (0x2)  | MatrixBPackedFloat16INTEL (0x8)
  ;         = 0x30000A = 3145738
  %call = call spir_func <8 x float> @_Z51__spirv_SubgroupScaledMatrixMultiplyAccumulateINTELiDv8_sDv8_iDv8_fcci(i32 16, <8 x i16> %sM8, <8 x i32> %iM8, <8 x float> %fM8, i8 %sA, i8 %sB, i32 3145738)
  %call2 = call spir_func i32 @_Z51__spirv_SubgroupScaledMatrixMultiplyAccumulateINTELiiDv8_iicc(i32 16, i32 %iA, <8 x i32> %iM8, i32 %iC, i8 %sA, i8 %sB)
  ret void
}

declare spir_func <8 x float> @_Z51__spirv_SubgroupScaledMatrixMultiplyAccumulateINTELiDv8_sDv8_iDv8_fcci(i32, <8 x i16>, <8 x i32>, <8 x float>, i8, i8, i32)

declare spir_func i32 @_Z51__spirv_SubgroupScaledMatrixMultiplyAccumulateINTELiiDv8_iicc(i32, i32, <8 x i32>, i32, i8, i8)

!opencl.spir.version = !{!0}
!spirv.Source = !{!1}

!0 = !{i32 1, i32 0}
!1 = !{i32 4, i32 100000}
