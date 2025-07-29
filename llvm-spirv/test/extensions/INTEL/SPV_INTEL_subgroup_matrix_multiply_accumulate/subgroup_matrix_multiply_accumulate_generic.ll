; generated with mma.cl:
; #pragma OPENCL EXTENSION cl_khr_fp16 : enable
; 
; // all combinations of parameter types
; int  __spirv_SubgroupMatrixMultiplyAccumulateINTEL(int K_Dim, int  Matrix_A, int8 Matrix_B, int  Matrix_C, int Operands);
; int2 __spirv_SubgroupMatrixMultiplyAccumulateINTEL(int K_Dim, int2 Matrix_A, int8 Matrix_B, int2 Matrix_C, int Operands);
; int4 __spirv_SubgroupMatrixMultiplyAccumulateINTEL(int K_Dim, int4 Matrix_A, int8 Matrix_B, int4 Matrix_C, int Operands);
; int8 __spirv_SubgroupMatrixMultiplyAccumulateINTEL(int K_Dim, int8 Matrix_A, int8 Matrix_B, int8 Matrix_C, int Operands);
; 
; float  __spirv_SubgroupMatrixMultiplyAccumulateINTEL(int K_Dim, int  Matrix_A, int8 Matrix_B, float  Matrix_C, int Operands);
; float2 __spirv_SubgroupMatrixMultiplyAccumulateINTEL(int K_Dim, int2 Matrix_A, int8 Matrix_B, float2 Matrix_C, int Operands);
; float4 __spirv_SubgroupMatrixMultiplyAccumulateINTEL(int K_Dim, int4 Matrix_A, int8 Matrix_B, float4 Matrix_C, int Operands);
; float8 __spirv_SubgroupMatrixMultiplyAccumulateINTEL(int K_Dim, int8 Matrix_A, int8 Matrix_B, float8 Matrix_C, int Operands);
; 
; int  __spirv_SubgroupMatrixMultiplyAccumulateINTEL(int K_Dim, short  Matrix_A, int8 Matrix_B, int  Matrix_C, int Operands);
; int2 __spirv_SubgroupMatrixMultiplyAccumulateINTEL(int K_Dim, short2 Matrix_A, int8 Matrix_B, int2 Matrix_C, int Operands);
; int4 __spirv_SubgroupMatrixMultiplyAccumulateINTEL(int K_Dim, short4 Matrix_A, int8 Matrix_B, int4 Matrix_C, int Operands);
; int8 __spirv_SubgroupMatrixMultiplyAccumulateINTEL(int K_Dim, short8 Matrix_A, int8 Matrix_B, int8 Matrix_C, int Operands);
; 
; float  __spirv_SubgroupMatrixMultiplyAccumulateINTEL(int K_Dim, short  Matrix_A, int8 Matrix_B, float  Matrix_C, int Operands);
; float2 __spirv_SubgroupMatrixMultiplyAccumulateINTEL(int K_Dim, short2 Matrix_A, int8 Matrix_B, float2 Matrix_C, int Operands);
; float4 __spirv_SubgroupMatrixMultiplyAccumulateINTEL(int K_Dim, short4 Matrix_A, int8 Matrix_B, float4 Matrix_C, int Operands);
; float8 __spirv_SubgroupMatrixMultiplyAccumulateINTEL(int K_Dim, short8 Matrix_A, int8 Matrix_B, float8 Matrix_C, int Operands);
; 
; half  __spirv_SubgroupMatrixMultiplyAccumulateINTEL(int K_Dim, short  Matrix_A, int8 Matrix_B, half  Matrix_C, int Operands);
; half2 __spirv_SubgroupMatrixMultiplyAccumulateINTEL(int K_Dim, short2 Matrix_A, int8 Matrix_B, half2 Matrix_C, int Operands);
; half4 __spirv_SubgroupMatrixMultiplyAccumulateINTEL(int K_Dim, short4 Matrix_A, int8 Matrix_B, half4 Matrix_C, int Operands);
; half8 __spirv_SubgroupMatrixMultiplyAccumulateINTEL(int K_Dim, short8 Matrix_A, int8 Matrix_B, half8 Matrix_C, int Operands);
; 
; short  __spirv_SubgroupMatrixMultiplyAccumulateINTEL(int K_Dim, short  Matrix_A, int8 Matrix_B, short  Matrix_C, int Operands);
; short2 __spirv_SubgroupMatrixMultiplyAccumulateINTEL(int K_Dim, short2 Matrix_A, int8 Matrix_B, short2 Matrix_C, int Operands);
; short4 __spirv_SubgroupMatrixMultiplyAccumulateINTEL(int K_Dim, short4 Matrix_A, int8 Matrix_B, short4 Matrix_C, int Operands);
; short8 __spirv_SubgroupMatrixMultiplyAccumulateINTEL(int K_Dim, short8 Matrix_A, int8 Matrix_B, short8 Matrix_C, int Operands);
; 
; float  __spirv_SubgroupMatrixMultiplyAccumulateINTEL(int K_Dim, float  Matrix_A, float8 Matrix_B, float  Matrix_C, int Operands);
; float2 __spirv_SubgroupMatrixMultiplyAccumulateINTEL(int K_Dim, float2 Matrix_A, float8 Matrix_B, float2 Matrix_C, int Operands);
; float4 __spirv_SubgroupMatrixMultiplyAccumulateINTEL(int K_Dim, float4 Matrix_A, float8 Matrix_B, float4 Matrix_C, int Operands);
; float8 __spirv_SubgroupMatrixMultiplyAccumulateINTEL(int K_Dim, float8 Matrix_A, float8 Matrix_B, float8 Matrix_C, int Operands);
; 
; // no operands
; float4 __spirv_SubgroupMatrixMultiplyAccumulateINTEL(int K_Dim, short4 Matrix_A, int8 Matrix_B, float4 Matrix_C);
; 
; void foo(int iM, int2 iM2, int4 iM4, int8 iM8,
;          short sM, short2 sM2, short4 sM4, short8 sM8,
;          float fM, float2 fM2, float4 fM4, float8 fM8,
;          half hM, half2 hM2, half4 hM4, half8 hM8) {
;     const int i = 42;
;     int D = __spirv_SubgroupMatrixMultiplyAccumulateINTEL(i, iM, iM8, iM, 0xA);
;     int2 D2 = __spirv_SubgroupMatrixMultiplyAccumulateINTEL(i, iM2, iM8, iM2, 0xA);
;     int4 D4 = __spirv_SubgroupMatrixMultiplyAccumulateINTEL(i, iM4, iM8, iM4, 0xA);
;     int8 D8 = __spirv_SubgroupMatrixMultiplyAccumulateINTEL(i, iM8, iM8, iM8, 0xA);
; 
;     float fD = __spirv_SubgroupMatrixMultiplyAccumulateINTEL(i, iM, iM8, fM, 0xA);
;     float2 fD2 = __spirv_SubgroupMatrixMultiplyAccumulateINTEL(i, iM2, iM8, fM2, 0xA);
;     float4 fD4 = __spirv_SubgroupMatrixMultiplyAccumulateINTEL(i, iM4, iM8, fM4, 0xA);
;     float8 fD8 = __spirv_SubgroupMatrixMultiplyAccumulateINTEL(i, iM8, iM8, fM8, 0xA);
; 
;     int sD = __spirv_SubgroupMatrixMultiplyAccumulateINTEL(i, sM, iM8, iM, 0xA);
;     int2 sD2 = __spirv_SubgroupMatrixMultiplyAccumulateINTEL(i, sM2, iM8, iM2, 0xA);
;     int4 sD4 = __spirv_SubgroupMatrixMultiplyAccumulateINTEL(i, sM4, iM8, iM4, 0xA);
;     int8 sD8 = __spirv_SubgroupMatrixMultiplyAccumulateINTEL(i, sM8, iM8, iM8, 0xA);
; 
;     float sfD = __spirv_SubgroupMatrixMultiplyAccumulateINTEL(i, sM, iM8, fM, 0xA);
;     float2 sfD2 = __spirv_SubgroupMatrixMultiplyAccumulateINTEL(i, sM2, iM8, fM2, 0xA);
;     float4 sfD4 = __spirv_SubgroupMatrixMultiplyAccumulateINTEL(i, sM4, iM8, fM4, 0xA);
;     float8 sfD8 = __spirv_SubgroupMatrixMultiplyAccumulateINTEL(i, sM8, iM8, fM8, 0xA);
; 
;     half hD = __spirv_SubgroupMatrixMultiplyAccumulateINTEL(i, sM, iM8, hM, 0xA);
;     half2 hD2 = __spirv_SubgroupMatrixMultiplyAccumulateINTEL(i, sM2, iM8, hM2, 0xA);
;     half4 hD4 = __spirv_SubgroupMatrixMultiplyAccumulateINTEL(i, sM4, iM8, hM4, 0xA);
;     half8 hD8 = __spirv_SubgroupMatrixMultiplyAccumulateINTEL(i, sM8, iM8, hM8, 0xA);
; 
;     short ssD = __spirv_SubgroupMatrixMultiplyAccumulateINTEL(i, sM, iM8, sM, 0xA);
;     short2 ssD2 = __spirv_SubgroupMatrixMultiplyAccumulateINTEL(i, sM2, iM8, sM2, 0xA);
;     short4 ssD4 = __spirv_SubgroupMatrixMultiplyAccumulateINTEL(i, sM4, iM8, sM4, 0xA);
;     short8 ssD8 = __spirv_SubgroupMatrixMultiplyAccumulateINTEL(i, sM8, iM8, sM8, 0xA);
; 
;     float ffD = __spirv_SubgroupMatrixMultiplyAccumulateINTEL(i, fM, fM8, fM, 0xA);
;     float2 ffD2 = __spirv_SubgroupMatrixMultiplyAccumulateINTEL(i, fM2, fM8, fM2, 0xA);
;     float4 ffD4 = __spirv_SubgroupMatrixMultiplyAccumulateINTEL(i, fM4, fM8, fM4, 0xA);
;     float8 ffD8 = __spirv_SubgroupMatrixMultiplyAccumulateINTEL(i, fM8, fM8, fM8, 0xA);
; 
;     float4 noOpD4 = __spirv_SubgroupMatrixMultiplyAccumulateINTEL(i, sM4, iM8, fM4);
; }
; clang -cc1 -cl-std=clc++2021 -triple spir64-unknown-unknown -emit-llvm -finclude-default-header mma.cl -o tmp.ll

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv --spirv-ext=+SPV_INTEL_subgroup_matrix_multiply_accumulate
; RUN: llvm-spirv %t.spv -o %t.spt --to-text
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV

; RUN: llvm-spirv %t.spv -o %t.rev.bc -r --spirv-target-env=SPV-IR
; RUN: llvm-dis %t.rev.bc -o %t.rev.ll
; RUN: FileCheck < %t.rev.ll %s --check-prefix=CHECK-LLVM

; RUN: not llvm-spirv %t.bc 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR
; CHECK-ERROR: RequiresExtension: Feature requires the following SPIR-V extension:
; CHECK-ERROR-NEXT: SPV_INTEL_subgroup_matrix_multiply_accumulate

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-unknown"

; CHECK-SPIRV: Capability SubgroupMatrixMultiplyAccumulateINTEL 
; CHECK-SPIRV: Extension "SPV_INTEL_subgroup_matrix_multiply_accumulate" 
; CHECK-SPIRV-DAG: TypeInt [[#Int32Ty:]] 32 0
; CHECK-SPIRV-DAG: TypeInt [[#Int16Ty:]] 16 0
; CHECK-SPIRV-DAG: Constant [[#Int32Ty]] [[#Const42:]] 42
; CHECK-SPIRV-DAG: TypeVoid [[#VoidTy:]]
; CHECK-SPIRV-DAG: TypeVector [[#Vec2Int32Ty:]] [[#Int32Ty]] 2
; CHECK-SPIRV-DAG: TypeVector [[#Vec4Int32Ty:]] [[#Int32Ty]] 4
; CHECK-SPIRV-DAG: TypeVector [[#Vec8Int32Ty:]] [[#Int32Ty]] 8
; CHECK-SPIRV-DAG: TypeVector [[#Vec2Int16Ty:]] [[#Int16Ty]] 2
; CHECK-SPIRV-DAG: TypeVector [[#Vec4Int16Ty:]] [[#Int16Ty]] 4
; CHECK-SPIRV-DAG: TypeVector [[#Vec8Int16Ty:]] [[#Int16Ty]] 8
; CHECK-SPIRV-DAG: TypeFloat [[#FloatTy:]] 32
; CHECK-SPIRV-DAG: TypeVector [[#Vec2FloatTy:]] [[#FloatTy]] 2
; CHECK-SPIRV-DAG: TypeVector [[#Vec4FloatTy:]] [[#FloatTy]] 4
; CHECK-SPIRV-DAG: TypeVector [[#Vec8FloatTy:]] [[#FloatTy]] 8
; CHECK-SPIRV-DAG: TypeFloat [[#HalfTy:]] 16
; CHECK-SPIRV-DAG: TypeVector [[#Vec2HalfTy:]] [[#HalfTy]] 2
; CHECK-SPIRV-DAG: TypeVector [[#Vec4HalfTy:]] [[#HalfTy]] 4
; CHECK-SPIRV-DAG: TypeVector [[#Vec8HalfTy:]] [[#HalfTy]] 8
; CHECK-SPIRV: FunctionParameter [[#Int32Ty]] [[#iM:]]
; CHECK-SPIRV: FunctionParameter [[#Vec2Int32Ty]] [[#iM2:]] 
; CHECK-SPIRV: FunctionParameter [[#Vec4Int32Ty]] [[#iM4:]] 
; CHECK-SPIRV: FunctionParameter [[#Vec8Int32Ty]] [[#iM8:]] 
; CHECK-SPIRV: FunctionParameter [[#Int16Ty]] [[#sM:]] 
; CHECK-SPIRV: FunctionParameter [[#Vec2Int16Ty]] [[#sM2:]] 
; CHECK-SPIRV: FunctionParameter [[#Vec4Int16Ty]] [[#sM4:]] 
; CHECK-SPIRV: FunctionParameter [[#Vec8Int16Ty]] [[#sM8:]] 
; CHECK-SPIRV: FunctionParameter [[#FloatTy]] [[#fM:]] 
; CHECK-SPIRV: FunctionParameter [[#Vec2FloatTy]] [[#fM2:]] 
; CHECK-SPIRV: FunctionParameter [[#Vec4FloatTy]] [[#fM4:]] 
; CHECK-SPIRV: FunctionParameter [[#Vec8FloatTy]] [[#fM8:]] 
; CHECK-SPIRV: FunctionParameter [[#HalfTy]] [[#hM:]] 
; CHECK-SPIRV: FunctionParameter [[#Vec2HalfTy]] [[#hM2:]] 
; CHECK-SPIRV: FunctionParameter [[#Vec4HalfTy]] [[#hM4:]] 
; CHECK-SPIRV: FunctionParameter [[#Vec8HalfTy]] [[#hM8:]] 
; CHECK-SPIRV: SubgroupMatrixMultiplyAccumulateINTEL [[#Int32Ty]] [[#Id1:]] [[#Const42]] [[#iM]] [[#iM8]] [[#iM]] 10 
; CHECK-SPIRV: SubgroupMatrixMultiplyAccumulateINTEL [[#Vec2Int32Ty]] [[#Id1:]] [[#Const42]] [[#iM2]] [[#iM8]] [[#iM2]] 10 
; CHECK-SPIRV: SubgroupMatrixMultiplyAccumulateINTEL [[#Vec4Int32Ty]] [[#Id1:]] [[#Const42]] [[#iM4]] [[#iM8]] [[#iM4]] 10 
; CHECK-SPIRV: SubgroupMatrixMultiplyAccumulateINTEL [[#Vec8Int32Ty]] [[#Id1:]] [[#Const42]] [[#iM8]] [[#iM8]] [[#iM8]] 10 
; CHECK-SPIRV: SubgroupMatrixMultiplyAccumulateINTEL [[#FloatTy]] [[#Id1:]] [[#Const42]] [[#iM]] [[#iM8]] [[#fM]] 10 
; CHECK-SPIRV: SubgroupMatrixMultiplyAccumulateINTEL [[#Vec2FloatTy]] [[#Id1:]] [[#Const42]] [[#iM2]] [[#iM8]] [[#fM2]] 10 
; CHECK-SPIRV: SubgroupMatrixMultiplyAccumulateINTEL [[#Vec4FloatTy]] [[#Id1:]] [[#Const42]] [[#iM4]] [[#iM8]] [[#fM4]] 10 
; CHECK-SPIRV: SubgroupMatrixMultiplyAccumulateINTEL [[#Vec8FloatTy]] [[#Id1:]] [[#Const42]] [[#iM8]] [[#iM8]] [[#fM8]] 10 
; CHECK-SPIRV: SubgroupMatrixMultiplyAccumulateINTEL [[#Int32Ty]] [[#Id1:]] [[#Const42]] [[#sM]] [[#iM8]] [[#iM]] 10 
; CHECK-SPIRV: SubgroupMatrixMultiplyAccumulateINTEL [[#Vec2Int32Ty]] [[#Id1:]] [[#Const42]] [[#sM2]] [[#iM8]] [[#iM2]] 10 
; CHECK-SPIRV: SubgroupMatrixMultiplyAccumulateINTEL [[#Vec4Int32Ty]] [[#Id1:]] [[#Const42]] [[#sM4]] [[#iM8]] [[#iM4]] 10 
; CHECK-SPIRV: SubgroupMatrixMultiplyAccumulateINTEL [[#Vec8Int32Ty]] [[#Id1:]] [[#Const42]] [[#sM8]] [[#iM8]] [[#iM8]] 10 
; CHECK-SPIRV: SubgroupMatrixMultiplyAccumulateINTEL [[#FloatTy]] [[#Id1:]] [[#Const42]] [[#sM]] [[#iM8]] [[#fM]] 10 
; CHECK-SPIRV: SubgroupMatrixMultiplyAccumulateINTEL [[#Vec2FloatTy]] [[#Id1:]] [[#Const42]] [[#sM2]] [[#iM8]] [[#fM2]] 10 
; CHECK-SPIRV: SubgroupMatrixMultiplyAccumulateINTEL [[#Vec4FloatTy]] [[#Id1:]] [[#Const42]] [[#sM4]] [[#iM8]] [[#fM4]] 10 
; CHECK-SPIRV: SubgroupMatrixMultiplyAccumulateINTEL [[#Vec8FloatTy]] [[#Id1:]] [[#Const42]] [[#sM8]] [[#iM8]] [[#fM8]] 10 
; CHECK-SPIRV: SubgroupMatrixMultiplyAccumulateINTEL [[#HalfTy]] [[#Id1:]] [[#Const42]] [[#sM]] [[#iM8]] [[#hM]] 10 
; CHECK-SPIRV: SubgroupMatrixMultiplyAccumulateINTEL [[#Vec2HalfTy]] [[#Id1:]] [[#Const42]] [[#sM2]] [[#iM8]] [[#hM2]] 10 
; CHECK-SPIRV: SubgroupMatrixMultiplyAccumulateINTEL [[#Vec4HalfTy]] [[#Id1:]] [[#Const42]] [[#sM4]] [[#iM8]] [[#hM4]] 10 
; CHECK-SPIRV: SubgroupMatrixMultiplyAccumulateINTEL [[#Vec8HalfTy]] [[#Id1:]] [[#Const42]] [[#sM8]] [[#iM8]] [[#hM8]] 10 
; CHECK-SPIRV: SubgroupMatrixMultiplyAccumulateINTEL [[#Int16Ty]] [[#Id1:]] [[#Const42]] [[#sM]] [[#iM8]] [[#sM]] 10 
; CHECK-SPIRV: SubgroupMatrixMultiplyAccumulateINTEL [[#Vec2Int16Ty]] [[#Id1:]] [[#Const42]] [[#sM2]] [[#iM8]] [[#sM2]] 10 
; CHECK-SPIRV: SubgroupMatrixMultiplyAccumulateINTEL [[#Vec4Int16Ty]] [[#Id1:]] [[#Const42]] [[#sM4]] [[#iM8]] [[#sM4]] 10 
; CHECK-SPIRV: SubgroupMatrixMultiplyAccumulateINTEL [[#Vec8Int16Ty]] [[#Id1:]] [[#Const42]] [[#sM8]] [[#iM8]] [[#sM8]] 10 
; CHECK-SPIRV: SubgroupMatrixMultiplyAccumulateINTEL [[#FloatTy]] [[#Id1:]] [[#Const42]] [[#fM]] [[#fM8]] [[#fM]] 10 
; CHECK-SPIRV: SubgroupMatrixMultiplyAccumulateINTEL [[#Vec2FloatTy]] [[#Id1:]] [[#Const42]] [[#fM2]] [[#fM8]] [[#fM2]] 10 
; CHECK-SPIRV: SubgroupMatrixMultiplyAccumulateINTEL [[#Vec4FloatTy]] [[#Id1:]] [[#Const42]] [[#fM4]] [[#fM8]] [[#fM4]] 10 
; CHECK-SPIRV: SubgroupMatrixMultiplyAccumulateINTEL [[#Vec8FloatTy]] [[#Id1:]] [[#Const42]] [[#fM8]] [[#fM8]] [[#fM8]] 10 
; CHECK-SPIRV: SubgroupMatrixMultiplyAccumulateINTEL [[#Vec4FloatTy]] [[#Id1:]] [[#Const42]] [[#sM4]] [[#iM8]] [[#fM4]] 

; CHECK-LLVM: %{{.*}} = call spir_func i32 @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiiDv8_iii(i32 42, i32 %{{.*}}, <8 x i32> %{{.*}}, i32 %{{.*}}, i32 10)
; CHECK-LLVM: %{{.*}} = call spir_func <2 x i32> @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv2_iDv8_iS_i(i32 42, <2 x i32> %{{.*}}, <8 x i32> %{{.*}}, <2 x i32> %{{.*}}, i32 10)
; CHECK-LLVM: %{{.*}} = call spir_func <4 x i32> @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv4_iDv8_iS_i(i32 42, <4 x i32> %{{.*}}, <8 x i32> %{{.*}}, <4 x i32> %{{.*}}, i32 10)
; CHECK-LLVM: %{{.*}} = call spir_func <8 x i32> @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv8_iS_S_i(i32 42, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}, i32 10)
; CHECK-LLVM: %{{.*}} = call spir_func float @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiiDv8_ifi(i32 42, i32 %{{.*}}, <8 x i32> %{{.*}}, float %{{.*}}, i32 10)
; CHECK-LLVM: %{{.*}} = call spir_func <2 x float> @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv2_iDv8_iDv2_fi(i32 42, <2 x i32> %{{.*}}, <8 x i32> %{{.*}}, <2 x float> %{{.*}}, i32 10)
; CHECK-LLVM: %{{.*}} = call spir_func <4 x float> @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv4_iDv8_iDv4_fi(i32 42, <4 x i32> %{{.*}}, <8 x i32> %{{.*}}, <4 x float> %{{.*}}, i32 10)
; CHECK-LLVM: %{{.*}} = call spir_func <8 x float> @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv8_iS_Dv8_fi(i32 42, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}, <8 x float> %{{.*}}, i32 10)
; CHECK-LLVM: %{{.*}} = call spir_func i32 @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELisDv8_iii(i32 42, i16  %{{.*}}, <8 x i32> %{{.*}}, i32 %{{.*}}, i32 10)
; CHECK-LLVM: %{{.*}} = call spir_func <2 x i32> @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv2_sDv8_iDv2_ii(i32 42, <2 x i16> %{{.*}}, <8 x i32> %{{.*}}, <2 x i32> %{{.*}}, i32 10)
; CHECK-LLVM: %{{.*}} = call spir_func <4 x i32> @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv4_sDv8_iDv4_ii(i32 42, <4 x i16> %{{.*}}, <8 x i32> %{{.*}}, <4 x i32> %{{.*}}, i32 10)
; CHECK-LLVM: %{{.*}} = call spir_func <8 x i32> @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv8_sDv8_iS0_i(i32 42, <8 x i16> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}, i32 10)
; CHECK-LLVM: %{{.*}} = call spir_func float @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELisDv8_ifi(i32 42, i16  %{{.*}}, <8 x i32> %{{.*}}, float %{{.*}}, i32 10)
; CHECK-LLVM: %{{.*}} = call spir_func <2 x float> @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv2_sDv8_iDv2_fi(i32 42, <2 x i16> %{{.*}}, <8 x i32> %{{.*}}, <2 x float> %{{.*}}, i32 10)
; CHECK-LLVM: %{{.*}} = call spir_func <4 x float> @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv4_sDv8_iDv4_fi(i32 42, <4 x i16> %{{.*}}, <8 x i32> %{{.*}}, <4 x float> %{{.*}}, i32 10)
; CHECK-LLVM: %{{.*}} = call spir_func <8 x float> @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv8_sDv8_iDv8_fi(i32 42, <8 x i16> %{{.*}}, <8 x i32> %{{.*}}, <8 x float> %{{.*}}, i32 10)
; CHECK-LLVM: %{{.*}} = call spir_func half @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELisDv8_iDhi(i32 42, i16  %{{.*}}, <8 x i32> %{{.*}}, half %{{.*}}, i32 10)
; CHECK-LLVM: %{{.*}} = call spir_func <2 x half> @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv2_sDv8_iDv2_Dhi(i32 42, <2 x i16> %{{.*}}, <8 x i32> %{{.*}}, <2 x half> %{{.*}}, i32 10)
; CHECK-LLVM: %{{.*}} = call spir_func <4 x half> @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv4_sDv8_iDv4_Dhi(i32 42, <4 x i16> %{{.*}}, <8 x i32> %{{.*}}, <4 x half> %{{.*}}, i32 10)
; CHECK-LLVM: %{{.*}} = call spir_func <8 x half> @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv8_sDv8_iDv8_Dhi(i32 42, <8 x i16> %{{.*}}, <8 x i32> %{{.*}}, <8 x half> %{{.*}}, i32 10)
; CHECK-LLVM: %{{.*}} = call spir_func  i16 @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELisDv8_isi(i32 42, i16  %{{.*}}, <8 x i32> %{{.*}}, i16  %{{.*}}, i32 10)
; CHECK-LLVM: %{{.*}} = call spir_func <2 x i16> @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv2_sDv8_iS_i(i32 42, <2 x i16> %{{.*}}, <8 x i32> %{{.*}}, <2 x i16> %{{.*}}, i32 10)
; CHECK-LLVM: %{{.*}} = call spir_func <4 x i16> @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv4_sDv8_iS_i(i32 42, <4 x i16> %{{.*}}, <8 x i32> %{{.*}}, <4 x i16> %{{.*}}, i32 10)
; CHECK-LLVM: %{{.*}} = call spir_func <8 x i16> @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv8_sDv8_iS_i(i32 42, <8 x i16> %{{.*}}, <8 x i32> %{{.*}}, <8 x i16> %{{.*}}, i32 10)
; CHECK-LLVM: %{{.*}} = call spir_func float @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELifDv8_ffi(i32 42, float %{{.*}}, <8 x float> %{{.*}}, float %{{.*}}, i32 10)
; CHECK-LLVM: %{{.*}} = call spir_func <2 x float> @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv2_fDv8_fS_i(i32 42, <2 x float> %{{.*}}, <8 x float> %{{.*}}, <2 x float> %{{.*}}, i32 10)
; CHECK-LLVM: %{{.*}} = call spir_func <4 x float> @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv4_fDv8_fS_i(i32 42, <4 x float> %{{.*}}, <8 x float> %{{.*}}, <4 x float> %{{.*}}, i32 10)
; CHECK-LLVM: %{{.*}} = call spir_func <8 x float> @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv8_fS_S_i(i32 42, <8 x float> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}, i32 10)
; CHECK-LLVM: %{{.*}} = call spir_func <4 x float> @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv4_sDv8_iDv4_f(i32 42, <4 x i16> %{{.*}}, <8 x i32> %{{.*}}, <4 x float> %{{.*}})

define spir_func void @foo(i32 %iM, <2 x i32> %iM2, <4 x i32> %iM4, <8 x i32> %iM8,
                           i16 signext %sM, <2 x i16> %sM2, <4 x i16> %sM4, <8 x i16> %sM8,
                           float %fM, <2 x float> %fM2, <4 x float> %fM4, <8 x float> %fM8,
                           half %hM, <2 x half> %hM2, <4 x half> %hM4, <8 x half> %hM8) {
entry:
  %call = call spir_func i32 @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiiDv8_iii(i32 42, i32 %iM, <8 x i32> %iM8, i32 %iM, i32 10)
  %call1 = call spir_func <2 x i32> @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv2_iDv8_iS_i(i32 42, <2 x i32> %iM2, <8 x i32> %iM8, <2 x i32> %iM2, i32 10)
  %call2 = call spir_func <4 x i32> @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv4_iDv8_iS_i(i32 42, <4 x i32> %iM4, <8 x i32> %iM8, <4 x i32> %iM4, i32 10)
  %call3 = call spir_func <8 x i32> @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv8_iS_S_i(i32 42, <8 x i32> %iM8, <8 x i32> %iM8, <8 x i32> %iM8, i32 10)
  %call4 = call spir_func float @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiiDv8_ifi(i32 42, i32 %iM, <8 x i32> %iM8, float %fM, i32 10)
  %call5 = call spir_func <2 x float> @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv2_iDv8_iDv2_fi(i32 42, <2 x i32> %iM2, <8 x i32> %iM8, <2 x float> %fM2, i32 10)
  %call6 = call spir_func <4 x float> @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv4_iDv8_iDv4_fi(i32 42, <4 x i32> %iM4, <8 x i32> %iM8, <4 x float> %fM4, i32 10)
  %call7 = call spir_func <8 x float> @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv8_iS_Dv8_fi(i32 42, <8 x i32> %iM8, <8 x i32> %iM8, <8 x float> %fM8, i32 10)
  %call8 = call spir_func i32 @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELisDv8_iii(i32 42, i16 signext %sM, <8 x i32> %iM8, i32 %iM, i32 10)
  %call9 = call spir_func <2 x i32> @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv2_sDv8_iDv2_ii(i32 42, <2 x i16> %sM2, <8 x i32> %iM8, <2 x i32> %iM2, i32 10)
  %call10 = call spir_func <4 x i32> @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv4_sDv8_iDv4_ii(i32 42, <4 x i16> %sM4, <8 x i32> %iM8, <4 x i32> %iM4, i32 10)
  %call11 = call spir_func <8 x i32> @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv8_sDv8_iS0_i(i32 42, <8 x i16> %sM8, <8 x i32> %iM8, <8 x i32> %iM8, i32 10)
  %call12 = call spir_func float @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELisDv8_ifi(i32 42, i16 signext %sM, <8 x i32> %iM8, float %fM, i32 10)
  %call13 = call spir_func <2 x float> @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv2_sDv8_iDv2_fi(i32 42, <2 x i16> %sM2, <8 x i32> %iM8, <2 x float> %fM2, i32 10)
  %call14 = call spir_func <4 x float> @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv4_sDv8_iDv4_fi(i32 42, <4 x i16> %sM4, <8 x i32> %iM8, <4 x float> %fM4, i32 10)
  %call15 = call spir_func <8 x float> @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv8_sDv8_iDv8_fi(i32 42, <8 x i16> %sM8, <8 x i32> %iM8, <8 x float> %fM8, i32 10)
  %call16 = call spir_func half @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELisDv8_iDhi(i32 42, i16 signext %sM, <8 x i32> %iM8, half %hM, i32 10)
  %call17 = call spir_func <2 x half> @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv2_sDv8_iDv2_Dhi(i32 42, <2 x i16> %sM2, <8 x i32> %iM8, <2 x half> %hM2, i32 10)
  %call18 = call spir_func <4 x half> @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv4_sDv8_iDv4_Dhi(i32 42, <4 x i16> %sM4, <8 x i32> %iM8, <4 x half> %hM4, i32 10)
  %call19 = call spir_func <8 x half> @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv8_sDv8_iDv8_Dhi(i32 42, <8 x i16> %sM8, <8 x i32> %iM8, <8 x half> %hM8, i32 10)
  %call20 = call spir_func signext i16 @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELisDv8_isi(i32 42, i16 signext %sM, <8 x i32> %iM8, i16 signext %sM, i32 10)
  %call21 = call spir_func <2 x i16> @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv2_sDv8_iS_i(i32 42, <2 x i16> %sM2, <8 x i32> %iM8, <2 x i16> %sM2, i32 10)
  %call22 = call spir_func <4 x i16> @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv4_sDv8_iS_i(i32 42, <4 x i16> %sM4, <8 x i32> %iM8, <4 x i16> %sM4, i32 10)
  %call23 = call spir_func <8 x i16> @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv8_sDv8_iS_i(i32 42, <8 x i16> %sM8, <8 x i32> %iM8, <8 x i16> %sM8, i32 10)
  %call24 = call spir_func float @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELifDv8_ffi(i32 42, float %fM, <8 x float> %fM8, float %fM, i32 10)
  %call25 = call spir_func <2 x float> @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv2_fDv8_fS_i(i32 42, <2 x float> %fM2, <8 x float> %fM8, <2 x float> %fM2, i32 10)
  %call26 = call spir_func <4 x float> @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv4_fDv8_fS_i(i32 42, <4 x float> %fM4, <8 x float> %fM8, <4 x float> %fM4, i32 10)
  %call27 = call spir_func <8 x float> @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv8_fS_S_i(i32 42, <8 x float> %fM8, <8 x float> %fM8, <8 x float> %fM8, i32 10)
  %call28 = call spir_func <4 x float> @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv4_sDv8_iDv4_f(i32 42, <4 x i16> %sM4, <8 x i32> %iM8, <4 x float> %fM4)
  ret void
}

declare spir_func i32 @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiiDv8_iii(i32, i32, <8 x i32>, i32, i32)
declare spir_func <2 x i32> @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv2_iDv8_iS_i(i32, <2 x i32>, <8 x i32>, <2 x i32>, i32)
declare spir_func <4 x i32> @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv4_iDv8_iS_i(i32, <4 x i32>, <8 x i32>, <4 x i32>, i32)
declare spir_func <8 x i32> @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv8_iS_S_i(i32, <8 x i32>, <8 x i32>, <8 x i32>, i32)
declare spir_func float @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiiDv8_ifi(i32, i32, <8 x i32>, float, i32)
declare spir_func <2 x float> @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv2_iDv8_iDv2_fi(i32, <2 x i32>, <8 x i32>, <2 x float>, i32)
declare spir_func <4 x float> @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv4_iDv8_iDv4_fi(i32, <4 x i32>, <8 x i32>, <4 x float>, i32)
declare spir_func <8 x float> @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv8_iS_Dv8_fi(i32, <8 x i32>, <8 x i32>, <8 x float>, i32)
declare spir_func i32 @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELisDv8_iii(i32, i16 signext, <8 x i32>, i32, i32)
declare spir_func <2 x i32> @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv2_sDv8_iDv2_ii(i32, <2 x i16>, <8 x i32>, <2 x i32>, i32)
declare spir_func <4 x i32> @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv4_sDv8_iDv4_ii(i32, <4 x i16>, <8 x i32>, <4 x i32>, i32)
declare spir_func <8 x i32> @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv8_sDv8_iS0_i(i32, <8 x i16>, <8 x i32>, <8 x i32>, i32)
declare spir_func float @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELisDv8_ifi(i32, i16 signext, <8 x i32>, float, i32)
declare spir_func <2 x float> @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv2_sDv8_iDv2_fi(i32, <2 x i16>, <8 x i32>, <2 x float>, i32)
declare spir_func <4 x float> @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv4_sDv8_iDv4_fi(i32, <4 x i16>, <8 x i32>, <4 x float>, i32)
declare spir_func <8 x float> @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv8_sDv8_iDv8_fi(i32, <8 x i16>, <8 x i32>, <8 x float>, i32)
declare spir_func half @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELisDv8_iDhi(i32, i16 signext, <8 x i32>, half, i32)
declare spir_func <2 x half> @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv2_sDv8_iDv2_Dhi(i32, <2 x i16>, <8 x i32>, <2 x half>, i32)
declare spir_func <4 x half> @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv4_sDv8_iDv4_Dhi(i32, <4 x i16>, <8 x i32>, <4 x half>, i32)
declare spir_func <8 x half> @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv8_sDv8_iDv8_Dhi(i32, <8 x i16>, <8 x i32>, <8 x half>, i32)
declare spir_func signext i16 @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELisDv8_isi(i32, i16 signext, <8 x i32>, i16 signext, i32)
declare spir_func <2 x i16> @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv2_sDv8_iS_i(i32, <2 x i16>, <8 x i32>, <2 x i16>, i32)
declare spir_func <4 x i16> @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv4_sDv8_iS_i(i32, <4 x i16>, <8 x i32>, <4 x i16>, i32)
declare spir_func <8 x i16> @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv8_sDv8_iS_i(i32, <8 x i16>, <8 x i32>, <8 x i16>, i32)
declare spir_func float @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELifDv8_ffi(i32, float, <8 x float>, float, i32)
declare spir_func <2 x float> @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv2_fDv8_fS_i(i32, <2 x float>, <8 x float>, <2 x float>, i32)
declare spir_func <4 x float> @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv4_fDv8_fS_i(i32, <4 x float>, <8 x float>, <4 x float>, i32)
declare spir_func <8 x float> @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv8_fS_S_i(i32, <8 x float>, <8 x float>, <8 x float>, i32)
declare spir_func <4 x float> @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv4_sDv8_iDv4_f(i32, <4 x i16>, <8 x i32>, <4 x float>)

!opencl.spir.version = !{!0}
!spirv.Source = !{!1}
!llvm.ident = !{!2}

!0 = !{i32 1, i32 0}
!1 = !{i32 4, i32 100000}
!2 = !{!"clang version 17.0.0"}
