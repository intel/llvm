; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_KHR_bfloat16 --spirv-ext=+SPV_INTEL_bfloat16_arithmetic -o %t.spv
; RUN: llvm-spirv %t.spv -to-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV

; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc -o - | FileCheck %s --check-prefix=CHECK-LLVM

; RUN: not llvm-spirv %t.bc --spirv-ext=+SPV_KHR_bfloat16 2>&1 >/dev/null | FileCheck %s --check-prefix=CHECK-ERROR
; RUN: not llvm-spirv %t.bc --spirv-ext=+SPV_INTEL_bfloat16_arithmetic 2>&1 >/dev/null | FileCheck %s --check-prefix=CHECK-ERROR
; CHECK-ERROR: RequiresExtension: Feature requires the following SPIR-V extension:

source_filename = "bfloat16.cpp"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spirv64-unknown-unknown"

; CHECK-SPIRV: Capability BFloat16TypeKHR
; CHECK-SPIRV: Capability BFloat16ArithmeticINTEL
; CHECK-SPIRV: Extension "SPV_INTEL_bfloat16_arithmetic"
; CHECK-SPIRV: Extension "SPV_KHR_bfloat16"
; CHECK-SPIRV: 4 TypeFloat [[BFLOAT:[0-9]+]] 16 0
; CHECK-SPIRV: 5 Function [[#]] [[#]] [[#]] [[#]]
; CHECK-SPIRV: 7 Phi [[BFLOAT]] [[#]] [[#]] [[#]] [[#]] [[#]]
; CHECK-SPIRV: 2 ReturnValue [[#]]
; CHECK-SPIRV: 4 Variable [[#]] [[ADDR1:[0-9]+]]
; CHECK-SPIRV: 4 Variable [[#]] [[ADDR2:[0-9]+]]
; CHECK-SPIRV: 4 Variable [[#]] [[ADDR3:[0-9]+]]
; CHECK-SPIRV: 6 Load [[BFLOAT]] [[DATA1:[0-9]+]] [[ADDR1]]
; CHECK-SPIRV: 6 Load [[BFLOAT]] [[DATA2:[0-9]+]] [[ADDR2]]
; CHECK-SPIRV: 6 Load [[BFLOAT]] [[DATA3:[0-9]+]] [[ADDR3]]
;                Undef
;                Constant
;                ConstantComposite
;                ConstantNull
;                SpecConstant
;                SpecConstantComposite
; CHECK-SPIRV: 4 ConvertFToU [[#]] [[#]] [[DATA1]]
; CHECK-SPIRV: 4 ConvertFToS [[#]] [[#]] [[DATA1]]
; CHECK-SPIRV: 4 ConvertSToF [[BFLOAT]] [[#]] [[#]]
; CHECK-SPIRV: 4 ConvertUToF [[BFLOAT]] [[#]] [[#]]
;                Bitcast
; CHECK-SPIRV: 4 FNegate [[BFLOAT]] [[#]] [[DATA1]]
; CHECK-SPIRV: 5 FAdd [[BFLOAT]] [[#]] [[DATA1]] [[DATA2]]
; CHECK-SPIRV: 5 FSub [[BFLOAT]] [[#]] [[DATA1]] [[DATA2]]
; CHECK-SPIRV: 5 FMul [[BFLOAT]] [[#]] [[DATA1]] [[DATA2]]
; CHECK-SPIRV: 5 FDiv [[BFLOAT]] [[#]] [[DATA1]] [[DATA2]]
; CHECK-SPIRV: 5 FRem [[BFLOAT]] [[#]] [[DATA1]] [[DATA2]]
;                FMod
;                VectorTimesScalar
; CHECK-SPIRV: 4 IsNan [[#]] [[#]] [[DATA1]]
; CHECK-SPIRV: 4 IsInf [[#]] [[#]] [[DATA1]]
;                IsFinite
; CHECK-SPIRV: 4 IsNormal [[#]] [[#]] [[DATA1]]
; CHECK-SPIRV: 5 Ordered [[#]] [[#]] [[DATA1]] [[DATA2]]
; CHECK-SPIRV: 5 Unordered [[#]] [[#]] [[DATA1]] [[DATA2]]
; CHECK-SPIRV: 6 Select [[BFLOAT]] [[#]] [[#]] [[DATA1]] [[DATA2]]
; CHECK-SPIRV: 5 FOrdEqual [[#]] [[#]] [[DATA1]] [[DATA2]]
; CHECK-SPIRV: 5 FUnordEqual [[#]] [[#]] [[DATA1]] [[DATA2]]
; CHECK-SPIRV: 5 FOrdNotEqual [[#]] [[#]] [[DATA1]] [[DATA2]]
; CHECK-SPIRV: 5 FUnordNotEqual [[#]] [[#]] [[DATA1]] [[DATA2]]
; CHECK-SPIRV: 5 FOrdLessThan [[#]] [[#]] [[DATA1]] [[DATA2]]
; CHECK-SPIRV: 5 FUnordLessThan [[#]] [[#]] [[DATA1]] [[DATA2]]
; CHECK-SPIRV: 5 FOrdGreaterThan [[#]] [[#]] [[DATA1]] [[DATA2]]
; CHECK-SPIRV: 5 FUnordGreaterThan [[#]] [[#]] [[DATA1]] [[DATA2]]
; CHECK-SPIRV: 5 FOrdLessThanEqual [[#]] [[#]] [[DATA1]] [[DATA2]]
; CHECK-SPIRV: 5 FUnordLessThanEqual [[#]] [[#]] [[DATA1]] [[DATA2]]
; CHECK-SPIRV: 5 FOrdGreaterThanEqual [[#]] [[#]] [[DATA1]] [[DATA2]]
; CHECK-SPIRV: 5 FUnordGreaterThanEqual [[#]] [[#]] [[DATA1]] [[DATA2]]
; CHECK-SPIRV: 6 ExtInst [[BFLOAT]] [[#]] [[#]] fabs [[DATA1]]
; CHECK-SPIRV: 8 ExtInst [[BFLOAT]] [[#]] [[#]] fclamp [[DATA1]] [[DATA2]] [[DATA3]]
; CHECK-SPIRV: 8 ExtInst [[BFLOAT]] [[#]] [[#]] fma [[DATA1]] [[DATA2]] [[DATA3]]
; CHECK-SPIRV: 7 ExtInst [[BFLOAT]] [[#]] [[#]] fmax [[DATA1]] [[DATA2]]
; CHECK-SPIRV: 7 ExtInst [[BFLOAT]] [[#]] [[#]] fmin [[DATA1]] [[DATA2]]
; CHECK-SPIRV: 8 ExtInst [[BFLOAT]] [[#]] [[#]] mad [[DATA1]] [[DATA2]] [[DATA3]]
; CHECK-SPIRV: 6 ExtInst [[BFLOAT]] [[#]] [[#]] nan [[DATA1]]
; CHECK-SPIRV: 6 ExtInst [[BFLOAT]] [[#]] [[#]] native_cos [[DATA1]]
; CHECK-SPIRV: 7 ExtInst [[BFLOAT]] [[#]] [[#]] native_divide [[DATA1]] [[DATA2]]
; CHECK-SPIRV: 6 ExtInst [[BFLOAT]] [[#]] [[#]] native_exp [[DATA1]]
; CHECK-SPIRV: 6 ExtInst [[BFLOAT]] [[#]] [[#]] native_exp10 [[DATA1]]
; CHECK-SPIRV: 6 ExtInst [[BFLOAT]] [[#]] [[#]] native_exp2 [[DATA1]]
; CHECK-SPIRV: 6 ExtInst [[BFLOAT]] [[#]] [[#]] native_log [[DATA1]]
; CHECK-SPIRV: 6 ExtInst [[BFLOAT]] [[#]] [[#]] native_log10 [[DATA1]]
; CHECK-SPIRV: 6 ExtInst [[BFLOAT]] [[#]] [[#]] native_log2 [[DATA1]]
; CHECK-SPIRV: 7 ExtInst [[BFLOAT]] [[#]] [[#]] native_powr [[DATA1]] [[DATA2]]
; CHECK-SPIRV: 6 ExtInst [[BFLOAT]] [[#]] [[#]] native_recip [[DATA1]]
; CHECK-SPIRV: 6 ExtInst [[BFLOAT]] [[#]] [[#]] native_rsqrt [[DATA1]]
; CHECK-SPIRV: 6 ExtInst [[BFLOAT]] [[#]] [[#]] native_sin [[DATA1]]
; CHECK-SPIRV: 6 ExtInst [[BFLOAT]] [[#]] [[#]] native_sqrt [[DATA1]]
; CHECK-SPIRV: 6 ExtInst [[BFLOAT]] [[#]] [[#]] native_tan [[DATA1]]

; CHECK-LLVM: define spir_func void @OpPhi(bfloat %data1, bfloat %data2)
; CHECK-LLVM: %OpPhi = phi bfloat [ %data1, %blockA ], [ %data2, %blockB ]
; CHECK-LLVM: ret bfloat %OpReturnValue
; CHECK-LLVM: [[ADDR1:[%a-z0-9]+]] = alloca bfloat
; CHECK-LLVM: [[ADDR2:[%a-z0-9]+]] = alloca bfloat
; CHECK-LLVM: [[ADDR3:[%a-z0-9]+]] = alloca bfloat
; CHECK-LLVM: [[DATA1:[%a-z0-9]+]] = load bfloat, ptr [[ADDR1]]
; CHECK-LLVM: [[DATA2:[%a-z0-9]+]] = load bfloat, ptr [[ADDR2]]
; CHECK-LLVM: [[DATA3:[%a-z0-9]+]] = load bfloat, ptr [[ADDR3]]
;             %OpUndef
;             %OpConstant
;             %OpConstantComposite
;             %OpConstantNull
;             %OpSpecConstant
;             %OpSpecConstantComposite
; CHECK-LLVM: %OpConvertFToU = fptoui bfloat [[DATA1]] to i32
; CHECK-LLVM: %OpConvertFToS = fptosi bfloat [[DATA1]] to i32
; CHECK-LLVM: %OpConvertSToF = sitofp i32 0 to bfloat
; CHECK-LLVM: %OpConvertUToF = uitofp i32 0 to bfloat
;             %OpBitcast
; CHECK-LLVM: %OpFNegate = fneg bfloat [[DATA1]]
; CHECK-LLVM: %OpFAdd = fadd bfloat [[DATA1]], [[DATA2]]
; CHECK-LLVM: %OpFSub = fsub bfloat [[DATA1]], [[DATA2]]
; CHECK-LLVM: %OpFMul = fmul bfloat [[DATA1]], [[DATA2]]
; CHECK-LLVM: %OpFDiv = fdiv bfloat [[DATA1]], [[DATA2]]
; CHECK-LLVM: %OpFRem = frem bfloat [[DATA1]], [[DATA2]]
;             %OpFMod
;             %OpVectorTimesScalar
; CHECK-LLVM: %[[#]] = call spir_func i32 @_Z5isnanDF16b(bfloat [[DATA1]])
; CHECK-LLVM: %[[#]] = call spir_func i32 @_Z5isinfDF16b(bfloat [[DATA1]])
;             %OpIsFinite
; CHECK-LLVM: %[[#]] = call spir_func i32 @_Z8isnormalDF16b(bfloat [[DATA1]])
; CHECK-LLVM: %OpOrdered = fcmp ord bfloat [[DATA1]], [[DATA2]]
; CHECK-LLVM: %OpUnordered = fcmp uno bfloat [[DATA1]], [[DATA2]]
; CHECK-LLVM: %OpSelect = select i1 true, bfloat [[DATA1]], bfloat [[DATA2]]
; CHECK-LLVM: %OpFOrdEqual = fcmp oeq bfloat [[DATA1]], [[DATA2]]
; CHECK-LLVM: %OpFUnordEqual = fcmp ueq bfloat [[DATA1]], [[DATA2]]
; CHECK-LLVM: %OpFOrdNotEqual = fcmp one bfloat [[DATA1]], [[DATA2]]
; CHECK-LLVM: %OpFUnordNotEqual = fcmp une bfloat [[DATA1]], [[DATA2]]
; CHECK-LLVM: %OpFOrdLessThan = fcmp olt bfloat [[DATA1]], [[DATA2]]
; CHECK-LLVM: %OpFUnordLessThan = fcmp ult bfloat [[DATA1]], [[DATA2]]
; CHECK-LLVM: %OpFOrdGreaterThan = fcmp ogt bfloat [[DATA1]], [[DATA2]]
; CHECK-LLVM: %OpFUnordGreaterThan = fcmp ugt bfloat [[DATA1]], [[DATA2]]
; CHECK-LLVM: %OpFOrdLessThanEqual = fcmp ole bfloat [[DATA1]], [[DATA2]]
; CHECK-LLVM: %OpFUnordLessThanEqual = fcmp ule bfloat [[DATA1]], [[DATA2]]
; CHECK-LLVM: %OpFOrdGreaterThanEqual = fcmp oge bfloat [[DATA1]], [[DATA2]]
; CHECK-LLVM: %OpFUnordGreaterThanEqual = fcmp uge bfloat [[DATA1]], [[DATA2]]
; CHECK-LLVM: %fabs = call spir_func bfloat @_Z4fabsDF16b(bfloat [[DATA1]])
; CHECK-LLVM: %fclamp = call spir_func bfloat @_Z5clampDF16bDF16bDF16b(bfloat [[DATA1]], bfloat [[DATA2]], bfloat [[DATA3]])
; CHECK-LLVM: %fma = call spir_func bfloat @_Z3fmaDF16bDF16bDF16b(bfloat [[DATA1]], bfloat [[DATA2]], bfloat [[DATA3]])
; CHECK-LLVM: %fmax = call spir_func bfloat @_Z4fmaxDF16bDF16b(bfloat [[DATA1]], bfloat [[DATA2]])
; CHECK-LLVM: %fmin = call spir_func bfloat @_Z4fminDF16bDF16b(bfloat [[DATA1]], bfloat [[DATA2]])
; CHECK-LLVM: %mad = call spir_func bfloat @_Z3madDF16bDF16bDF16b(bfloat [[DATA1]], bfloat [[DATA2]], bfloat [[DATA3]])
; CHECK-LLVM: %nan = call spir_func bfloat @_Z3nanDF16b(bfloat [[DATA1]])
; CHECK-LLVM: %native_cos = call spir_func bfloat @_Z10native_cosDF16b(bfloat [[DATA1]])
; CHECK-LLVM: %native_divide = call spir_func bfloat @_Z13native_divideDF16bDF16b(bfloat [[DATA1]], bfloat [[DATA2]])
; CHECK-LLVM: %native_exp = call spir_func bfloat @_Z10native_expDF16b(bfloat [[DATA1]])
; CHECK-LLVM: %native_exp10 = call spir_func bfloat @_Z12native_exp10DF16b(bfloat [[DATA1]])
; CHECK-LLVM: %native_exp2 = call spir_func bfloat @_Z11native_exp2DF16b(bfloat [[DATA1]])
; CHECK-LLVM: %native_log = call spir_func bfloat @_Z10native_logDF16b(bfloat [[DATA1]])
; CHECK-LLVM: %native_log10 = call spir_func bfloat @_Z12native_log10DF16b(bfloat [[DATA1]])
; CHECK-LLVM: %native_log2 = call spir_func bfloat @_Z11native_log2DF16b(bfloat [[DATA1]])
; CHECK-LLVM: %native_powr = call spir_func bfloat @_Z11native_powrDF16bDF16b(bfloat [[DATA1]], bfloat [[DATA2]])
; CHECK-LLVM: %native_recip = call spir_func bfloat @_Z12native_recipDF16b(bfloat [[DATA1]])
; CHECK-LLVM: %native_rsqrt = call spir_func bfloat @_Z12native_rsqrtDF16b(bfloat [[DATA1]])
; CHECK-LLVM: %native_sin = call spir_func bfloat @_Z10native_sinDF16b(bfloat [[DATA1]])
; CHECK-LLVM: %native_sqrt = call spir_func bfloat @_Z11native_sqrtDF16b(bfloat [[DATA1]])
; CHECK-LLVM: %native_tan = call spir_func bfloat @_Z10native_tanDF16b(bfloat [[DATA1]])

declare spir_func bfloat @_Z5clampDF16bDF16bDF16b(bfloat, bfloat, bfloat)
declare spir_func bfloat @_Z3nanDF16b(bfloat)
declare spir_func bfloat @_Z10native_cosDF16b(bfloat)
declare spir_func bfloat @_Z13native_divideDF16bDF16b(bfloat, bfloat)
declare spir_func bfloat @_Z10native_expDF16b(bfloat)
declare spir_func bfloat @_Z12native_exp10DF16b(bfloat)
declare spir_func bfloat @_Z11native_exp2DF16b(bfloat)
declare spir_func bfloat @_Z10native_logDF16b(bfloat)
declare spir_func bfloat @_Z12native_log10DF16b(bfloat)
declare spir_func bfloat @_Z11native_log2DF16b(bfloat)
declare spir_func bfloat @_Z11native_powrDF16bDF16b(bfloat, bfloat)
declare spir_func bfloat @_Z12native_recipDF16b(bfloat)
declare spir_func bfloat @_Z12native_rsqrtDF16b(bfloat)
declare spir_func bfloat @_Z10native_sinDF16b(bfloat)
declare spir_func bfloat @_Z11native_sqrtDF16b(bfloat)
declare spir_func bfloat @_Z10native_tanDF16b(bfloat)

define spir_func void @OpPhi(bfloat %data1, bfloat %data2) {
  br label %blockA
blockA:
  br label %phi
blockB:
  br label %phi
phi:
  %OpPhi = phi bfloat [ %data1, %blockA ], [ %data2, %blockB ]
  ret void
}

define spir_func bfloat @OpReturnValue(bfloat %OpReturnValue) {
  ret bfloat %OpReturnValue
}

define spir_kernel void @testMath() {
entry:
  %addr1 = alloca bfloat
  %addr2 = alloca bfloat
  %addr3 = alloca bfloat
  %data1 = load bfloat, ptr %addr1
  %data2 = load bfloat, ptr %addr2
  %data3 = load bfloat, ptr %addr3
  ; %OpUndef
  ; %OpConstant
  ; %OpConstantComposite
  ; %OpConstantNull
  ; %OpSpecConstant
  ; %OpSpecConstantComposite
  %OpConvertFToU = fptoui bfloat %data1 to i32
  %OpConvertFToS = fptosi bfloat %data1 to i32
  %OpConvertSToF = sitofp i32 0 to bfloat
  %OpConvertUToF = uitofp i32 0 to bfloat
  ; %OpBitcast
  %OpFNegate = fneg bfloat %data1
  %OpFAdd = fadd bfloat %data1, %data2
  %OpFSub = fsub bfloat %data1, %data2
  %OpFMul = fmul bfloat %data1, %data2
  %OpFDiv = fdiv bfloat %data1, %data2
  %OpFRem = frem bfloat %data1, %data2
  ; %OpFMod
  ; %OpVectorTimesScalar
  %OpIsNan = call i1 @llvm.is.fpclass.bfloat(bfloat %data1, i32 3)
  %OpIsInf = call i1 @llvm.is.fpclass.bfloat(bfloat %data1, i32 516)
  ; %OpIsFinite
  %OpIsNormal = call i1 @llvm.is.fpclass.bfloat(bfloat %data1, i32 264)
  %OpOrdered = fcmp ord bfloat %data1, %data2
  %OpUnordered = fcmp uno bfloat %data1, %data2
  %OpSelect = select i1 true, bfloat %data1, bfloat %data2
  %OpFOrdEqual = fcmp oeq bfloat %data1, %data2
  %OpFUnordEqual = fcmp ueq bfloat %data1, %data2
  %OpFOrdNotEqual = fcmp one bfloat %data1, %data2
  %OpFUnordNotEqual = fcmp une bfloat %data1, %data2
  %OpFOrdLessThan = fcmp olt bfloat %data1, %data2
  %OpFUnordLessThan = fcmp ult bfloat %data1, %data2
  %OpFOrdGreaterThan = fcmp ogt bfloat %data1, %data2
  %OpFUnordGreaterThan = fcmp ugt bfloat %data1, %data2
  %OpFOrdLessThanEqual = fcmp ole bfloat %data1, %data2
  %OpFUnordLessThanEqual = fcmp ule bfloat %data1, %data2
  %OpFOrdGreaterThanEqual = fcmp oge bfloat %data1, %data2
  %OpFUnordGreaterThanEqual = fcmp uge bfloat %data1, %data2
  %fabs = call bfloat @llvm.fabs.bfloat(bfloat %data1)
  %fclamp = call spir_func bfloat @_Z5clampDF16bDF16bDF16b(bfloat %data1, bfloat %data2, bfloat %data3)
  %fma = call bfloat @llvm.fma.bfloat(bfloat %data1, bfloat %data2, bfloat %data3)
  %fmax = call bfloat @llvm.maxnum.bfloat(bfloat %data1, bfloat %data2)
  %fmin = call bfloat @llvm.minnum.bfloat(bfloat %data1, bfloat %data2)
  %mad = call bfloat @llvm.fmuladd.bfloat(bfloat %data1, bfloat %data2, bfloat %data3)
  %nan = call spir_func bfloat @_Z3nanDF16b(bfloat %data1)
  %native_cos = call spir_func bfloat @_Z10native_cosDF16b(bfloat %data1)
  %native_divide = call spir_func bfloat @_Z13native_divideDF16bDF16b(bfloat %data1, bfloat %data2)
  %native_exp = call spir_func bfloat @_Z10native_expDF16b(bfloat %data1)
  %native_exp10 = call spir_func bfloat @_Z12native_exp10DF16b(bfloat %data1)
  %native_exp2 = call spir_func bfloat @_Z11native_exp2DF16b(bfloat %data1)
  %native_log = call spir_func bfloat @_Z10native_logDF16b(bfloat %data1)
  %native_log10 = call spir_func bfloat @_Z12native_log10DF16b(bfloat %data1)
  %native_log2 = call spir_func bfloat @_Z11native_log2DF16b(bfloat %data1)
  %native_powr = call spir_func bfloat @_Z11native_powrDF16bDF16b(bfloat %data1, bfloat %data2)
  %native_recip = call spir_func bfloat @_Z12native_recipDF16b(bfloat %data1)
  %native_rsqrt = call spir_func bfloat @_Z12native_rsqrtDF16b(bfloat %data1)
  %native_sin = call spir_func bfloat @_Z10native_sinDF16b(bfloat %data1)
  %native_sqrt = call spir_func bfloat @_Z11native_sqrtDF16b(bfloat %data1)
  %native_tan = call spir_func bfloat @_Z10native_tanDF16b(bfloat %data1)
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
