; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc

; Just check that reverse translation works
; RUN: llvm-dis < %t.rev.bc

;; Test for llvm.is.fpclass translation
;; it's a bit rewritten subset of is_fpclass.ll test from llvm.org

; CHECK-SPIRV-DAG: TypeInt [[#Int32Ty:]] 32 0
; CHECK-SPIRV-DAG: TypeInt [[#Int64Ty:]] 64 0
; CHECK-SPIRV-DAG: TypeInt [[#Int16Ty:]] 16 0
; CHECK-SPIRV-DAG: TypeBool [[#BoolTy:]]
; CHECK-SPIRV-DAG: TypeVector [[#VecBoolTy:]] [[#BoolTy]] 2
; CHECK-SPIRV-DAG: TypeVector [[#Int32VecTy:]] [[#Int32Ty]] 2
; CHECK-SPIRV-DAG: TypeVector [[#Int16VecTy:]] [[#Int16Ty]] 2
; CHECK-SPIRV-DAG: TypeFloat [[#DoubleTy:]] 64
; CHECK-SPIRV-DAG: Constant [[#Int32Ty]] [[#QNanBitConst:]] 2143289344
; CHECK-SPIRV-DAG: Constant [[#Int32Ty]] [[#MantissaConst:]] 8388607
; CHECK-SPIRV-DAG: Constant [[#Int32Ty]] [[#ZeroConst:]] 0
; CHECK-SPIRV-DAG: Constant [[#Int32Ty]] [[#MaskToClearSignBit:]] 2147483647
; CHECK-SPIRV-DAG: Constant [[#Int32Ty]] [[#NegatedZeroConst:]] 2147483648
; CHECK-SPIRV-DAG: Constant [[#Int64Ty]] [[#QNanBitConst64:]] 0 2146959360
; CHECK-SPIRV-DAG: Constant [[#Int64Ty]] [[#MantissaConst64:]] 4294967295 1048575
; CHECK-SPIRV-DAG: Constant [[#Int64Ty]] [[#ZeroConst64:]] 0 0
; CHECK-SPIRV-DAG: Constant [[#Int16Ty]] [[#QNanBitConst16:]] 32256
; CHECK-SPIRV-DAG: Constant [[#Int16Ty]] [[#MantissaConst16:]] 1023
; CHECK-SPIRV-DAG: Constant [[#DoubleTy]] [[#DoubleConst:]] 0 1072693248
; CHECK-SPIRV-DAG: ConstantComposite [[#Int16VecTy:]] [[#QNanBitConstVec16:]] [[#QNanBitConst16]] [[#QNanBitConst16]]
; CHECK-SPIRV-DAG: ConstantComposite [[#Int16VecTy:]] [[#MantissaConstVec16:]] [[#MantissaConst16]] [[#MantissaConst16]]
; CHECK-SPIRV-DAG: ConstantNull [[#Int16VecTy]] [[#ZeroConst16:]]
; CHECK-SPIRV-DAG: ConstantTrue [[#BoolTy]] [[#True:]]
; CHECK-SPIRV-DAG: ConstantFalse [[#BoolTy]] [[#False:]]
; CHECK-SPIRV-DAG: Name [[#NoMaskFunc:]] "test_class_no_mask_f32"
; CHECK-SPIRV-DAG: Name [[#FullMaskFunc:]] "test_class_full_mask_f32"
; CHECK-SPIRV-DAG: Name [[#NanFunc:]] "test_class_isnan_f32"
; CHECK-SPIRV-DAG: Name [[#VecNanFunc:]] "test_class_isnan_v2f32"
; CHECK-SPIRV-DAG: Name [[#SNanFunc:]] "test_class_issnan_f32"
; CHECK-SPIRV-DAG: Name [[#QNanFunc:]] "test_class_isqnan_f32"
; CHECK-SPIRV-DAG: Name [[#InfFunc:]] "test_class_is_inf_f32"
; CHECK-SPIRV-DAG: Name [[#InfVecFunc:]] "test_class_is_inf_v2f32"
; CHECK-SPIRV-DAG: Name [[#PosInfFunc:]] "test_class_is_pinf_f32"
; CHECK-SPIRV-DAG: Name [[#PosInfVecFunc:]] "test_class_is_pinf_v2f32"
; CHECK-SPIRV-DAG: Name [[#NegInfFunc:]] "test_class_is_ninf_f32"
; CHECK-SPIRV-DAG: Name [[#NegInfVecFunc:]] "test_class_is_ninf_v2f32"
; CHECK-SPIRV-DAG: Name [[#NormFunc:]] "test_class_is_normal"
; CHECK-SPIRV-DAG: Name [[#PosNormFunc:]] "test_constant_class_pnormal"
; CHECK-SPIRV-DAG: Name [[#NegNormFunc:]] "test_constant_class_nnormal"
; CHECK-SPIRV-DAG: Name [[#SubnormFunc:]] "test_class_subnormal"
; CHECK-SPIRV-DAG: Name [[#PosSubnormFunc:]] "test_class_possubnormal"
; CHECK-SPIRV-DAG: Name [[#NegSubnormFunc:]] "test_class_negsubnormal"
; CHECK-SPIRV-DAG: Name [[#ZeroFunc:]] "test_class_zero"
; CHECK-SPIRV-DAG: Name [[#PosZeroFunc:]] "test_class_poszero"
; CHECK-SPIRV-DAG: Name [[#NegZeroFunc:]] "test_class_negzero"
; CHECK-SPIRV-DAG: Name [[#NegInfOrNanFunc:]] "test_class_is_ninf_or_nan_f32"
; CHECK-SPIRV-DAG: Name [[#ComplexFunc1:]] "test_class_neginf_posnormal_negsubnormal_poszero_snan_f64"
; CHECK-SPIRV-DAG: Name [[#ComplexFunc2:]] "test_class_neginf_posnormal_negsubnormal_poszero_snan_v2f16"
; CHECK-SPIRV-DAG: Name [[#NotNanFunc:]] "test_class_inverted_is_not_nan_f32"

; ModuleID = 'fpclass.bc'
source_filename = "fpclass.ll"
target triple = "spir64-unknown-unknown"

; check for no mask
define i1 @test_class_no_mask_f32(float %x) {
; CHECK-SPIRV: Function [[#]] [[#NoMaskFunc]]
; CHECK-SPIRV-NEXT: FunctionParameter
; CHECK-SPIRV-EMPTY:
; CHECK-SPIRV-NEXT: Label
; CHECK-SPIRV-NEXT: ReturnValue [[#False]]
  %val = call i1 @llvm.is.fpclass.f32(float %x, i32 0)
  ret i1 %val
}

; check for full mask
define i1 @test_class_full_mask_f32(float %x) {
; CHECK-SPIRV: Function [[#]] [[#FullMaskFunc]]
; CHECK-SPIRV-NEXT: FunctionParameter
; CHECK-SPIRV-EMPTY:
; CHECK-SPIRV-NEXT: Label
; CHECK-SPIRV-NEXT: ReturnValue [[#True]]
  %val = call i1 @llvm.is.fpclass.f32(float %x, i32 1023)
  ret i1 %val
}

; check for nan
define i1 @test_class_isnan_f32(float %x) {
; CHECK-SPIRV: Function [[#]] [[#NanFunc]]
; CHECK-SPIRV-NEXT: FunctionParameter [[#]] [[#Val:]]
; CHECK-SPIRV-EMPTY:
; CHECK-SPIRV-NEXT: Label
; CHECK-SPIRV-NEXT: IsNan [[#BoolTy]] [[#IsNan:]] [[#Val]]
; CHECK-SPIRV-NEXT: ReturnValue [[#IsNan]]
  %val = call i1 @llvm.is.fpclass.f32(float %x, i32 3)
  ret i1 %val
}

define <2 x i1> @test_class_isnan_v2f32(<2 x float> %x) {
; CHECK-SPIRV: Function [[#]] [[#VecNanFunc]]
; CHECK-SPIRV-NEXT: FunctionParameter [[#]] [[#Val:]]
; CHECK-SPIRV-EMPTY:
; CHECK-SPIRV-NEXT: Label
; CHECK-SPIRV-NEXT: IsNan [[#VecBoolTy]] [[#IsNan:]] [[#Val]]
; CHECK-SPIRV-NEXT: ReturnValue [[#IsNan]]
  %val = call <2 x i1> @llvm.is.fpclass.v2f32(<2 x float> %x, i32 3)
  ret <2 x i1> %val
}

; check for snan
define i1 @test_class_issnan_f32(float %x) {
; CHECK-SPIRV: Function [[#]] [[#SNanFunc]]
; CHECK-SPIRV-NEXT: FunctionParameter [[#]] [[#Val:]]
; CHECK-SPIRV-EMPTY:
; CHECK-SPIRV-NEXT: Label
; CHECK-SPIRV-NEXT: Bitcast [[#Int32Ty]] [[#BitCast:]] [[#Val]]
; CHECK-SPIRV-NEXT: UGreaterThanEqual [[#]] [[#GECheck:]] [[#BitCast]] [[#QNanBitConst]]
; CHECK-SPIRV-NEXT: IsNan [[#BoolTy]] [[#IsNan:]] [[#Val]]
; CHECK-SPIRV-NEXT: LogicalNot [[#]] [[#Not:]] [[#GECheck]]
; CHECK-SPIRV-NEXT: LogicalAnd [[#]] [[#And:]] [[#IsNan]] [[#Not]]
; CHECK-SPIRV-NEXT: ReturnValue [[#And]]
  %val = call i1 @llvm.is.fpclass.f32(float %x, i32 1)
  ret i1 %val
}

; check for qnan
define i1 @test_class_isqnan_f32(float %x) {
; CHECK-SPIRV: Function [[#]] [[#QNanFunc]]
; CHECK-SPIRV-NEXT: FunctionParameter [[#]] [[#Val:]]
; CHECK-SPIRV-EMPTY:
; CHECK-SPIRV-NEXT: Label
; CHECK-SPIRV-NEXT: Bitcast [[#Int32Ty]] [[#BitCast:]] [[#Val]]
; CHECK-SPIRV-NEXT: UGreaterThanEqual [[#]] [[#GECheck:]] [[#BitCast]] [[#QNanBitConst]]
; CHECK-SPIRV-NEXT: ReturnValue [[#GECheck]]
  %val = call i1 @llvm.is.fpclass.f32(float %x, i32 2)
  ret i1 %val
}

; check for inf
define i1 @test_class_is_inf_f32(float %x) {
; CHECK-SPIRV: Function [[#]] [[#InfFunc]]
; CHECK-SPIRV-NEXT: FunctionParameter [[#]] [[#Val:]]
; CHECK-SPIRV-EMPTY:
; CHECK-SPIRV-NEXT: Label
; CHECK-SPIRV-NEXT: IsInf [[#BoolTy]] [[#IsInf:]] [[#Val]]
; CHECK-SPIRV-NEXT: ReturnValue [[#IsInf]]
  %val = call i1 @llvm.is.fpclass.f32(float %x, i32 516)
  ret i1 %val
}

define <2 x i1> @test_class_is_inf_v2f32(<2 x float> %x) {
; CHECK-SPIRV: Function [[#]] [[#InfVecFunc]]
; CHECK-SPIRV-NEXT: FunctionParameter [[#]] [[#Val:]]
; CHECK-SPIRV-EMPTY:
; CHECK-SPIRV-NEXT: Label
; CHECK-SPIRV-NEXT: IsInf [[#VecBoolTy]] [[#IsInf:]] [[#Val]]
; CHECK-SPIRV-NEXT: ReturnValue [[#IsInf]]
  %val = call <2 x i1> @llvm.is.fpclass.v2f32(<2 x float> %x, i32 516)
  ret <2 x i1> %val
}

; check for pos inf
define i1 @test_class_is_pinf_f32(float %x) {
; CHECK-SPIRV: Function [[#]] [[#PosInfFunc]]
; CHECK-SPIRV-NEXT: FunctionParameter [[#]] [[#Val:]]
; CHECK-SPIRV-EMPTY:
; CHECK-SPIRV-NEXT: Label
; CHECK-SPIRV-NEXT: IsInf [[#BoolTy]] [[#IsInf:]] [[#Val]]
; CHECK-SPIRV-NEXT: SignBitSet [[#BoolTy]] [[#Sign:]] [[#Val]]
; CHECK-SPIRV-NEXT: LogicalNot [[#BoolTy]] [[#Not:]] [[#Sign]]
; CHECK-SPIRV-NEXT: LogicalAnd [[#BoolTy]] [[#And:]] [[#Not:]] [[#IsInf]]
; CHECK-SPIRV-NEXT: ReturnValue [[#And]]
  %val = call i1 @llvm.is.fpclass.f32(float %x, i32 512)
  ret i1 %val
}

define <2 x i1> @test_class_is_pinf_v2f32(<2 x float> %x) {
; CHECK-SPIRV: Function [[#]] [[#PosInfVecFunc]]
; CHECK-SPIRV-NEXT: FunctionParameter [[#]] [[#Val:]]
; CHECK-SPIRV-EMPTY:
; CHECK-SPIRV-NEXT: Label
; CHECK-SPIRV-NEXT: IsInf [[#VecBoolTy]] [[#IsInf:]] [[#Val]]
; CHECK-SPIRV-NEXT: SignBitSet [[#VecBoolTy]] [[#Sign:]] [[#Val]]
; CHECK-SPIRV-NEXT: LogicalNot [[#VecBoolTy]] [[#Not:]] [[#Sign]]
; CHECK-SPIRV-NEXT: LogicalAnd [[#VecBoolTy]] [[#And:]] [[#Not:]] [[#IsInf]]
; CHECK-SPIRV-NEXT: ReturnValue [[#And]]
  %val = call <2 x i1> @llvm.is.fpclass.v2f32(<2 x float> %x, i32 512)
  ret <2 x i1> %val
}

; check for neg inf
define i1 @test_class_is_ninf_f32(float %x) {
; CHECK-SPIRV: Function [[#]] [[#NegInfFunc]]
; CHECK-SPIRV-NEXT: FunctionParameter [[#]] [[#Val:]]
; CHECK-SPIRV-EMPTY:
; CHECK-SPIRV-NEXT: Label
; CHECK-SPIRV-NEXT: IsInf [[#BoolTy]] [[#IsInf:]] [[#Val]]
; CHECK-SPIRV-NEXT: SignBitSet [[#BoolTy]] [[#Sign:]] [[#Val]]
; CHECK-SPIRV-NEXT: LogicalAnd [[#BoolTy]] [[#And:]] [[#Sign]] [[#IsInf]]
; CHECK-SPIRV-NEXT: ReturnValue [[#And]]
  %val = call i1 @llvm.is.fpclass.f32(float %x, i32 4)
  ret i1 %val
}

define <2 x i1> @test_class_is_ninf_v2f32(<2 x float> %x) {
; CHECK-SPIRV: Function [[#]] [[#NegInfVecFunc]]
; CHECK-SPIRV-NEXT: FunctionParameter [[#]] [[#Val:]]
; CHECK-SPIRV-EMPTY:
; CHECK-SPIRV-NEXT: Label
; CHECK-SPIRV-NEXT: IsInf [[#VecBoolTy]] [[#IsInf:]] [[#Val]]
; CHECK-SPIRV-NEXT: SignBitSet [[#VecBoolTy]] [[#Sign:]] [[#Val]]
; CHECK-SPIRV-NEXT: LogicalAnd [[#VecBoolTy]] [[#And:]] [[#Sign]] [[#IsInf]]
; CHECK-SPIRV-NEXT: ReturnValue [[#And]]
  %val = call <2 x i1> @llvm.is.fpclass.v2f32(<2 x float> %x, i32 4)
  ret <2 x i1> %val
}

; check for normal
define i1 @test_class_is_normal(float %x) {
; CHECK-SPIRV: Function [[#]] [[#NormFunc]]
; CHECK-SPIRV-NEXT: FunctionParameter [[#]] [[#Val:]]
; CHECK-SPIRV-EMPTY:
; CHECK-SPIRV-NEXT: Label
; CHECK-SPIRV-NEXT: IsNormal [[#BoolTy]] [[#IsNormal:]] [[#Val]]
; CHECK-SPIRV-NEXT: ReturnValue [[#IsNormal]]
  %val = call i1 @llvm.is.fpclass.f32(float %x, i32 264)
  ret i1 %val
}

; check for pos normal
define i1 @test_constant_class_pnormal() {
; CHECK-SPIRV: Function [[#]] [[#PosNormFunc]]
; CHECK-SPIRV-EMPTY:
; CHECK-SPIRV-NEXT: Label
; CHECK-SPIRV-NEXT: IsNormal [[#BoolTy]] [[#IsNormal:]] [[#DoubleConst]]
; CHECK-SPIRV-NEXT: SignBitSet [[#BoolTy]] [[#Sign:]] [[#DoubleConst]]
; CHECK-SPIRV-NEXT: LogicalNot [[#BoolTy]] [[#Not:]] [[#Sign]]
; CHECK-SPIRV-NEXT: LogicalAnd [[#BoolTy]] [[#And:]] [[#Not]] [[#IsNormal]]
; CHECK-SPIRV-NEXT: ReturnValue [[#And]]
  %val = call i1 @llvm.is.fpclass.f64(double 1.000000e+00, i32 256)
  ret i1 %val
}
; check for neg normal
define i1 @test_constant_class_nnormal() {
; CHECK-SPIRV: Function [[#]] [[#NegNormFunc]]
; CHECK-SPIRV-EMPTY:
; CHECK-SPIRV-NEXT: Label
; CHECK-SPIRV-NEXT: IsNormal [[#BoolTy]] [[#IsNormal:]] [[#DoubleConst]]
; CHECK-SPIRV-NEXT: SignBitSet [[#BoolTy]] [[#Sign:]] [[#DoubleConst]]
; CHECK-SPIRV-NEXT: LogicalAnd [[#BoolTy]] [[#And:]] [[#Sign]] [[#IsNormal]]
; CHECK-SPIRV-NEXT: ReturnValue [[#And]]
  %val = call i1 @llvm.is.fpclass.f64(double 1.000000e+00, i32 8)
  ret i1 %val
}

; check for subnormal
define i1 @test_class_subnormal(float %arg) {
; CHECK-SPIRV: Function [[#]] [[#SubnormFunc]]
; CHECK-SPIRV-NEXT: FunctionParameter [[#]] [[#Val:]]
; CHECK-SPIRV-EMPTY:
; CHECK-SPIRV-NEXT: Label
; CHECK-SPIRV-NEXT: Bitcast [[#Int32Ty]] [[#BitCast:]] [[#Val]]
; CHECK-SPIRV-NEXT: ISub [[#Int32Ty]] [[#Sub:]] [[#BitCast]] [[#MantissaConst:]]
; CHECK-SPIRV-NEXT: ULessThan [[#BoolTy]] [[#Less:]] [[#Sub]] [[#MantissaConst:]]
; CHECK-SPIRV-NEXT: ReturnValue [[#Less]]
  %val = call i1 @llvm.is.fpclass.f32(float %arg, i32 144)
  ret i1 %val
}

; check for pos subnormal
define i1 @test_class_possubnormal(float %arg) {
; CHECK-SPIRV: Function [[#]] [[#PosSubnormFunc]]
; CHECK-SPIRV-NEXT: FunctionParameter [[#]] [[#Val:]]
; CHECK-SPIRV-EMPTY:
; CHECK-SPIRV-NEXT: Label
; CHECK-SPIRV-NEXT: Bitcast [[#Int32Ty]] [[#BitCast:]] [[#Val]]
; CHECK-SPIRV-NEXT: ISub [[#Int32Ty]] [[#Sub:]] [[#BitCast]] [[#MantissaConst:]]
; CHECK-SPIRV-NEXT: ULessThan [[#BoolTy]] [[#Less:]] [[#Sub]] [[#MantissaConst:]]
; CHECK-SPIRV-NEXT: SignBitSet [[#BoolTy]] [[#Sign:]] [[#Val]]
; CHECK-SPIRV-NEXT: LogicalNot [[#BoolTy]] [[#Not:]] [[#Sign]]
; CHECK-SPIRV-NEXT: LogicalAnd [[#BoolTy]] [[#And:]] [[#Not]] [[#Less]]
; CHECK-SPIRV-NEXT: ReturnValue [[#And]]
  %val = call i1 @llvm.is.fpclass.f32(float %arg, i32 128)
  ret i1 %val
}

; check for neg subnormal
define i1 @test_class_negsubnormal(float %arg) {
; CHECK-SPIRV: Function [[#]] [[#NegSubnormFunc]]
; CHECK-SPIRV-NEXT: FunctionParameter [[#]] [[#Val:]]
; CHECK-SPIRV-EMPTY:
; CHECK-SPIRV-NEXT: Label
; CHECK-SPIRV-NEXT: Bitcast [[#Int32Ty]] [[#BitCast:]] [[#Val]]
; CHECK-SPIRV-NEXT: ISub [[#Int32Ty]] [[#Sub:]] [[#BitCast]] [[#MantissaConst:]]
; CHECK-SPIRV-NEXT: ULessThan [[#BoolTy]] [[#Less:]] [[#Sub]] [[#MantissaConst:]]
; CHECK-SPIRV-NEXT: SignBitSet [[#BoolTy]] [[#Sign:]] [[#Val]]
; CHECK-SPIRV-NEXT: LogicalAnd [[#BoolTy]] [[#And:]] [[#Sign]] [[#Less]]
; CHECK-SPIRV-NEXT: ReturnValue [[#And]]
  %val = call i1 @llvm.is.fpclass.f32(float %arg, i32 16)
  ret i1 %val
}

; check for zero
define i1 @test_class_zero(float %arg) {
; CHECK-SPIRV: Function [[#]] [[#ZeroFunc]]
; CHECK-SPIRV-NEXT: FunctionParameter [[#]] [[#Val:]]
; CHECK-SPIRV-EMPTY:
; CHECK-SPIRV-NEXT: Label
; CHECK-SPIRV-NEXT: Bitcast [[#Int32Ty]] [[#BitCast:]] [[#Val]]
; CHECK-SPIRV-NEXT: BitwiseAnd [[#Int32Ty]] [[#BitwiseAndRes:]] [[#BitCast]] [[#MaskToClearSignBit]]
; CHECK-SPIRV-NEXT: IEqual [[#BoolTy]] [[#EqualPos:]] [[#BitwiseAndRes]] [[#ZeroConst]]
; CHECK-SPIRV-NEXT: ReturnValue [[#EqualPos]]
  %val = call i1 @llvm.is.fpclass.f32(float %arg, i32 96)
  ret i1 %val
}

; check for pos zero
define i1 @test_class_poszero(float %arg) {
; CHECK-SPIRV: Function [[#]] [[#PosZeroFunc]]
; CHECK-SPIRV-NEXT: FunctionParameter [[#]] [[#Val:]]
; CHECK-SPIRV-EMPTY:
; CHECK-SPIRV-NEXT: Label
; CHECK-SPIRV-NEXT: Bitcast [[#Int32Ty]] [[#BitCast:]] [[#Val]]
; CHECK-SPIRV-NEXT: IEqual [[#BoolTy]] [[#Equal:]] [[#BitCast]] [[#ZeroConst]]
; CHECK-SPIRV-NEXT: ReturnValue [[#Equal]]
  %val = call i1 @llvm.is.fpclass.f32(float %arg, i32 64)
  ret i1 %val
}

; check for neg zero
define i1 @test_class_negzero(float %arg) {
; CHECK-SPIRV: Function [[#]] [[#NegZeroFunc]]
; CHECK-SPIRV-NEXT: FunctionParameter [[#]] [[#Val:]]
; CHECK-SPIRV-EMPTY:
; CHECK-SPIRV-NEXT: Label
; CHECK-SPIRV-NEXT: Bitcast [[#Int32Ty]] [[#BitCast:]] [[#Val]]
; CHECK-SPIRV-NEXT: IEqual [[#BoolTy]] [[#Equal:]] [[#BitCast]] [[#NegatedZeroConst]]
; CHECK-SPIRV-NEXT: ReturnValue [[#Equal]]
  %val = call i1 @llvm.is.fpclass.f32(float %arg, i32 32)
  ret i1 %val
}

; check for neg inf or nan
define i1 @test_class_is_ninf_or_nan_f32(float %x) {
; CHECK-SPIRV: Function [[#]] [[#NegInfOrNanFunc]]
; CHECK-SPIRV-NEXT: FunctionParameter [[#]] [[#Val:]]
; CHECK-SPIRV-EMPTY:
; CHECK-SPIRV-NEXT: Label
; CHECK-SPIRV-NEXT: IsNan [[#BoolTy]] [[#IsNan:]] [[#Val]]
; CHECK-SPIRV-NEXT: IsInf [[#BoolTy]] [[#IsInf:]] [[#Val]]
; CHECK-SPIRV-NEXT: SignBitSet [[#BoolTy]] [[#Sign:]] [[#Val]]
; CHECK-SPIRV-NEXT: LogicalAnd [[#BoolTy]] [[#And:]] [[#Sign]] [[#IsInf]]
; CHECK-SPIRV-NEXT: LogicalOr [[#BoolTy]] [[#Or:]] [[#IsNan]] [[#And]]
; CHECK-SPIRV-NEXT: ReturnValue [[#Or]]
  %val = call i1 @llvm.is.fpclass.f32(float %x, i32 7)
  ret i1 %val
}

; check for neg inf, pos normal, neg subnormal pos zero and snan scalar
define i1 @test_class_neginf_posnormal_negsubnormal_poszero_snan_f64(double %arg) {
; CHECK-SPIRV: Function [[#]] [[#ComplexFunc1]]
; CHECK-SPIRV-NEXT: FunctionParameter [[#]] [[#Val:]]
; CHECK-SPIRV-EMPTY:
; CHECK-SPIRV-NEXT: Label
; CHECK-SPIRV-NEXT: Bitcast [[#Int64Ty]] [[#BitCast1:]] [[#Val]]
; CHECK-SPIRV-NEXT: UGreaterThanEqual [[#BoolTy]] [[#GECheck:]] [[#BitCast1]] [[#QNanBitConst64]]
; CHECK-SPIRV-NEXT: IsNan [[#BoolTy]] [[#IsNan:]] [[#Val]]
; CHECK-SPIRV-NEXT: LogicalNot [[#BoolTy]] [[#Not1:]] [[#GECheck]]
; CHECK-SPIRV-NEXT: LogicalAnd [[#BoolTy]] [[#And1:]] [[#IsNan]] [[#Not1]]
; CHECK-SPIRV-NEXT: IsInf [[#BoolTy]] [[#IsInf:]] [[#Val]]
; CHECK-SPIRV-NEXT: SignBitSet [[#BoolTy]] [[#Sign:]] [[#Val]]
; CHECK-SPIRV-NEXT: LogicalAnd [[#BoolTy]] [[#And2:]] [[#Sign]] [[#IsInf]]
; CHECK-SPIRV-NEXT: IsNormal [[#BoolTy]] [[#IsNormal:]] [[#Val]]
; CHECK-SPIRV-NEXT: LogicalNot [[#BoolTy]] [[#Not2:]] [[#Sign]]
; CHECK-SPIRV-NEXT: LogicalAnd [[#BoolTy]] [[#And3:]] [[#Not2]] [[#IsNormal]]
; CHECK-SPIRV-NEXT: Bitcast [[#Int64Ty]] [[#BitCast2:]] [[#Val]]
; CHECK-SPIRV-NEXT: ISub [[#Int64Ty]] [[#Sub:]] [[#BitCast2]] [[#MantissaConst64]]
; CHECK-SPIRV-NEXT: ULessThan [[#BoolTy]] [[#Less:]] [[#Sub]] [[#MantissaConst64]]
; CHECK-SPIRV-NEXT: LogicalAnd [[#BoolTy]] [[#And4:]] [[#Sign]] [[#Less]]
; CHECK-SPIRV-NEXT: Bitcast [[#Int64Ty]] [[#BitCast3:]] [[#Val]]
; CHECK-SPIRV-NEXT: IEqual [[#BoolTy]] [[#Equal:]] [[#BitCast3]] [[#ZeroConst64]]
; CHECK-SPIRV-NEXT: LogicalOr [[#BoolTy]] [[#Or1:]] [[#And1]] [[#And2]]
; CHECK-SPIRV-NEXT: LogicalOr [[#BoolTy]] [[#Or2:]] [[#Or1]] [[#And3]]
; CHECK-SPIRV-NEXT: LogicalOr [[#BoolTy]] [[#Or3:]] [[#Or2]] [[#And4]]
; CHECK-SPIRV-NEXT: LogicalOr [[#BoolTy]] [[#Or4:]] [[#Or3]] [[#Equal]]
; CHECK-SPIRV-NEXT: ReturnValue [[#Or4]]
  %val = call i1 @llvm.is.fpclass.f64(double %arg, i32 341)
  ret i1 %val
}

; check for neg inf, pos normal, neg subnormal pos zero and snan vector
define <2 x i1> @test_class_neginf_posnormal_negsubnormal_poszero_snan_v2f16(<2 x half> %arg) {
; CHECK-SPIRV: Function [[#]] [[#ComplexFunc2]]
; CHECK-SPIRV-NEXT: FunctionParameter [[#]] [[#Val:]]
; CHECK-SPIRV-EMPTY:
; CHECK-SPIRV-NEXT: Label
; CHECK-SPIRV-NEXT: Bitcast [[#Int16VecTy]] [[#BitCast1:]] [[#Val]]
; CHECK-SPIRV-NEXT: UGreaterThanEqual [[#VecBoolTy]] [[#GECheck:]] [[#BitCast1]] [[#QNanBitConstVec16]]
; CHECK-SPIRV-NEXT: IsNan [[#VecBoolTy]] [[#IsNan:]] [[#Val]]
; CHECK-SPIRV-NEXT: LogicalNot [[#VecBoolTy]] [[#Not1:]] [[#GECheck]]
; CHECK-SPIRV-NEXT: LogicalAnd [[#VecBoolTy]] [[#And1:]] [[#IsNan]] [[#Not1]]
; CHECK-SPIRV-NEXT: IsInf [[#VecBoolTy]] [[#IsInf:]] [[#Val]]
; CHECK-SPIRV-NEXT: SignBitSet [[#VecBoolTy]] [[#Sign:]] [[#Val]]
; CHECK-SPIRV-NEXT: LogicalAnd [[#VecBoolTy]] [[#And2:]] [[#Sign]] [[#IsInf]]
; CHECK-SPIRV-NEXT: IsNormal [[#VecBoolTy]] [[#IsNormal:]] [[#Val]]
; CHECK-SPIRV-NEXT: LogicalNot [[#VecBoolTy]] [[#Not2:]] [[#Sign]]
; CHECK-SPIRV-NEXT: LogicalAnd [[#VecBoolTy]] [[#And3:]] [[#Not2]] [[#IsNormal]]
; CHECK-SPIRV-NEXT: Bitcast [[#Int16VecTy]] [[#BitCast2:]] [[#Val]]
; CHECK-SPIRV-NEXT: ISub [[#Int16VecTy]] [[#Sub:]] [[#BitCast2]] [[#MantissaConstVec16]]
; CHECK-SPIRV-NEXT: ULessThan [[#VecBoolTy]] [[#Less:]] [[#Sub]] [[#MantissaConstVec16]]
; CHECK-SPIRV-NEXT: LogicalAnd [[#VecBoolTy]] [[#And4:]] [[#Sign]] [[#Less]]
; CHECK-SPIRV-NEXT: Bitcast [[#Int16VecTy]] [[#BitCast3:]] [[#Val]]
; CHECK-SPIRV-NEXT: IEqual [[#VecBoolTy]] [[#Equal:]] [[#BitCast3]] [[#ZeroConst16]]
; CHECK-SPIRV-NEXT: LogicalOr [[#VecBoolTy]] [[#Or1:]] [[#And1]] [[#And2]]
; CHECK-SPIRV-NEXT: LogicalOr [[#VecBoolTy]] [[#Or2:]] [[#Or1]] [[#And3]]
; CHECK-SPIRV-NEXT: LogicalOr [[#VecBoolTy]] [[#Or3:]] [[#Or2]] [[#And4]]
; CHECK-SPIRV-NEXT: LogicalOr [[#VecBoolTy]] [[#Or4:]] [[#Or3]] [[#Equal]]
; CHECK-SPIRV-NEXT: ReturnValue [[#Or4]]
  %val = call <2 x i1> @llvm.is.fpclass.v2f16(<2 x half> %arg, i32 341)
  ret <2 x i1> %val
}

; inverted check for not nan
define i1 @test_class_inverted_is_not_nan_f32(float %x) {
; CHECK-SPIRV: Function [[#]] [[#NotNanFunc]]
; CHECK-SPIRV-NEXT: FunctionParameter [[#]] [[#Val:]]
; CHECK-SPIRV-EMPTY:
; CHECK-SPIRV-NEXT: Label
; CHECK-SPIRV-NEXT: IsNan [[#BoolTy]] [[#IsNan:]] [[#Val]]
; CHECK-SPIRV-NEXT: LogicalNot [[#BoolTy]] [[#Not:]] [[#IsNan]]
; CHECK-SPIRV-NEXT: ReturnValue [[#Not]]
  %val = call i1 @llvm.is.fpclass.f32(float %x, i32 1020)
  ret i1 %val
}

declare i1 @llvm.is.fpclass.f32(float, i32 immarg)

declare i1 @llvm.is.fpclass.f64(double, i32 immarg)

declare <2 x i1> @llvm.is.fpclass.v2f32(<2 x float>, i32 immarg)

declare <2 x i1> @llvm.is.fpclass.v2f16(<2 x half>, i32 immarg)
