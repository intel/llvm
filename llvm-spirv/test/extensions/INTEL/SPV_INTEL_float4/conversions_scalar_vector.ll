; This tests checks if FP4 and FP8 scalar and vector conversions specified by
; __builtin_spirv_* external function calls translated correctly. It doesn't
; include Clamp*, Biased*, ClampBiased* conversions (it's part of another test
; file).

; RUN: llvm-spirv %s -o %t.spv --spirv-ext=+SPV_EXT_float8,+SPV_INTEL_float4,+SPV_INTEL_int4,+SPV_KHR_bfloat16
; RUN: llvm-spirv %t.spv -o %t.spt --to-text
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.spv -o %t.rev.bc -r --spirv-target-env=SPV-IR
; RUN: llvm-dis %t.rev.bc -o %t.rev.ll
; RUN: FileCheck < %t.rev.ll %s --check-prefix=CHECK-LLVM

; CHECK-SPIRV-DAG: Capability Int4TypeINTEL
; CHECK-SPIRV-DAG: Capability Float8EXT

; CHECK-SPIRV-DAG: Extension "SPV_INTEL_int4"
; CHECK-SPIRV-DAG: Extension "SPV_EXT_float8"
; CHECK-SPIRV-DAG: Extension "SPV_INTEL_float4"

; CHECK-SPIRV-DAG: Name [[#fp4e2m1_hf8_scalar:]] "fp4e2m1_hf8_scalar"
; CHECK-SPIRV-DAG: Name [[#fp4e2m1_hf8_vector:]] "fp4e2m1_hf8_vector"
; CHECK-SPIRV-DAG: Name [[#fp4e2m1_bf8_scalar:]] "fp4e2m1_bf8_scalar"
; CHECK-SPIRV-DAG: Name [[#fp4e2m1_bf8_vector:]] "fp4e2m1_bf8_vector"
; CHECK-SPIRV-DAG: Name [[#fp4e2m1_hf16_scalar:]] "fp4e2m1_hf16_scalar"
; CHECK-SPIRV-DAG: Name [[#fp4e2m1_hf16_vector:]] "fp4e2m1_hf16_vector"
; CHECK-SPIRV-DAG: Name [[#fp4e2m1_bf16_scalar:]] "fp4e2m1_bf16_scalar"
; CHECK-SPIRV-DAG: Name [[#fp4e2m1_bf16_vector:]] "fp4e2m1_bf16_vector"

; CHECK-SPIRV-DAG: Name [[#hf16_fp4e2m1_scalar:]] "hf16_fp4e2m1_scalar"
; CHECK-SPIRV-DAG: Name [[#hf16_fp4e2m1_vector:]] "hf16_fp4e2m1_vector"
; CHECK-SPIRV-DAG: Name [[#bf16_fp4e2m1_scalar:]] "bf16_fp4e2m1_scalar"
; CHECK-SPIRV-DAG: Name [[#bf16_fp4e2m1_vector:]] "bf16_fp4e2m1_vector"

; CHECK-SPIRV-DAG: TypeInt [[#Int8Ty:]] 8 0
; CHECK-SPIRV-DAG: TypeVector [[#Int8VecTy:]] [[#Int8Ty]] 8

; CHECK-SPIRV-DAG: TypeInt [[#Int4Ty:]] 4 0
; CHECK-SPIRV-DAG: TypeVector [[#Int4VecTy:]] [[#Int4Ty]] 8
; CHECK-SPIRV-DAG: Constant [[#Int4Ty]] [[#Int4Const:]] 1
; CHECK-SPIRV-DAG: ConstantComposite [[#Int4VecTy]] [[#Int4VecConst:]] [[#Int4Const]] [[#Int4Const]] [[#Int4Const]] [[#Int4Const]] [[#Int4Const]] [[#Int4Const]] [[#Int4Const]] [[#Int4Const]]

; CHECK-SPIRV-DAG: TypeFloat [[#E2M1Ty:]] 4 6214
; CHECK-SPIRV-DAG: TypeVector [[#E2M1VecTy:]] [[#E2M1Ty]] 8

; CHECK-SPIRV-DAG: TypeFloat [[#HFloat8Ty:]] 8 4214
; CHECK-SPIRV-DAG: TypeVector [[#HFloat8VecTy:]] [[#HFloat8Ty]] 8

; CHECK-SPIRV-DAG: TypeFloat [[#BFloat8Ty:]] 8 4215
; CHECK-SPIRV-DAG: TypeVector [[#BFloat8VecTy:]] [[#BFloat8Ty]] 8

; CHECK-SPIRV-DAG: TypeFloat [[#HFloat16Ty:]] 16 {{$}}
; CHECK-SPIRV-DAG: TypeVector [[#HFloat16VecTy:]] [[#HFloat16Ty]] 8
; CHECK-SPIRV-DAG: Constant [[#HFloat16Ty]] [[#HalfConst:]] 15360
; CHECK-SPIRV-DAG: ConstantComposite [[#HFloat16VecTy]] [[#HalfVecConst:]] [[#HalfConst]] [[#HalfConst]] [[#HalfConst]] [[#HalfConst]] [[#HalfConst]] [[#HalfConst]] [[#HalfConst]] [[#HalfConst]]

; CHECK-SPIRV-DAG: TypeFloat [[#BFloat16Ty:]] 16 0
; CHECK-SPIRV-DAG: TypeVector [[#BFloat16VecTy:]] [[#BFloat16Ty]] 8
; CHECK-SPIRV-DAG: Constant [[#BFloat16Ty]] [[#BfloatConst:]] 16256
; CHECK-SPIRV-DAG: ConstantComposite [[#BFloat16VecTy]] [[#BfloatVecConst:]] [[#BfloatConst]] [[#BfloatConst]] [[#BfloatConst]] [[#BfloatConst]] [[#BfloatConst]] [[#BfloatConst]] [[#BfloatConst]] [[#BfloatConst]]

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

; Followings tests are for 4-bit upconversions

; CHECK-SPIRV: Function [[#]] [[#fp4e2m1_hf8_scalar]] [[#]]
; CHECK-SPIRV: Bitcast [[#E2M1Ty]] [[#Cast1:]] [[#Int4Const]]
; CHECK-SPIRV: FConvert [[#HFloat8Ty]] [[#Conv:]] [[#Cast1]]
; CHECK-SPIRV: Bitcast [[#Int8Ty]] [[#Cast2:]] [[#Conv]]
; CHECK-SPIRV: ReturnValue [[#Cast2]]

; CHECK-LLVM-LABEL: fp4e2m1_hf8_scalar
; CHECK-LLVM: %[[#Call:]] = call i8 @_Z38__builtin_spirv_ConvertE2M1ToE4M3INTELi(i4 1)
; CHECK-LLVM: ret i8 %[[#Call]]

define spir_func i8 @fp4e2m1_hf8_scalar() {
entry:
  %0 = call i8 @_Z38__builtin_spirv_ConvertE2M1ToE4M3INTELi(i4 1)
  ret i8 %0
}

declare dso_local spir_func i8 @_Z38__builtin_spirv_ConvertE2M1ToE4M3INTELi(i4)

; CHECK-SPIRV: Function [[#]] [[#fp4e2m1_hf8_vector]] [[#]]
; CHECK-SPIRV: Bitcast [[#E2M1VecTy]] [[#Cast1:]] [[#Int4VecConst]]
; CHECK-SPIRV: FConvert [[#HFloat8VecTy]] [[#Conv:]] [[#Cast1]]
; CHECK-SPIRV: Bitcast [[#Int8VecTy]] [[#Cast2:]] [[#Conv]]
; CHECK-SPIRV: ReturnValue [[#Cast2]]

; CHECK-LLVM-LABEL: fp4e2m1_hf8_vector
; CHECK-LLVM: %[[#Call:]] = call <8 x i8> @_Z38__builtin_spirv_ConvertE2M1ToE4M3INTELDv8_i(<8 x i4> splat (i4 1))
; CHECK-LLVM: ret <8 x i8> %[[#Call]]

define spir_func <8 x i8> @fp4e2m1_hf8_vector() {
entry:
  %0 = call <8 x i8> @_Z38__builtin_spirv_ConvertE2M1ToE4M3INTELDv8_i(<8 x i4> <i4 1, i4 1, i4 1, i4 1, i4 1, i4 1, i4 1, i4 1>)
  ret <8 x i8> %0
}

declare dso_local spir_func <8 x i8> @_Z38__builtin_spirv_ConvertE2M1ToE4M3INTELDv8_i(<8 x i4>)

; CHECK-SPIRV: Function [[#]] [[#fp4e2m1_bf8_scalar]] [[#]]
; CHECK-SPIRV: Bitcast [[#E2M1Ty]] [[#Cast1:]] [[#Int4Const]]
; CHECK-SPIRV: FConvert [[#BFloat8Ty]] [[#Conv:]] [[#Cast1]]
; CHECK-SPIRV: Bitcast [[#Int8Ty]] [[#Cast2:]] [[#Conv]]
; CHECK-SPIRV: ReturnValue [[#Cast2]]

; CHECK-LLVM-LABEL: fp4e2m1_bf8_scalar
; CHECK-LLVM: %[[#Call:]] = call i8 @_Z38__builtin_spirv_ConvertE2M1ToE5M2INTELi(i4 1)
; CHECK-LLVM: ret i8 %[[#Call]]

define spir_func i8 @fp4e2m1_bf8_scalar() {
entry:
  %0 = call i8 @_Z38__builtin_spirv_ConvertE2M1ToE5M2INTELi(i4 1)
  ret i8 %0
}

declare dso_local spir_func i8 @_Z38__builtin_spirv_ConvertE2M1ToE5M2INTELi(i4)

; CHECK-SPIRV: Function [[#]] [[#fp4e2m1_bf8_vector]] [[#]]
; CHECK-SPIRV: Bitcast [[#E2M1VecTy]] [[#Cast1:]] [[#Int4VecConst]]
; CHECK-SPIRV: FConvert [[#BFloat8VecTy]] [[#Conv:]] [[#Cast1]]
; CHECK-SPIRV: Bitcast [[#Int8VecTy]] [[#Cast2:]] [[#Conv]]
; CHECK-SPIRV: ReturnValue [[#Cast2]]

; CHECK-LLVM-LABEL: fp4e2m1_bf8_vector
; CHECK-LLVM: %[[#Call:]] = call <8 x i8> @_Z38__builtin_spirv_ConvertE2M1ToE5M2INTELDv8_i(<8 x i4> splat (i4 1))
; CHECK-LLVM: ret <8 x i8> %[[#Call]]

define spir_func <8 x i8> @fp4e2m1_bf8_vector() {
entry:
  %0 = call <8 x i8> @_Z38__builtin_spirv_ConvertE2M1ToE5M2INTELDv8_i(<8 x i4> <i4 1, i4 1, i4 1, i4 1, i4 1, i4 1, i4 1, i4 1>)
  ret <8 x i8> %0
}

declare dso_local spir_func <8 x i8> @_Z38__builtin_spirv_ConvertE2M1ToE5M2INTELDv8_i(<8 x i4>)

; CHECK-SPIRV: Function [[#]] [[#fp4e2m1_hf16_scalar]] [[#]]
; CHECK-SPIRV: Bitcast [[#E2M1Ty]] [[#Cast1:]] [[#Int4Const]]
; CHECK-SPIRV: FConvert [[#HFloat16Ty]] [[#Conv:]] [[#Cast1]]
; CHECK-SPIRV: ReturnValue [[#Conv]]

; CHECK-LLVM-LABEL: fp4e2m1_hf16_scalar
; CHECK-LLVM: %[[#Call:]] = call half @_Z38__builtin_spirv_ConvertE2M1ToFP16INTELi(i4 1)
; CHECK-LLVM: ret half %[[#Call]]

define spir_func half @fp4e2m1_hf16_scalar() {
entry:
  %0 = call half @_Z38__builtin_spirv_ConvertE2M1ToFP16INTELi(i4 1)
  ret half %0
}

declare dso_local spir_func half @_Z38__builtin_spirv_ConvertE2M1ToFP16INTELi(i4)

; CHECK-SPIRV: Function [[#]] [[#fp4e2m1_hf16_vector]] [[#]]
; CHECK-SPIRV: Bitcast [[#E2M1VecTy]] [[#Cast1:]] [[#Int4VecConst]]
; CHECK-SPIRV: FConvert [[#HFloat16VecTy]] [[#Conv:]] [[#Cast1]]
; CHECK-SPIRV: ReturnValue [[#Conv]]

; CHECK-LLVM-LABEL: fp4e2m1_hf16_vector
; CHECK-LLVM: %[[#Call:]] = call <8 x half> @_Z38__builtin_spirv_ConvertE2M1ToFP16INTELDv8_i(<8 x i4> splat (i4 1))
; CHECK-LLVM: ret <8 x half> %[[#Call]]

define spir_func <8 x half> @fp4e2m1_hf16_vector() {
entry:
  %0 = call <8 x half> @_Z38__builtin_spirv_ConvertE2M1ToFP16INTELDv8_i(<8 x i4> <i4 1, i4 1, i4 1, i4 1, i4 1, i4 1, i4 1, i4 1>)
  ret <8 x half> %0
}

declare dso_local spir_func <8 x half> @_Z38__builtin_spirv_ConvertE2M1ToFP16INTELDv8_i(<8 x i4>)

; CHECK-SPIRV: Function [[#]] [[#fp4e2m1_bf16_scalar]] [[#]]
; CHECK-SPIRV: Bitcast [[#E2M1Ty]] [[#Cast1:]] [[#Int4Const]]
; CHECK-SPIRV: FConvert [[#BFloat16Ty]] [[#Conv:]] [[#Cast1]]
; CHECK-SPIRV: ReturnValue [[#Conv]]

; CHECK-LLVM-LABEL: fp4e2m1_bf16_scalar
; CHECK-LLVM: %[[#Call:]] = call bfloat @_Z38__builtin_spirv_ConvertE2M1ToBF16INTELi(i4 1)
; CHECK-LLVM: ret bfloat %[[#Call]]

define spir_func bfloat @fp4e2m1_bf16_scalar() {
entry:
  %0 = call bfloat @_Z38__builtin_spirv_ConvertE2M1ToBF16INTELi(i4 1)
  ret bfloat %0
}

declare dso_local spir_func bfloat @_Z38__builtin_spirv_ConvertE2M1ToBF16INTELi(i4)

; CHECK-SPIRV: Function [[#]] [[#fp4e2m1_bf16_vector]] [[#]]
; CHECK-SPIRV: Bitcast [[#E2M1VecTy]] [[#Cast1:]] [[#Int4VecConst]]
; CHECK-SPIRV: FConvert [[#BFloat16VecTy]] [[#Conv:]] [[#Cast1]]
; CHECK-SPIRV: ReturnValue [[#Conv]]

; CHECK-LLVM-LABEL: fp4e2m1_bf16_vector
; CHECK-LLVM: %[[#Call:]] = call <8 x bfloat> @_Z38__builtin_spirv_ConvertE2M1ToBF16INTELDv8_i(<8 x i4> splat (i4 1))
; CHECK-LLVM: ret <8 x bfloat> %[[#Call]]

define spir_func <8 x bfloat> @fp4e2m1_bf16_vector() {
entry:
  %0 = call <8 x bfloat> @_Z38__builtin_spirv_ConvertE2M1ToBF16INTELDv8_i(<8 x i4> <i4 1, i4 1, i4 1, i4 1, i4 1, i4 1, i4 1, i4 1>)
  ret <8 x bfloat> %0
}

declare dso_local spir_func <8 x bfloat> @_Z38__builtin_spirv_ConvertE2M1ToBF16INTELDv8_i(<8 x i4>)

; Following tests are for 4-bit roundings

; CHECK-SPIRV: Function [[#]] [[#hf16_fp4e2m1_scalar]] [[#]]
; CHECK-SPIRV: FConvert [[#E2M1Ty]] [[#Conv:]] [[#HalfConst]]
; CHECK-SPIRV: Bitcast [[#Int4Ty]] [[#Cast:]] [[#Conv]]
; CHECK-SPIRV: ReturnValue [[#Cast]]

; CHECK-LLVM-LABEL: hf16_fp4e2m1_scalar
; CHECK-LLVM: %[[#Call:]] = call i4 @_Z38__builtin_spirv_ConvertFP16ToE2M1INTELDh(half 0xH3C00)
; CHECK-LLVM: ret i4 %[[#Call]]

define spir_func i4 @hf16_fp4e2m1_scalar() {
entry:
  %0 = call i4 @_Z38__builtin_spirv_ConvertFP16ToE2M1INTELDh(half 1.0)
  ret i4 %0
}

declare dso_local spir_func i4 @_Z38__builtin_spirv_ConvertFP16ToE2M1INTELDh(half)

; CHECK-SPIRV: Function [[#]] [[#hf16_fp4e2m1_vector]] [[#]]
; CHECK-SPIRV: FConvert [[#E2M1VecTy]] [[#Conv:]] [[#HalfVecConst]]
; CHECK-SPIRV: Bitcast [[#Int4VecTy]] [[#Cast:]] [[#Conv]]
; CHECK-SPIRV: ReturnValue [[#Cast]]

; CHECK-LLVM-LABEL: hf16_fp4e2m1_vector
; CHECK-LLVM: %[[#Call:]] = call <8 x i4> @_Z38__builtin_spirv_ConvertFP16ToE2M1INTELDv8_Dh(<8 x half> splat (half 0xH3C00))
; CHECK-LLVM: ret <8 x i4> %[[#Call]]

define spir_func <8 x i4> @hf16_fp4e2m1_vector() {
entry:
  %0 = call <8 x i4> @_Z38__builtin_spirv_ConvertFP16ToE2M1INTELDv8_Dh(<8 x half> <half 1.0, half 1.0, half 1.0, half 1.0, half 1.0, half 1.0, half 1.0, half 1.0>)
  ret <8 x i4> %0
}

declare dso_local spir_func <8 x i4> @_Z38__builtin_spirv_ConvertFP16ToE2M1INTELDv8_Dh(<8 x half>)

; CHECK-SPIRV: Function [[#]] [[#bf16_fp4e2m1_scalar]] [[#]]
; CHECK-SPIRV: FConvert [[#E2M1Ty]] [[#Conv:]] [[#BfloatConst]]
; CHECK-SPIRV: Bitcast [[#Int4Ty]] [[#Cast:]] [[#Conv]]
; CHECK-SPIRV: ReturnValue [[#Cast]]

; CHECK-LLVM-LABEL: bf16_fp4e2m1_scalar
; CHECK-LLVM: %[[#Call:]] = call i4 @_Z38__builtin_spirv_ConvertBF16ToE2M1INTELDF16b(bfloat 0xR3F80)
; CHECK-LLVM: ret i4 %[[#Call]]

define spir_func i4 @bf16_fp4e2m1_scalar() {
entry:
  %0 = call i4 @_Z38__builtin_spirv_ConvertBF16ToE2M1INTELDF16b(bfloat 1.0)
  ret i4 %0
}

declare dso_local spir_func i4 @_Z38__builtin_spirv_ConvertBF16ToE2M1INTELDF16b(bfloat)

; CHECK-SPIRV: Function [[#]] [[#bf16_fp4e2m1_vector]] [[#]]
; CHECK-SPIRV: FConvert [[#E2M1VecTy]] [[#Conv:]] [[#BfloatVecConst]]
; CHECK-SPIRV: Bitcast [[#Int4VecTy]] [[#Cast:]] [[#Conv]]
; CHECK-SPIRV: ReturnValue [[#Cast]]

; CHECK-LLVM-LABEL: bf16_fp4e2m1_vector
; CHECK-LLVM: %[[#Call:]] = call <8 x i4> @_Z38__builtin_spirv_ConvertBF16ToE2M1INTELDv8_DF16b(<8 x bfloat> splat (bfloat 0xR3F80))
; CHECK-LLVM: ret <8 x i4> %[[#Call]]

define spir_func <8 x i4> @bf16_fp4e2m1_vector() {
entry:
  %0 = call <8 x i4> @_Z38__builtin_spirv_ConvertBF16ToE2M1INTELDv8_DF16b(<8 x bfloat> <bfloat 1.0, bfloat 1.0, bfloat 1.0, bfloat 1.0, bfloat 1.0, bfloat 1.0, bfloat 1.0, bfloat 1.0>)
  ret <8 x i4> %0
}

declare dso_local spir_func <8 x i4> @_Z38__builtin_spirv_ConvertBF16ToE2M1INTELDv8_DF16b(<8 x bfloat>)

