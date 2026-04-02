; This tests checks if FP8 scalar and vector conversions specified by
; __builtin_spirv_* external function calls translated correctly.

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv --spirv-ext=+SPV_EXT_float8,+SPV_KHR_bfloat16
; RUN: llvm-spirv %t.spv -o %t.spt --to-text
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.spv -o %t.rev.bc -r --spirv-target-env=SPV-IR
; RUN: llvm-dis %t.rev.bc -o %t.rev.ll
; RUN: FileCheck < %t.rev.ll %s --check-prefix=CHECK-LLVM

; TODO: RUNx: spirv-val

; CHECK-SPIRV-DAG: Capability Float8EXT

; CHECK-SPIRV-DAG: Extension "SPV_EXT_float8"

; CHECK-SPIRV-DAG: Name [[#e4m3_hf16_scalar:]] "e4m3_hf16_scalar"
; CHECK-SPIRV-DAG: Name [[#e4m3_hf16_vector:]] "e4m3_hf16_vector"
; CHECK-SPIRV-DAG: Name [[#e5m2_hf16_scalar:]] "e5m2_hf16_scalar"
; CHECK-SPIRV-DAG: Name [[#e5m2_hf16_vector:]] "e5m2_hf16_vector"
; CHECK-SPIRV-DAG: Name [[#e4m3_bf16_scalar:]] "e4m3_bf16_scalar"
; CHECK-SPIRV-DAG: Name [[#e4m3_bf16_vector:]] "e4m3_bf16_vector"
; CHECK-SPIRV-DAG: Name [[#e5m2_bf16_scalar:]] "e5m2_bf16_scalar"
; CHECK-SPIRV-DAG: Name [[#e5m2_bf16_vector:]] "e5m2_bf16_vector"
; CHECK-SPIRV-DAG: Name [[#hf16_e4m3_scalar:]] "hf16_e4m3_scalar"
; CHECK-SPIRV-DAG: Name [[#hf16_e4m3_vector:]] "hf16_e4m3_vector"
; CHECK-SPIRV-DAG: Name [[#hf16_e5m2_scalar:]] "hf16_e5m2_scalar"
; CHECK-SPIRV-DAG: Name [[#hf16_e5m2_vector:]] "hf16_e5m2_vector"
; CHECK-SPIRV-DAG: Name [[#bf16_e4m3_scalar:]] "bf16_e4m3_scalar"
; CHECK-SPIRV-DAG: Name [[#bf16_e4m3_vector:]] "bf16_e4m3_vector"
; CHECK-SPIRV-DAG: Name [[#bf16_e5m2_scalar:]] "bf16_e5m2_scalar"
; CHECK-SPIRV-DAG: Name [[#bf16_e5m2_vector:]] "bf16_e5m2_vector"

; CHECK-SPIRV-DAG: TypeInt [[#Int8Ty:]] 8 0
; CHECK-SPIRV-DAG: TypeVector [[#Int8VecTy:]] [[#Int8Ty]] 8
; CHECK-SPIRV-DAG: Constant [[#Int8Ty]] [[#Int8Const:]] 1
; CHECK-SPIRV-DAG: ConstantComposite [[#Int8VecTy]] [[#Int8VecConst:]] [[#Int8Const]] [[#Int8Const]] [[#Int8Const]] [[#Int8Const]] [[#Int8Const]] [[#Int8Const]] [[#Int8Const]] [[#Int8Const]]

; CHECK-SPIRV-DAG: TypeFloat [[#E4M3Ty:]] 8 4214
; CHECK-SPIRV-DAG: TypeVector [[#E4M3VecTy:]] [[#E4M3Ty]] 8

; CHECK-SPIRV-DAG: TypeFloat [[#E5M2Ty:]] 8 4215
; CHECK-SPIRV-DAG: TypeVector [[#E5M2VecTy:]] [[#E5M2Ty]] 8

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

; CHECK-SPIRV: Function [[#]] [[#e4m3_hf16_scalar]] [[#]]
; CHECK-SPIRV: Bitcast [[#E4M3Ty]] [[#Cast1:]] [[#Int8Const]]
; CHECK-SPIRV: FConvert [[#HFloat16Ty]] [[#Conv:]] [[#Cast1]]
; CHECK-SPIRV: ReturnValue [[#Conv]]

; CHECK-LLVM-LABEL: e4m3_hf16_scalar
; CHECK-LLVM: %[[#Call:]] = call half @_Z36__builtin_spirv_ConvertE4M3ToFP16EXTc(i8 1)
; CHECK-LLVM: ret half %[[#Call]]

define spir_func half @e4m3_hf16_scalar() {
entry:
  %0 = call half @_Z36__builtin_spirv_ConvertE4M3ToFP16EXTc(i8 1)
  ret half %0
}

declare dso_local spir_func half @_Z36__builtin_spirv_ConvertE4M3ToFP16EXTc(i8)

; CHECK-SPIRV: Function [[#]] [[#e4m3_hf16_vector]] [[#]]
; CHECK-SPIRV: Bitcast [[#E4M3VecTy]] [[#Cast1:]] [[#Int8VecConst]]
; CHECK-SPIRV: FConvert [[#HFloat16VecTy]] [[#Conv:]] [[#Cast1]]
; CHECK-SPIRV: ReturnValue [[#Conv]]

; CHECK-LLVM-LABEL: e4m3_hf16_vector
; CHECK-LLVM: %[[#Call:]] = call <8 x half> @_Z36__builtin_spirv_ConvertE4M3ToFP16EXTDv8_c(<8 x i8> splat (i8 1))
; CHECK-LLVM: ret <8 x half> %[[#Call]]

define spir_func <8 x half> @e4m3_hf16_vector() {
entry:
  %0 = call <8 x half> @_Z36__builtin_spirv_ConvertE4M3ToFP16EXTDv8_i(<8 x i8> <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>)
  ret <8 x half> %0
}

declare dso_local spir_func <8 x half> @_Z36__builtin_spirv_ConvertE4M3ToFP16EXTDv8_i(<8 x i8>)

; CHECK-SPIRV: Function [[#]] [[#e5m2_hf16_scalar]] [[#]]
; CHECK-SPIRV: Bitcast [[#E5M2Ty]] [[#Cast1:]] [[#Int8Const]]
; CHECK-SPIRV: FConvert [[#HFloat16Ty]] [[#Conv:]] [[#Cast1]]
; CHECK-SPIRV: ReturnValue [[#Conv]]

; CHECK-LLVM-LABEL: e5m2_hf16_scalar
; CHECK-LLVM: %[[#Call:]] = call half @_Z36__builtin_spirv_ConvertE5M2ToFP16EXTc(i8 1)
; CHECK-LLVM: ret half %[[#Call]]

define spir_func half @e5m2_hf16_scalar() {
entry:
  %0 = call half @_Z36__builtin_spirv_ConvertE5M2ToFP16EXTc(i8 1)
  ret half %0
}

declare dso_local spir_func half @_Z36__builtin_spirv_ConvertE5M2ToFP16EXTc(i8)

; CHECK-SPIRV: Function [[#]] [[#e5m2_hf16_vector]] [[#]]
; CHECK-SPIRV: Bitcast [[#E5M2VecTy]] [[#Cast1:]] [[#Int8VecConst]]
; CHECK-SPIRV: FConvert [[#HFloat16VecTy]] [[#Conv:]] [[#Cast1]]
; CHECK-SPIRV: ReturnValue [[#Conv]]

; CHECK-LLVM-LABEL: e5m2_hf16_vector
; CHECK-LLVM: %[[#Call:]] = call <8 x half> @_Z36__builtin_spirv_ConvertE5M2ToFP16EXTDv8_c(<8 x i8> splat (i8 1))
; CHECK-LLVM: ret <8 x half> %[[#Call]]

define spir_func <8 x half> @e5m2_hf16_vector() {
entry:
  %0 = call <8 x half> @_Z36__builtin_spirv_ConvertE5M2ToFP16EXTDv8_i(<8 x i8> <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>)
  ret <8 x half> %0
}

declare dso_local spir_func <8 x half> @_Z36__builtin_spirv_ConvertE5M2ToFP16EXTDv8_i(<8 x i8>)

; CHECK-SPIRV: Function [[#]] [[#e4m3_bf16_scalar]] [[#]]
; CHECK-SPIRV: Bitcast [[#E4M3Ty]] [[#Cast1:]] [[#Int8Const]]
; CHECK-SPIRV: FConvert [[#BFloat16Ty]] [[#Conv:]] [[#Cast1]]
; CHECK-SPIRV: ReturnValue [[#Conv]]

; CHECK-LLVM-LABEL: e4m3_bf16_scalar
; CHECK-LLVM: %[[#Call:]] = call bfloat @_Z36__builtin_spirv_ConvertE4M3ToBF16EXTc(i8 1)
; CHECK-LLVM: ret bfloat %[[#Call]]

define spir_func bfloat @e4m3_bf16_scalar() {
entry:
  %0 = call bfloat @_Z36__builtin_spirv_ConvertE4M3ToBF16EXTc(i8 1)
  ret bfloat %0
}

declare dso_local spir_func bfloat @_Z36__builtin_spirv_ConvertE4M3ToBF16EXTc(i8)

; CHECK-SPIRV: Function [[#]] [[#e4m3_bf16_vector]] [[#]]
; CHECK-SPIRV: Bitcast [[#E4M3VecTy]] [[#Cast1:]] [[#Int8VecConst]]
; CHECK-SPIRV: FConvert [[#BFloat16VecTy]] [[#Conv:]] [[#Cast1]]
; CHECK-SPIRV: ReturnValue [[#Conv]]

; CHECK-LLVM-LABEL: e4m3_bf16_vector
; CHECK-LLVM: %[[#Call:]] = call <8 x bfloat> @_Z36__builtin_spirv_ConvertE4M3ToBF16EXTDv8_c(<8 x i8> splat (i8 1))
; CHECK-LLVM: ret <8 x bfloat> %[[#Call]]

define spir_func <8 x bfloat> @e4m3_bf16_vector() {
entry:
  %0 = call <8 x bfloat> @_Z36__builtin_spirv_ConvertE4M3ToBF16EXTDv8_i(<8 x i8> <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>)
  ret <8 x bfloat> %0
}

declare dso_local spir_func <8 x bfloat> @_Z36__builtin_spirv_ConvertE4M3ToBF16EXTDv8_i(<8 x i8>)

; CHECK-SPIRV: Function [[#]] [[#e5m2_bf16_scalar]] [[#]]
; CHECK-SPIRV: Bitcast [[#E5M2Ty]] [[#Cast1:]] [[#Int8Const]]
; CHECK-SPIRV: FConvert [[#BFloat16Ty]] [[#Conv:]] [[#Cast1]]
; CHECK-SPIRV: ReturnValue [[#Conv]]

; CHECK-LLVM-LABEL: e5m2_bf16_scalar
; CHECK-LLVM: %[[#Call:]] = call bfloat @_Z36__builtin_spirv_ConvertE5M2ToBF16EXTc(i8 1)
; CHECK-LLVM: ret bfloat %[[#Call]]

define spir_func bfloat @e5m2_bf16_scalar() {
entry:
  %0 = call bfloat @_Z36__builtin_spirv_ConvertE5M2ToBF16EXTc(i8 1)
  ret bfloat %0
}

declare dso_local spir_func bfloat @_Z36__builtin_spirv_ConvertE5M2ToBF16EXTc(i8)

; CHECK-SPIRV: Function [[#]] [[#e5m2_bf16_vector]] [[#]]
; CHECK-SPIRV: Bitcast [[#E5M2VecTy]] [[#Cast1:]] [[#Int8VecConst]]
; CHECK-SPIRV: FConvert [[#BFloat16VecTy]] [[#Conv:]] [[#Cast1]]
; CHECK-SPIRV: ReturnValue [[#Conv]]

; CHECK-LLVM-LABEL: e5m2_bf16_vector
; CHECK-LLVM: %[[#Call:]] = call <8 x bfloat> @_Z36__builtin_spirv_ConvertE5M2ToBF16EXTDv8_c(<8 x i8> splat (i8 1))
; CHECK-LLVM: ret <8 x bfloat> %[[#Call]]

define spir_func <8 x bfloat> @e5m2_bf16_vector() {
entry:
  %0 = call <8 x bfloat> @_Z36__builtin_spirv_ConvertE5M2ToBF16EXTDv8_i(<8 x i8> <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>)
  ret <8 x bfloat> %0
}

declare dso_local spir_func <8 x bfloat> @_Z36__builtin_spirv_ConvertE5M2ToBF16EXTDv8_i(<8 x i8>)

; CHECK-SPIRV: Function [[#]] [[#hf16_e4m3_scalar]] [[#]]
; CHECK-SPIRV: FConvert [[#E4M3Ty]] [[#Conv:]] [[#HalfConst]]
; CHECK-SPIRV: Bitcast [[#Int8Ty]] [[#Cast1:]] [[#Conv]]
; CHECK-SPIRV: ReturnValue [[#Cast1]]

; CHECK-LLVM-LABEL: hf16_e4m3_scalar
; CHECK-LLVM: %[[#Call:]] = call i8 @_Z36__builtin_spirv_ConvertFP16ToE4M3EXTDh(half 0xH3C00)
; CHECK-LLVM: ret i8 %[[#Call]]

define spir_func i8 @hf16_e4m3_scalar() {
entry:
  %0 = call i8 @_Z36__builtin_spirv_ConvertFP16ToE4M3EXTDh(half 0xH3C00)
  ret i8 %0
}

declare dso_local spir_func i8 @_Z36__builtin_spirv_ConvertFP16ToE4M3EXTDh(half)

; CHECK-SPIRV: Function [[#]] [[#hf16_e4m3_vector]] [[#]]
; CHECK-SPIRV: FConvert [[#E4M3VecTy]] [[#Conv:]] [[#HalfVecConst]]
; CHECK-SPIRV: Bitcast [[#Int8VecTy]] [[#Cast1:]] [[#Conv]]
; CHECK-SPIRV: ReturnValue [[#Cast1]]

; CHECK-LLVM-LABEL: hf16_e4m3_vector
; CHECK-LLVM: %[[#Call:]] = call <8 x i8> @_Z36__builtin_spirv_ConvertFP16ToE4M3EXTDv8_Dh(<8 x half> splat (half 0xH3C00))
; CHECK-LLVM: ret <8 x i8> %[[#Call]]

define spir_func <8 x i8> @hf16_e4m3_vector() {
entry:
  %0 = call <8 x i8> @_Z36__builtin_spirv_ConvertFP16ToE4M3EXTDv8_Dh(<8 x half> <half 0xH3C00, half 0xH3C00, half 0xH3C00, half 0xH3C00, half 0xH3C00, half 0xH3C00, half 0xH3C00, half 0xH3C00>)
  ret <8 x i8> %0
}

declare dso_local spir_func <8 x i8> @_Z36__builtin_spirv_ConvertFP16ToE4M3EXTDv8_Dh(<8 x half>)

; CHECK-SPIRV: Function [[#]] [[#hf16_e5m2_scalar]] [[#]]
; CHECK-SPIRV: FConvert [[#E5M2Ty]] [[#Conv:]] [[#HalfConst]]
; CHECK-SPIRV: Bitcast [[#Int8Ty]] [[#Cast1:]] [[#Conv]]
; CHECK-SPIRV: ReturnValue [[#Cast1]]

; CHECK-LLVM-LABEL: hf16_e5m2_scalar
; CHECK-LLVM: %[[#Call:]] = call i8 @_Z36__builtin_spirv_ConvertFP16ToE5M2EXTDh(half 0xH3C00)
; CHECK-LLVM: ret i8 %[[#Call]]

define spir_func i8 @hf16_e5m2_scalar() {
entry:
  %0 = call i8 @_Z36__builtin_spirv_ConvertFP16ToE5M2EXTDh(half 0xH3C00)
  ret i8 %0
}

declare dso_local spir_func i8 @_Z36__builtin_spirv_ConvertFP16ToE5M2EXTDh(half)

; CHECK-SPIRV: Function [[#]] [[#hf16_e5m2_vector]] [[#]]
; CHECK-SPIRV: FConvert [[#E5M2VecTy]] [[#Conv:]] [[#HalfVecConst]]
; CHECK-SPIRV: Bitcast [[#Int8VecTy]] [[#Cast1:]] [[#Conv]]
; CHECK-SPIRV: ReturnValue [[#Cast1]]

; CHECK-LLVM-LABEL: hf16_e5m2_vector
; CHECK-LLVM: %[[#Call:]] = call <8 x i8> @_Z36__builtin_spirv_ConvertFP16ToE5M2EXTDv8_Dh(<8 x half> splat (half 0xH3C00))
; CHECK-LLVM: ret <8 x i8> %[[#Call]]

define spir_func <8 x i8> @hf16_e5m2_vector() {
entry:
  %0 = call <8 x i8> @_Z36__builtin_spirv_ConvertFP16ToE5M2EXTDv8_Dh(<8 x half> <half 0xH3C00, half 0xH3C00, half 0xH3C00, half 0xH3C00, half 0xH3C00, half 0xH3C00, half 0xH3C00, half 0xH3C00>)
  ret <8 x i8> %0
}

declare dso_local spir_func <8 x i8> @_Z36__builtin_spirv_ConvertFP16ToE5M2EXTDv8_Dh(<8 x half>)

; CHECK-SPIRV: Function [[#]] [[#bf16_e4m3_scalar]] [[#]]
; CHECK-SPIRV: FConvert [[#E4M3Ty]] [[#Conv:]] [[#BfloatConst]]
; CHECK-SPIRV: Bitcast [[#Int8Ty]] [[#Cast1:]] [[#Conv]]
; CHECK-SPIRV: ReturnValue [[#Cast1]]

; CHECK-LLVM-LABEL: bf16_e4m3_scalar
; CHECK-LLVM: %[[#Call:]] = call i8 @_Z36__builtin_spirv_ConvertBF16ToE4M3EXTDF16b(bfloat 0xR3F80)
; CHECK-LLVM: ret i8 %[[#Call]]

define spir_func i8 @bf16_e4m3_scalar() {
entry:
  %0 = call i8 @_Z36__builtin_spirv_ConvertBF16ToE4M3EXTDF16b(bfloat 0xR3F80)
  ret i8 %0
}

declare dso_local spir_func i8 @_Z36__builtin_spirv_ConvertBF16ToE4M3EXTDF16b(bfloat)

; CHECK-SPIRV: Function [[#]] [[#bf16_e4m3_vector]] [[#]]
; CHECK-SPIRV: FConvert [[#E4M3VecTy]] [[#Conv:]] [[#BfloatVecConst]]
; CHECK-SPIRV: Bitcast [[#Int8VecTy]] [[#Cast1:]] [[#Conv]]
; CHECK-SPIRV: ReturnValue [[#Cast1]]

; CHECK-LLVM-LABEL: bf16_e4m3_vector
; CHECK-LLVM: %[[#Call:]] = call <8 x i8> @_Z36__builtin_spirv_ConvertBF16ToE4M3EXTDv8_DF16b(<8 x bfloat> splat (bfloat 0xR3F80))
; CHECK-LLVM: ret <8 x i8> %[[#Call]]

define spir_func <8 x i8> @bf16_e4m3_vector() {
entry:
  %0 = call <8 x i8> @_Z36__builtin_spirv_ConvertBF16ToE4M3EXTDv8_DF16b(<8 x bfloat> <bfloat 0xR3F80, bfloat 0xR3F80, bfloat 0xR3F80, bfloat 0xR3F80, bfloat 0xR3F80, bfloat 0xR3F80, bfloat 0xR3F80, bfloat 0xR3F80>)
  ret <8 x i8> %0
}

declare dso_local spir_func <8 x i8> @_Z36__builtin_spirv_ConvertBF16ToE4M3EXTDv8_DF16b(<8 x bfloat>)

; CHECK-SPIRV: Function [[#]] [[#bf16_e5m2_scalar]] [[#]]
; CHECK-SPIRV: FConvert [[#E5M2Ty]] [[#Conv:]] [[#BfloatConst]]
; CHECK-SPIRV: Bitcast [[#Int8Ty]] [[#Cast1:]] [[#Conv]]
; CHECK-SPIRV: ReturnValue [[#Cast1]]

; CHECK-LLVM-LABEL: bf16_e5m2_scalar
; CHECK-LLVM: %[[#Call:]] = call i8 @_Z36__builtin_spirv_ConvertBF16ToE5M2EXTDF16b(bfloat 0xR3F80)
; CHECK-LLVM: ret i8 %[[#Call]]

define spir_func i8 @bf16_e5m2_scalar() {
entry:
  %0 = call i8 @_Z36__builtin_spirv_ConvertBF16ToE5M2EXTDF16b(bfloat 0xR3F80)
  ret i8 %0
}

declare dso_local spir_func i8 @_Z36__builtin_spirv_ConvertBF16ToE5M2EXTDF16b(bfloat)

; CHECK-SPIRV: Function [[#]] [[#bf16_e5m2_vector]] [[#]]
; CHECK-SPIRV: FConvert [[#E5M2VecTy]] [[#Conv:]] [[#BfloatVecConst]]
; CHECK-SPIRV: Bitcast [[#Int8VecTy]] [[#Cast1:]] [[#Conv]]
; CHECK-SPIRV: ReturnValue [[#Cast1]]

; CHECK-LLVM-LABEL: bf16_e5m2_vector
; CHECK-LLVM: %[[#Call:]] = call <8 x i8> @_Z36__builtin_spirv_ConvertBF16ToE5M2EXTDv8_DF16b(<8 x bfloat> splat (bfloat 0xR3F80))
; CHECK-LLVM: ret <8 x i8> %[[#Call]]

define spir_func <8 x i8> @bf16_e5m2_vector() {
entry:
  %0 = call <8 x i8> @_Z36__builtin_spirv_ConvertBF16ToE5M2EXTDv8_DF16b(<8 x bfloat> <bfloat 0xR3F80, bfloat 0xR3F80, bfloat 0xR3F80, bfloat 0xR3F80, bfloat 0xR3F80, bfloat 0xR3F80, bfloat 0xR3F80, bfloat 0xR3F80>)
  ret <8 x i8> %0
}

declare dso_local spir_func <8 x i8> @_Z36__builtin_spirv_ConvertBF16ToE5M2EXTDv8_DF16b(<8 x bfloat>)

