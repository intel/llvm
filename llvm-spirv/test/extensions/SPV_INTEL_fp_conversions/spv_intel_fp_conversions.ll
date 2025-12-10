; This tests checks if FP4 and FP8 conversions specified by
; __builtin_spirv_* external function calls translated correctly.
; This test is for Clamp*, Stochastic*, ClampStochastic* conversions.
; Packed and vector conversions are tested for general case, this test is only
; for scalar

; RUN: llvm-spirv %s -o %t.spv --spirv-ext=+SPV_EXT_float8,+SPV_INTEL_float4,+SPV_INTEL_int4,+SPV_KHR_bfloat16,+SPV_INTEL_fp_conversions
; RUN: llvm-spirv %t.spv -o %t.spt --to-text
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.spv -o %t.rev.bc -r
; RUN: llvm-dis %t.rev.bc -o %t.rev.ll
; RUN: FileCheck < %t.rev.ll %s --check-prefix=CHECK-LLVM

; CHECK-SPIRV-DAG: Capability Int4TypeINTEL
; CHECK-SPIRV-DAG: Capability Float8EXT
; CHECK-SPIRV-DAG: Capability FloatConversionsINTEL

; CHECK-SPIRV-DAG: Extension "SPV_INTEL_int4"
; CHECK-SPIRV-DAG: Extension "SPV_EXT_float8"
; CHECK-SPIRV-DAG: Extension "SPV_INTEL_float4"
; CHECK-SPIRV-DAG: Extension "SPV_INTEL_fp_conversions"

; CHECK-SPIRV-DAG: Name [[#hf16_hf8_clamp:]] "hf16_hf8_clamp"
; CHECK-SPIRV-DAG: Name [[#hf16_bf8_clamp:]] "hf16_bf8_clamp"
; CHECK-SPIRV-DAG: Name [[#bf16_hf8_clamp:]] "bf16_hf8_clamp"
; CHECK-SPIRV-DAG: Name [[#bf16_bf8_clamp:]] "bf16_bf8_clamp"

; CHECK-SPIRV-DAG: Name [[#hf16_bf8_stochastic:]] "hf16_bf8_stochastic"
; CHECK-SPIRV-DAG: Name [[#hf16_hf8_stochastic:]] "hf16_hf8_stochastic"
; CHECK-SPIRV-DAG: Name [[#bf16_bf8_stochastic:]] "bf16_bf8_stochastic"
; CHECK-SPIRV-DAG: Name [[#bf16_hf8_stochastic:]] "bf16_hf8_stochastic"
; CHECK-SPIRV-DAG: Name [[#hf16_fp4e2m1_stochastic:]] "hf16_fp4e2m1_stochastic"
; CHECK-SPIRV-DAG: Name [[#bf16_fp4e2m1_stochastic:]] "bf16_fp4e2m1_stochastic"
; CHECK-SPIRV-DAG: Name [[#hf16_int4_stochastic:]] "hf16_int4_stochastic"
; CHECK-SPIRV-DAG: Name [[#bf16_int4_stochastic:]] "bf16_int4_stochastic"
; CHECK-SPIRV-DAG: Name [[#hf16_bf8_clamp_stochastic:]] "hf16_bf8_clamp_stochastic"
; CHECK-SPIRV-DAG: Name [[#bf16_bf8_clamp_stochastic:]] "bf16_bf8_clamp_stochastic"

; CHECK-SPIRV-DAG: Name [[#hf16_bf8_stochastic_last_seed:]] "hf16_bf8_stochastic_last_seed"
; CHECK-SPIRV-DAG: Name [[#hf16_int4_stochastic_last_seed:]] "hf16_int4_stochastic_last_seed"
; CHECK-SPIRV-DAG: Name [[#hf16_bf8_clamp_stochastic_last_seed:]] "hf16_bf8_clamp_stochastic_last_seed"

; CHECK-SPIRV-DAG: TypeInt [[#Int8Ty:]] 8 0
; CHECK-SPIRV-DAG: TypeInt [[#Int32Ty:]] 32 0
; CHECK-SPIRV-DAG: Constant [[#Int32Ty]] [[#Int32Const:]] 1
; CHECK-SPIRV-DAG: TypeInt [[#Int4Ty:]] 4 0

; CHECK-SPIRV-DAG: TypeFloat [[#E2M1Ty:]] 4 6214
; CHECK-SPIRV-DAG: TypeFloat [[#HFloat8Ty:]] 8 4214
; CHECK-SPIRV-DAG: TypeFloat [[#BFloat8Ty:]] 8 4215

; CHECK-SPIRV-DAG: TypeFloat [[#HFloat16Ty:]] 16 {{$}}
; CHECK-SPIRV-DAG: Constant [[#HFloat16Ty]] [[#HalfConst:]] 15360

; CHECK-SPIRV-DAG: TypeFloat [[#BFloat16Ty:]] 16 0
; CHECK-SPIRV-DAG: Constant [[#BFloat16Ty]] [[#BfloatConst:]] 16256

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

; Followings tests are for clamp rounding

; CHECK-SPIRV: Function [[#]] [[#hf16_hf8_clamp]] [[#]]
; CHECK-SPIRV: ClampConvertFToFINTEL [[#HFloat8Ty]] [[#Conv:]] [[#HalfConst]]
; CHECK-SPIRV: Bitcast [[#Int8Ty]] [[#Cast:]] [[#Conv]]
; CHECK-SPIRV: ReturnValue [[#Cast]]

; CHECK-LLVM-LABEL: hf16_hf8_clamp
; CHECK-LLVM: %[[#Call:]] = call i8 @_Z43__builtin_spirv_ClampConvertFP16ToE4M3INTELDh(half 0xH3C00)
; CHECK-LLVM: ret i8 %[[#Call]]

define spir_func i8 @hf16_hf8_clamp() {
entry:
  %0 = call i8 @_Z43__builtin_spirv_ClampConvertFP16ToE4M3INTELDh(half 1.0)
  ret i8 %0
}

declare dso_local spir_func i8 @_Z43__builtin_spirv_ClampConvertFP16ToE4M3INTELDh(half)

; CHECK-SPIRV: Function [[#]] [[#hf16_bf8_clamp]] [[#]]
; CHECK-SPIRV: ClampConvertFToFINTEL [[#BFloat8Ty]] [[#Conv:]] [[#HalfConst]]
; CHECK-SPIRV: Bitcast [[#Int8Ty]] [[#Cast:]] [[#Conv]]
; CHECK-SPIRV: ReturnValue [[#Cast]]

; CHECK-LLVM-LABEL: hf16_bf8_clamp
; CHECK-LLVM: %[[#Call:]] = call i8 @_Z43__builtin_spirv_ClampConvertFP16ToE5M2INTELDh(half 0xH3C00)
; CHECK-LLVM: ret i8 %[[#Call]]

define spir_func i8 @hf16_bf8_clamp() {
entry:
  %0 = call i8 @_Z43__builtin_spirv_ClampConvertFP16ToE5M2INTELDh(half 1.0)
  ret i8 %0
}

declare dso_local spir_func i8 @_Z43__builtin_spirv_ClampConvertFP16ToE5M2INTELDh(half)

; CHECK-SPIRV: Function [[#]] [[#bf16_hf8_clamp]] [[#]]
; CHECK-SPIRV: ClampConvertFToFINTEL [[#HFloat8Ty]] [[#Conv:]] [[#BfloatConst]]
; CHECK-SPIRV: Bitcast [[#Int8Ty]] [[#Cast:]] [[#Conv]]
; CHECK-SPIRV: ReturnValue [[#Cast]]

; CHECK-LLVM-LABEL: bf16_hf8_clamp
; CHECK-LLVM: %[[#Call:]] = call i8 @_Z43__builtin_spirv_ClampConvertBF16ToE4M3INTELDF16b(bfloat 0xR3F80)
; CHECK-LLVM: ret i8 %[[#Call]]

define spir_func i8 @bf16_hf8_clamp() {
entry:
  %0 = call i8 @_Z43__builtin_spirv_ClampConvertBF16ToE4M3INTELDF16b(bfloat 1.0)
  ret i8 %0
}

declare dso_local spir_func i8 @_Z43__builtin_spirv_ClampConvertBF16ToE4M3INTELDF16b(bfloat)

; CHECK-SPIRV: Function [[#]] [[#bf16_bf8_clamp]] [[#]]
; CHECK-SPIRV: ClampConvertFToFINTEL [[#BFloat8Ty]] [[#Conv:]] [[#BfloatConst]]
; CHECK-SPIRV: Bitcast [[#Int8Ty]] [[#Cast:]] [[#Conv]]
; CHECK-SPIRV: ReturnValue [[#Cast]]

; CHECK-LLVM-LABEL: bf16_bf8_clamp
; CHECK-LLVM: %[[#Call:]] = call i8 @_Z43__builtin_spirv_ClampConvertBF16ToE5M2INTELDF16b(bfloat 0xR3F80)
; CHECK-LLVM: ret i8 %[[#Call]]

define spir_func i8 @bf16_bf8_clamp() {
entry:
  %0 = call i8 @_Z43__builtin_spirv_ClampConvertBF16ToE5M2INTELDF16b(bfloat 1.0)
  ret i8 %0
}

declare dso_local spir_func i8 @_Z43__builtin_spirv_ClampConvertBF16ToE5M2INTELDF16b(bfloat)

; CHECK-SPIRV: Function [[#]] [[#hf16_bf8_stochastic]] [[#]]
; CHECK-SPIRV: StochasticRoundFToFINTEL [[#BFloat8Ty]] [[#Conv:]] [[#HalfConst]] [[#Int32Const]]
; CHECK-SPIRV: Bitcast [[#Int8Ty]] [[#Cast:]] [[#Conv]]
; CHECK-SPIRV: ReturnValue [[#Cast]]

; CHECK-LLVM-LABEL: hf16_bf8_stochastic
; CHECK-LLVM: %[[#Call:]] = call i8 @_Z46__builtin_spirv_StochasticRoundFP16ToE5M2INTELDhi(half 0xH3C00, i32 1)
; CHECK-LLVM: ret i8 %[[#Call]]

define spir_func i8 @hf16_bf8_stochastic() {
entry:
  %0 = call i8 @_Z46__builtin_spirv_StochasticRoundFP16ToE5M2INTELDhi(half 1.0, i32 1)
  ret i8 %0
}

declare dso_local spir_func i8 @_Z46__builtin_spirv_StochasticRoundFP16ToE5M2INTELDhi(half, i32)

; CHECK-SPIRV: Function [[#]] [[#hf16_hf8_stochastic]] [[#]]
; CHECK-SPIRV: StochasticRoundFToFINTEL [[#HFloat8Ty]] [[#Conv:]] [[#HalfConst]] [[#Int32Const]]
; CHECK-SPIRV: Bitcast [[#Int8Ty]] [[#Cast:]] [[#Conv]]
; CHECK-SPIRV: ReturnValue [[#Cast]]

; CHECK-LLVM-LABEL: hf16_hf8_stochastic
; CHECK-LLVM: %[[#Call:]] = call i8 @_Z46__builtin_spirv_StochasticRoundFP16ToE4M3INTELDhi(half 0xH3C00, i32 1)
; CHECK-LLVM: ret i8 %[[#Call]]

define spir_func i8 @hf16_hf8_stochastic() {
entry:
  %0 = call i8 @_Z46__builtin_spirv_StochasticRoundFP16ToE4M3INTELDhi(half 1.0, i32 1)
  ret i8 %0
}

declare dso_local spir_func i8 @_Z46__builtin_spirv_StochasticRoundFP16ToE4M3INTELDhi(half, i32)

; CHECK-SPIRV: Function [[#]] [[#bf16_bf8_stochastic]] [[#]]
; CHECK-SPIRV: StochasticRoundFToFINTEL [[#BFloat8Ty]] [[#Conv:]] [[#BfloatConst]] [[#Int32Const]]
; CHECK-SPIRV: Bitcast [[#Int8Ty]] [[#Cast:]] [[#Conv]]
; CHECK-SPIRV: ReturnValue [[#Cast]]

; CHECK-LLVM-LABEL: bf16_bf8_stochastic
; CHECK-LLVM: %[[#Call:]] = call i8 @_Z46__builtin_spirv_StochasticRoundBF16ToE5M2INTELDF16bi(bfloat 0xR3F80, i32 1)
; CHECK-LLVM: ret i8 %[[#Call]]

define spir_func i8 @bf16_bf8_stochastic() {
entry:
  %0 = call i8 @_Z46__builtin_spirv_StochasticRoundBF16ToE5M2INTELDF16bi(bfloat 1.0, i32 1)
  ret i8 %0
}

declare dso_local spir_func i8 @_Z46__builtin_spirv_StochasticRoundBF16ToE5M2INTELDF16bi(bfloat, i32)

; CHECK-SPIRV: Function [[#]] [[#bf16_hf8_stochastic]] [[#]]
; CHECK-SPIRV: StochasticRoundFToFINTEL [[#HFloat8Ty]] [[#Conv:]] [[#BfloatConst]] [[#Int32Const]]
; CHECK-SPIRV: Bitcast [[#Int8Ty]] [[#Cast:]] [[#Conv]]
; CHECK-SPIRV: ReturnValue [[#Cast]]

; CHECK-LLVM-LABEL: bf16_hf8_stochastic
; CHECK-LLVM: %[[#Call:]] = call i8 @_Z46__builtin_spirv_StochasticRoundBF16ToE4M3INTELDF16bi(bfloat 0xR3F80, i32 1)
; CHECK-LLVM: ret i8 %[[#Call]]

define spir_func i8 @bf16_hf8_stochastic() {
entry:
  %0 = call i8 @_Z46__builtin_spirv_StochasticRoundBF16ToE4M3INTELDF16bi(bfloat 1.0, i32 1)
  ret i8 %0
}

declare dso_local spir_func i8 @_Z46__builtin_spirv_StochasticRoundBF16ToE4M3INTELDF16bi(bfloat, i32)

; CHECK-SPIRV: Function [[#]] [[#hf16_fp4e2m1_stochastic]] [[#]]
; CHECK-SPIRV: StochasticRoundFToFINTEL [[#E2M1Ty]] [[#Conv:]] [[#HalfConst]] [[#Int32Const]]
; CHECK-SPIRV: Bitcast [[#Int4Ty]] [[#Cast:]] [[#Conv]]
; CHECK-SPIRV: ReturnValue [[#Cast]]

; CHECK-LLVM-LABEL: hf16_fp4e2m1_stochastic
; CHECK-LLVM: %[[#Call:]] = call i4 @_Z46__builtin_spirv_StochasticRoundFP16ToE2M1INTELDhi(half 0xH3C00, i32 1)
; CHECK-LLVM: ret i4 %[[#Call]]

define spir_func i4 @hf16_fp4e2m1_stochastic() {
entry:
  %0 = call i4 @_Z46__builtin_spirv_StochasticRoundFP16ToE2M1INTELDhi(half 1.0, i32 1)
  ret i4 %0
}

declare dso_local spir_func i4 @_Z46__builtin_spirv_StochasticRoundFP16ToE2M1INTELDhi(half, i32)

; CHECK-SPIRV: Function [[#]] [[#bf16_fp4e2m1_stochastic]] [[#]]
; CHECK-SPIRV: StochasticRoundFToFINTEL [[#E2M1Ty]] [[#Conv:]] [[#BfloatConst]] [[#Int32Const]]
; CHECK-SPIRV: Bitcast [[#Int4Ty]] [[#Cast:]] [[#Conv]]
; CHECK-SPIRV: ReturnValue [[#Cast]]

; CHECK-LLVM-LABEL: bf16_fp4e2m1_stochastic
; CHECK-LLVM: %[[#Call:]] = call i4 @_Z46__builtin_spirv_StochasticRoundBF16ToE2M1INTELDF16bi(bfloat 0xR3F80, i32 1)
; CHECK-LLVM: ret i4 %[[#Call]]

define spir_func i4 @bf16_fp4e2m1_stochastic() {
entry:
  %0 = call i4 @_Z46__builtin_spirv_StochasticRoundBF16ToE2M1INTELDF16bi(bfloat 1.0, i32 1)
  ret i4 %0
}

declare dso_local spir_func i4 @_Z46__builtin_spirv_StochasticRoundBF16ToE2M1INTELDF16bi(bfloat, i32)

; CHECK-SPIRV: Function [[#]] [[#hf16_int4_stochastic]] [[#]]
; CHECK-SPIRV: ClampStochasticRoundFToSINTEL [[#Int4Ty]] [[#Conv:]] [[#HalfConst]] [[#Int32Const]]
; CHECK-SPIRV: ReturnValue [[#Conv]]

; CHECK-LLVM-LABEL: hf16_int4_stochastic
; CHECK-LLVM: %[[#Call:]] = call i4 @_Z51__builtin_spirv_ClampStochasticRoundFP16ToInt4INTELDhi(half 0xH3C00, i32 1)
; CHECK-LLVM: ret i4 %[[#Call]]

define spir_func i4 @hf16_int4_stochastic() {
entry:
  %0 = call i4 @_Z51__builtin_spirv_ClampStochasticRoundFP16ToInt4INTELhs(half 1.0, i32 1)
  ret i4 %0
}

declare dso_local spir_func i4 @_Z51__builtin_spirv_ClampStochasticRoundFP16ToInt4INTELhs(half, i32)

; CHECK-SPIRV: Function [[#]] [[#bf16_int4_stochastic]] [[#]]
; CHECK-SPIRV: ClampStochasticRoundFToSINTEL [[#Int4Ty]] [[#Conv:]] [[#BfloatConst]] [[#Int32Const]]
; CHECK-SPIRV: ReturnValue [[#Conv]]

; CHECK-LLVM-LABEL: bf16_int4_stochastic
; CHECK-LLVM: %[[#Call:]] = call i4 @_Z51__builtin_spirv_ClampStochasticRoundBF16ToInt4INTELDF16bi(bfloat 0xR3F80, i32 1)
; CHECK-LLVM: ret i4 %[[#Call]]

define spir_func i4 @bf16_int4_stochastic() {
entry:
  %0 = call i4 @_Z51__builtin_spirv_ClampStochasticRoundBF16ToInt4INTELDF16bi(bfloat 1.0, i32 1)
  ret i4 %0
}

declare dso_local spir_func i4 @_Z51__builtin_spirv_ClampStochasticRoundBF16ToInt4INTELDF16bi(bfloat, i32)

; CHECK-SPIRV: Function [[#]] [[#hf16_bf8_clamp_stochastic]] [[#]]
; CHECK-SPIRV: ClampStochasticRoundFToFINTEL [[#BFloat8Ty]] [[#Conv:]] [[#HalfConst]] [[#Int32Const]]
; CHECK-SPIRV: Bitcast [[#Int8Ty]] [[#Cast:]] [[#Conv]]
; CHECK-SPIRV: ReturnValue [[#Cast]]

; CHECK-LLVM-LABEL: hf16_bf8_clamp_stochastic
; CHECK-LLVM: %[[#Call:]] = call i8 @_Z51__builtin_spirv_ClampStochasticRoundFP16ToE5M2INTELDhi(half 0xH3C00, i32 1)
; CHECK-LLVM: ret i8 %[[#Call]]

define spir_func i8 @hf16_bf8_clamp_stochastic() {
entry:
  %0 = call i8 @_Z51__builtin_spirv_ClampStochasticRoundFP16ToE5M2INTELDhi(half 1.0, i32 1)
  ret i8 %0
}

declare dso_local spir_func i8 @_Z51__builtin_spirv_ClampStochasticRoundFP16ToE5M2INTELDhi(half, i32)

; CHECK-SPIRV: Function [[#]] [[#bf16_bf8_clamp_stochastic]] [[#]]
; CHECK-SPIRV: ClampStochasticRoundFToFINTEL [[#BFloat8Ty]] [[#Conv:]] [[#BfloatConst]] [[#Int32Const]]
; CHECK-SPIRV: Bitcast [[#Int8Ty]] [[#Cast:]] [[#Conv]]
; CHECK-SPIRV: ReturnValue [[#Cast]]

; CHECK-LLVM-LABEL: bf16_bf8_clamp_stochastic
; CHECK-LLVM: %[[#Call:]] = call i8 @_Z51__builtin_spirv_ClampStochasticRoundBF16ToE5M2INTELDF16bi(bfloat 0xR3F80, i32 1)
; CHECK-LLVM: ret i8 %[[#Call]]

define spir_func i8 @bf16_bf8_clamp_stochastic() {
entry:
  %0 = call i8 @_Z51__builtin_spirv_ClampStochasticRoundBF16ToE5M2INTELDF16bi(bfloat 1.0, i32 1)
  ret i8 %0
}

declare dso_local spir_func i8 @_Z51__builtin_spirv_ClampStochasticRoundBF16ToE5M2INTELDF16bi(bfloat, i32)

; CHECK-SPIRV: Function [[#]] [[#hf16_bf8_stochastic_last_seed]] [[#]]
; CHECK-SPIRV: Variable [[#]] [[#Ptr:]]
; CHECK-SPIRV: StochasticRoundFToFINTEL [[#BFloat8Ty]] [[#Conv:]] [[#HalfConst]] [[#Int32Const]] [[#Ptr]]
; CHECK-SPIRV: Bitcast [[#Int8Ty]] [[#Cast:]] [[#Conv]]
; CHECK-SPIRV: ReturnValue [[#Cast]]

; CHECK-LLVM-LABEL: hf16_bf8_stochastic_last_seed
; CHECK-LLVM: %[[#Ptr:]] = alloca i32
; CHECK-LLVM: %[[#Call:]] = call i8 @_Z46__builtin_spirv_StochasticRoundFP16ToE5M2INTELDhiPi(half 0xH3C00, i32 1, ptr %[[#Ptr]])
; CHECK-LLVM: ret i8 %[[#Call]]

define spir_func i8 @hf16_bf8_stochastic_last_seed() {
entry:
  %0 = alloca i32
  %1 = call i8 @_Z46__builtin_spirv_StochasticRoundFP16ToE5M2INTELDhiPi(half 1.0, i32 1, ptr %0)
  ret i8 %1
}

declare dso_local spir_func i8 @_Z46__builtin_spirv_StochasticRoundFP16ToE5M2INTELDhiPi(half, i32, ptr)

; CHECK-SPIRV: Function [[#]] [[#hf16_int4_stochastic_last_seed]] [[#]]
; CHECK-SPIRV: Variable [[#]] [[#Ptr:]]
; CHECK-SPIRV: ClampStochasticRoundFToSINTEL [[#Int4Ty]] [[#Conv:]] [[#HalfConst]] [[#Int32Const]] [[#Ptr]]
; CHECK-SPIRV: ReturnValue [[#Conv]]

; CHECK-LLVM-LABEL: hf16_int4_stochastic_last_seed
; CHECK-LLVM: %[[#Ptr:]] = alloca i32
; CHECK-LLVM: %[[#Call:]] = call i4 @_Z51__builtin_spirv_ClampStochasticRoundFP16ToInt4INTELDhiPi(half 0xH3C00, i32 1, ptr %[[#Ptr]])
; CHECK-LLVM: ret i4 %[[#Call]]

define spir_func i4 @hf16_int4_stochastic_last_seed() {
entry:
  %0 = alloca i32
  %1 = call i4 @_Z51__builtin_spirv_ClampStochasticRoundFP16ToInt4INTELhiPi(half 1.0, i32 1, ptr %0)
  ret i4 %1
}

declare dso_local spir_func i4 @_Z51__builtin_spirv_ClampStochasticRoundFP16ToInt4INTELhiPi(half, i32, ptr)

; CHECK-SPIRV: Function [[#]] [[#hf16_bf8_clamp_stochastic_last_seed]] [[#]]
; CHECK-SPIRV: Variable [[#]] [[#Ptr:]]
; CHECK-SPIRV: ClampStochasticRoundFToFINTEL [[#BFloat8Ty]] [[#Conv:]] [[#HalfConst]] [[#Int32Const]] [[#Ptr]]
; CHECK-SPIRV: Bitcast [[#Int8Ty]] [[#Cast:]] [[#Conv]]
; CHECK-SPIRV: ReturnValue [[#Cast]]

; CHECK-LLVM-LABEL: hf16_bf8_clamp_stochastic_last_seed
; CHECK-LLVM: %[[#Ptr:]] = alloca i32
; CHECK-LLVM: %[[#Call:]] = call i8 @_Z51__builtin_spirv_ClampStochasticRoundFP16ToE5M2INTELDhiPi(half 0xH3C00, i32 1, ptr %[[#Ptr]])
; CHECK-LLVM: ret i8 %[[#Call]]

define spir_func i8 @hf16_bf8_clamp_stochastic_last_seed() {
entry:
  %0 = alloca i32
  %1 = call i8 @_Z51__builtin_spirv_ClampStochasticRoundFP16ToE5M2INTELDhiPi(half 1.0, i32 1, ptr %0)
  ret i8 %1
}

declare dso_local spir_func i8 @_Z51__builtin_spirv_ClampStochasticRoundFP16ToE5M2INTELDhiPi(half, i32, ptr)
