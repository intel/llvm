; This tests checks if FP4 and Int4 packed conversions specified by
; __builtin_spirv_* external function calls translated correctly.
; Not all of the instructions are tested here, only one per the following
; cases:
; 1. from packed FP4 to ... :
;   a. packed in 32-bit
;   b. packed in 8-bit
; 2. to packed FP4 from ... :
;   a. packed in 32-bit
;   b. packed in 8-bit

; RUN: llvm-spirv %s -o %t.spv --spirv-ext=+SPV_EXT_float8,+SPV_INTEL_float4,+SPV_INTEL_int4,+SPV_KHR_bfloat16
; RUN: llvm-spirv %t.spv -o %t.spt --to-text
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.spv -o %t.rev.bc -r --spirv-target-env=SPV-IR
; RUN: llvm-dis %t.rev.bc -o %t.rev.ll
; RUN: FileCheck < %t.rev.ll %s --check-prefix=CHECK-LLVM

; CHECK-SPIRV-DAG: Capability Float8EXT
; CHECK-SPIRV-DAG: Capability Float4E2M1INTEL
; CHECK-SPIRV-DAG: Extension "SPV_INTEL_float4"
; CHECK-SPIRV-DAG: Extension "SPV_EXT_float8"

; CHECK-SPIRV-DAG: Name [[#fp4e2m1_hf8_32:]] "fp4e2m1_hf8_32"
; CHECK-SPIRV-DAG: Name [[#fp4e2m1_hf8_8:]] "fp4e2m1_hf8_8"
; CHECK-SPIRV-DAG: Name [[#hf16_fp4e2m1_32:]] "hf16_fp4e2m1_32"
; CHECK-SPIRV-DAG: Name [[#hf16_fp4e2m1_8:]] "hf16_fp4e2m1_8"

; CHECK-SPIRV-DAG: TypeInt [[#Int32Ty:]] 32 0
; CHECK-SPIRV-DAG: Constant [[#Int32Ty]] [[#Int32Const:]] 1

; CHECK-SPIRV-DAG: TypeInt [[#Int8Ty:]] 8 0
; CHECK-SPIRV-DAG: TypeVector [[#Int8Vec8Ty:]] [[#Int8Ty]] 8
; CHECK-SPIRV-DAG: TypeVector [[#Int8Vec2Ty:]] [[#Int8Ty]] 2
; CHECK-SPIRV-DAG: Constant [[#Int8Ty]] [[#Int8Const:]] 1

; CHECK-SPIRV-DAG: TypeFloat [[#E2M1Ty:]] 4 6214
; CHECK-SPIRV-DAG: TypeVector [[#E2M1Vec8Ty:]] [[#E2M1Ty]] 8
; CHECK-SPIRV-DAG: TypeVector [[#E2M1Vec2Ty:]] [[#E2M1Ty]] 2

; CHECK-SPIRV-DAG: TypeFloat [[#Float8E4M3Ty:]] 8 4214
; CHECK-SPIRV-DAG: TypeVector [[#Float8E4M3Vec8Ty:]] [[#Float8E4M3Ty]] 8
; CHECK-SPIRV-DAG: TypeVector [[#Float8E4M3Vec2Ty:]] [[#Float8E4M3Ty]] 2

; CHECK-SPIRV-DAG: TypeFloat [[#HFloat16Ty:]] 16 {{$}}
; CHECK-SPIRV-DAG: TypeVector [[#HFloat16Vec8Ty:]] [[#HFloat16Ty]] 8
; CHECK-SPIRV-DAG: TypeVector [[#HFloat16Vec2Ty:]] [[#HFloat16Ty]] 2
; CHECK-SPIRV-DAG: Constant [[#HFloat16Ty]] [[#HFloat16Const:]] 15360
; CHECK-SPIRV-DAG: ConstantComposite [[#HFloat16Vec8Ty]] [[#HFloat16Vec8Const:]] [[#HFloat16Const]] [[#HFloat16Const]] [[#HFloat16Const]] [[#HFloat16Const]] [[#HFloat16Const]] [[#HFloat16Const]] [[#HFloat16Const]] [[#HFloat16Const]]
; CHECK-SPIRV-DAG: ConstantComposite [[#HFloat16Vec2Ty]] [[#HFloat16Vec2Const:]] [[#HFloat16Const]] [[#HFloat16Const]]

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

; Packed in 32-bit integer

; CHECK-SPIRV: Function [[#]] [[#fp4e2m1_hf8_32]] [[#]]
; CHECK-SPIRV: Bitcast [[#E2M1Vec8Ty]] [[#Cast1:]] [[#Int32Const]]
; CHECK-SPIRV: FConvert [[#Float8E4M3Vec8Ty]] [[#Conv:]] [[#Cast1]]
; CHECK-SPIRV: Bitcast [[#Int8Vec8Ty]] [[#Cast2:]] [[#Conv]]
; CHECK-SPIRV: ReturnValue [[#Cast2]]

; CHECK-LLVM-LABEL: fp4e2m1_hf8_32
; CHECK-LLVM: %[[#Cast:]] = bitcast i32 1 to <8 x i4>
; CHECK-LLVM: %[[#Call:]] = call <8 x i8> @_Z38__builtin_spirv_ConvertE2M1ToE4M3INTELDv8_i(<8 x i4> %[[#Cast]])
; CHECK-LLVM: ret <8 x i8> %[[#Call]]

define spir_func <8 x i8> @fp4e2m1_hf8_32() {
entry:
  %0 = call spir_func <8 x i8> @_Z38__builtin_spirv_ConvertE2M1ToE4M3INTELi(i32 1)
  ret <8 x i8> %0
}

declare dso_local spir_func <8 x i8> @_Z38__builtin_spirv_ConvertE2M1ToE4M3INTELi(i32)

; CHECK-SPIRV: Function [[#]] [[#hf16_fp4e2m1_32]] [[#]]
; CHECK-SPIRV: FConvert [[#E2M1Vec8Ty]] [[#Conv:]] [[#HFloat16Vec8Const]]
; CHECK-SPIRV: Bitcast [[#Int32Ty]] [[#Cast2:]] [[#Conv]]
; CHECK-SPIRV: ReturnValue [[#Cast2]]

; CHECK-LLVM-LABEL: hf16_fp4e2m1_32
; CHECK-LLVM: %[[#Call:]] = call <8 x i4> @_Z38__builtin_spirv_ConvertFP16ToE2M1INTELDv8_Dh(<8 x half> splat (half 0xH3C00))
; CHECK-LLVM: %[[#Cast:]] = bitcast <8 x i4> %[[#Call]] to i32
; CHECK-LLVM: ret i32 %[[#Cast]]

define spir_func i32 @hf16_fp4e2m1_32() {
entry:
  %0 = call i32 @_Z38__builtin_spirv_ConvertFP16ToE2M1INTELDv8_Dh(<8 x half> <half 1.0, half 1.0, half 1.0, half 1.0, half 1.0, half 1.0, half 1.0, half 1.0>)
  ret i32 %0
}

declare dso_local spir_func i32 @_Z38__builtin_spirv_ConvertFP16ToE2M1INTELDv8_Dh(<8 x half>)

; Packed in 8-bit integer

; CHECK-SPIRV: Function [[#]] [[#fp4e2m1_hf8_8]] [[#]]
; CHECK-SPIRV: Bitcast [[#E2M1Vec2Ty]] [[#Cast1:]] [[#Int8Const]]
; CHECK-SPIRV: FConvert [[#Float8E4M3Vec2Ty]] [[#Conv:]] [[#Cast1]]
; CHECK-SPIRV: Bitcast [[#Int8Vec2Ty]] [[#Cast2:]] [[#Conv]]
; CHECK-SPIRV: ReturnValue [[#Cast2]]

; CHECK-LLVM-LABEL: fp4e2m1_hf8_8
; CHECK-LLVM: %[[#Cast:]] = bitcast i8 1 to <2 x i4>
; CHECK-LLVM: %[[#Call:]] = call <2 x i8> @_Z38__builtin_spirv_ConvertE2M1ToE4M3INTELDv2_i(<2 x i4> %[[#Cast]])
; CHECK-LLVM: ret <2 x i8> %[[#Call]]

define spir_func <2 x i8> @fp4e2m1_hf8_8() {
entry:
  %0 = call spir_func <2 x i8> @_Z38__builtin_spirv_ConvertE2M1ToE4M3INTELc(i8 1)
  ret <2 x i8> %0
}

declare dso_local spir_func <2 x i8> @_Z38__builtin_spirv_ConvertE2M1ToE4M3INTELc(i8)

; CHECK-SPIRV: Function [[#]] [[#hf16_fp4e2m1_8]] [[#]]
; CHECK-SPIRV: FConvert [[#E2M1Vec2Ty]] [[#Conv:]] [[#HFloat16Vec2Const]]
; CHECK-SPIRV: Bitcast [[#Int8Ty]] [[#Cast2:]] [[#Conv]]
; CHECK-SPIRV: ReturnValue [[#Cast2]]

; CHECK-LLVM-LABEL: hf16_fp4e2m1_8
; CHECK-LLVM: %[[#Call:]] = call <2 x i4> @_Z38__builtin_spirv_ConvertFP16ToE2M1INTELDv2_Dh(<2 x half> splat (half 0xH3C00))
; CHECK-LLVM: %[[#Cast:]] = bitcast <2 x i4> %[[#Call]] to i8
; CHECK-LLVM: ret i8 %[[#Cast]]

define spir_func i8 @hf16_fp4e2m1_8() {
entry:
  %0 = call i8 @_Z38__builtin_spirv_ConvertFP16ToE2M1INTELDv2_Dh(<2 x half> <half 1.0, half 1.0>)
  ret i8 %0
}

declare dso_local spir_func i8 @_Z38__builtin_spirv_ConvertFP16ToE2M1INTELDv2_Dh(<2 x half>)
