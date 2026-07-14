; Clamping FP-to-FP8 conversions: when SPV_EXT_float8 is enabled, the
; __builtin_spirv_ClampConvert<Src>To<E4M3|E5M2>INTEL names map to OpFConvert
; decorated with SaturatedToLargestFloat8NormalConversionEXT, and round-trip
; back to the same builtin names.

; RUN: llvm-spirv %s -o %t.spv --spirv-ext=+SPV_EXT_float8,+SPV_KHR_bfloat16
; RUN: spirv-val %t.spv
; RUN: llvm-spirv %t.spv --to-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.spv -r -o - | llvm-dis | FileCheck %s --check-prefix=CHECK-LLVM

; CHECK-SPIRV-DAG: Capability Float16Buffer
; CHECK-SPIRV-DAG: Capability Float8EXT
; CHECK-SPIRV-DAG: Extension "SPV_EXT_float8"

; CHECK-SPIRV-DAG: TypeFloat [[#HalfTy:]] 16 {{$}}
; CHECK-SPIRV-DAG: TypeFloat [[#BF16Ty:]] 16 0
; CHECK-SPIRV-DAG: TypeFloat [[#E4M3Ty:]] 8 4214
; CHECK-SPIRV-DAG: TypeFloat [[#E5M2Ty:]] 8 4215

; CHECK-SPIRV-DAG: Decorate [[#E4M3FromHalf:]] SaturatedToLargestFloat8NormalConversionEXT
; CHECK-SPIRV-DAG: Decorate [[#E5M2FromHalf:]] SaturatedToLargestFloat8NormalConversionEXT
; CHECK-SPIRV-DAG: Decorate [[#E4M3FromBF:]] SaturatedToLargestFloat8NormalConversionEXT
; CHECK-SPIRV-DAG: Decorate [[#E5M2FromBF:]] SaturatedToLargestFloat8NormalConversionEXT

; CHECK-SPIRV: FunctionParameter [[#HalfTy]] [[#H:]]
; CHECK-SPIRV: FunctionParameter [[#BF16Ty]] [[#B:]]
; CHECK-SPIRV: FConvert [[#E4M3Ty]] [[#E4M3FromHalf]] [[#H]]
; CHECK-SPIRV: FConvert [[#E5M2Ty]] [[#E5M2FromHalf]] [[#H]]
; CHECK-SPIRV: FConvert [[#E4M3Ty]] [[#E4M3FromBF]] [[#B]]
; CHECK-SPIRV: FConvert [[#E5M2Ty]] [[#E5M2FromBF]] [[#B]]

; CHECK-LLVM: call spir_func i8 @_Z43__builtin_spirv_ClampConvertFP16ToE4M3INTELDh(half
; CHECK-LLVM: call spir_func i8 @_Z43__builtin_spirv_ClampConvertFP16ToE5M2INTELDh(half
; CHECK-LLVM: call spir_func i8 @_Z43__builtin_spirv_ClampConvertBF16ToE4M3INTELDF16b(bfloat
; CHECK-LLVM: call spir_func i8 @_Z43__builtin_spirv_ClampConvertBF16ToE5M2INTELDF16b(bfloat

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-unknown"

define spir_func void @test(half %h, bfloat %b) {
entry:
  %h_e4m3 = call spir_func i8 @_Z43__builtin_spirv_ClampConvertFP16ToE4M3INTELDh(half %h)
  %h_e5m2 = call spir_func i8 @_Z43__builtin_spirv_ClampConvertFP16ToE5M2INTELDh(half %h)
  %b_e4m3 = call spir_func i8 @_Z43__builtin_spirv_ClampConvertBF16ToE4M3INTELDF16b(bfloat %b)
  %b_e5m2 = call spir_func i8 @_Z43__builtin_spirv_ClampConvertBF16ToE5M2INTELDF16b(bfloat %b)
  store i8 %h_e4m3, ptr addrspace(1) null
  store i8 %h_e5m2, ptr addrspace(1) null
  store i8 %b_e4m3, ptr addrspace(1) null
  store i8 %b_e5m2, ptr addrspace(1) null
  ret void
}

declare spir_func i8 @_Z43__builtin_spirv_ClampConvertFP16ToE4M3INTELDh(half)
declare spir_func i8 @_Z43__builtin_spirv_ClampConvertFP16ToE5M2INTELDh(half)
declare spir_func i8 @_Z43__builtin_spirv_ClampConvertBF16ToE4M3INTELDF16b(bfloat)
declare spir_func i8 @_Z43__builtin_spirv_ClampConvertBF16ToE5M2INTELDF16b(bfloat)
