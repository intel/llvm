; A FPRoundingMode decoration on a conversion to/from Float4E2M1EXT
; must land on the FConvert result, not on the bitcast used to bridge
; the mini-float value to its packed i4 LLVM representation.
; Check that it round-trips through the real builtin rather than
; the generic mangler.

; RUN: llvm-spirv %s -spirv-ext=+SPV_EXT_ocp_microscaling_types,+SPV_INTEL_int4 -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv %t.spv --to-text -o - | FileCheck %s
; RUN: llvm-spirv -r --spirv-target-env=SPV-IR %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc -o - | FileCheck %s --check-prefix=CHECK-LLVM

; CHECK-DAG: Decorate [[#TOCONV:]] FPRoundingMode 0
; CHECK-DAG: FConvert [[#]] [[#TOCONV]] [[#]]
; CHECK-DAG: Bitcast [[#]] [[#]] [[#TOCONV]]

; CHECK-DAG: Decorate [[#FROMCONV:]] FPRoundingMode 0
; CHECK-DAG: Bitcast [[#]] [[#FROMBC:]] [[#]]
; CHECK-DAG: FConvert [[#]] [[#FROMCONV]] [[#FROMBC]]

; CHECK-LLVM-DAG: call spir_func i4 @_Z36__builtin_spirv_ConvertFP16ToE2M1EXTDh(half {{.*}}, !spirv.Decorations ![[#DECO:]]
; CHECK-LLVM-DAG: call spir_func half @_Z36__builtin_spirv_ConvertE2M1ToFP16EXTi(i4 {{.*}}, !spirv.Decorations ![[#DECO]]
; CHECK-LLVM: ![[#DECO]] = !{![[#RTE:]]}
; CHECK-LLVM: ![[#RTE]] = !{i32 39, i32 0}

target triple = "spir64-unknown-unknown"

declare dso_local spir_func i4 @_Z36__builtin_spirv_ConvertFP16ToE2M1EXTDh(half)
declare dso_local spir_func half @_Z36__builtin_spirv_ConvertE2M1ToFP16EXTi(i4)

define spir_func i4 @to_e2m1(half %val) {
  %conv = call i4 @_Z36__builtin_spirv_ConvertFP16ToE2M1EXTDh(half %val), !spirv.Decorations !1
  ret i4 %conv
}

define spir_func half @from_e2m1(i4 %val) {
  %conv = call half @_Z36__builtin_spirv_ConvertE2M1ToFP16EXTi(i4 %val), !spirv.Decorations !1
  ret half %conv
}

!1 = !{!2}
!2 = !{i32 39, i32 0}
