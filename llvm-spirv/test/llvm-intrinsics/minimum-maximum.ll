; llvm.minimum/llvm.maximum return NaN if either operand is NaN and order -0.0
; before +0.0. OpenCL.std fmin/fmax instead return the non-NaN operand and may
; return either zero, so the translator adds the missing fix-ups around them,
; skipping the ones made unnecessary by the nnan/nsz fast-math flags.

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv -s %t.bc -o - | llvm-dis -o - | FileCheck %s
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv

; The declarations precede their uses, so they are still used when visited by
; SPIRVRegularizeLLVM. Make sure they don't end up as imported functions.
; RUN: llvm-spirv %t.bc -spirv-allow-unknown-intrinsics -spirv-text -o - \
; RUN:   | FileCheck %s --check-prefix=CHECK-SPIRV \
; RUN:     --implicit-check-not=llvm.minimum --implicit-check-not=llvm.maximum

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

declare float @llvm.minimum.f32(float, float)
declare float @llvm.maximum.f32(float, float)
declare half @llvm.minimum.f16(half, half)
declare double @llvm.maximum.f64(double, double)
declare <4 x float> @llvm.minimum.v4f32(<4 x float>, <4 x float>)
declare <2 x double> @llvm.maximum.v2f64(<2 x double>, <2 x double>)
declare <2 x half> @llvm.minimum.v2f16(<2 x half>, <2 x half>)
declare <2 x half> @llvm.maximum.v2f16(<2 x half>, <2 x half>)

; CHECK-SPIRV-DAG: TypeBool [[#Bool:]]
; CHECK-SPIRV-DAG: TypeFloat [[#Float:]] 32
; CHECK-SPIRV-DAG: Constant [[#Float]] [[#FloatOne:]] 1065353216
; CHECK-SPIRV-DAG: Constant [[#Float]] [[#FloatZero:]] 0{{ *$}}
; CHECK-SPIRV-DAG: Constant [[#Float]] [[#FloatNaN:]] 2143289344

; CHECK-SPIRV: Function [[#Float]] [[#]]
; CHECK-SPIRV-NEXT: FunctionParameter [[#Float]] [[#X:]]
; CHECK-SPIRV-NEXT: FunctionParameter [[#Float]] [[#Y:]]
; CHECK-SPIRV: ExtInst [[#Float]] [[#M:]] [[#]] fmin [[#X]] [[#Y]]
; CHECK-SPIRV-NEXT: ExtInst [[#Float]] [[#XSign:]] [[#]] copysign [[#FloatOne]] [[#X]]
; CHECK-SPIRV-NEXT: FOrdLessThan [[#Bool]] [[#XNeg:]] [[#XSign]] [[#FloatZero]]
; CHECK-SPIRV-NEXT: Select [[#Float]] [[#S:]] [[#XNeg]] [[#X]] [[#Y]]
; CHECK-SPIRV-NEXT: ExtInst [[#Float]] [[#MS:]] [[#]] copysign [[#M]] [[#S]]
; CHECK-SPIRV-NEXT: Unordered [[#Bool]] [[#Uno:]] [[#X]] [[#Y]]
; CHECK-SPIRV-NEXT: Select [[#Float]] [[#Res:]] [[#Uno]] [[#FloatNaN]] [[#MS]]
; CHECK-SPIRV-NEXT: ReturnValue [[#Res]]

; CHECK-LABEL: define spir_func float @minimum_f32(
; CHECK-NEXT: %[[M:[0-9]+]] = call float @llvm.minnum.f32(float %x, float %y)
; CHECK-NEXT: %[[XSIGN:[0-9]+]] = call float @llvm.copysign.f32(float 1.000000e+00, float %x)
; CHECK-NEXT: %[[XNEG:[0-9]+]] = fcmp olt float %[[XSIGN]], 0.000000e+00
; CHECK-NEXT: %[[S:[0-9]+]] = select i1 %[[XNEG]], float %x, float %y
; CHECK-NEXT: %[[MS:[0-9]+]] = call float @llvm.copysign.f32(float %[[M]], float %[[S]])
; CHECK-NEXT: %[[UNO:[0-9]+]] = fcmp uno float %x, %y
; CHECK-NEXT: %r = select i1 %[[UNO]], float +qnan, float %[[MS]]
; CHECK-NEXT: ret float %r
define spir_func float @minimum_f32(float %x, float %y) {
  %r = call float @llvm.minimum.f32(float %x, float %y)
  ret float %r
}

; CHECK-LABEL: define spir_func float @maximum_f32(
; CHECK-NEXT: %[[M:[0-9]+]] = call float @llvm.maxnum.f32(float %x, float %y)
; CHECK-NEXT: %[[XSIGN:[0-9]+]] = call float @llvm.copysign.f32(float 1.000000e+00, float %x)
; CHECK-NEXT: %[[XNEG:[0-9]+]] = fcmp olt float %[[XSIGN]], 0.000000e+00
; CHECK-NEXT: %[[S:[0-9]+]] = select i1 %[[XNEG]], float %y, float %x
; CHECK-NEXT: %[[MS:[0-9]+]] = call float @llvm.copysign.f32(float %[[M]], float %[[S]])
; CHECK-NEXT: %[[UNO:[0-9]+]] = fcmp uno float %x, %y
; CHECK-NEXT: %r = select i1 %[[UNO]], float +qnan, float %[[MS]]
; CHECK-NEXT: ret float %r
define spir_func float @maximum_f32(float %x, float %y) {
  %r = call float @llvm.maximum.f32(float %x, float %y)
  ret float %r
}

; CHECK-LABEL: define spir_func half @minimum_f16(
; CHECK-NEXT: %[[M:[0-9]+]] = call half @llvm.minnum.f16(half %x, half %y)
; CHECK-NEXT: %[[XSIGN:[0-9]+]] = call half @llvm.copysign.f16(half 1.000000e+00, half %x)
; CHECK-NEXT: %[[XNEG:[0-9]+]] = fcmp olt half %[[XSIGN]], 0.000000e+00
; CHECK-NEXT: %[[S:[0-9]+]] = select i1 %[[XNEG]], half %x, half %y
; CHECK-NEXT: %[[MS:[0-9]+]] = call half @llvm.copysign.f16(half %[[M]], half %[[S]])
; CHECK-NEXT: %[[UNO:[0-9]+]] = fcmp uno half %x, %y
; CHECK-NEXT: %r = select i1 %[[UNO]], half +qnan, half %[[MS]]
; CHECK-NEXT: ret half %r
define spir_func half @minimum_f16(half %x, half %y) {
  %r = call half @llvm.minimum.f16(half %x, half %y)
  ret half %r
}

; CHECK-LABEL: define spir_func double @maximum_f64(
; CHECK-NEXT: %[[M:[0-9]+]] = call double @llvm.maxnum.f64(double %x, double %y)
; CHECK-NEXT: %[[XSIGN:[0-9]+]] = call double @llvm.copysign.f64(double 1.000000e+00, double %x)
; CHECK-NEXT: %[[XNEG:[0-9]+]] = fcmp olt double %[[XSIGN]], 0.000000e+00
; CHECK-NEXT: %[[S:[0-9]+]] = select i1 %[[XNEG]], double %y, double %x
; CHECK-NEXT: %[[MS:[0-9]+]] = call double @llvm.copysign.f64(double %[[M]], double %[[S]])
; CHECK-NEXT: %[[UNO:[0-9]+]] = fcmp uno double %x, %y
; CHECK-NEXT: %r = select i1 %[[UNO]], double +qnan, double %[[MS]]
; CHECK-NEXT: ret double %r
define spir_func double @maximum_f64(double %x, double %y) {
  %r = call double @llvm.maximum.f64(double %x, double %y)
  ret double %r
}

; CHECK-LABEL: define spir_func <4 x float> @minimum_v4f32(
; CHECK-NEXT: %[[M:[0-9]+]] = call <4 x float> @llvm.minnum.v4f32(<4 x float> %x, <4 x float> %y)
; CHECK-NEXT: %[[XSIGN:[0-9]+]] = call <4 x float> @llvm.copysign.v4f32(<4 x float> splat (float 1.000000e+00), <4 x float> %x)
; CHECK-NEXT: %[[XNEG:[0-9]+]] = fcmp olt <4 x float> %[[XSIGN]], zeroinitializer
; CHECK-NEXT: %[[S:[0-9]+]] = select <4 x i1> %[[XNEG]], <4 x float> %x, <4 x float> %y
; CHECK-NEXT: %[[MS:[0-9]+]] = call <4 x float> @llvm.copysign.v4f32(<4 x float> %[[M]], <4 x float> %[[S]])
; CHECK-NEXT: %[[UNO:[0-9]+]] = fcmp uno <4 x float> %x, %y
; CHECK-NEXT: %r = select <4 x i1> %[[UNO]], <4 x float> splat (float +qnan), <4 x float> %[[MS]]
; CHECK-NEXT: ret <4 x float> %r
define spir_func <4 x float> @minimum_v4f32(<4 x float> %x, <4 x float> %y) {
  %r = call <4 x float> @llvm.minimum.v4f32(<4 x float> %x, <4 x float> %y)
  ret <4 x float> %r
}

; CHECK-LABEL: define spir_func <2 x double> @maximum_v2f64(
; CHECK-NEXT: %[[M:[0-9]+]] = call <2 x double> @llvm.maxnum.v2f64(<2 x double> %x, <2 x double> %y)
; CHECK-NEXT: %[[XSIGN:[0-9]+]] = call <2 x double> @llvm.copysign.v2f64(<2 x double> splat (double 1.000000e+00), <2 x double> %x)
; CHECK-NEXT: %[[XNEG:[0-9]+]] = fcmp olt <2 x double> %[[XSIGN]], zeroinitializer
; CHECK-NEXT: %[[S:[0-9]+]] = select <2 x i1> %[[XNEG]], <2 x double> %y, <2 x double> %x
; CHECK-NEXT: %[[MS:[0-9]+]] = call <2 x double> @llvm.copysign.v2f64(<2 x double> %[[M]], <2 x double> %[[S]])
; CHECK-NEXT: %[[UNO:[0-9]+]] = fcmp uno <2 x double> %x, %y
; CHECK-NEXT: %r = select <2 x i1> %[[UNO]], <2 x double> splat (double +qnan), <2 x double> %[[MS]]
; CHECK-NEXT: ret <2 x double> %r
define spir_func <2 x double> @maximum_v2f64(<2 x double> %x, <2 x double> %y) {
  %r = call <2 x double> @llvm.maximum.v2f64(<2 x double> %x, <2 x double> %y)
  ret <2 x double> %r
}

; CHECK-LABEL: define spir_func <2 x half> @maximum_v2f16(
; CHECK-NEXT: %[[M:[0-9]+]] = call <2 x half> @llvm.maxnum.v2f16(<2 x half> %x, <2 x half> %y)
; CHECK-NEXT: %[[XSIGN:[0-9]+]] = call <2 x half> @llvm.copysign.v2f16(<2 x half> splat (half 1.000000e+00), <2 x half> %x)
; CHECK-NEXT: %[[XNEG:[0-9]+]] = fcmp olt <2 x half> %[[XSIGN]], zeroinitializer
; CHECK-NEXT: %[[S:[0-9]+]] = select <2 x i1> %[[XNEG]], <2 x half> %y, <2 x half> %x
; CHECK-NEXT: %[[MS:[0-9]+]] = call <2 x half> @llvm.copysign.v2f16(<2 x half> %[[M]], <2 x half> %[[S]])
; CHECK-NEXT: %[[UNO:[0-9]+]] = fcmp uno <2 x half> %x, %y
; CHECK-NEXT: %r = select <2 x i1> %[[UNO]], <2 x half> splat (half +qnan), <2 x half> %[[MS]]
; CHECK-NEXT: ret <2 x half> %r
define spir_func <2 x half> @maximum_v2f16(<2 x half> %x, <2 x half> %y) {
  %r = call <2 x half> @llvm.maximum.v2f16(<2 x half> %x, <2 x half> %y)
  ret <2 x half> %r
}

; nnan: no NaN fix-up.
; CHECK-LABEL: define spir_func float @minimum_f32_nnan(
; CHECK-NEXT: %[[M:[0-9]+]] = call nnan float @llvm.minnum.f32(float %x, float %y)
; CHECK-NEXT: %[[XSIGN:[0-9]+]] = call float @llvm.copysign.f32(float 1.000000e+00, float %x)
; CHECK-NEXT: %[[XNEG:[0-9]+]] = fcmp olt float %[[XSIGN]], 0.000000e+00
; CHECK-NEXT: %[[S:[0-9]+]] = select i1 %[[XNEG]], float %x, float %y
; CHECK-NEXT: %r = call float @llvm.copysign.f32(float %[[M]], float %[[S]])
; CHECK-NEXT: ret float %r
define spir_func float @minimum_f32_nnan(float %x, float %y) {
  %r = call nnan float @llvm.minimum.f32(float %x, float %y)
  ret float %r
}

; nsz: no signed-zero fix-up.
; CHECK-LABEL: define spir_func float @maximum_f32_nsz(
; CHECK-NEXT: %[[M:[0-9]+]] = call nsz float @llvm.maxnum.f32(float %x, float %y)
; CHECK-NEXT: %[[UNO:[0-9]+]] = fcmp uno float %x, %y
; CHECK-NEXT: %r = select i1 %[[UNO]], float +qnan, float %[[M]]
; CHECK-NEXT: ret float %r
define spir_func float @maximum_f32_nsz(float %x, float %y) {
  %r = call nsz float @llvm.maximum.f32(float %x, float %y)
  ret float %r
}

; nnan nsz: plain fmin.
; CHECK-LABEL: define spir_func <2 x half> @minimum_v2f16_nnan_nsz(
; CHECK-NEXT: %r = call nnan nsz <2 x half> @llvm.minnum.v2f16(<2 x half> %x, <2 x half> %y)
; CHECK-NEXT: ret <2 x half> %r
define spir_func <2 x half> @minimum_v2f16_nnan_nsz(<2 x half> %x, <2 x half> %y) {
  %r = call nnan nsz <2 x half> @llvm.minimum.v2f16(<2 x half> %x, <2 x half> %y)
  ret <2 x half> %r
}

; CHECK-LABEL: define spir_func float @maximum_f32_fast(
; CHECK-NEXT: %r = call fast float @llvm.maxnum.f32(float %x, float %y)
; CHECK-NEXT: ret float %r
define spir_func float @maximum_f32_fast(float %x, float %y) {
  %r = call fast float @llvm.maximum.f32(float %x, float %y)
  ret float %r
}
