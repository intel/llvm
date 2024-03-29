; RUN: llvm-as %s -o %t.bc

;; Check that an error is reported if a fpbuiltin-max-error attribute is encountered without the SPV_INTEL_fp_max_error
;; extension.
; RUN: not llvm-spirv %t.bc --spirv-allow-unknown-intrinsics=llvm.fpbuiltin -o %t.spv 2>&1  | FileCheck %s --check-prefix=CHECK_NO_CAPABILITY_ERROR
; CHECK_NO_CAPABILITY_ERROR: RequiresExtension: Feature requires the following SPIR-V extension:
; CHECK_NO_CAPABILITY_ERROR-NEXT: SPV_INTEL_fp_max_error

;; Check that fpbuiltin-max-error is translated and reverse-translated properly
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_INTEL_fp_max_error --spirv-allow-unknown-intrinsics=llvm.fpbuiltin -o %t.spv
; RUN: llvm-spirv %t.spv -to-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV

; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; CHECK-SPIRV: Capability FPMaxErrorINTEL
; CHECK-SPIRV: Extension "SPV_INTEL_fp_max_error"
; CHECK-SPIRV: ExtInstImport [[#OCLEXTID:]] "OpenCL.std"

; CHECK-SPIRV: Name [[#T1:]] "t1"
; CHECK-SPIRV: Name [[#T2:]] "t2"
; CHECK-SPIRV: Name [[#T3:]] "t3"
; CHECK-SPIRV: Name [[#T4:]] "t4"
; CHECK-SPIRV: Name [[#T5:]] "t5"
; CHECK-SPIRV: Name [[#T6:]] "t6"
; CHECK-SPIRV: Name [[#T7:]] "t7"
; CHECK-SPIRV: Name [[#T8:]] "t8"
; CHECK-SPIRV: Name [[#T9:]] "t9"
; CHECK-SPIRV: Name [[#T10:]] "t10"
; CHECK-SPIRV: Name [[#T11:]] "t11"
; CHECK-SPIRV: Name [[#T12:]] "t12"
; CHECK-SPIRV: Name [[#T13:]] "t13"
; CHECK-SPIRV: Name [[#T14:]] "t14"
; CHECK-SPIRV: Name [[#T15:]] "t15"
; CHECK-SPIRV: Name [[#T16:]] "t16"
; CHECK-SPIRV: Name [[#T17:]] "t17"
; CHECK-SPIRV: Name [[#T18:]] "t18"
; CHECK-SPIRV: Name [[#T19:]] "t19"
; CHECK-SPIRV: Name [[#T20:]] "t20"
; CHECK-SPIRV: Name [[#T21:]] "t21"
; CHECK-SPIRV: Name [[#T22:]] "t22"
; CHECK-SPIRV: Name [[#T23:]] "t23"
; CHECK-SPIRV: Name [[#T24:]] "t24"
; CHECK-SPIRV: Name [[#T25:]] "t25"
; CHECK-SPIRV: Name [[#T26:]] "t26"
; CHECK-SPIRV: Name [[#T27:]] "t27"
; CHECK-SPIRV: Name [[#T28:]] "t28"
; CHECK-SPIRV: Name [[#T29:]] "t29"
; CHECK-SPIRV: Name [[#T30:]] "t30"
; CHECK-SPIRV: Name [[#T31:]] "t31"
; CHECK-SPIRV: Name [[#T32:]] "t32"
; CHECK-SPIRV: Name [[#T33:]] "t33"

; CHECK-SPIRV: Decorate [[#T3]] FPMaxErrorDecorationINTEL 1056964608
; CHECK-SPIRV: Decorate [[#T4]] FPMaxErrorDecorationINTEL 1065353216
; CHECK-SPIRV: Decorate [[#T5]] FPMaxErrorDecorationINTEL 1065353216
; CHECK-SPIRV: Decorate [[#T6]] FPMaxErrorDecorationINTEL 1075838976
; CHECK-SPIRV: Decorate [[#T7]] FPMaxErrorDecorationINTEL 1075838976
; CHECK-SPIRV: Decorate [[#T8]] FPMaxErrorDecorationINTEL 1075838976
; CHECK-SPIRV: Decorate [[#T9]] FPMaxErrorDecorationINTEL 1075838976
; CHECK-SPIRV: Decorate [[#T10]] FPMaxErrorDecorationINTEL 1075838976
; CHECK-SPIRV: Decorate [[#T11]] FPMaxErrorDecorationINTEL 1075838976
; CHECK-SPIRV: Decorate [[#T12]] FPMaxErrorDecorationINTEL 1075838976
; CHECK-SPIRV: Decorate [[#T13]] FPMaxErrorDecorationINTEL 1075838976
; CHECK-SPIRV: Decorate [[#T14]] FPMaxErrorDecorationINTEL 1075838976
; CHECK-SPIRV: Decorate [[#T15]] FPMaxErrorDecorationINTEL 1075838976
; CHECK-SPIRV: Decorate [[#T16]] FPMaxErrorDecorationINTEL 1075838976
; CHECK-SPIRV: Decorate [[#T17]] FPMaxErrorDecorationINTEL 1075838976
; CHECK-SPIRV: Decorate [[#T18]] FPMaxErrorDecorationINTEL 1075838976
; CHECK-SPIRV: Decorate [[#T19]] FPMaxErrorDecorationINTEL 1075838976
; CHECK-SPIRV: Decorate [[#T20]] FPMaxErrorDecorationINTEL 1075838976
; CHECK-SPIRV: Decorate [[#T21]] FPMaxErrorDecorationINTEL 1075838976
; CHECK-SPIRV: Decorate [[#T22]] FPMaxErrorDecorationINTEL 1075838976
; CHECK-SPIRV: Decorate [[#T23]] FPMaxErrorDecorationINTEL 1075838976
; CHECK-SPIRV: Decorate [[#T24]] FPMaxErrorDecorationINTEL 1075838976
; CHECK-SPIRV: Decorate [[#T25]] FPMaxErrorDecorationINTEL 1075838976
; CHECK-SPIRV: Decorate [[#T26]] FPMaxErrorDecorationINTEL 1075838976
; CHECK-SPIRV: Decorate [[#T27]] FPMaxErrorDecorationINTEL 1075838976
; CHECK-SPIRV: Decorate [[#T28]] FPMaxErrorDecorationINTEL 1075838976
; CHECK-SPIRV: Decorate [[#T29]] FPMaxErrorDecorationINTEL 1075838976
; CHECK-SPIRV: Decorate [[#T30]] FPMaxErrorDecorationINTEL 1082130432
; CHECK-SPIRV: Decorate [[#T31]] FPMaxErrorDecorationINTEL 1082130432
; CHECK-SPIRV: Decorate [[#T32]] FPMaxErrorDecorationINTEL 1082130432
; CHECK-SPIRV: Decorate [[#T33]] FPMaxErrorDecorationINTEL 1166016512

; CHECK-SPIRV: 3 TypeFloat [[#FTYPE:]] 32

; CHECK-SPIRV: FAdd [[#FTYPE]] [[#T1]]
; CHECK-SPIRV: FSub [[#FTYPE]] [[#T2]]
; CHECK-SPIRV: FMul [[#FTYPE]] [[#T3]]
; CHECK-SPIRV: FDiv [[#FTYPE]] [[#T4]]
; CHECK-SPIRV: FRem [[#FTYPE]] [[#T5]]
; CHECK-SPIRV: ExtInst [[#FTYPE]] [[#T6]] [[#OCLEXTID]] sin
; CHECK-SPIRV: ExtInst [[#FTYPE]] [[#T7]] [[#OCLEXTID]] cos
; CHECK-SPIRV: ExtInst [[#FTYPE]] [[#T8]] [[#OCLEXTID]] tan
; CHECK-SPIRV: ExtInst [[#FTYPE]] [[#T9]] [[#OCLEXTID]] sinh
; CHECK-SPIRV: ExtInst [[#FTYPE]] [[#T10]] [[#OCLEXTID]] cosh
; CHECK-SPIRV: ExtInst [[#FTYPE]] [[#T11]] [[#OCLEXTID]] tanh
; CHECK-SPIRV: ExtInst [[#FTYPE]] [[#T12]] [[#OCLEXTID]] asin
; CHECK-SPIRV: ExtInst [[#FTYPE]] [[#T13]] [[#OCLEXTID]] acos
; CHECK-SPIRV: ExtInst [[#FTYPE]] [[#T14]] [[#OCLEXTID]] atan
; CHECK-SPIRV: ExtInst [[#FTYPE]] [[#T15]] [[#OCLEXTID]] asinh
; CHECK-SPIRV: ExtInst [[#FTYPE]] [[#T16]] [[#OCLEXTID]] acosh
; CHECK-SPIRV: ExtInst [[#FTYPE]] [[#T17]] [[#OCLEXTID]] atanh
; CHECK-SPIRV: ExtInst [[#FTYPE]] [[#T18]] [[#OCLEXTID]] exp
; CHECK-SPIRV: ExtInst [[#FTYPE]] [[#T19]] [[#OCLEXTID]] exp2
; CHECK-SPIRV: ExtInst [[#FTYPE]] [[#T20]] [[#OCLEXTID]] exp10
; CHECK-SPIRV: ExtInst [[#FTYPE]] [[#T21]] [[#OCLEXTID]] expm1
; CHECK-SPIRV: ExtInst [[#FTYPE]] [[#T22]] [[#OCLEXTID]] log
; CHECK-SPIRV: ExtInst [[#FTYPE]] [[#T23]] [[#OCLEXTID]] log2
; CHECK-SPIRV: ExtInst [[#FTYPE]] [[#T24]] [[#OCLEXTID]] log10
; CHECK-SPIRV: ExtInst [[#FTYPE]] [[#T25]] [[#OCLEXTID]] log1p
; CHECK-SPIRV: ExtInst [[#FTYPE]] [[#T26]] [[#OCLEXTID]] sqrt
; CHECK-SPIRV: ExtInst [[#FTYPE]] [[#T27]] [[#OCLEXTID]] rsqrt
; CHECK-SPIRV: ExtInst [[#FTYPE]] [[#T28]] [[#OCLEXTID]] erf
; CHECK-SPIRV: ExtInst [[#FTYPE]] [[#T29]] [[#OCLEXTID]] erfc
; CHECK-SPIRV: ExtInst [[#FTYPE]] [[#T30]] [[#OCLEXTID]] atan2
; CHECK-SPIRV: ExtInst [[#FTYPE]] [[#T31]] [[#OCLEXTID]] ldexp
; CHECK-SPIRV: ExtInst [[#FTYPE]] [[#T32]] [[#OCLEXTID]] pow
; CHECK-SPIRV: ExtInst [[#FTYPE]] [[#T33]] [[#OCLEXTID]] hypot

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

define void @test_fp_max_error_decoration(float %f1, float %f2, float %f3, i32 %f4) {
entry:
; CHECK-LLVM-NOT: fadd float %f1, %f2, !fpbuiltin-max-error
; CHECK-LLVM-NOT: fsub float %f1, %f2, !fpbuiltin-max-error
; CHECK-LLVM: fmul float %f1, %f2, !fpbuiltin-max-error ![[#ME1:]]
; CHECK-LLVM: fdiv float %f1, %f2, !fpbuiltin-max-error ![[#ME2:]]
; CHECK-LLVM: frem float %f1, %f2, !fpbuiltin-max-error ![[#ME2]]
  %t1 = call float @llvm.fpbuiltin.fadd.f32(float %f1, float %f2)
  %t2 = call float @llvm.fpbuiltin.fsub.f32(float %f1, float %f2)
  %t3 = call float @llvm.fpbuiltin.fmul.f32(float %f1, float %f2) #0
  %t4 = call float @llvm.fpbuiltin.fdiv.f32(float %f1, float %f2) #1
  %t5 = call float @llvm.fpbuiltin.frem.f32(float %f1, float %f2) #1

; CHECK-LLVM: call spir_func float @_Z3sinf(float %f1) #[[#AT3:]]
; CHECK-LLVM: call spir_func float @_Z3cosf(float %f1) #[[#AT3]]
; CHECK-LLVM: call spir_func float @_Z3tanf(float %f1) #[[#AT3]]
  %t6 = call float @llvm.fpbuiltin.sin.f32(float %f1) #2
  %t7 = call float @llvm.fpbuiltin.cos.f32(float %f1) #2
  %t8 = call float @llvm.fpbuiltin.tan.f32(float %f1) #2

; CHECK-LLVM: call spir_func float @_Z4sinhf(float %f1) #[[#AT3]]
; CHECK-LLVM: call spir_func float @_Z4coshf(float %f1) #[[#AT3]]
; CHECK-LLVM: call spir_func float @_Z4tanhf(float %f1) #[[#AT3]]
  %t9 = call float @llvm.fpbuiltin.sinh.f32(float %f1) #2
  %t10 = call float @llvm.fpbuiltin.cosh.f32(float %f1) #2
  %t11 = call float @llvm.fpbuiltin.tanh.f32(float %f1) #2

; CHECK-LLVM: call spir_func float @_Z4asinf(float %f1) #[[#AT3]]
; CHECK-LLVM: call spir_func float @_Z4acosf(float %f1) #[[#AT3]]
; CHECK-LLVM: call spir_func float @_Z4atanf(float %f1) #[[#AT3]]
  %t12 = call float @llvm.fpbuiltin.asin.f32(float %f1) #2
  %t13 = call float @llvm.fpbuiltin.acos.f32(float %f1) #2
  %t14 = call float @llvm.fpbuiltin.atan.f32(float %f1) #2

; CHECK-LLVM:15 = call spir_func float @_Z5asinhf(float %f1) #[[#AT3]]
; CHECK-LLVM:16 = call spir_func float @_Z5acoshf(float %f1) #[[#AT3]]
; CHECK-LLVM:17 = call spir_func float @_Z5atanhf(float %f1) #[[#AT3]]
  %t15 = call float @llvm.fpbuiltin.asinh.f32(float %f1) #2
  %t16 = call float @llvm.fpbuiltin.acosh.f32(float %f1) #2
  %t17 = call float @llvm.fpbuiltin.atanh.f32(float %f1) #2

; CHECK-LLVM:18 = call spir_func float @_Z3expf(float %f1) #[[#AT3]]
; CHECK-LLVM:19 = call spir_func float @_Z4exp2f(float %f1) #[[#AT3]]
; CHECK-LLVM:20 = call spir_func float @_Z5exp10f(float %f1) #[[#AT3]]
; CHECK-LLVM:21 = call spir_func float @_Z5expm1f(float %f1) #[[#AT3]]
  %t18 = call float @llvm.fpbuiltin.exp.f32(float %f1) #2
  %t19 = call float @llvm.fpbuiltin.exp2.f32(float %f1) #2
  %t20 = call float @llvm.fpbuiltin.exp10.f32(float %f1) #2
  %t21 = call float @llvm.fpbuiltin.expm1.f32(float %f1) #2

; CHECK-LLVM: call spir_func float @_Z3logf(float %f1) #[[#AT3]]
; CHECK-LLVM: call spir_func float @_Z4log2f(float %f1) #[[#AT3]]
; CHECK-LLVM: call spir_func float @_Z5log10f(float %f1) #[[#AT3]]
; CHECK-LLVM: call spir_func float @_Z5log1pf(float %f1) #[[#AT3]]
  %t22 = call float @llvm.fpbuiltin.log.f32(float %f1) #2
  %t23 = call float @llvm.fpbuiltin.log2.f32(float %f1) #2
  %t24 = call float @llvm.fpbuiltin.log10.f32(float %f1) #2
  %t25 = call float @llvm.fpbuiltin.log1p.f32(float %f1) #2

; CHECK-LLVM: call spir_func float @_Z4sqrtf(float %f1) #[[#AT3]]
; CHECK-LLVM: call spir_func float @_Z5rsqrtf(float %f1) #[[#AT3]]
; CHECK-LLVM: call spir_func float @_Z3erff(float %f1) #[[#AT3]]
; CHECK-LLVM: call spir_func float @_Z4erfcf(float %f1) #[[#AT3]]
  %t26 = call float @llvm.fpbuiltin.sqrt.f32(float %f1) #2
  %t27 = call float @llvm.fpbuiltin.rsqrt.f32(float %f1) #2
  %t28 = call float @llvm.fpbuiltin.erf.f32(float %f1) #2
  %t29 = call float @llvm.fpbuiltin.erfc.f32(float %f1) #2

; CHECK-LLVM: call spir_func float @_Z5atan2ff(float %f1, float %f2) #[[#AT4:]]
; CHECK-LLVM: call spir_func float @_Z5ldexpfi(float %f1, i32 %f4) #[[#AT4]]
; CHECK-LLVM: call spir_func float @_Z3powff(float %f1, float %f2) #[[#AT4]]
  %t30 = call float @llvm.fpbuiltin.atan2.f32(float %f1, float %f2) #3
  %t31 = call float @llvm.fpbuiltin.ldexp.f32.i32(float %f1, i32 %f4) #3
  %t32 = call float @llvm.fpbuiltin.pow.f32(float %f1, float %f2) #3
  
  ; CHECK-LLVM: call spir_func float @_Z5hypotff(float %f1, float %f2) #[[#AT5:]]
  %t33 = call float @llvm.fpbuiltin.hypot.f32(float %f1, float %f2) #4

  ret void
}

declare float @llvm.fpbuiltin.fadd.f32(float, float)
declare float @llvm.fpbuiltin.fsub.f32(float, float)
declare float @llvm.fpbuiltin.fmul.f32(float, float)
declare float @llvm.fpbuiltin.fdiv.f32(float, float)
declare float @llvm.fpbuiltin.frem.f32(float, float)

declare float @llvm.fpbuiltin.sin.f32(float)
declare float @llvm.fpbuiltin.cos.f32(float)
declare float @llvm.fpbuiltin.tan.f32(float)
declare float @llvm.fpbuiltin.sinh.f32(float)
declare float @llvm.fpbuiltin.cosh.f32(float)
declare float @llvm.fpbuiltin.tanh.f32(float)
declare float @llvm.fpbuiltin.asin.f32(float)
declare float @llvm.fpbuiltin.acos.f32(float)
declare float @llvm.fpbuiltin.atan.f32(float)
declare float @llvm.fpbuiltin.asinh.f32(float)
declare float @llvm.fpbuiltin.acosh.f32(float)
declare float @llvm.fpbuiltin.atanh.f32(float)
declare float @llvm.fpbuiltin.exp.f32(float)
declare float @llvm.fpbuiltin.exp2.f32(float)
declare float @llvm.fpbuiltin.exp10.f32(float)
declare float @llvm.fpbuiltin.expm1.f32(float)
declare float @llvm.fpbuiltin.log.f32(float)
declare float @llvm.fpbuiltin.log2.f32(float)
declare float @llvm.fpbuiltin.log10.f32(float)
declare float @llvm.fpbuiltin.log1p.f32(float)
declare float @llvm.fpbuiltin.sqrt.f32(float)
declare float @llvm.fpbuiltin.rsqrt.f32(float)
declare float @llvm.fpbuiltin.erf.f32(float)
declare float @llvm.fpbuiltin.erfc.f32(float)

declare float @llvm.fpbuiltin.atan2.f32(float, float)
declare float @llvm.fpbuiltin.hypot.f32(float, float)
declare float @llvm.fpbuiltin.pow.f32(float, float)
declare float @llvm.fpbuiltin.ldexp.f32.i32(float, i32)

; CHECK-LLVM: attributes #[[#AT3]] = {{{.*}} "fpbuiltin-max-error"="2.5{{0+}}" {{.*}}}
; CHECK-LLVM: attributes #[[#AT4]] = {{{.*}} "fpbuiltin-max-error"="4.0{{0+}}" {{.*}}}
; CHECK-LLVM: attributes #[[#AT5]] = {{{.*}} "fpbuiltin-max-error"="4096.0{{0+}}" {{.*}}}
; CHECK-LLVM: ![[#ME1]] = !{!"0.500000"}
; CHECK-LLVM: ![[#ME2]] = !{!"1.000000"}

attributes #0 = { "fpbuiltin-max-error"="0.5" }
attributes #1 = { "fpbuiltin-max-error"="1.0" }
attributes #2 = { "fpbuiltin-max-error"="2.5" }
attributes #3 = { "fpbuiltin-max-error"="4.0" }
attributes #4 = { "fpbuiltin-max-error"="4096.0" }
