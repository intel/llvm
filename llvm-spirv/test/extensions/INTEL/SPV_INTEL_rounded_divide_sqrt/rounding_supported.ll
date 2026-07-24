; Verifies that FPRoundingMode decorations on OpFDiv and sqrt (OpenCL.std
; extended instruction) trigger the RoundedDivideSqrtINTEL capability and the
; SPV_INTEL_rounded_divide_sqrt extension, and round-trip back through both
; reverse-translation paths.

; RUN: llvm-spirv %s -o %t.spv --spirv-ext=+SPV_INTEL_rounded_divide_sqrt
; RUN: spirv-val %t.spv
; RUN: llvm-spirv %t.spv -o %t.spt --to-text
; RUN: FileCheck %s --input-file %t.spt --check-prefix=CHECK-SPIRV

; OCL path: there is no OCL representation for the rounding mode on fdiv/sqrt, so it is dropped.
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc -o - | FileCheck %s --check-prefixes=CHECK-LLVM,CHECK-LLVM-OCL \
; RUN:   --implicit-check-not=FPRoundingMode --implicit-check-not=spirv.Decorations

; SPV-IR path: the rounding mode is preserved as an
; !spirv.Decorations FPRoundingMode metadata attached to the fdiv/sqrt.
; RUN: llvm-spirv -r %t.spv --spirv-target-env=SPV-IR -o %t.rev.spvir.bc
; RUN: llvm-dis %t.rev.spvir.bc -o - | FileCheck %s --check-prefixes=CHECK-LLVM,CHECK-LLVM-SPV

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

; CHECK-SPIRV: Capability RoundedDivideSqrtINTEL
; CHECK-SPIRV: Extension "SPV_INTEL_rounded_divide_sqrt"

; CHECK-SPIRV-DAG: TypeFloat [[#HALF:]] 16
; CHECK-SPIRV-DAG: TypeFloat [[#FLOAT:]] 32
; CHECK-SPIRV-DAG: TypeFloat [[#DOUBLE:]] 64

; CHECK-SPIRV-DAG: TypeVector [[#HALFV:]] [[#HALF]] 2
; CHECK-SPIRV-DAG: TypeVector [[#FLOATV:]] [[#FLOAT]] 4
; CHECK-SPIRV-DAG: TypeVector [[#DOUBLEV:]] [[#DOUBLE]] 3

; All four rounding modes on the scalar-typed divides.
; CHECK-SPIRV-DAG: Decorate [[#H_RTE:]] FPRoundingMode 0
; CHECK-SPIRV-DAG: Decorate [[#F_RTZ:]] FPRoundingMode 1
; CHECK-SPIRV-DAG: Decorate [[#D_RTP:]] FPRoundingMode 2
; CHECK-SPIRV-DAG: Decorate [[#D_RTN:]] FPRoundingMode 3
; CHECK-SPIRV-DAG: FDiv [[#HALF]] [[#H_RTE]]
; CHECK-SPIRV-DAG: FDiv [[#FLOAT]] [[#F_RTZ]]
; CHECK-SPIRV-DAG: FDiv [[#DOUBLE]] [[#D_RTP]]
; CHECK-SPIRV-DAG: FDiv [[#DOUBLE]] [[#D_RTN]]
; CHECK-LLVM-LABEL: @test_fdiv_scalar
; CHECK-LLVM: fdiv half
; CHECK-LLVM-SPV-SAME: !spirv.Decorations ![[#RTE:]]
; CHECK-LLVM: fdiv float
; CHECK-LLVM-SPV-SAME: !spirv.Decorations ![[#RTZ:]]
; CHECK-LLVM: fdiv double
; CHECK-LLVM-SPV-SAME: !spirv.Decorations ![[#RTP:]]
; CHECK-LLVM: fdiv double
; CHECK-LLVM-SPV-SAME: !spirv.Decorations ![[#RTN:]]
define spir_kernel void @test_fdiv_scalar(half %h0, half %h1, float %f0, float %f1, double %d0, double %d1) {
entry:
  %h_rte = call half @llvm.experimental.constrained.fdiv.f16(half %h0, half %h1, metadata !"round.tonearest", metadata !"fpexcept.strict")
  %f_rtz = call float @llvm.experimental.constrained.fdiv.f32(float %f0, float %f1, metadata !"round.towardzero", metadata !"fpexcept.strict")
  %d_rtp = call double @llvm.experimental.constrained.fdiv.f64(double %d0, double %d1, metadata !"round.upward", metadata !"fpexcept.strict")
  %d_rtn = call double @llvm.experimental.constrained.fdiv.f64(double %d0, double %d1, metadata !"round.downward", metadata !"fpexcept.strict")
  ret void
}

; All four rounding modes on the vector-typed divides.
; CHECK-SPIRV-DAG: Decorate [[#HV_RTE:]] FPRoundingMode 0
; CHECK-SPIRV-DAG: Decorate [[#FV_RTZ:]] FPRoundingMode 1
; CHECK-SPIRV-DAG: Decorate [[#DV_RTP:]] FPRoundingMode 2
; CHECK-SPIRV-DAG: Decorate [[#DV_RTN:]] FPRoundingMode 3
; CHECK-SPIRV-DAG: FDiv [[#HALFV]] [[#HV_RTE]]
; CHECK-SPIRV-DAG: FDiv [[#FLOATV]] [[#FV_RTZ]]
; CHECK-SPIRV-DAG: FDiv [[#DOUBLEV]] [[#DV_RTP]]
; CHECK-SPIRV-DAG: FDiv [[#DOUBLEV]] [[#DV_RTN]]

; CHECK-LLVM-LABEL: @test_fdiv_vector
; CHECK-LLVM: fdiv <2 x half>
; CHECK-LLVM-SPV-SAME: !spirv.Decorations ![[#RTE]]
; CHECK-LLVM: fdiv <4 x float>
; CHECK-LLVM-SPV-SAME: !spirv.Decorations ![[#RTZ]]
; CHECK-LLVM: fdiv <3 x double>
; CHECK-LLVM-SPV-SAME: !spirv.Decorations ![[#RTP]]
; CHECK-LLVM: fdiv <3 x double>
; CHECK-LLVM-SPV-SAME: !spirv.Decorations ![[#RTN]]
define spir_kernel void @test_fdiv_vector(<2 x half> %h0, <2 x half> %h1, <4 x float> %f0, <4 x float> %f1, <3 x double> %d0, <3 x double> %d1) {
entry:
  %h_rte = call <2 x half> @llvm.experimental.constrained.fdiv.v2f16(<2 x half> %h0, <2 x half> %h1, metadata !"round.tonearest", metadata !"fpexcept.strict")
  %f_rtz = call <4 x float> @llvm.experimental.constrained.fdiv.v4f32(<4 x float> %f0, <4 x float> %f1, metadata !"round.towardzero", metadata !"fpexcept.strict")
  %d_rtp = call <3 x double> @llvm.experimental.constrained.fdiv.v3f64(<3 x double> %d0, <3 x double> %d1, metadata !"round.upward", metadata !"fpexcept.strict")
  %d_rtn = call <3 x double> @llvm.experimental.constrained.fdiv.v3f64(<3 x double> %d0, <3 x double> %d1, metadata !"round.downward", metadata !"fpexcept.strict")
  ret void
}

; All four rounding modes on the scalar-typed constrained sqrts.
; CHECK-SPIRV-DAG: Decorate [[#S_RTE:]] FPRoundingMode 0
; CHECK-SPIRV-DAG: Decorate [[#S_RTZ:]] FPRoundingMode 1
; CHECK-SPIRV-DAG: Decorate [[#S_RTP:]] FPRoundingMode 2
; CHECK-SPIRV-DAG: Decorate [[#S_RTN:]] FPRoundingMode 3
; CHECK-SPIRV-DAG: ExtInst [[#HALF]]   [[#S_RTE]] [[#]] sqrt
; CHECK-SPIRV-DAG: ExtInst [[#FLOAT]]  [[#S_RTZ]] [[#]] sqrt
; CHECK-SPIRV-DAG: ExtInst [[#DOUBLE]] [[#S_RTP]] [[#]] sqrt
; CHECK-SPIRV-DAG: ExtInst [[#DOUBLE]] [[#S_RTN]] [[#]] sqrt

; CHECK-LLVM-OCL-LABEL: @test_sqrt_scalar
; CHECK-LLVM-OCL: call spir_func half @_Z4sqrtDh
; CHECK-LLVM-OCL: call spir_func float @_Z4sqrtf
; CHECK-LLVM-OCL: call spir_func double @_Z4sqrtd
; CHECK-LLVM-OCL: call spir_func double @_Z4sqrtd

; CHECK-LLVM-SPV-LABEL: @test_sqrt_scalar
; CHECK-LLVM-SPV: call spir_func half @_Z16__spirv_ocl_sqrtDh(half %{{.*}}, !spirv.Decorations ![[#RTE]]
; CHECK-LLVM-SPV: call spir_func float @_Z16__spirv_ocl_sqrtf(float %{{.*}}, !spirv.Decorations ![[#RTZ]]
; CHECK-LLVM-SPV: call spir_func double @_Z16__spirv_ocl_sqrtd(double %{{.*}}, !spirv.Decorations ![[#RTP]]
; CHECK-LLVM-SPV: call spir_func double @_Z16__spirv_ocl_sqrtd(double %{{.*}}, !spirv.Decorations ![[#RTN]]
define spir_kernel void @test_sqrt_scalar(half %h, float %f, double %d) {
entry:
  %h_rte = call half   @llvm.experimental.constrained.sqrt.f16(half %h,   metadata !"round.tonearest", metadata !"fpexcept.strict")
  %f_rtz = call float  @llvm.experimental.constrained.sqrt.f32(float %f,  metadata !"round.towardzero", metadata !"fpexcept.strict")
  %d_rtp = call double @llvm.experimental.constrained.sqrt.f64(double %d, metadata !"round.upward",     metadata !"fpexcept.strict")
  %d_rtn = call double @llvm.experimental.constrained.sqrt.f64(double %d, metadata !"round.downward",   metadata !"fpexcept.strict")
  ret void
}

; All four rounding modes on vector-typed constrained sqrts.
; CHECK-SPIRV-DAG: Decorate [[#SV_RTE:]] FPRoundingMode 0
; CHECK-SPIRV-DAG: Decorate [[#SV_RTZ:]] FPRoundingMode 1
; CHECK-SPIRV-DAG: Decorate [[#SV_RTP:]] FPRoundingMode 2
; CHECK-SPIRV-DAG: Decorate [[#SV_RTN:]] FPRoundingMode 3
; CHECK-SPIRV-DAG: ExtInst [[#HALFV]]   [[#SV_RTE]] [[#]] sqrt
; CHECK-SPIRV-DAG: ExtInst [[#FLOATV]]  [[#SV_RTZ]] [[#]] sqrt
; CHECK-SPIRV-DAG: ExtInst [[#DOUBLEV]] [[#SV_RTP]] [[#]] sqrt
; CHECK-SPIRV-DAG: ExtInst [[#DOUBLEV]] [[#SV_RTN]] [[#]] sqrt

; CHECK-LLVM-OCL-LABEL: @test_sqrt_vector
; CHECK-LLVM-OCL: call spir_func <2 x half> @_Z4sqrtDv2_Dh
; CHECK-LLVM-OCL: call spir_func <4 x float> @_Z4sqrtDv4_f
; CHECK-LLVM-OCL: call spir_func <3 x double> @_Z4sqrtDv3_d
; CHECK-LLVM-OCL: call spir_func <3 x double> @_Z4sqrtDv3_d

; CHECK-LLVM-SPV-LABEL: @test_sqrt_vector
; CHECK-LLVM-SPV: call spir_func <2 x half> @_Z16__spirv_ocl_sqrtDv2_Dh(<2 x half> %{{.*}}, !spirv.Decorations ![[#RTE]]
; CHECK-LLVM-SPV: call spir_func <4 x float> @_Z16__spirv_ocl_sqrtDv4_f(<4 x float> %{{.*}}, !spirv.Decorations ![[#RTZ]]
; CHECK-LLVM-SPV: call spir_func <3 x double> @_Z16__spirv_ocl_sqrtDv3_d(<3 x double> %{{.*}}, !spirv.Decorations ![[#RTP]]
; CHECK-LLVM-SPV: call spir_func <3 x double> @_Z16__spirv_ocl_sqrtDv3_d(<3 x double> %{{.*}}, !spirv.Decorations ![[#RTN]]
define spir_kernel void @test_sqrt_vector(<2 x half> %h, <4 x float> %f, <3 x double> %d) {
entry:
  %h_rte = call <2 x half>   @llvm.experimental.constrained.sqrt.v2f16(<2 x half> %h,   metadata !"round.tonearest", metadata !"fpexcept.strict")
  %f_rtz = call <4 x float>  @llvm.experimental.constrained.sqrt.v4f32(<4 x float> %f,  metadata !"round.towardzero", metadata !"fpexcept.strict")
  %d_rtp = call <3 x double> @llvm.experimental.constrained.sqrt.v3f64(<3 x double> %d, metadata !"round.upward",     metadata !"fpexcept.strict")
  %d_rtn = call <3 x double> @llvm.experimental.constrained.sqrt.v3f64(<3 x double> %d, metadata !"round.downward",   metadata !"fpexcept.strict")
  ret void
}

declare half @llvm.experimental.constrained.fdiv.f16(half, half, metadata, metadata)
declare float @llvm.experimental.constrained.fdiv.f32(float, float, metadata, metadata)
declare double @llvm.experimental.constrained.fdiv.f64(double, double, metadata, metadata)

declare <2 x half> @llvm.experimental.constrained.fdiv.v2f16(<2 x half>, <2 x half>, metadata, metadata)
declare <4 x float> @llvm.experimental.constrained.fdiv.v4f32(<4 x float>, <4 x float>, metadata, metadata)
declare <3 x double> @llvm.experimental.constrained.fdiv.v3f64(<3 x double>, <3 x double>, metadata, metadata)

declare half   @llvm.experimental.constrained.sqrt.f16(half, metadata, metadata)
declare float  @llvm.experimental.constrained.sqrt.f32(float, metadata, metadata)
declare double @llvm.experimental.constrained.sqrt.f64(double, metadata, metadata)

declare <2 x half>   @llvm.experimental.constrained.sqrt.v2f16(<2 x half>, metadata, metadata)
declare <4 x float>  @llvm.experimental.constrained.sqrt.v4f32(<4 x float>, metadata, metadata)
declare <3 x double> @llvm.experimental.constrained.sqrt.v3f64(<3 x double>, metadata, metadata)

; Verify the FPRoundingMode (Decoration 39) metadata nodes captured above:
; mode 0 = RTE, 1 = RTZ, 2 = RTP, 3 = RTN.
; CHECK-LLVM-SPV-DAG: ![[#RTE]] = !{![[#RTEX:]]}
; CHECK-LLVM-SPV-DAG: ![[#RTEX]] = !{i32 39, i32 0}
; CHECK-LLVM-SPV-DAG: ![[#RTZ]] = !{![[#RTZX:]]}
; CHECK-LLVM-SPV-DAG: ![[#RTZX]] = !{i32 39, i32 1}
; CHECK-LLVM-SPV-DAG: ![[#RTP]] = !{![[#RTPX:]]}
; CHECK-LLVM-SPV-DAG: ![[#RTPX]] = !{i32 39, i32 2}
; CHECK-LLVM-SPV-DAG: ![[#RTN]] = !{![[#RTNX:]]}
; CHECK-LLVM-SPV-DAG: ![[#RTNX]] = !{i32 39, i32 3}
