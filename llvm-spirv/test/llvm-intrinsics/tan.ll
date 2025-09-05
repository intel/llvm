; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o - | FileCheck %s
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir"

; CHECK: ExtInstImport [[#ExtInstSetId:]] "OpenCL.std"

; CHECK: Name [[#HalfArg:]] "h"
; CHECK: Name [[#FloatArg:]] "f"
; CHECK: Name [[#DoubleArg:]] "d"

; CHECK: TypeFloat [[#Half:]] 16
; CHECK: TypeFloat [[#Float:]] 32
; CHECK: TypeFloat [[#Double:]] 64

; CHECK: TypeVector [[#Half4:]] [[#Half]] 4
; CHECK: TypeVector [[#Float4:]] [[#Float]] 4
; CHECK: TypeVector [[#Double4:]] [[#Double]] 4

; CHECK: ConstantComposite [[#Half4]] [[#Half4Arg:]]
; CHECK: ConstantComposite [[#Float4]] [[#Float4Arg:]]
; CHECK: ConstantComposite [[#Double4]] [[#Double4Arg:]]

; CHECK: ExtInst [[#Half]] {{[0-9]+}} [[#ExtInstSetId]] tan [[#HalfArg]]
; CHECK: ExtInst [[#Float]] {{[0-9]+}} [[#ExtInstSetId]] tan [[#FloatArg]]
; CHECK: ExtInst [[#Double]] {{[0-9]+}} [[#ExtInstSetId]] tan [[#DoubleArg]]

; CHECK: ExtInst [[#Half4]] {{[0-9]+}} [[#ExtInstSetId]] tan [[#Half4Arg]]
; CHECK: ExtInst [[#Float4]] {{[0-9]+}} [[#ExtInstSetId]] tan [[#Float4Arg]]
; CHECK: ExtInst [[#Double4]] {{[0-9]+}} [[#ExtInstSetId]] tan [[#Double4Arg]]

; CHECK: ExtInst [[#Half]] {{[0-9]+}} [[#ExtInstSetId]] tan [[#HalfArg]]
; CHECK: ExtInst [[#Float]] {{[0-9]+}} [[#ExtInstSetId]] native_tan [[#FloatArg]]
; CHECK: ExtInst [[#Double]] {{[0-9]+}} [[#ExtInstSetId]] tan [[#DoubleArg]]

; CHECK: ExtInst [[#Half4]] {{[0-9]+}} [[#ExtInstSetId]] tan [[#Half4Arg]]
; CHECK: ExtInst [[#Float4]] {{[0-9]+}} [[#ExtInstSetId]] native_tan [[#Float4Arg]]
; CHECK: ExtInst [[#Double4]] {{[0-9]+}} [[#ExtInstSetId]] tan [[#Double4Arg]]

; Function Attrs: nounwind readnone
define dso_local spir_func void @foo(half %h, float %f, double %d) local_unnamed_addr {
entry:
  %0 = call half @llvm.tan.f16(half %h)
  %1 = call float @llvm.tan.f32(float %f)
  %2 = call double @llvm.tan.f64(double %d)
  %3 = call <4 x half> @llvm.tan.v4f16(<4 x half> <half 5.000000e-01, half 10.000000e-01, half 15.000000e-01, half 20.000000e-01>)
  %4 = call <4 x float> @llvm.tan.v4f32(<4 x float> <float 5.000000e-01, float 10.000000e-01, float 15.000000e-01, float 20.000000e-01>)
  %5 = call <4 x double> @llvm.tan.v4f64(<4 x double> <double 5.000000e-01, double 2.000000e-01, double 3.000000e-01, double 4.000000e-01>)
  %6 = call afn half @llvm.tan.f16(half %h)
  %7 = call afn float @llvm.tan.f32(float %f)
  %8 = call afn double @llvm.tan.f64(double %d)
  %9 = call afn <4 x half> @llvm.tan.v4f16(<4 x half> <half 5.000000e-01, half 10.000000e-01, half 15.000000e-01, half 20.000000e-01>)
  %10 = call afn <4 x float> @llvm.tan.v4f32(<4 x float> <float 5.000000e-01, float 10.000000e-01, float 15.000000e-01, float 20.000000e-01>)
  %11 = call afn <4 x double> @llvm.tan.v4f64(<4 x double> <double 5.000000e-01, double 2.000000e-01, double 3.000000e-01, double 4.000000e-01>)
  ret void
}

; Function Attrs: nounwind readnone speculatable willreturn
declare half @llvm.tan.f16(half)

; Function Attrs: nounwind readnone speculatable willreturn
declare float @llvm.tan.f32(float)

; Function Attrs: nounwind readnone speculatable willreturn
declare double @llvm.tan.f64(double)

; Function Attrs: nounwind readnone speculatable willreturn
declare <4 x half> @llvm.tan.v4f16(<4 x half>)

; Function Attrs: nounwind readnone speculatable willreturn
declare <4 x float> @llvm.tan.v4f32(<4 x float>)

; Function Attrs: nounwind readnone speculatable willreturn
declare <4 x double> @llvm.tan.v4f64(<4 x double>)
