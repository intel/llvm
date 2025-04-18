; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o - | FileCheck %s
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-unknown"

; CHECK: ExtInstImport [[extinst_id:[0-9]+]] "OpenCL.std"

; CHECK: 3 TypeFloat [[var0:[0-9]+]] 16
; CHECK: 3 TypeFloat [[var1:[0-9]+]] 32
; CHECK: 3 TypeFloat [[var2:[0-9]+]] 64
; CHECK: 4 TypeVector [[var3:[0-9]+]] [[var1]] 4

; CHECK: Function
; CHECK: ExtInst [[var0]] {{[0-9]+}} [[extinst_id]] fabs
; CHECK: FunctionEnd

define spir_func half @TestFabs16(half %x) local_unnamed_addr {
entry:
  %t = tail call half @llvm.fabs.f16(half %x)
  ret half %t
}

; CHECK: Function
; CHECK: ExtInst [[var1]] {{[0-9]+}} [[extinst_id]] fabs
; CHECK: FunctionEnd

define spir_func float @TestFabs32(float %x) local_unnamed_addr {
entry:
  %t = tail call float @llvm.fabs.f32(float %x)
  ret float %t
}

; CHECK: Function
; CHECK: ExtInst [[var2]] {{[0-9]+}} [[extinst_id]] fabs
; CHECK: FunctionEnd

define spir_func double @TestFabs64(double %x) local_unnamed_addr {
entry:
  %t = tail call double @llvm.fabs.f64(double %x)
  ret double %t
}

; CHECK: Function
; CHECK: ExtInst [[var3]] {{[0-9]+}} [[extinst_id]] fabs
; CHECK: FunctionEnd

; Function Attrs: nounwind readnone
define spir_func <4 x float> @TestFabsVec(<4 x float> %x) local_unnamed_addr {
entry:
  %t = tail call <4 x float> @llvm.fabs.v4f32(<4 x float> %x)
  ret <4 x float> %t
}

declare half @llvm.fabs.f16(half)
declare float @llvm.fabs.f32(float)
declare double @llvm.fabs.f64(double)
declare <4 x float> @llvm.fabs.v4f32(<4 x float>)

; We checked several types with fabs, but the type check works the same for
; all intrinsics being translated, so for the rest we'll just test one type.

; CHECK: Function
; CHECK: ExtInst [[var1]] {{[0-9]+}} [[extinst_id]] ceil
; CHECK: FunctionEnd

define spir_func float @TestCeil(float %x) local_unnamed_addr {
entry:
  %t = tail call float @llvm.ceil.f32(float %x)
  ret float %t
}

declare float @llvm.ceil.f32(float)

; CHECK: Function
; CHECK: FunctionParameter {{[0-9]+}} [[x:[0-9]+]]
; CHECK: FunctionParameter {{[0-9]+}} [[n:[0-9]+]]
; CHECK: ExtInst [[var1]] {{[0-9]+}} [[extinst_id]] pown [[x]] [[n]]
; CHECK: FunctionEnd

define spir_func float @TestPowi(float %x, i32 %n) local_unnamed_addr {
entry:
  %t = tail call float @llvm.powi.f32(float %x, i32 %n)
  ret float %t
}

declare float @llvm.powi.f32(float, i32)

; CHECK: Function
; CHECK: ExtInst [[var1]] {{[0-9]+}} [[extinst_id]] sin
; CHECK: FunctionEnd

define spir_func float @TestSin(float %x) local_unnamed_addr {
entry:
  %t = tail call float @llvm.sin.f32(float %x)
  ret float %t
}

declare float @llvm.sin.f32(float)

; CHECK: Function
; CHECK: ExtInst [[var1]] {{[0-9]+}} [[extinst_id]] cos
; CHECK: FunctionEnd

define spir_func float @TestCos(float %x) local_unnamed_addr {
entry:
  %t = tail call float @llvm.cos.f32(float %x)
  ret float %t
}

declare float @llvm.cos.f32(float)

; CHECK: Function
; CHECK: FunctionParameter {{[0-9]+}} [[x:[0-9]+]]
; CHECK: FunctionParameter {{[0-9]+}} [[y:[0-9]+]]
; CHECK: ExtInst [[var1]] {{[0-9]+}} [[extinst_id]] pow [[x]] [[y]]
; CHECK: FunctionEnd

define spir_func float @TestPow(float %x, float %y) local_unnamed_addr {
entry:
  %t = tail call float @llvm.pow.f32(float %x, float %y)
  ret float %t
}

declare float @llvm.pow.f32(float, float)

; CHECK: Function
; CHECK: ExtInst [[var1]] {{[0-9]+}} [[extinst_id]] exp
; CHECK: FunctionEnd

define spir_func float @TestExp(float %x) local_unnamed_addr {
entry:
  %t = tail call float @llvm.exp.f32(float %x)
  ret float %t
}

declare float @llvm.exp.f32(float)

; CHECK: Function
; CHECK: ExtInst [[var1]] {{[0-9]+}} [[extinst_id]] exp2
; CHECK: FunctionEnd

define spir_func float @TestExp2(float %x) local_unnamed_addr {
entry:
  %t = tail call float @llvm.exp2.f32(float %x)
  ret float %t
}

declare float @llvm.exp2.f32(float)

; CHECK: Function
; CHECK: ExtInst [[var1]] {{[0-9]+}} [[extinst_id]] log
; CHECK: FunctionEnd

define spir_func float @TestLog(float %x) local_unnamed_addr {
entry:
  %t = tail call float @llvm.log.f32(float %x)
  ret float %t
}

declare float @llvm.log.f32(float)

; CHECK: Function
; CHECK: ExtInst [[var1]] {{[0-9]+}} [[extinst_id]] log10
; CHECK: FunctionEnd

define spir_func float @TestLog10(float %x) local_unnamed_addr {
entry:
  %t = tail call float @llvm.log10.f32(float %x)
  ret float %t
}

declare float @llvm.log10.f32(float)

; CHECK: Function
; CHECK: ExtInst [[var1]] {{[0-9]+}} [[extinst_id]] log2
; CHECK: FunctionEnd

define spir_func float @TestLog2(float %x) local_unnamed_addr {
entry:
  %t = tail call float @llvm.log2.f32(float %x)
  ret float %t
}

declare float @llvm.log2.f32(float)

; CHECK: Function
; CHECK: FunctionParameter {{[0-9]+}} [[x:[0-9]+]]
; CHECK: FunctionParameter {{[0-9]+}} [[y:[0-9]+]]
; CHECK: ExtInst {{[0-9]+}} [[res:[0-9]+]] {{[0-9]+}} fmin [[x]] [[y]]
; CHECK: ReturnValue [[res]]

define spir_func float @TestMinNum(float %x, float %y) {
entry:
  %t = call float @llvm.minnum.f32(float %x, float %y)
  ret float %t
}

declare float @llvm.minnum.f32(float, float)

; CHECK: Function
; CHECK: FunctionParameter {{[0-9]+}} [[x:[0-9]+]]
; CHECK: FunctionParameter {{[0-9]+}} [[y:[0-9]+]]
; CHECK: ExtInst {{[0-9]+}} [[res:[0-9]+]] {{[0-9]+}} fmax [[x]] [[y]]
; CHECK: ReturnValue [[res]]

define spir_func float @TestMaxNum(float %x, float %y) {
entry:
  %t = call float @llvm.maxnum.f32(float %x, float %y)
  ret float %t
}

declare float @llvm.maxnum.f32(float, float)

; CHECK: Function
; CHECK: FunctionParameter {{[0-9]+}} [[x:[0-9]+]]
; CHECK: FunctionParameter {{[0-9]+}} [[y:[0-9]+]]
; CHECK: ExtInst {{[0-9]+}} [[res:[0-9]+]] {{[0-9]+}} fmin [[x]] [[y]]
; CHECK: ReturnValue [[res]]

define spir_func float @TestMinimum(float %x, float %y) {
entry:
  %t = call float @llvm.minimum.f32(float %x, float %y)
  ret float %t
}

declare float @llvm.minimum.f32(float, float)

; CHECK: Function
; CHECK: FunctionParameter {{[0-9]+}} [[x:[0-9]+]]
; CHECK: FunctionParameter {{[0-9]+}} [[y:[0-9]+]]
; CHECK: ExtInst {{[0-9]+}} [[res:[0-9]+]] {{[0-9]+}} fmax [[x]] [[y]]
; CHECK: ReturnValue [[res]]

define spir_func float @TestMaximum(float %x, float %y) {
entry:
  %t = call float @llvm.maximum.f32(float %x, float %y)
  ret float %t
}

declare float @llvm.maximum.f32(float, float)

; CHECK: Function
; CHECK: FunctionParameter {{[0-9]+}} [[x:[0-9]+]]
; CHECK: FunctionParameter {{[0-9]+}} [[y:[0-9]+]]
; CHECK: ExtInst [[var1]] {{[0-9]+}} [[extinst_id]] copysign [[x]] [[y]]
; CHECK: FunctionEnd

define spir_func float @TestCopysign(float %x, float %y) local_unnamed_addr {
entry:
  %t = tail call float @llvm.copysign.f32(float %x, float %y)
  ret float %t
}

declare float @llvm.copysign.f32(float, float)

; CHECK: Function
; CHECK: ExtInst [[var1]] {{[0-9]+}} [[extinst_id]] floor
; CHECK: FunctionEnd

define spir_func float @TestFloor(float %x) local_unnamed_addr {
entry:
  %t = tail call float @llvm.floor.f32(float %x)
  ret float %t
}

declare float @llvm.floor.f32(float)

; CHECK: Function
; CHECK: ExtInst [[var1]] {{[0-9]+}} [[extinst_id]] trunc
; CHECK: FunctionEnd

define spir_func float @TestTrunc(float %x) local_unnamed_addr {
entry:
  %t = tail call float @llvm.trunc.f32(float %x)
  ret float %t
}

declare float @llvm.trunc.f32(float)

; CHECK: Function
; CHECK: ExtInst [[var1]] {{[0-9]+}} [[extinst_id]] rint
; CHECK: FunctionEnd

define spir_func float @TestRint(float %x) local_unnamed_addr {
entry:
  %t = tail call float @llvm.rint.f32(float %x)
  ret float %t
}

declare float @llvm.rint.f32(float)

; It is intentional that nearbyint translates to rint.
; CHECK: Function
; CHECK: ExtInst [[var1]] {{[0-9]+}} [[extinst_id]] rint
; CHECK: FunctionEnd

define spir_func float @TestNearbyint(float %x) local_unnamed_addr {
entry:
  %t = tail call float @llvm.nearbyint.f32(float %x)
  ret float %t
}

declare float @llvm.nearbyint.f32(float)

; CHECK: Function
; CHECK: ExtInst [[var1]] {{[0-9]+}} [[extinst_id]] round
; CHECK: FunctionEnd

define spir_func float @TestRound(float %x) local_unnamed_addr {
entry:
  %t = tail call float @llvm.round.f32(float %x)
  ret float %t
}

declare float @llvm.round.f32(float)

; It is intentional that roundeven translates to rint.
; CHECK: Function
; CHECK: ExtInst [[var1]] {{[0-9]+}} [[extinst_id]] rint
; CHECK: FunctionEnd

define spir_func float @TestRoundEven(float %x) local_unnamed_addr {
entry:
  %t = tail call float @llvm.roundeven.f32(float %x)
  ret float %t
}

declare float @llvm.roundeven.f32(float)

; CHECK: Function
; CHECK: FunctionParameter {{[0-9]+}} [[x:[0-9]+]]
; CHECK: FunctionParameter {{[0-9]+}} [[y:[0-9]+]]
; CHECK: FunctionParameter {{[0-9]+}} [[z:[0-9]+]]
; CHECK: ExtInst [[var1]] {{[0-9]+}} [[extinst_id]] fma [[x]] [[y]] [[z]]
; CHECK: FunctionEnd

define spir_func float @TestFma(float %x, float %y, float %z) {
entry:
  %t = tail call float @llvm.fma.f32(float %x, float %y, float %z)
  ret float %t
}

declare float @llvm.fma.f32(float, float, float)

; CHECK: Function
; CHECK: FunctionParameter {{[0-9]+}} [[x:[0-9]+]]
; CHECK: ExtInst [[var1]] {{[0-9]+}} [[extinst_id]] acos [[x]]
; CHECK: FunctionEnd

define spir_func float @TestAcos(float %x) {
entry:
  %t = tail call float @llvm.acos.f32(float %x)
  ret float %t
}

declare float @llvm.acos.f32(float)

; CHECK: Function
; CHECK: FunctionParameter {{[0-9]+}} [[x:[0-9]+]]
; CHECK: ExtInst [[var1]] {{[0-9]+}} [[extinst_id]] asin [[x]]
; CHECK: FunctionEnd

define spir_func float @TestAsin(float %x) {
entry:
  %t = tail call float @llvm.asin.f32(float %x)
  ret float %t
}

declare float @llvm.asin.f32(float)

; CHECK: Function
; CHECK: FunctionParameter {{[0-9]+}} [[x:[0-9]+]]
; CHECK: ExtInst [[var1]] {{[0-9]+}} [[extinst_id]] atan [[x]]
; CHECK: FunctionEnd

define spir_func float @TestAtan(float %x) {
entry:
  %t = tail call float @llvm.atan.f32(float %x)
  ret float %t
}

declare float @llvm.atan.f32(float)

; CHECK: Function
; CHECK: FunctionParameter {{[0-9]+}} [[x:[0-9]+]]
; CHECK: ExtInst [[var1]] {{[0-9]+}} [[extinst_id]] cosh [[x]]
; CHECK: FunctionEnd

define spir_func float @TestCosh(float %x) {
entry:
  %t = tail call float @llvm.cosh.f32(float %x)
  ret float %t
}

declare float @llvm.cosh.f32(float)

; CHECK: Function
; CHECK: FunctionParameter {{[0-9]+}} [[x:[0-9]+]]
; CHECK: ExtInst [[var1]] {{[0-9]+}} [[extinst_id]] sinh [[x]]
; CHECK: FunctionEnd

define spir_func float @TestSinh(float %x) {
entry:
  %t = tail call float @llvm.sinh.f32(float %x)
  ret float %t
}

declare float @llvm.sinh.f32(float)

; CHECK: Function
; CHECK: FunctionParameter {{[0-9]+}} [[x:[0-9]+]]
; CHECK: ExtInst [[var1]] {{[0-9]+}} [[extinst_id]] tanh [[x]]
; CHECK: FunctionEnd

define spir_func float @TestTanh(float %x) {
entry:
  %t = tail call float @llvm.tanh.f32(float %x)
  ret float %t
}

declare float @llvm.tanh.f32(float)

; CHECK: Function
; CHECK: FunctionParameter {{[0-9]+}} [[x:[0-9]+]]
; CHECK: FunctionParameter {{[0-9]+}} [[y:[0-9]+]]
; CHECK: ExtInst [[var1]] {{[0-9]+}} [[extinst_id]] atan2 [[y]] [[x]]
; CHECK: FunctionEnd

define spir_func float @TestAtan2(float %x, float %y) {
entry:
  %t = tail call float @llvm.atan2.f32(float %y, float %x)
  ret float %t
}

declare float @llvm.atan2.f32(float, float)

; CHECK: Function [[ResTy:[0-9]+]]
; CHECK: FunctionParameter {{[0-9]+}} [[x:[0-9]+]]
; CHECK: Variable [[PtrTy:[0-9]+]] [[Ptr:[0-9]+]] 7
; CHECK: ExtInst [[var2]] [[ResFirstElem:[0-9]+]] [[extinst_id]] modf [[x]] [[Ptr]]
; CHECK: Load [[var2]] [[ResSecondElem:[0-9]+]] [[Ptr]]
; CHECK: CompositeConstruct [[ResTy]] [[RetVal:[0-9]+]] [[ResFirstElem]] [[ResSecondElem]]
; CHECK: ReturnValue [[RetVal]]
; CHECK: FunctionEnd


define spir_func {double, double} @TestModf(double %x) {
entry:
  %t = tail call {double, double} @llvm.modf.f64(double %x)
  ret {double, double} %t
}

declare {double, double} @llvm.modf.f64(double)
