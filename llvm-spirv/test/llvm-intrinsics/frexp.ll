; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc
; RUN: FileCheck < %t.rev.ll %s --check-prefix=CHECK-LLVM

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-unknown"

; CHECK-SPIRV: ExtInstImport [[#ExtInstSetId:]] "OpenCL.std"

; CHECK-SPIRV: TypeInt [[#TypeInt:]] 32
; CHECK-SPIRV: TypeFloat [[#TypeFloat:]] 32
; CHECK-SPIRV: TypeStruct [[#TypeStrFloatInt:]] [[#TypeFloat]] [[#TypeInt]]
; CHECK-SPIRV: TypePointer [[#TypeIntPtr:]] 7 [[#TypeInt]]

; CHECK-SPIRV: TypeFloat [[#TypeDouble:]] 64
; CHECK-SPIRV: TypeStruct [[#TypeStrDoubleInt:]] [[#TypeDouble]] [[#TypeInt]]

; CHECK-SPIRV: TypeVector [[#VecFloat2:]] [[#TypeFloat]] 2
; CHECK-SPIRV: TypeVector [[#VecInt2:]] [[#TypeInt]] 2
; CHECK-SPIRV: TypeStruct [[#TypeStrFloatIntVec2:]] [[#VecFloat2]] [[#VecInt2]]

; CHECK-SPIRV: TypeVector [[#VecFloat4:]] [[#TypeFloat]] 4
; CHECK-SPIRV: TypeVector [[#VecInt4:]] [[#TypeInt]] 4
; CHECK-SPIRV: TypeStruct [[#TypeStrFloatIntVec4:]] [[#VecFloat4]] [[#VecInt4]]

; CHECK-SPIRV: TypeVector [[#VecDouble2:]] [[#TypeDouble]] 2
; CHECK-SPIRV: TypeStruct [[#TypeStrDoubleIntVec2:]] [[#VecDouble2]] [[#VecInt2]]

; CHECK-SPIRV: Constant [[#TypeFloat]] [[#NegatedZeroConst:]] 2147483648
; CHECK-SPIRV: Undef [[#TypeDouble]] [[#UndefDouble:]]
; CHECK-SPIRV: ConstantNull [[#VecFloat2]] [[#NullVecFloat2:]]
; CHECK-SPIRV: Constant [[#TypeFloat]] [[#ZeroConstFloat:]] 0
; CHECK-SPIRV: ConstantComposite [[#VecFloat2]] [[#ZeroesCompositeFloat:]] [[#ZeroConstFloat]] [[#NegatedZeroConst]]

; CHECK-LLVM: %[[StrTypeFloatInt:[a-z0-9.]+]] = type { float, i32 }
; CHECK-LLVM: %[[StrTypeDoubleInt:[a-z0-9.]+]] = type { double, i32 }
; CHECK-LLVM: %[[StrTypeFloatIntVec2:[a-z0-9.]+]] = type { <2 x float>, <2 x i32> }
; CHECK-LLVM: %[[StrTypeFloatIntVec4:[a-z0-9.]+]] = type { <4 x float>, <4 x i32> }
; CHECK-LLVM: %[[StrTypeDoubleIntVec2:[a-z0-9.]+]] = type { <2 x double>, <2 x i32> }

declare { float, i32 } @llvm.frexp.f32.i32(float)
declare { double, i32 } @llvm.frexp.f64.i32(double)
declare { <2 x float>, <2 x i32> } @llvm.frexp.v2f32.v2i32(<2 x float>)
declare { <4 x float>, <4 x i32> } @llvm.frexp.v4f32.v4i32(<4 x float>)
declare { <2 x double>, <2 x i32> } @llvm.frexp.v2f64.v2i32(<2 x double>)

; CHECK-SPIRV: Function [[#TypeStrFloatInt:]]
; CHECK-SPIRV: Variable [[#TypeIntPtr]] [[#IntVar:]] 7
; CHECK-SPIRV: ExtInst [[#TypeFloat]] [[#FrexpId:]] [[#ExtInstSetId]] frexp [[#NegatedZeroConst]] [[#IntVar]]
; CHECK-SPIRV: Load [[#]] [[#LoadId:]] [[#]]
; CHECK-SPIRV: CompositeConstruct [[#TypeStrFloatInt]] [[#ComposConstr:]] [[#FrexpId]] [[#LoadId]]
; CHECK-SPIRV: ReturnValue [[#ComposConstr]]

; CHECK-LLVM: %[[#IntVar:]] = alloca i32
; CHECK-LLVM: %[[Frexp:[a-z0-9]+]] = call spir_func float @_Z5frexpfPi(float -0.000000e+00, ptr %[[#IntVar]])
; CHECK-LLVM: %[[#LoadIntVar:]] = load i32, ptr %[[#IntVar]]
; CHECK-LLVM: %[[#AllocaStrFloatInt:]] = alloca %[[StrTypeFloatInt]]
; CHECK-LLVM: %[[GEPFloat:[a-z0-9]+]] = getelementptr inbounds %structtype, ptr %[[#AllocaStrFloatInt]], i32 0, i32 0
; CHECK-LLVM: store float %[[Frexp]], ptr %[[GEPFloat]]
; CHECK-LLVM: %[[GEPInt:[a-z0-9]+]] = getelementptr inbounds %structtype, ptr %[[#AllocaStrFloatInt]], i32 0, i32 1
; CHECK-LLVM: store i32 %[[#LoadIntVar]], ptr %[[GEPInt]]
; CHECK-LLVM: %[[LoadStrFloatInt:[a-z0-9]+]] = load %[[StrTypeFloatInt]], ptr %[[#AllocaStrFloatInt]]
; CHECK-LLVM: ret %[[StrTypeFloatInt]] %[[LoadStrFloatInt]]
define { float, i32 } @frexp_negzero() {
  %ret = call { float, i32 } @llvm.frexp.f32.i32(float -0.0)
  ret { float, i32 } %ret
}

; CHECK-SPIRV: ExtInst [[#TypeDouble]] [[#]] [[#ExtInstSetId]] frexp [[#UndefDouble]] [[#]]
; CHECK-LLVM: call spir_func double @_Z5frexpdPi(double undef, ptr %[[#]])
; CHECK-LLVM: ret %[[StrTypeDoubleInt]]
define { double, i32 } @frexp_undef() {
  %ret = call { double, i32 } @llvm.frexp.f64.i32(double undef)
  ret { double, i32 } %ret
}

; CHECK-SPIRV: ExtInst [[#VecFloat2]] [[#]] [[#ExtInstSetId]] frexp [[#NullVecFloat2]] [[#]]
; CHECK-LLVM: call spir_func <2 x float> @_Z5frexpDv2_fPDv2_i(<2 x float> zeroinitializer, ptr %[[#]])
; CHECK-LLVM: ret %[[StrTypeFloatIntVec2]]
define { <2 x float>, <2 x i32> } @frexp_zero_vector() {
  %ret = call { <2 x float>, <2 x i32> } @llvm.frexp.v2f32.v2i32(<2 x float> zeroinitializer)
  ret { <2 x float>, <2 x i32> } %ret
}

; CHECK-SPIRV: ExtInst [[#VecFloat2]] [[#]] [[#ExtInstSetId]] frexp [[#ZeroesCompositeFloat]] [[#]]
; CHECK-LLVM: call spir_func <2 x float> @_Z5frexpDv2_fPDv2_i(<2 x float> <float 0.000000e+00, float -0.000000e+00>, ptr %[[#]])
; CHECK-LLVM: ret %[[StrTypeFloatIntVec2]]
define { <2 x float>, <2 x i32> } @frexp_zero_negzero_vector() {
  %ret = call { <2 x float>, <2 x i32> } @llvm.frexp.v2f32.v2i32(<2 x float> <float 0.0, float -0.0>)
  ret { <2 x float>, <2 x i32> } %ret
}

; CHECK-SPIRV: ExtInst [[#VecFloat4]] [[#]] [[#ExtInstSetId]] frexp [[#]] [[#]]
; CHECK-LLVM: call spir_func <4 x float> @_Z5frexpDv4_fPDv4_i(<4 x float> <float 1.600000e+01, float -3.200000e+01, float undef, float 9.999000e+03>, ptr %[[#]])
; CHECK-LLVM: ret %[[StrTypeFloatIntVec4]]
define { <4 x float>, <4 x i32> } @frexp_nonsplat_vector() {
  %ret = call { <4 x float>, <4 x i32> } @llvm.frexp.v4f32.v4i32(<4 x float> <float 16.0, float -32.0, float undef, float 9999.0>)
  ret { <4 x float>, <4 x i32> } %ret
}

; CHECK-SPIRV: ExtInst [[#TypeFloat]] [[#]] [[#ExtInstSetId]] frexp [[#]] [[#]]
; CHECK-SPIRV: ExtInst [[#TypeFloat]] [[#]] [[#ExtInstSetId]] frexp [[#]] [[#]]
; CHECK-LLVM: %[[#IntVar1:]] = alloca i32
; CHECK-LLVM: %[[Frexp0:[a-z0-9.]+]] = call spir_func float @_Z5frexpfPi(float %x, ptr %[[#IntVar1]])
; CHECK-LLVM: %[[#IntVar2:]] = alloca i32
; CHECK-LLVM: %[[Frexp1:[a-z0-9.]+]] = call spir_func float @_Z5frexpfPi(float %[[Frexp0]], ptr %[[#IntVar2]])
; CHECK-LLVM: %[[#LoadIntVar:]] = load i32, ptr %[[#IntVar2]]
; CHECK-LLVM: %[[#AllocaStrFloatInt:]] = alloca %[[StrTypeFloatInt]]
; CHECK-LLVM: %[[GEPFloat:[a-z0-9]+]] = getelementptr inbounds %structtype, ptr %[[#AllocaStrFloatInt]], i32 0, i32 0
; CHECK-LLVM: store float %[[Frexp1]], ptr %[[GEPFloat]]
; CHECK-LLVM: %[[GEPInt:[a-z0-9]+]] = getelementptr inbounds %structtype, ptr %[[#AllocaStrFloatInt]], i32 0, i32 1
; CHECK-LLVM: store i32 %[[#LoadIntVar]], ptr %[[GEPInt]]
; CHECK-LLVM: %[[LoadStrFloatInt:[a-z0-9]+]] = load %[[StrTypeFloatInt]], ptr %[[#AllocaStrFloatInt]]
; CHECK-LLVM: ret %[[StrTypeFloatInt]] %[[LoadStrFloatInt]]
define { float, i32 } @frexp_frexp(float %x) {
  %frexp0 = call { float, i32 } @llvm.frexp.f32.i32(float %x)
  %frexp0.0 = extractvalue { float, i32 } %frexp0, 0
  %frexp1 = call { float, i32 } @llvm.frexp.f32.i32(float %frexp0.0)
  ret { float, i32 } %frexp1
}

; CHECK-SPIRV: ExtInst [[#VecDouble2]] [[#]] [[#ExtInstSetId]] frexp [[#]] [[#]]
; CHECK-SPIRV: ExtInst [[#VecDouble2]] [[#]] [[#ExtInstSetId]] frexp [[#]] [[#]]
; CHECK-LLVM: %[[Frexp0:[a-z0-9.]+]] = call spir_func <2 x double> @_Z5frexpDv2_dPDv2_i(<2 x double> %x, ptr %[[#]])
; CHECK-LLVM: call spir_func <2 x double> @_Z5frexpDv2_dPDv2_i(<2 x double> %[[Frexp0]], ptr %[[#]])
; CHECK-LLVM: ret %[[StrTypeDoubleIntVec2]]
define { <2 x double>, <2 x i32> } @frexp_frexp_vector(<2 x double> %x) {
  %frexp0 = call { <2 x double>, <2 x i32> } @llvm.frexp.v2f64.v2i32(<2 x double> %x)
  %frexp0.0 = extractvalue { <2 x double>, <2 x i32> } %frexp0, 0
  %frexp1 = call { <2 x double>, <2 x i32> } @llvm.frexp.v2f64.v2i32(<2 x double> %frexp0.0)
  ret { <2 x double>, <2 x i32> } %frexp1
}

; CHECK-SPIRV: ExtInst [[#TypeFloat]] [[#]] [[#ExtInstSetId]] frexp [[#]] [[#]]
; CHECK-LLVM: %[[#IntVar:]] = alloca i32
; CHECK-LLVM: %[[Frexp:[a-z0-9.]+]] = call spir_func float @_Z5frexpfPi(float %x, ptr %[[#IntVar]])
; CHECK-LLVM: %[[LoadVar:[a-z0-9.]+]] = load i32, ptr %[[#IntVar]]
; CHECK-LLVM: ret i32 %[[LoadVar]]
define i32 @frexp_frexp_get_int(float %x) {
  %frexp0 = call { float, i32 } @llvm.frexp.f32.i32(float %x)
  %frexp0.0 = extractvalue { float, i32 } %frexp0, 1
  ret i32 %frexp0.0
}
