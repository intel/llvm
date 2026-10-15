; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text
; RUN: FileCheck < %t.spt %s --check-prefixes=CHECK-SPIRV,CHECK-SPIRV-TYPED-PTR
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc
; RUN: FileCheck < %t.rev.ll %s --check-prefix=CHECK-LLVM

; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_KHR_untyped_pointers -spirv-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefixes=CHECK-SPIRV,CHECK-SPIRV-UNTYPED-PTR
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_KHR_untyped_pointers -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc
; RUN: FileCheck < %t.rev.ll %s --check-prefix=CHECK-LLVM

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-unknown"

; CHECK-SPIRV: ExtInstImport [[#ExtInstSetId:]] "OpenCL.std"

; CHECK-SPIRV: TypeInt [[#TypeInt:]] 32
; CHECK-SPIRV: TypeFloat [[#TypeFloat:]] 32
; CHECK-SPIRV: TypeStruct [[#TypeStrFloatInt:]] [[#TypeFloat]] [[#TypeInt]]
; CHECK-SPIRV-TYPED-PTR: TypePointer [[#TypeIntPtr:]] 7 [[#TypeInt]]
; CHECK-SPIRV-UNTYPED-PTR: TypeUntypedPointerKHR [[#TypePtr:]] 7

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
; CHECK-SPIRV-TYPED-PTR: Variable [[#TypeIntPtr]] [[#IntVar:]] 7
; CHECK-SPIRV-UNTYPED-PTR: UntypedVariableKHR [[#TypePtr]] [[#IntVar:]] 7 [[#TypeInt]]
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

; CHECK-SPIRV: ExtInst [[#TypeFloat]] [[#Frexp0:]] [[#ExtInstSetId]] frexp [[#]] [[#IntVar0:]]
; CHECK-SPIRV: Load [[#TypeInt]] [[#Exp0:]] [[#IntVar0]]
; CHECK-SPIRV: CompositeConstruct [[#TypeStrFloatInt]] [[#Res0:]] [[#Frexp0]] [[#Exp0]]
; CHECK-SPIRV: CompositeExtract [[#TypeFloat]] [[#Mant0:]] [[#Res0]] 0
; CHECK-SPIRV: ExtInst [[#TypeFloat]] [[#Frexp1:]] [[#ExtInstSetId]] frexp [[#Mant0]] [[#IntVar1:]]
; CHECK-SPIRV: Load [[#TypeInt]] [[#Exp1:]] [[#IntVar1]]
; CHECK-SPIRV: CompositeConstruct [[#TypeStrFloatInt]] [[#Res1:]] [[#Frexp1]] [[#Exp1]]
; CHECK-SPIRV: ReturnValue [[#Res1]]

; CHECK-LLVM: %[[#IntVar1:]] = alloca i32
; CHECK-LLVM: %[[#IntVar2:]] = alloca i32
; CHECK-LLVM: %[[Frexp0:[a-z0-9.]+]] = call spir_func float @_Z5frexpfPi(float %x, ptr %[[#IntVar1]])
; CHECK-LLVM: %[[#LoadIntVar0:]] = load i32, ptr %[[#IntVar1]]
; CHECK-LLVM: %[[#AllocaStr0:]] = alloca %[[StrTypeFloatInt]]
; CHECK-LLVM: %[[GEPFloat0:[a-z0-9]+]] = getelementptr inbounds %[[StrTypeFloatInt]], ptr %[[#AllocaStr0]], i32 0, i32 0
; CHECK-LLVM: store float %[[Frexp0]], ptr %[[GEPFloat0]]
; CHECK-LLVM: %[[GEPInt0:[a-z0-9]+]] = getelementptr inbounds %[[StrTypeFloatInt]], ptr %[[#AllocaStr0]], i32 0, i32 1
; CHECK-LLVM: store i32 %[[#LoadIntVar0]], ptr %[[GEPInt0]]
; CHECK-LLVM: %[[Str0:[a-z0-9.]+]] = load %[[StrTypeFloatInt]], ptr %[[#AllocaStr0]]
; CHECK-LLVM: %[[Mant0:[a-z0-9.]+]] = extractvalue %[[StrTypeFloatInt]] %[[Str0]], 0
; CHECK-LLVM: %[[Frexp1:[a-z0-9.]+]] = call spir_func float @_Z5frexpfPi(float %[[Mant0]], ptr %[[#IntVar2]])
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

; CHECK-SPIRV: ExtInst [[#VecDouble2]] [[#Frexp0:]] [[#ExtInstSetId]] frexp [[#]] [[#IntVar0:]]
; CHECK-SPIRV: Load [[#VecInt2]] [[#Exp0:]] [[#IntVar0]]
; CHECK-SPIRV: CompositeConstruct [[#TypeStrDoubleIntVec2]] [[#Res0:]] [[#Frexp0]] [[#Exp0]]
; CHECK-SPIRV: CompositeExtract [[#VecDouble2]] [[#Mant0:]] [[#Res0]] 0
; CHECK-SPIRV: ExtInst [[#VecDouble2]] [[#Frexp1:]] [[#ExtInstSetId]] frexp [[#Mant0]] [[#IntVar1:]]
; CHECK-SPIRV: Load [[#VecInt2]] [[#Exp1:]] [[#IntVar1]]
; CHECK-SPIRV: CompositeConstruct [[#TypeStrDoubleIntVec2]] [[#Res1:]] [[#Frexp1]] [[#Exp1]]
; CHECK-SPIRV: ReturnValue [[#Res1]]

; CHECK-LLVM: %[[FrexpVec0:[a-z0-9.]+]] = call spir_func <2 x double> @_Z5frexpDv2_dPDv2_i(<2 x double> %x, ptr %[[#IntVarVec0:]])
; CHECK-LLVM: %[[ExpVec0:[a-z0-9.]+]] = load <2 x i32>, ptr %[[#IntVarVec0]]
; CHECK-LLVM: %[[#AllocaStrVec0:]] = alloca %[[StrTypeDoubleIntVec2]]
; CHECK-LLVM: %[[GEPVecDouble:[a-z0-9]+]] = getelementptr inbounds %[[StrTypeDoubleIntVec2]], ptr %[[#AllocaStrVec0]], i32 0, i32 0
; CHECK-LLVM: store <2 x double> %[[FrexpVec0]], ptr %[[GEPVecDouble]]
; CHECK-LLVM: %[[GEPVecInt:[a-z0-9]+]] = getelementptr inbounds %[[StrTypeDoubleIntVec2]], ptr %[[#AllocaStrVec0]], i32 0, i32 1
; CHECK-LLVM: store <2 x i32> %[[ExpVec0]], ptr %[[GEPVecInt]]
; CHECK-LLVM: %[[StrVec0:[a-z0-9.]+]] = load %[[StrTypeDoubleIntVec2]], ptr %[[#AllocaStrVec0]]
; CHECK-LLVM: %[[Mant0:[a-z0-9.]+]] = extractvalue %[[StrTypeDoubleIntVec2]] %[[StrVec0]], 0
; CHECK-LLVM: call spir_func <2 x double> @_Z5frexpDv2_dPDv2_i(<2 x double> %[[Mant0]], ptr %[[#]])
; CHECK-LLVM: ret %[[StrTypeDoubleIntVec2]]
define { <2 x double>, <2 x i32> } @frexp_frexp_vector(<2 x double> %x) {
  %frexp0 = call { <2 x double>, <2 x i32> } @llvm.frexp.v2f64.v2i32(<2 x double> %x)
  %frexp0.0 = extractvalue { <2 x double>, <2 x i32> } %frexp0, 0
  %frexp1 = call { <2 x double>, <2 x i32> } @llvm.frexp.v2f64.v2i32(<2 x double> %frexp0.0)
  ret { <2 x double>, <2 x i32> } %frexp1
}

; CHECK-SPIRV: ExtInst [[#TypeFloat]] [[#FrexpId:]] [[#ExtInstSetId]] frexp [[#]] [[#IntVar:]]
; CHECK-SPIRV: Load [[#TypeInt]] [[#LoadId:]] [[#IntVar]]
; CHECK-SPIRV: CompositeConstruct [[#TypeStrFloatInt]] [[#Res:]] [[#FrexpId]] [[#LoadId]]
; CHECK-SPIRV: CompositeExtract [[#TypeInt]] [[#Exp:]] [[#Res]] 1
; CHECK-SPIRV: ReturnValue [[#Exp]]

; CHECK-LLVM: %[[#IntVar:]] = alloca i32
; CHECK-LLVM: %[[Frexp:[a-z0-9.]+]] = call spir_func float @_Z5frexpfPi(float %x, ptr %[[#IntVar]])
; CHECK-LLVM: %[[LoadVar:[a-z0-9.]+]] = load i32, ptr %[[#IntVar]]
; CHECK-LLVM: %[[#AllocaStr:]] = alloca %[[StrTypeFloatInt]]
; CHECK-LLVM: %[[GEPFloat:[a-z0-9]+]] = getelementptr inbounds %[[StrTypeFloatInt]], ptr %[[#AllocaStr]], i32 0, i32 0
; CHECK-LLVM: store float %[[Frexp]], ptr %[[GEPFloat]]
; CHECK-LLVM: %[[GEPInt:[a-z0-9]+]] = getelementptr inbounds %[[StrTypeFloatInt]], ptr %[[#AllocaStr]], i32 0, i32 1
; CHECK-LLVM: store i32 %[[LoadVar]], ptr %[[GEPInt]]
; CHECK-LLVM: %[[Str:[a-z0-9.]+]] = load %[[StrTypeFloatInt]], ptr %[[#AllocaStr]]
; CHECK-LLVM: %[[Exp:[a-z0-9.]+]] = extractvalue %[[StrTypeFloatInt]] %[[Str]], 1
; CHECK-LLVM: ret i32 %[[Exp]]
define i32 @frexp_frexp_get_int(float %x) {
  %frexp0 = call { float, i32 } @llvm.frexp.f32.i32(float %x)
  %frexp0.0 = extractvalue { float, i32 } %frexp0, 1
  ret i32 %frexp0.0
}

; CHECK-SPIRV: Function [[#TypeInt]]
; CHECK-SPIRV: Label
; CHECK-SPIRV-TYPED-PTR-NEXT: Variable [[#TypeIntPtr]] [[#IntVar:]] 7
; CHECK-SPIRV-UNTYPED-PTR-NEXT: UntypedVariableKHR [[#TypePtr]] [[#IntVar:]] 7 [[#TypeInt]]
; CHECK-SPIRV-NEXT: Branch
; CHECK-SPIRV: ExtInst [[#TypeFloat]] [[#]] [[#ExtInstSetId]] frexp [[#]] [[#IntVar]]

; CHECK-LLVM-LABEL: define spir_func i32 @frexp_non_entry_block
; CHECK-LLVM: entry:
; CHECK-LLVM-NEXT: %[[#IntVar:]] = alloca i32
; CHECK-LLVM-NEXT: br label %next
; CHECK-LLVM: call spir_func float @_Z5frexpfPi(float %x, ptr %[[#IntVar]])
; CHECK-LLVM: load i32, ptr %[[#IntVar]]
define i32 @frexp_non_entry_block(float %x) {
entry:
  br label %next

next:
  %frexp = call { float, i32 } @llvm.frexp.f32.i32(float %x)
  %exp = extractvalue { float, i32 } %frexp, 1
  ret i32 %exp
}

; The whole struct, not just the mantissa, has to reach an aggregate store.

; CHECK-SPIRV: ExtInst [[#TypeFloat]] [[#FrexpId:]] [[#ExtInstSetId]] frexp [[#]] [[#IntVar:]]
; CHECK-SPIRV: Load [[#TypeInt]] [[#LoadId:]] [[#IntVar]]
; CHECK-SPIRV: CompositeConstruct [[#TypeStrFloatInt]] [[#Res:]] [[#FrexpId]] [[#LoadId]]
; CHECK-SPIRV: Store [[#]] [[#Res]]

; CHECK-LLVM-LABEL: define spir_func void @frexp_store(
; CHECK-LLVM: %[[Frexp:[a-z0-9.]+]] = call spir_func float @_Z5frexpfPi(float %x, ptr %[[#IntVar:]])
; CHECK-LLVM: %[[Exp:[a-z0-9.]+]] = load i32, ptr %[[#IntVar]]
; CHECK-LLVM: %[[#AllocaStr:]] = alloca %[[StrTypeFloatInt]]
; CHECK-LLVM: %[[GEPFloat:[a-z0-9]+]] = getelementptr inbounds %[[StrTypeFloatInt]], ptr %[[#AllocaStr]], i32 0, i32 0
; CHECK-LLVM: store float %[[Frexp]], ptr %[[GEPFloat]]
; CHECK-LLVM: %[[GEPInt:[a-z0-9]+]] = getelementptr inbounds %[[StrTypeFloatInt]], ptr %[[#AllocaStr]], i32 0, i32 1
; CHECK-LLVM: store i32 %[[Exp]], ptr %[[GEPInt]]
; CHECK-LLVM: %[[Res:[a-z0-9.]+]] = load %[[StrTypeFloatInt]], ptr %[[#AllocaStr]]
; CHECK-LLVM: store %[[StrTypeFloatInt]] %[[Res]], ptr %p
define void @frexp_store(float %x, ptr %p) {
  %r = call { float, i32 } @llvm.frexp.f32.i32(float %x)
  store { float, i32 } %r, ptr %p
  ret void
}

; CHECK-SPIRV: ExtInst [[#VecFloat2]] [[#FrexpId:]] [[#ExtInstSetId]] frexp [[#]] [[#IntVar:]]
; CHECK-SPIRV: Load [[#VecInt2]] [[#LoadId:]] [[#IntVar]]
; CHECK-SPIRV: CompositeConstruct [[#TypeStrFloatIntVec2]] [[#Res:]] [[#FrexpId]] [[#LoadId]]
; CHECK-SPIRV: Store [[#]] [[#Res]]

; CHECK-LLVM-LABEL: define spir_func void @frexp_store_vector(
; CHECK-LLVM: %[[Frexp:[a-z0-9.]+]] = call spir_func <2 x float> @_Z5frexpDv2_fPDv2_i(<2 x float> %x, ptr %[[#IntVar:]])
; CHECK-LLVM: %[[Exp:[a-z0-9.]+]] = load <2 x i32>, ptr %[[#IntVar]]
; CHECK-LLVM: %[[#AllocaStr:]] = alloca %[[StrTypeFloatIntVec2]]
; CHECK-LLVM: %[[GEPVecFloat:[a-z0-9]+]] = getelementptr inbounds %[[StrTypeFloatIntVec2]], ptr %[[#AllocaStr]], i32 0, i32 0
; CHECK-LLVM: store <2 x float> %[[Frexp]], ptr %[[GEPVecFloat]]
; CHECK-LLVM: %[[GEPVecInt:[a-z0-9]+]] = getelementptr inbounds %[[StrTypeFloatIntVec2]], ptr %[[#AllocaStr]], i32 0, i32 1
; CHECK-LLVM: store <2 x i32> %[[Exp]], ptr %[[GEPVecInt]]
; CHECK-LLVM: %[[Res:[a-z0-9.]+]] = load %[[StrTypeFloatIntVec2]], ptr %[[#AllocaStr]]
; CHECK-LLVM: store %[[StrTypeFloatIntVec2]] %[[Res]], ptr %p
define void @frexp_store_vector(<2 x float> %x, ptr %p) {
  %r = call { <2 x float>, <2 x i32> } @llvm.frexp.v2f32.v2i32(<2 x float> %x)
  store { <2 x float>, <2 x i32> } %r, ptr %p
  ret void
}

; CHECK-SPIRV: ExtInst [[#TypeFloat]] [[#FrexpId:]] [[#ExtInstSetId]] frexp [[#]] [[#IntVar:]]
; CHECK-SPIRV: Load [[#TypeInt]] [[#LoadId:]] [[#IntVar]]
; CHECK-SPIRV: CompositeConstruct [[#TypeStrFloatInt]] [[#Res:]] [[#FrexpId]] [[#LoadId]]
; CHECK-SPIRV: CompositeInsert [[#TypeStrFloatInt]] [[#Ins:]] [[#]] [[#Res]] 0
; CHECK-SPIRV: ReturnValue [[#Ins]]

; CHECK-LLVM-LABEL: @frexp_insertvalue(
; CHECK-LLVM: %[[Frexp:[a-z0-9.]+]] = call spir_func float @_Z5frexpfPi(float %x, ptr %[[#IntVar:]])
; CHECK-LLVM: %[[Exp:[a-z0-9.]+]] = load i32, ptr %[[#IntVar]]
; CHECK-LLVM: %[[#AllocaStr:]] = alloca %[[StrTypeFloatInt]]
; CHECK-LLVM: %[[GEPFloat:[a-z0-9]+]] = getelementptr inbounds %[[StrTypeFloatInt]], ptr %[[#AllocaStr]], i32 0, i32 0
; CHECK-LLVM: store float %[[Frexp]], ptr %[[GEPFloat]]
; CHECK-LLVM: %[[GEPInt:[a-z0-9]+]] = getelementptr inbounds %[[StrTypeFloatInt]], ptr %[[#AllocaStr]], i32 0, i32 1
; CHECK-LLVM: store i32 %[[Exp]], ptr %[[GEPInt]]
; CHECK-LLVM: %[[Res:[a-z0-9.]+]] = load %[[StrTypeFloatInt]], ptr %[[#AllocaStr]]
; CHECK-LLVM: %[[Ins:[a-z0-9.]+]] = insertvalue %[[StrTypeFloatInt]] %[[Res]], float %y, 0
; CHECK-LLVM: ret %[[StrTypeFloatInt]] %[[Ins]]
define { float, i32 } @frexp_insertvalue(float %x, float %y) {
  %r = call { float, i32 } @llvm.frexp.f32.i32(float %x)
  %i = insertvalue { float, i32 } %r, float %y, 0
  ret { float, i32 } %i
}
