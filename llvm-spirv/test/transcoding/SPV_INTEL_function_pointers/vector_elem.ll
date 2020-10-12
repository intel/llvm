; RUN: llvm-as < %s | llvm-spirv -spirv-text --spirv-ext=+SPV_INTEL_function_pointers | FileCheck %s --check-prefix=CHECK-SPIRV

; CHECK-SPIRV-DAG: 6 Name [[F1:[0-9+]]] "_Z2f1u2CMvb32_j"
; CHECK-SPIRV-DAG: 6 Name [[F2:[0-9+]]] "_Z2f2u2CMvb32_j"
; CHECK-SPIRV-DAG: 4 Name [[Funcs:[0-9]+]] "Funcs"

; CHECK-SPIRV: 4 TypeInt [[TypeInt32:[0-9]+]] 32 0
; CHECK-SPIRV: 4 TypeFunction [[TypeFunc:[0-9]+]] [[TypeInt32]] [[TypeInt32]]
; CHECK-SPIRV: 4 TypePointer [[TypePtr:[0-9]+]] {{[0-9]+}} [[TypeFunc]]
; CHECK-SPIRV: 4 TypeVector [[TypeVec:[0-9]+]] [[TypePtr]] [[TypeInt32]]
; CHECK-SPIRV-DAG: 3 Undef [[TypeVec]] [[TypeUndef:[0-9]+]]
; CHECK-SPIRV-DAG: 4 ConstFunctionPointerINTEL [[TypePtr]] [[F1Ptr:[0-9]+]] [[F1]]
; CHECK-SPIRV-DAG: 4 ConstFunctionPointerINTEL [[TypePtr]] [[F2Ptr:[0-9]+]] [[F2]]

; CHECK-SPIRV: 6 CompositeInsert [[TypeVec]] [[NewVec0:[0-9]+]] [[F1Ptr]] [[TypeUndef]] 0
; CHECK-SPIRV: 6 CompositeInsert [[TypeVec]] [[NewVec1:[0-9]+]] [[F2Ptr]] [[NewVec0]] 1
; CHECK-SPIRV: 5 Store [[Funcs]] [[NewVec1]] [[TypeInt32]] {{[0-9+]}}


target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

; Function Attrs: noinline norecurse nounwind readnone
define internal i32 @_Z2f1u2CMvb32_j(i32 %x) {
entry:
  ret i32 %x
}
; Function Attrs: noinline norecurse nounwind readnone
define internal i32 @_Z2f2u2CMvb32_j(i32 %x) {
entry:
  ret i32 %x
}

; Function Attrs: noinline nounwind
define dllexport void @vadd() {
entry:
  %Funcs = alloca <2 x i32 (i32)*>, align 16
  %0 = insertelement <2 x i32 (i32)*> undef, i32 (i32)* @_Z2f1u2CMvb32_j, i32 0
  %1 = insertelement <2 x i32 (i32)*> %0, i32 (i32)* @_Z2f2u2CMvb32_j, i32 1
  store <2 x i32 (i32)*> %1, <2 x i32 (i32)*>* %Funcs, align 16
  ret void
}
