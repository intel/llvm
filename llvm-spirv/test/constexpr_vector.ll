
; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv --spirv-ext=+SPV_INTEL_function_pointers
; RUN: llvm-spirv %t.spv -to-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc
; RUN: FileCheck < %t.rev.ll %s --check-prefix=CHECK-LLVM


; CHECK-SPIRV-DAG: 4 Name [[#Funcs:]] "Funcs"
; CHECK-SPIRV-DAG: 4 Name [[#Funcs1:]] "Funcs1"
; CHECK-SPIRV-DAG: 6 Name [[#F1:]] "_Z2f1u2CMvb32_j"
; CHECK-SPIRV-DAG: 6 Name [[#F2:]] "_Z2f2u2CMvb32_j"

; CHECK-SPIRV-DAG: 4 TypeInt [[TypeInt8:[0-9]+]] 8 0
; CHECK-SPIRV-DAG: 4 TypeInt [[TypeInt32:[0-9]+]] 32 0
; CHECK-SPIRV-DAG: 4 TypeInt [[TypeInt64:[0-9]+]] 64 0

; 117 is OpConvertPtrToU opcode
; 81 is OpCompositeExtract opcode
; CHECK-SPIRV: SpecConstantOp [[#]] [[#FIRST_ARG:]] 117 [[#F1Ptr:]]
; CHECK-SPIRV: SpecConstantOp [[#]] [[#VEC_0:]] 81 [[#FIRST_COMPOSITE:]] 0
; CHECK-SPIRV: SpecConstantOp [[#]] [[#VEC_1:]] 81 [[#FIRST_COMPOSITE]] 1
; CHECK-SPIRV: SpecConstantOp [[#]] [[#VEC_2:]] 81 [[#FIRST_COMPOSITE]] 2
; CHECK-SPIRV: SpecConstantOp [[#]] [[#VEC_3:]] 81 [[#FIRST_COMPOSITE]] 3
; CHECK-SPIRV: SpecConstantOp [[#]] [[#VEC_4:]] 81 [[#FIRST_COMPOSITE]] 4
; CHECK-SPIRV: SpecConstantOp [[#]] [[#VEC_5:]] 81 [[#FIRST_COMPOSITE]] 5
; CHECK-SPIRV: SpecConstantOp [[#]] [[#VEC_6:]] 81 [[#FIRST_COMPOSITE]] 6
; CHECK-SPIRV: SpecConstantOp [[#]] [[#VEC_7:]] 81 [[#FIRST_COMPOSITE]] 7

; 117 is OpConvertPtrToU opcode
; 81 is OpCompositeExtract opcode
; CHECK-SPIRV: SpecConstantOp [[#]] [[#SECOND_ARG:]] 117 [[#F2Ptr:]]
; CHECK-SPIRV: SpecConstantOp [[#]] [[#VEC_8:]] 81 [[#SECOND_COMPOSITE:]] 0
; CHECK-SPIRV: SpecConstantOp [[#]] [[#VEC_9:]] 81 [[#SECOND_COMPOSITE]] 1
; CHECK-SPIRV: SpecConstantOp [[#]] [[#VEC_10:]] 81 [[#SECOND_COMPOSITE]] 2
; CHECK-SPIRV: SpecConstantOp [[#]] [[#VEC_11:]] 81 [[#SECOND_COMPOSITE]] 3
; CHECK-SPIRV: SpecConstantOp [[#]] [[#VEC_12:]] 81 [[#SECOND_COMPOSITE]] 4
; CHECK-SPIRV: SpecConstantOp [[#]] [[#VEC_13:]] 81 [[#SECOND_COMPOSITE]] 5
; CHECK-SPIRV: SpecConstantOp [[#]] [[#VEC_14:]] 81 [[#SECOND_COMPOSITE]] 6
; CHECK-SPIRV: SpecConstantOp [[#]] [[#VEC_15:]] 81 [[#SECOND_COMPOSITE]] 7

; CHECK-SPIRV-DAG: 4 TypeVector [[#TypeVec16:]] [[TypeInt8]] 16
; CHECK-SPIRV-DAG: 4 TypeVector [[#TypeVec8:]] [[TypeInt8]] 8
; CHECK-SPIRV-DAG: 4 TypeVector [[#TypeVec64:]] [[TypeInt64]] 2
; CHECK-SPIRV-DAG: 4 TypePointer [[#StorePtr:]] 7 [[#TypeVec16]]

; 124 is OpBitcast opcode
; CHECK-SPIRV-DAG: 4 ConstantFunctionPointerINTEL [[#FuncPtrTy:]] [[#F1Ptr]] [[#F1]]
; CHECK-SPIRV: SpecConstantOp [[#TypeVec8]] [[#FIRST_COMPOSITE]] 124 [[#FIRST_ARG]]
; CHECK-SPIRV-DAG: 4 ConstantFunctionPointerINTEL [[#FuncPtrTy]] [[#F2Ptr]] [[#F2]]
; CHECK-SPIRV: SpecConstantOp [[#TypeVec8]] [[#SECOND_COMPOSITE]] 124 [[#SECOND_ARG]]
; CHECK-SPIRV: ConstantComposite [[#TypeVec16]] [[#FUNCS_COMPOSITE:]] [[#VEC_0]] [[#VEC_1]] [[#VEC_2]] [[#VEC_3]] [[#VEC_4]] [[#VEC_5]] [[#VEC_6]] [[#VEC_7]] [[#VEC_8]] [[#VEC_9]] [[#VEC_10]] [[#VEC_11]] [[#VEC_12]] [[#VEC_13]] [[#VEC_14]] [[#VEC_15]]
; CHECK-SPIRV: ConstantComposite [[#TypeVec64]] [[#FUNCS1_COMPOSITE:]] [[#FIRST_ARG]] [[#SECOND_ARG]]

; CHECK-SPIRV: 5 Store [[#Funcs]] [[#FUNCS_COMPOSITE]] [[TypeInt32]] [[#StorePtr]]
; CHECK-SPIRV: 5 Store [[#Funcs1]] [[#FUNCS1_COMPOSITE]] [[TypeInt32]] [[#StorePtr]]

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

; CHECK-LLVM: define spir_func void @vadd()
; CHECK-LLVM: %Funcs = alloca <16 x i8>, align 16
; CHECK-LLVM: store <16 x i8> <i8 extractelement (<8 x i8> bitcast (i64 ptrtoint (i32 (i32)* @_Z2f1u2CMvb32_j to i64) to <8 x i8>), i32 0), i8 extractelement (<8 x i8> bitcast (i64 ptrtoint (i32 (i32)* @_Z2f1u2CMvb32_j to i64) to <8 x i8>), i32 1), i8 extractelement (<8 x i8> bitcast (i64 ptrtoint (i32 (i32)* @_Z2f1u2CMvb32_j to i64) to <8 x i8>), i32 2), i8 extractelement (<8 x i8> bitcast (i64 ptrtoint (i32 (i32)* @_Z2f1u2CMvb32_j to i64) to <8 x i8>), i32 3), i8 extractelement (<8 x i8> bitcast (i64 ptrtoint (i32 (i32)* @_Z2f1u2CMvb32_j to i64) to <8 x i8>), i32 4), i8 extractelement (<8 x i8> bitcast (i64 ptrtoint (i32 (i32)* @_Z2f1u2CMvb32_j to i64) to <8 x i8>), i32 5), i8 extractelement (<8 x i8> bitcast (i64 ptrtoint (i32 (i32)* @_Z2f1u2CMvb32_j to i64) to <8 x i8>), i32 6), i8 extractelement (<8 x i8> bitcast (i64 ptrtoint (i32 (i32)* @_Z2f1u2CMvb32_j to i64) to <8 x i8>), i32 7), i8 extractelement (<8 x i8> bitcast (i64 ptrtoint (i32 (i32)* @_Z2f2u2CMvb32_j to i64) to <8 x i8>), i32 0), i8 extractelement (<8 x i8> bitcast (i64 ptrtoint (i32 (i32)* @_Z2f2u2CMvb32_j to i64) to <8 x i8>), i32 1), i8 extractelement (<8 x i8> bitcast (i64 ptrtoint (i32 (i32)* @_Z2f2u2CMvb32_j to i64) to <8 x i8>), i32 2), i8 extractelement (<8 x i8> bitcast (i64 ptrtoint (i32 (i32)* @_Z2f2u2CMvb32_j to i64) to <8 x i8>), i32 3), i8 extractelement (<8 x i8> bitcast (i64 ptrtoint (i32 (i32)* @_Z2f2u2CMvb32_j to i64) to <8 x i8>), i32 4), i8 extractelement (<8 x i8> bitcast (i64 ptrtoint (i32 (i32)* @_Z2f2u2CMvb32_j to i64) to <8 x i8>), i32 5), i8 extractelement (<8 x i8> bitcast (i64 ptrtoint (i32 (i32)* @_Z2f2u2CMvb32_j to i64) to <8 x i8>), i32 6), i8 extractelement (<8 x i8> bitcast (i64 ptrtoint (i32 (i32)* @_Z2f2u2CMvb32_j to i64) to <8 x i8>), i32 7)>, <16 x i8>* %Funcs, align 16
; CHECK-LLVM: %Funcs1 = alloca <2 x i64>, align 16
; CHECK-LLVM: store <2 x i64> <i64 ptrtoint (i32 (i32)* @_Z2f1u2CMvb32_j to i64), i64 ptrtoint (i32 (i32)* @_Z2f2u2CMvb32_j to i64)>, <2 x i64>* %Funcs1, align 16
; Function Attrs: noinline nounwind
define dllexport void @vadd() {
entry:
  %Funcs = alloca <16 x i8>, align 16
  store <16 x i8> <
    i8 extractelement (<8 x i8> bitcast (i64 ptrtoint (i32 (i32)* @_Z2f1u2CMvb32_j to i64) to <8 x i8>), i32 0),
    i8 extractelement (<8 x i8> bitcast (i64 ptrtoint (i32 (i32)* @_Z2f1u2CMvb32_j to i64) to <8 x i8>), i32 1),
    i8 extractelement (<8 x i8> bitcast (i64 ptrtoint (i32 (i32)* @_Z2f1u2CMvb32_j to i64) to <8 x i8>), i32 2),
    i8 extractelement (<8 x i8> bitcast (i64 ptrtoint (i32 (i32)* @_Z2f1u2CMvb32_j to i64) to <8 x i8>), i32 3),
    i8 extractelement (<8 x i8> bitcast (i64 ptrtoint (i32 (i32)* @_Z2f1u2CMvb32_j to i64) to <8 x i8>), i32 4),
    i8 extractelement (<8 x i8> bitcast (i64 ptrtoint (i32 (i32)* @_Z2f1u2CMvb32_j to i64) to <8 x i8>), i32 5),
    i8 extractelement (<8 x i8> bitcast (i64 ptrtoint (i32 (i32)* @_Z2f1u2CMvb32_j to i64) to <8 x i8>), i32 6),
    i8 extractelement (<8 x i8> bitcast (i64 ptrtoint (i32 (i32)* @_Z2f1u2CMvb32_j to i64) to <8 x i8>), i32 7),
    i8 extractelement (<8 x i8> bitcast (i64 ptrtoint (i32 (i32)* @_Z2f2u2CMvb32_j to i64) to <8 x i8>), i32 0),
    i8 extractelement (<8 x i8> bitcast (i64 ptrtoint (i32 (i32)* @_Z2f2u2CMvb32_j to i64) to <8 x i8>), i32 1),
    i8 extractelement (<8 x i8> bitcast (i64 ptrtoint (i32 (i32)* @_Z2f2u2CMvb32_j to i64) to <8 x i8>), i32 2),
    i8 extractelement (<8 x i8> bitcast (i64 ptrtoint (i32 (i32)* @_Z2f2u2CMvb32_j to i64) to <8 x i8>), i32 3),
    i8 extractelement (<8 x i8> bitcast (i64 ptrtoint (i32 (i32)* @_Z2f2u2CMvb32_j to i64) to <8 x i8>), i32 4),
    i8 extractelement (<8 x i8> bitcast (i64 ptrtoint (i32 (i32)* @_Z2f2u2CMvb32_j to i64) to <8 x i8>), i32 5),
    i8 extractelement (<8 x i8> bitcast (i64 ptrtoint (i32 (i32)* @_Z2f2u2CMvb32_j to i64) to <8 x i8>), i32 6),
    i8 extractelement (<8 x i8> bitcast (i64 ptrtoint (i32 (i32)* @_Z2f2u2CMvb32_j to i64) to <8 x i8>), i32 7)
    >, <16 x i8>* %Funcs, align 16
  %Funcs1 = alloca <2 x i64>, align 16
  store <2 x i64> <
    i64 ptrtoint (i32 (i32)* @_Z2f1u2CMvb32_j to i64),
    i64 ptrtoint (i32 (i32)* @_Z2f2u2CMvb32_j to i64)
    >, <2 x i64>* %Funcs1, align 16
  ret void
}
