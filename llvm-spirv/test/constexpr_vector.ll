; RUN: llvm-as < %s | llvm-spirv -s | llvm-dis | FileCheck %s --check-prefix=CHECK-LLVM

; CHECK-LLVM: define dllexport void @vadd() {
; CHECK-LLVM-NEXT: entry:
; CHECK-LLVM-NEXT:   %Funcs = alloca <16 x i8>, align 16
; CHECK-LLVM-NEXT:   %0 = ptrtoint i32 (i32)* @_Z2f1u2CMvb32_j to i64
; CHECK-LLVM-NEXT:   %1 = bitcast i64 %0 to <8 x i8>
; CHECK-LLVM-NEXT:   %2 = extractelement <8 x i8> %1, i32 0
; CHECK-LLVM-NEXT:   %3 = extractelement <8 x i8> %1, i32 1
; CHECK-LLVM-NEXT:   %4 = extractelement <8 x i8> %1, i32 2
; CHECK-LLVM-NEXT:   %5 = extractelement <8 x i8> %1, i32 3
; CHECK-LLVM-NEXT:   %6 = extractelement <8 x i8> %1, i32 4
; CHECK-LLVM-NEXT:   %7 = extractelement <8 x i8> %1, i32 5
; CHECK-LLVM-NEXT:   %8 = extractelement <8 x i8> %1, i32 6
; CHECK-LLVM-NEXT:   %9 = extractelement <8 x i8> %1, i32 7
; CHECK-LLVM-NEXT:   %10 = ptrtoint i32 (i32)* @_Z2f2u2CMvb32_j to i64
; CHECK-LLVM-NEXT:   %11 = bitcast i64 %10 to <8 x i8>
; CHECK-LLVM-NEXT:   %12 = extractelement <8 x i8> %11, i32 0
; CHECK-LLVM-NEXT:   %13 = extractelement <8 x i8> %11, i32 1
; CHECK-LLVM-NEXT:   %14 = extractelement <8 x i8> %11, i32 2
; CHECK-LLVM-NEXT:   %15 = extractelement <8 x i8> %11, i32 3
; CHECK-LLVM-NEXT:   %16 = extractelement <8 x i8> %11, i32 4
; CHECK-LLVM-NEXT:   %17 = extractelement <8 x i8> %11, i32 5
; CHECK-LLVM-NEXT:   %18 = extractelement <8 x i8> %11, i32 6
; CHECK-LLVM-NEXT:   %19 = extractelement <8 x i8> %11, i32 7
; CHECK-LLVM-NEXT:   %20 = insertelement <16 x i8> undef, i8 %2, i32 0
; CHECK-LLVM-NEXT:   %21 = insertelement <16 x i8> %20, i8 %3, i32 1
; CHECK-LLVM-NEXT:   %22 = insertelement <16 x i8> %21, i8 %4, i32 2
; CHECK-LLVM-NEXT:   %23 = insertelement <16 x i8> %22, i8 %5, i32 3
; CHECK-LLVM-NEXT:   %24 = insertelement <16 x i8> %23, i8 %6, i32 4
; CHECK-LLVM-NEXT:   %25 = insertelement <16 x i8> %24, i8 %7, i32 5
; CHECK-LLVM-NEXT:   %26 = insertelement <16 x i8> %25, i8 %8, i32 6
; CHECK-LLVM-NEXT:   %27 = insertelement <16 x i8> %26, i8 %9, i32 7
; CHECK-LLVM-NEXT:   %28 = insertelement <16 x i8> %27, i8 %12, i32 8
; CHECK-LLVM-NEXT:   %29 = insertelement <16 x i8> %28, i8 %13, i32 9
; CHECK-LLVM-NEXT:   %30 = insertelement <16 x i8> %29, i8 %14, i32 10
; CHECK-LLVM-NEXT:   %31 = insertelement <16 x i8> %30, i8 %15, i32 11
; CHECK-LLVM-NEXT:   %32 = insertelement <16 x i8> %31, i8 %16, i32 12
; CHECK-LLVM-NEXT:   %33 = insertelement <16 x i8> %32, i8 %17, i32 13
; CHECK-LLVM-NEXT:   %34 = insertelement <16 x i8> %33, i8 %18, i32 14
; CHECK-LLVM-NEXT:   %35 = insertelement <16 x i8> %34, i8 %19, i32 15
; CHECK-LLVM-NEXT:   store <16 x i8> %35, <16 x i8>* %Funcs, align 16
; CHECK-LLVM-NEXT:   ret void
; CHECK-LLVM-NEXT: }

; RUN: llvm-as < %s | llvm-spirv -spirv-text | FileCheck %s --check-prefix=CHECK-SPIRV

; CHECK-SPIRV: 4 Name [[Funcs:[0-9]+]] "Funcs"

; CHECK-SPIRV-DAG: 4 TypeInt [[TypeInt8:[0-9]+]] 8 0
; CHECK-SPIRV-DAG: 4 TypeInt [[TypeInt32:[0-9]+]] 32 0
; CHECK-SPIRV-DAG: 4 TypeInt [[TypeInt64:[0-9]+]] 64 0
; CHECK-SPIRV-DAG: 4 TypeVector [[TypeVec16:[0-9]+]] [[TypeInt8]] 16
; CHECK-SPIRV-DAG: 4 TypeVector [[TypeVec8:[0-9]+]] [[TypeInt8]] 8
; CHECK-SPIRV-DAG: 4 TypePointer [[StorePtr:[0-9]+]] 7 [[TypeVec16]]
; CHECK-SPIRV-DAG: 3 Undef [[TypeVec16]] [[TypeUndef:[0-9]+]]

; CHECK-SPIRV: 4 ConvertPtrToU [[TypeInt64]] [[Ptr1:[0-9]+]] {{[0-9]+}}
; CHECK-SPIRV-NEXT: 4 Bitcast [[TypeVec8]] [[Vec1:[0-9]+]] [[Ptr1]]
; CHECK-SPIRV-NEXT: 5 CompositeExtract [[TypeInt8]] [[v00:[0-9]+]] [[Vec1]] 0
; CHECK-SPIRV-NEXT: 5 CompositeExtract [[TypeInt8]] [[v01:[0-9]+]] [[Vec1]] 1
; CHECK-SPIRV-NEXT: 5 CompositeExtract [[TypeInt8]] [[v02:[0-9]+]] [[Vec1]] 2
; CHECK-SPIRV-NEXT: 5 CompositeExtract [[TypeInt8]] [[v03:[0-9]+]] [[Vec1]] 3
; CHECK-SPIRV-NEXT: 5 CompositeExtract [[TypeInt8]] [[v04:[0-9]+]] [[Vec1]] 4
; CHECK-SPIRV-NEXT: 5 CompositeExtract [[TypeInt8]] [[v05:[0-9]+]] [[Vec1]] 5
; CHECK-SPIRV-NEXT: 5 CompositeExtract [[TypeInt8]] [[v06:[0-9]+]] [[Vec1]] 6
; CHECK-SPIRV-NEXT: 5 CompositeExtract [[TypeInt8]] [[v07:[0-9]+]] [[Vec1]] 7
; CHECK-SPIRV-NEXT: 4 ConvertPtrToU [[TypeInt64]] [[Ptr2:[0-9]+]] {{[0-9]+}}
; CHECK-SPIRV-NEXT: 4 Bitcast [[TypeVec8]] [[Vec2:[0-9]+]] [[Ptr2]]
; CHECK-SPIRV-NEXT: 5 CompositeExtract [[TypeInt8]] [[v10:[0-9]+]] [[Vec2]] 0
; CHECK-SPIRV-NEXT: 5 CompositeExtract [[TypeInt8]] [[v11:[0-9]+]] [[Vec2]] 1
; CHECK-SPIRV-NEXT: 5 CompositeExtract [[TypeInt8]] [[v12:[0-9]+]] [[Vec2]] 2
; CHECK-SPIRV-NEXT: 5 CompositeExtract [[TypeInt8]] [[v13:[0-9]+]] [[Vec2]] 3
; CHECK-SPIRV-NEXT: 5 CompositeExtract [[TypeInt8]] [[v14:[0-9]+]] [[Vec2]] 4
; CHECK-SPIRV-NEXT: 5 CompositeExtract [[TypeInt8]] [[v15:[0-9]+]] [[Vec2]] 5
; CHECK-SPIRV-NEXT: 5 CompositeExtract [[TypeInt8]] [[v16:[0-9]+]] [[Vec2]] 6
; CHECK-SPIRV-NEXT: 5 CompositeExtract [[TypeInt8]] [[v17:[0-9]+]] [[Vec2]] 7
; CHECK-SPIRV-NEXT: 6 CompositeInsert [[TypeVec16]] [[NewVec0:[0-9]+]] [[v00]] [[TypeUndef]] 0
; CHECK-SPIRV-NEXT: 6 CompositeInsert [[TypeVec16]] [[NewVec1:[0-9]+]] [[v01]] [[NewVec0]] 1
; CHECK-SPIRV-NEXT: 6 CompositeInsert [[TypeVec16]] [[NewVec2:[0-9]+]] [[v02]] [[NewVec1]] 2
; CHECK-SPIRV-NEXT: 6 CompositeInsert [[TypeVec16]] [[NewVec3:[0-9]+]] [[v03]] [[NewVec2]] 3
; CHECK-SPIRV-NEXT: 6 CompositeInsert [[TypeVec16]] [[NewVec4:[0-9]+]] [[v04]] [[NewVec3]] 4
; CHECK-SPIRV-NEXT: 6 CompositeInsert [[TypeVec16]] [[NewVec5:[0-9]+]] [[v05]] [[NewVec4]] 5
; CHECK-SPIRV-NEXT: 6 CompositeInsert [[TypeVec16]] [[NewVec6:[0-9]+]] [[v06]] [[NewVec5]] 6
; CHECK-SPIRV-NEXT: 6 CompositeInsert [[TypeVec16]] [[NewVec7:[0-9]+]] [[v07]] [[NewVec6]] 7
; CHECK-SPIRV-NEXT: 6 CompositeInsert [[TypeVec16]] [[NewVec8:[0-9]+]] [[v10]] [[NewVec7]] 8
; CHECK-SPIRV-NEXT: 6 CompositeInsert [[TypeVec16]] [[NewVec9:[0-9]+]] [[v11]] [[NewVec8]] 9
; CHECK-SPIRV-NEXT: 6 CompositeInsert [[TypeVec16]] [[NewVec10:[0-9]+]] [[v12]] [[NewVec9]] 10
; CHECK-SPIRV-NEXT: 6 CompositeInsert [[TypeVec16]] [[NewVec11:[0-9]+]] [[v13]] [[NewVec10]] 11
; CHECK-SPIRV-NEXT: 6 CompositeInsert [[TypeVec16]] [[NewVec12:[0-9]+]] [[v14]] [[NewVec11]] 12
; CHECK-SPIRV-NEXT: 6 CompositeInsert [[TypeVec16]] [[NewVec13:[0-9]+]] [[v15]] [[NewVec12]] 13
; CHECK-SPIRV-NEXT: 6 CompositeInsert [[TypeVec16]] [[NewVec14:[0-9]+]] [[v16]] [[NewVec13]] 14
; CHECK-SPIRV-NEXT: 6 CompositeInsert [[TypeVec16]] [[NewVec15:[0-9]+]] [[v17]] [[NewVec14]] 15
; CHECK-SPIRV-NEXT: 5 Store [[Funcs]] [[NewVec15]] [[TypeInt32]] [[StorePtr]]

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
  %Funcs = alloca <16 x i8>, align 16
  store <16 x i8> <i8 extractelement (<8 x i8> bitcast (i64 ptrtoint (i32 (i32)* @_Z2f1u2CMvb32_j to i64) to <8 x i8>), i32 0), i8 extractelement (<8 x i8> bitcast (i64 ptrtoint (i32 (i32)* @_Z2f1u2CMvb32_j to i64) to <8 x i8>), i32 1), i8 extractelement (<8 x i8> bitcast (i64 ptrtoint (i32 (i32)* @_Z2f1u2CMvb32_j to i64) to <8 x i8>), i32 2), i8 extractelement (<8 x i8> bitcast (i64 ptrtoint (i32 (i32)* @_Z2f1u2CMvb32_j to i64) to <8 x i8>), i32 3), i8 extractelement (<8 x i8> bitcast (i64 ptrtoint (i32 (i32)* @_Z2f1u2CMvb32_j to i64) to <8 x i8>), i32 4), i8 extractelement (<8 x i8> bitcast (i64 ptrtoint (i32 (i32)* @_Z2f1u2CMvb32_j to i64) to <8 x i8>), i32 5), i8 extractelement (<8 x i8> bitcast (i64 ptrtoint (i32 (i32)* @_Z2f1u2CMvb32_j to i64) to <8 x i8>), i32 6), i8 extractelement (<8 x i8> bitcast (i64 ptrtoint (i32 (i32)* @_Z2f1u2CMvb32_j to i64) to <8 x i8>), i32 7), i8 extractelement (<8 x i8> bitcast (i64 ptrtoint (i32 (i32)* @_Z2f2u2CMvb32_j to i64) to <8 x i8>), i32 0), i8 extractelement (<8 x i8> bitcast (i64 ptrtoint (i32 (i32)* @_Z2f2u2CMvb32_j to i64) to <8 x i8>), i32 1), i8 extractelement (<8 x i8> bitcast (i64 ptrtoint (i32 (i32)* @_Z2f2u2CMvb32_j to i64) to <8 x i8>), i32 2), i8 extractelement (<8 x i8> bitcast (i64 ptrtoint (i32 (i32)* @_Z2f2u2CMvb32_j to i64) to <8 x i8>), i32 3), i8 extractelement (<8 x i8> bitcast (i64 ptrtoint (i32 (i32)* @_Z2f2u2CMvb32_j to i64) to <8 x i8>), i32 4), i8 extractelement (<8 x i8> bitcast (i64 ptrtoint (i32 (i32)* @_Z2f2u2CMvb32_j to i64) to <8 x i8>), i32 5), i8 extractelement (<8 x i8> bitcast (i64 ptrtoint (i32 (i32)* @_Z2f2u2CMvb32_j to i64) to <8 x i8>), i32 6), i8 extractelement (<8 x i8> bitcast (i64 ptrtoint (i32 (i32)* @_Z2f2u2CMvb32_j to i64) to <8 x i8>), i32 7)>, <16 x i8>* %Funcs, align 16
  ret void
}
