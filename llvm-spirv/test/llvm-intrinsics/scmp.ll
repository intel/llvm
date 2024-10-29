; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-G1"
target triple = "spir64-unknown-unknown"
; CHECK-SPIRV: TypeInt [[#TypeI8:]] 8 0 
; CHECK-SPIRV: TypeInt [[#TypeI16:]] 16 0 
; CHECK-SPIRV: TypeInt [[#TypeI32:]] 32 0 
; CHECK-SPIRV: TypeInt [[#TypeI64:]] 64 0 

; CHECK-SPIRV: Constant [[#TypeI8]] [[#CmpI8ConstOne:]] 1 
; CHECK-SPIRV: Constant [[#TypeI8]] [[#CmpI8ConstZero:]] 0 
; CHECK-SPIRV: Constant [[#TypeI8]] [[#CmpI8ConstMinusOne:]] 255 

; CHECK-SPIRV: Constant [[#TypeI16]] [[#CmpI16ConstOne:]] 1 
; CHECK-SPIRV: Constant [[#TypeI16]] [[#CmpI16ConstZero:]] 0 
; CHECK-SPIRV: Constant [[#TypeI16]] [[#CmpI16ConstMinusOne:]] 65535 

; CHECK-SPIRV: Constant [[#TypeI32]] [[#CmpI32ConstOne:]] 1 
; CHECK-SPIRV: Constant [[#TypeI32]] [[#CmpI32ConstZero:]] 0 
; CHECK-SPIRV: Constant [[#TypeI32]] [[#CmpI32ConstMinusOne:]] 4294967295 

; CHECK-SPIRV: Constant [[#TypeI64]] [[#CmpI64ConstOne:]] 1 
; CHECK-SPIRV: Constant [[#TypeI64]] [[#CmpI64ConstZero:]] 0 
; CHECK-SPIRV: Constant [[#TypeI64]] [[#CmpI64ConstMinusOne:]] 4294967295 4294967295 

; CHECK-SPIRV: Constant [[#TypeI8]] [[#CmpV4I8ConstOneElem:]] 1 
; CHECK-SPIRV: Constant [[#TypeI8]] [[#CmpV4I8ConstZeroElem:]] 0 
; CHECK-SPIRV: Constant [[#TypeI8]] [[#CmpV4I8ConstMinusOneElem:]] 255 

; CHECK-SPIRV: Constant [[#TypeI16]] [[#CmpV4I16ConstOneElem:]] 1 
; CHECK-SPIRV: Constant [[#TypeI16]] [[#CmpV4I16ConstZeroElem:]] 0 
; CHECK-SPIRV: Constant [[#TypeI16]] [[#CmpV4I16ConstMinusOneElem:]] 65535 

; CHECK-SPIRV: Constant [[#TypeI32]] [[#CmpV4I32ConstMinusOneElem:]] 4294967295 

; CHECK-SPIRV: Constant [[#TypeI64]] [[#CmpV4I64ConstOneElem:]] 1 
; CHECK-SPIRV: Constant [[#TypeI64]] [[#CmpV4I64ConstZeroElem:]] 0 
; CHECK-SPIRV: Constant [[#TypeI64]] [[#CmpV4I64ConstMinusOneElem:]] 4294967295 

; CHECK-SPIRV: Constant [[#TypeI16]] [[#CmpI16I8ConstOne:]] 1 
; CHECK-SPIRV: Constant [[#TypeI16]] [[#CmpI16I8ConstZero:]] 0 
; CHECK-SPIRV: Constant [[#TypeI16]] [[#CmpI16I8ConstMinusOne:]] 65535 

; CHECK-SPIRV: Constant [[#TypeI32]] [[#CmpI32I8ConstMinusOne:]] 4294967295 

; CHECK-SPIRV: Constant [[#TypeI64]] [[#CmpI64I8ConstOne:]] 1 
; CHECK-SPIRV: Constant [[#TypeI64]] [[#CmpI64I8ConstZero:]] 0 
; CHECK-SPIRV: Constant [[#TypeI64]] [[#CmpI64I8ConstMinusOne:]] 4294967295 

; CHECK-SPIRV: Constant [[#TypeI8]] [[#CmpI8I16ConstOne:]] 1 
; CHECK-SPIRV: Constant [[#TypeI8]] [[#CmpI8I16ConstZero:]] 0 
; CHECK-SPIRV: Constant [[#TypeI8]] [[#CmpI8I16ConstMinusOne:]] 255 

; CHECK-SPIRV: Constant [[#TypeI32]] [[#CmpI32I16ConstMinusOne:]] 4294967295 

; CHECK-SPIRV: Constant [[#TypeI64]] [[#CmpI64I16ConstOne:]] 1 
; CHECK-SPIRV: Constant [[#TypeI64]] [[#CmpI64I16ConstZero:]] 0 
; CHECK-SPIRV: Constant [[#TypeI64]] [[#CmpI64I16ConstMinusOne:]] 4294967295 4294967295 

; CHECK-SPIRV: Constant [[#TypeI8]] [[#CmpI8I32ConstOne:]] 1 
; CHECK-SPIRV: Constant [[#TypeI8]] [[#CmpI8I32ConstZero:]] 0 
; CHECK-SPIRV: Constant [[#TypeI8]] [[#CmpI8I32ConstMinusOne:]] 255 

; CHECK-SPIRV: Constant [[#TypeI16]] [[#CmpI16I32ConstOne:]] 1 
; CHECK-SPIRV: Constant [[#TypeI16]] [[#CmpI16I32ConstZero:]] 0 
; CHECK-SPIRV: Constant [[#TypeI16]] [[#CmpI16I32ConstMinusOne:]] 65535 

; CHECK-SPIRV: Constant [[#TypeI64]] [[#CmpI64I32ConstOne:]] 1 
; CHECK-SPIRV: Constant [[#TypeI64]] [[#CmpI64I32ConstZero:]] 0 
; CHECK-SPIRV: Constant [[#TypeI64]] [[#CmpI64I32ConstMinusOne:]] 4294967295 4294967295 

; CHECK-SPIRV: Constant [[#TypeI8]] [[#CmpI8I64ConstOne:]] 1 
; CHECK-SPIRV: Constant [[#TypeI8]] [[#CmpI8I64ConstZero:]] 0 
; CHECK-SPIRV: Constant [[#TypeI8]] [[#CmpI8I64ConstMinusOne:]] 255 

; CHECK-SPIRV: Constant [[#TypeI16]] [[#CmpI16I64ConstOne:]] 1 
; CHECK-SPIRV: Constant [[#TypeI16]] [[#CmpI16I64ConstZero:]] 0 
; CHECK-SPIRV: Constant [[#TypeI16]] [[#CmpI16I64ConstMinusOne:]] 65535 

; CHECK-SPIRV: Constant [[#TypeI32]] [[#CmpI32I64ConstMinusOne:]] 4294967295 

; CHECK-SPIRV: TypeBool [[#TypeBool:]]
; CHECK-SPIRV: TypeVector [[#TypeV4I8:]] [[#TypeI8]] 4 
; CHECK-SPIRV: TypeVector [[#TypeVBool:]] [[#TypeBool]] 4
; CHECK-SPIRV: TypeVector [[#TypeV4I16:]] [[#TypeI16]] 4 
; CHECK-SPIRV: TypeVector [[#TypeV4I32:]] [[#TypeI32]] 4 
; CHECK-SPIRV: TypeVector [[#TypeV4I64:]] [[#TypeI64]] 4 

; CHECK-SPIRV: ConstantComposite [[#TypeV4I8]] [[#CmpV4I8ConstOne:]] [[#CmpV4I8ConstOneElem]] [[#CmpV4I8ConstOneElem]] [[#CmpV4I8ConstOneElem]]
; CHECK-SPIRV: ConstantComposite [[#TypeV4I8]] [[#CmpV4I8ConstZero:]] [[#CmpV4I8ConstZeroElem]] [[#CmpV4I8ConstZeroElem]] [[#CmpV4I8ConstZeroElem]]
; CHECK-SPIRV: ConstantComposite [[#TypeV4I8]] [[#CmpV4I8ConstMinusOne:]] [[#CmpV4I8ConstMinusOneElem]] [[#CmpV4I8ConstMinusOneElem]] [[#CmpV4I8ConstMinusOneElem]]

; CHECK-SPIRV: ConstantComposite [[#TypeV4I16]] [[#CmpV4I16ConstOne:]] [[#CmpV4I16ConstOneElem]] [[#CmpV4I16ConstOneElem]] [[#CmpV4I16ConstOneElem]]
; CHECK-SPIRV: ConstantComposite [[#TypeV4I16]] [[#CmpV4I16ConstZero:]] [[#CmpV4I16ConstZeroElem]] [[#CmpV4I16ConstZeroElem]] [[#CmpV4I16ConstZeroElem]]
; CHECK-SPIRV: ConstantComposite [[#TypeV4I16]] [[#CmpV4I16ConstMinusOne:]] [[#CmpV4I16ConstMinusOneElem]] [[#CmpV4I16ConstMinusOneElem]] [[#CmpV4I16ConstMinusOneElem]]

; CHECK-SPIRV: ConstantComposite [[#TypeV4I32]] [[#CmpV4I32ConstOne:]] [[#CmpI32ConstOne]] [[#CmpI32ConstOne]] [[#CmpI32ConstOne]]
; CHECK-SPIRV: ConstantComposite [[#TypeV4I32]] [[#CmpV4I32ConstZero:]] [[#CmpI32ConstZero]] [[#CmpI32ConstZero]] [[#CmpI32ConstZero]]
; CHECK-SPIRV: ConstantComposite [[#TypeV4I32]] [[#CmpV4I32ConstMinusOne:]] [[#CmpV4I32ConstMinusOneElem]] [[#CmpV4I32ConstMinusOneElem]] [[#CmpV4I32ConstMinusOneElem]]

; CHECK-SPIRV: ConstantComposite [[#TypeV4I64]] [[#CmpV4I64ConstOne:]] [[#CmpV4I64ConstOneElem]] [[#CmpV4I64ConstOneElem]] [[#CmpV4I64ConstOneElem]]
; CHECK-SPIRV: ConstantComposite [[#TypeV4I64]] [[#CmpV4I64ConstZero:]] [[#CmpV4I64ConstZeroElem]] [[#CmpV4I64ConstZeroElem]] [[#CmpV4I64ConstZeroElem]]
; CHECK-SPIRV: ConstantComposite [[#TypeV4I64]] [[#CmpV4I64ConstMinusOne:]] [[#CmpV4I64ConstMinusOneElem]] [[#CmpV4I64ConstMinusOneElem]] [[#CmpV4I64ConstMinusOneElem]]

; CHECK-SPIRV: SLessThanEqual [[#TypeBool]] [[#CmpI8Res1:]] [[#]] [[#]]
; CHECK-SPIRV: SLessThan [[#TypeBool]] [[#CmpI8Res2:]] [[#]] [[#]]
; CHECK-SPIRV: Select [[#TypeI8]] [[#SelI8Res1:]] [[#CmpI8Res2]] [[#CmpI8ConstMinusOne]] [[#CmpI8ConstZero]]
; CHECK-SPIRV: Select [[#TypeI8]] [[#SelI8Res2:]] [[#CmpI8Res1]] [[#SelI8Res1]] [[#CmpI8ConstOne]] 
; CHECK-SPIRV: ReturnValue [[#SelI8Res2]] 
define dso_local noundef range(i8 -1, 2) i8 @compare_i8(i8 noundef %0, i8 noundef %1) local_unnamed_addr #0 {
  %3 = tail call i8 @llvm.scmp.i8.i8(i8 %0, i8 %1)
  ret i8 %3
}

; CHECK-SPIRV: SLessThanEqual [[#TypeBool]] [[#CmpI16Res1:]] [[#]] [[#]]
; CHECK-SPIRV: SLessThan [[#TypeBool]] [[#CmpI16Res2:]] [[#]] [[#]]
; CHECK-SPIRV: Select [[#TypeI16]] [[#SelI16Res1:]] [[#CmpI16Res2]] [[#CmpI16ConstMinusOne]] [[#CmpI16ConstZero]]
; CHECK-SPIRV: Select [[#TypeI16]] [[#SelI16Res2:]] [[#CmpI16Res1]] [[#SelI16Res1]] [[#CmpI16ConstOne]] 
; CHECK-SPIRV: ReturnValue [[#SelI16Res2]] 
define dso_local noundef range(i16 -1, 2) i16 @compare_i16(i16 noundef %0, i16 noundef %1) local_unnamed_addr #0 {
  %3 = tail call i16 @llvm.scmp.i16.i16(i16 %0, i16 %1)
  ret i16 %3
}

; CHECK-SPIRV: SLessThanEqual [[#TypeBool]] [[#CmpI32Res1:]] [[#]] [[#]]
; CHECK-SPIRV: SLessThan [[#TypeBool]] [[#CmpI32Res2:]] [[#]] [[#]]
; CHECK-SPIRV: Select [[#TypeI32]] [[#SelI32Res1:]] [[#CmpI32Res2]] [[#CmpI32ConstMinusOne]] [[#CmpI32ConstZero]]
; CHECK-SPIRV: Select [[#TypeI32]] [[#SelI32Res2:]] [[#CmpI32Res1]] [[#SelI32Res1]] [[#CmpI32ConstOne]] 
; CHECK-SPIRV: ReturnValue [[#SelI32Res2]] 
define dso_local noundef range(i32 -1, 2) i32 @compare_i32(i32 noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = tail call i32 @llvm.scmp.i32.i32(i32 %0, i32 %1)
  ret i32 %3
}

; CHECK-SPIRV: SLessThanEqual [[#TypeBool]] [[#CmpI64Res1:]] [[#]] [[#]]
; CHECK-SPIRV: SLessThan [[#TypeBool]] [[#CmpI64Res2:]] [[#]] [[#]]
; CHECK-SPIRV: Select [[#TypeI64]] [[#SelI64Res1:]] [[#CmpI64Res2]] [[#CmpI64ConstMinusOne]] [[#CmpI64ConstZero]]
; CHECK-SPIRV: Select [[#TypeI64]] [[#SelI64Res2:]] [[#CmpI64Res1]] [[#SelI64Res1]] [[#CmpI64ConstOne]] 
; CHECK-SPIRV: ReturnValue [[#SelI64Res2]] 
define dso_local noundef range(i64 -1, 2) i64 @compare_i64(i64 noundef %0, i64 noundef %1) local_unnamed_addr #0 {
  %3 = tail call i64 @llvm.scmp.i64.i64(i64 %0, i64 %1)
  ret i64 %3
}

; CHECK-SPIRV: SLessThanEqual [[#TypeVBool]] [[#CmpV4I8Res1:]] [[#]] [[#]]
; CHECK-SPIRV: SLessThan [[#TypeVBool]] [[#CmpV4I8Res2:]] [[#]] [[#]]
; CHECK-SPIRV: Select [[#TypeV4I8]] [[#SelV4I8Res1:]] [[#CmpV4I8Res2]] [[#CmpV4I8ConstMinusOne]] [[#CmpV4I8ConstZero]]
; CHECK-SPIRV: Select [[#TypeV4I8]] [[#SelV4I8Res2:]] [[#CmpV4I8Res1]] [[#SelV4I8Res1]] [[#CmpV4I8ConstOne]] 
; CHECK-SPIRV: ReturnValue [[#SelV4I8Res2]] 
define dso_local noundef range(i8 -1, 2) <4 x i8> @compare_v4i8(<4 x i8> noundef %0, <4 x i8> noundef %1) local_unnamed_addr #0 {
  %3 = tail call <4 x i8> @llvm.scmp.v4i8.v4i8(<4 x i8> %0, <4 x i8> %1)
  ret <4 x i8> %3
}

; CHECK-SPIRV: SLessThanEqual [[#TypeVBool]] [[#CmpV4I16Res1:]] [[#]] [[#]]
; CHECK-SPIRV: SLessThan [[#TypeVBool]] [[#CmpV4I16Res2:]] [[#]] [[#]]
; CHECK-SPIRV: Select [[#TypeV4I16]] [[#SelV4I16Res1:]] [[#CmpV4I16Res2]] [[#CmpV4I16ConstMinusOne]] [[#CmpV4I16ConstZero]]
; CHECK-SPIRV: Select [[#TypeV4I16]] [[#SelV4I16Res2:]] [[#CmpV4I16Res1]] [[#SelV4I16Res1]] [[#CmpV4I16ConstOne]] 
; CHECK-SPIRV: ReturnValue [[#SelV4I16Res2]] 
define dso_local noundef range(i16 -1, 2) <4 x i16> @compare_v4i16(<4 x i16> noundef %0, <4 x i16> noundef %1) local_unnamed_addr #0 {
  %3 = tail call <4 x i16> @llvm.scmp.v4i16.v4i16(<4 x i16> %0, <4 x i16> %1)
  ret <4 x i16> %3
}

; CHECK-SPIRV: SLessThanEqual [[#TypeVBool]] [[#CmpV4I32Res1:]] [[#]] [[#]]
; CHECK-SPIRV: SLessThan [[#TypeVBool]] [[#CmpV4I32Res2:]] [[#]] [[#]]
; CHECK-SPIRV: Select [[#TypeV4I32]] [[#SelV4I32Res1:]] [[#CmpV4I32Res2]] [[#CmpV4I32ConstMinusOne]] [[#CmpV4I32ConstZero]]
; CHECK-SPIRV: Select [[#TypeV4I32]] [[#SelV4I32Res2:]] [[#CmpV4I32Res1]] [[#SelV4I32Res1]] [[#CmpV4I32ConstOne]] 
; CHECK-SPIRV: ReturnValue [[#SelV4I32Res2]] 
define dso_local noundef range(i32 -1, 2) <4 x i32> @compare_v4i32(<4 x i32> noundef %0, <4 x i32> noundef %1) local_unnamed_addr #0 {
  %3 = tail call <4 x i32> @llvm.scmp.v4i32.v4i32(<4 x i32> %0, <4 x i32> %1)
  ret <4 x i32> %3
}

; CHECK-SPIRV: SLessThanEqual [[#TypeVBool]] [[#CmpV4I64Res1:]] [[#]] [[#]]
; CHECK-SPIRV: SLessThan [[#TypeVBool]] [[#CmpV4I64Res2:]] [[#]] [[#]]
; CHECK-SPIRV: Select [[#TypeV4I64]] [[#SelV4I64Res1:]] [[#CmpV4I64Res2]] [[#CmpV4I64ConstMinusOne]] [[#CmpV4I64ConstZero]]
; CHECK-SPIRV: Select [[#TypeV4I64]] [[#SelV4I64Res2:]] [[#CmpV4I64Res1]] [[#SelV4I64Res1]] [[#CmpV4I64ConstOne]] 
; CHECK-SPIRV: ReturnValue [[#SelV4I64Res2]] 
define dso_local noundef range(i64 -1, 2) <4 x i64> @compare_v4i64(<4 x i64> noundef %0, <4 x i64> noundef %1) local_unnamed_addr #0 {
  %3 = tail call <4 x i64> @llvm.scmp.v4i64.v4i64(<4 x i64> %0, <4 x i64> %1)
  ret <4 x i64> %3
}

; CHECK-SPIRV: SLessThanEqual [[#TypeBool]] [[#CmpI16I8Res1:]] [[#]] [[#]]
; CHECK-SPIRV: SLessThan [[#TypeBool]] [[#CmpI16I8Res2:]] [[#]] [[#]]
; CHECK-SPIRV: Select [[#TypeI16]] [[#SelI16I8Res1:]] [[#CmpI16I8Res2]] [[#CmpI16I8ConstMinusOne]] [[#CmpI16I8ConstZero]]
; CHECK-SPIRV: Select [[#TypeI16]] [[#SelI16I8Res2:]] [[#CmpI16I8Res1]] [[#SelI16I8Res1]] [[#CmpI16I8ConstOne]] 
; CHECK-SPIRV: ReturnValue [[#SelI16I8Res2]] 
define dso_local noundef range(i16 -1, 2) i16 @compare_i16i8(i8 noundef %0, i8 noundef %1) local_unnamed_addr #0 {
  %3 = tail call i16 @llvm.scmp.i16.i8(i8 %0, i8 %1)
  ret i16 %3
}

; CHECK-SPIRV: SLessThanEqual [[#TypeBool]] [[#CmpI32I8Res1:]] [[#]] [[#]]
; CHECK-SPIRV: SLessThan [[#TypeBool]] [[#CmpI32I8Res2:]] [[#]] [[#]]
; CHECK-SPIRV: Select [[#TypeI32]] [[#SelI32I8Res1:]] [[#CmpI32I8Res2]] [[#CmpI32I8ConstMinusOne]] [[#CmpI32ConstZero]]
; CHECK-SPIRV: Select [[#TypeI32]] [[#SelI32I8Res2:]] [[#CmpI32I8Res1]] [[#SelI32I8Res1]] [[#CmpI32ConstOne]] 
; CHECK-SPIRV: ReturnValue [[#SelI32I8Res2]] 
define dso_local noundef range(i32 -1, 2) i32 @compare_i32i8(i8 noundef %0, i8 noundef %1) local_unnamed_addr #0 {
  %3 = tail call i32 @llvm.scmp.i32.i8(i8 %0, i8 %1)
  ret i32 %3
}

; CHECK-SPIRV: SLessThanEqual [[#TypeBool]] [[#CmpI64I8Res1:]] [[#]] [[#]]
; CHECK-SPIRV: SLessThan [[#TypeBool]] [[#CmpI64I8Res2:]] [[#]] [[#]]
; CHECK-SPIRV: Select [[#TypeI64]] [[#SelI64I8Res1:]] [[#CmpI64I8Res2]] [[#CmpI64I8ConstMinusOne]] [[#CmpI64I8ConstZero]]
; CHECK-SPIRV: Select [[#TypeI64]] [[#SelI64I8Res2:]] [[#CmpI64I8Res1]] [[#SelI64I8Res1]] [[#CmpI64I8ConstOne]] 
; CHECK-SPIRV: ReturnValue [[#SelI64I8Res2]] 
define dso_local noundef range(i64 -1, 2) i64 @compare_i64i8(i8 noundef %0, i8 noundef %1) local_unnamed_addr #0 {
  %3 = tail call i64 @llvm.scmp.i64.i8(i8 %0, i8 %1)
  ret i64 %3
}

; CHECK-SPIRV: SLessThanEqual [[#TypeBool]] [[#CmpI8I16Res1:]] [[#]] [[#]]
; CHECK-SPIRV: SLessThan [[#TypeBool]] [[#CmpI8I16Res2:]] [[#]] [[#]]
; CHECK-SPIRV: Select [[#TypeI8]] [[#SelI8I16Res1:]] [[#CmpI8I16Res2]] [[#CmpI8I16ConstMinusOne]] [[#CmpI8I16ConstZero]]
; CHECK-SPIRV: Select [[#TypeI8]] [[#SelI8I16Res2:]] [[#CmpI8I16Res1]] [[#SelI8I16Res1]] [[#CmpI8I16ConstOne]] 
; CHECK-SPIRV: ReturnValue [[#SelI8I16Res2]] 
define dso_local noundef range(i8 -1, 2) i8 @compare_i8i16(i16 noundef %0, i16 noundef %1) local_unnamed_addr #0 {
  %3 = tail call i8 @llvm.scmp.i8.i16(i16 %0, i16 %1)
  ret i8 %3
}

; CHECK-SPIRV: SLessThanEqual [[#TypeBool]] [[#CmpI32I16Res1:]] [[#]] [[#]]
; CHECK-SPIRV: SLessThan [[#TypeBool]] [[#CmpI32I16Res2:]] [[#]] [[#]]
; CHECK-SPIRV: Select [[#TypeI32]] [[#SelI32I16Res1:]] [[#CmpI32I16Res2]] [[#CmpI32I16ConstMinusOne]] [[#CmpI32ConstZero]]
; CHECK-SPIRV: Select [[#TypeI32]] [[#SelI32I16Res2:]] [[#CmpI32I16Res1]] [[#SelI32I16Res1]] [[#CmpI32ConstOne]] 
; CHECK-SPIRV: ReturnValue [[#SelI32I16Res2]] 
define dso_local noundef range(i32 -1, 2) i32 @compare_i32i16(i16 noundef %0, i16 noundef %1) local_unnamed_addr #0 {
  %3 = tail call i32 @llvm.scmp.i32.i16(i16 %0, i16 %1)
  ret i32 %3
}

; CHECK-SPIRV: SLessThanEqual [[#TypeBool]] [[#CmpI64I16Res1:]] [[#]] [[#]]
; CHECK-SPIRV: SLessThan [[#TypeBool]] [[#CmpI64I16Res2:]] [[#]] [[#]]
; CHECK-SPIRV: Select [[#TypeI64]] [[#SelI64I16Res1:]] [[#CmpI64I16Res2]] [[#CmpI64I16ConstMinusOne]] [[#CmpI64I16ConstZero]]
; CHECK-SPIRV: Select [[#TypeI64]] [[#SelI64I16Res2:]] [[#CmpI64I16Res1]] [[#SelI64I16Res1]] [[#CmpI64I16ConstOne]] 
; CHECK-SPIRV: ReturnValue [[#SelI64I16Res2]] 
define dso_local noundef range(i64 -1, 2) i64 @compare_i64i16(i16 noundef %0, i16 noundef %1) local_unnamed_addr #0 {
  %3 = tail call i64 @llvm.scmp.i64.i16(i16 %0, i16 %1)
  ret i64 %3
}

; CHECK-SPIRV: SLessThanEqual [[#TypeBool]] [[#CmpI8I32Res1:]] [[#]] [[#]]
; CHECK-SPIRV: SLessThan [[#TypeBool]] [[#CmpI8I32Res2:]] [[#]] [[#]]
; CHECK-SPIRV: Select [[#TypeI8]] [[#SelI8I32Res1:]] [[#CmpI8I32Res2]] [[#CmpI8I32ConstMinusOne]] [[#CmpI8I32ConstZero]]
; CHECK-SPIRV: Select [[#TypeI8]] [[#SelI8I32Res2:]] [[#CmpI8I32Res1]] [[#SelI8I32Res1]] [[#CmpI8I32ConstOne]] 
; CHECK-SPIRV: ReturnValue [[#SelI8I32Res2]] 
define dso_local noundef range(i8 -1, 2) i8 @compare_i8i32(i32 noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = tail call i8 @llvm.scmp.i8.i32(i32 %0, i32 %1)
  ret i8 %3
}

; CHECK-SPIRV: SLessThanEqual [[#TypeBool]] [[#CmpI16I32Res1:]] [[#]] [[#]]
; CHECK-SPIRV: SLessThan [[#TypeBool]] [[#CmpI16I32Res2:]] [[#]] [[#]]
; CHECK-SPIRV: Select [[#TypeI16]] [[#SelI16I32Res1:]] [[#CmpI16I32Res2]] [[#CmpI16I32ConstMinusOne]] [[#CmpI16I32ConstZero]]
; CHECK-SPIRV: Select [[#TypeI16]] [[#SelI16I32Res2:]] [[#CmpI16I32Res1]] [[#SelI16I32Res1]] [[#CmpI16I32ConstOne]] 
; CHECK-SPIRV: ReturnValue [[#SelI16I32Res2]] 
define dso_local noundef range(i16 -1, 2) i16 @compare_i16i32(i32 noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = tail call i16 @llvm.scmp.i16.i32(i32 %0, i32 %1)
  ret i16 %3
}

; CHECK-SPIRV: SLessThanEqual [[#TypeBool]] [[#CmpI64I32Res1:]] [[#]] [[#]]
; CHECK-SPIRV: SLessThan [[#TypeBool]] [[#CmpI64I32Res2:]] [[#]] [[#]]
; CHECK-SPIRV: Select [[#TypeI64]] [[#SelI64I32Res1:]] [[#CmpI64I32Res2]] [[#CmpI64I32ConstMinusOne]] [[#CmpI64I32ConstZero]]
; CHECK-SPIRV: Select [[#TypeI64]] [[#SelI64I32Res2:]] [[#CmpI64I32Res1]] [[#SelI64I32Res1]] [[#CmpI64I32ConstOne]] 
; CHECK-SPIRV: ReturnValue [[#SelI64I32Res2]] 
define dso_local noundef range(i64 -1, 2) i64 @compare_i64i32(i32 noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = tail call i64 @llvm.scmp.i64.i32(i32 %0, i32 %1)
  ret i64 %3
}

; CHECK-SPIRV: SLessThanEqual [[#TypeBool]] [[#CmpI8I64Res1:]] [[#]] [[#]]
; CHECK-SPIRV: SLessThan [[#TypeBool]] [[#CmpI8I64Res2:]] [[#]] [[#]]
; CHECK-SPIRV: Select [[#TypeI8]] [[#SelI8I64Res1:]] [[#CmpI8I64Res2]] [[#CmpI8I64ConstMinusOne]] [[#CmpI8I64ConstZero]]
; CHECK-SPIRV: Select [[#TypeI8]] [[#SelI8I64Res2:]] [[#CmpI8I64Res1]] [[#SelI8I64Res1]] [[#CmpI8I64ConstOne]] 
; CHECK-SPIRV: ReturnValue [[#SelI8I64Res2]] 
define dso_local noundef range(i8 -1, 2) i8 @compare_i8i64(i64 noundef %0, i64 noundef %1) local_unnamed_addr #0 {
  %3 = tail call i8 @llvm.scmp.i8.i64(i64 %0, i64 %1)
  ret i8 %3
}

; CHECK-SPIRV: SLessThanEqual [[#TypeBool]] [[#CmpI16I64Res1:]] [[#]] [[#]]
; CHECK-SPIRV: SLessThan [[#TypeBool]] [[#CmpI16I64Res2:]] [[#]] [[#]]
; CHECK-SPIRV: Select [[#TypeI16]] [[#SelI16I64Res1:]] [[#CmpI16I64Res2]] [[#CmpI16I64ConstMinusOne]] [[#CmpI16I64ConstZero]]
; CHECK-SPIRV: Select [[#TypeI16]] [[#SelI16I64Res2:]] [[#CmpI16I64Res1]] [[#SelI16I64Res1]] [[#CmpI16I64ConstOne]] 
; CHECK-SPIRV: ReturnValue [[#SelI16I64Res2]] 
define dso_local noundef range(i16 -1, 2) i16 @compare_i16i64(i64 noundef %0, i64 noundef %1) local_unnamed_addr #0 {
  %3 = tail call i16 @llvm.scmp.i16.i64(i64 %0, i64 %1)
  ret i16 %3
}

; CHECK-SPIRV: SLessThanEqual [[#TypeBool]] [[#CmpI32I64Res1:]] [[#]] [[#]]
; CHECK-SPIRV: SLessThan [[#TypeBool]] [[#CmpI32I64Res2:]] [[#]] [[#]]
; CHECK-SPIRV: Select [[#TypeI32]] [[#SelI32I64Res1:]] [[#CmpI32I64Res2]] [[#CmpI32I64ConstMinusOne]] [[#CmpI32ConstZero]]
; CHECK-SPIRV: Select [[#TypeI32]] [[#SelI32I64Res2:]] [[#CmpI32I64Res1]] [[#SelI32I64Res1]] [[#CmpI32ConstOne]] 
; CHECK-SPIRV: ReturnValue [[#SelI32I64Res2]] 
define dso_local noundef range(i32 -1, 2) i32 @compare_i32i64(i64 noundef %0, i64 noundef %1) local_unnamed_addr #0 {
  %3 = tail call i32 @llvm.scmp.i32.i64(i64 %0, i64 %1)
  ret i32 %3
}

; Res type equal to arg type, scalar.
declare i8 @llvm.scmp.i8.i8(i8, i8) #1
declare i16 @llvm.scmp.i16.i16(i16, i16) #1
declare i32 @llvm.scmp.i32.i32(i32, i32) #1
declare i64 @llvm.scmp.i64.i64(i64, i64) #1

; Res type equal to arg type, array.
declare <4 x i8> @llvm.scmp.v4i8.v4i8(<4 x i8>, <4 x i8>) #1
declare <4 x i16> @llvm.scmp.v4i16.v4i16(<4 x i16>, <4 x i16>) #1
declare <4 x i32> @llvm.scmp.v4i32.v4i32(<4 x i32>, <4 x i32>) #1
declare <4 x i64> @llvm.scmp.v4i64.v4i64(<4 x i64>, <4 x i64>) #1

; Res type different than arg type, scalar.
declare i16 @llvm.scmp.i16.i8(i8, i8) #1
declare i32 @llvm.scmp.i32.i8(i8, i8) #1
declare i64 @llvm.scmp.i64.i8(i8, i8) #1
declare i8 @llvm.scmp.i8.i16(i16, i16) #1
declare i32 @llvm.scmp.i32.i16(i16, i16) #1
declare i64 @llvm.scmp.i64.i16(i16, i16) #1
declare i8 @llvm.scmp.i8.i32(i32, i32) #1
declare i16 @llvm.scmp.i16.i32(i32, i32) #1
declare i64 @llvm.scmp.i64.i32(i32, i32) #1
declare i8 @llvm.scmp.i8.i64(i64, i64) #1
declare i16 @llvm.scmp.i16.i64(i64, i64) #1
declare i32 @llvm.scmp.i32.i64(i64, i64) #1

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
