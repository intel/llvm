; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o - | FileCheck --check-prefix CHECK-SPIRV %s
; RUN: llvm-spirv %t.bc -o %t.spv
; Current implementation doesn't comply with specification and should be fixed in future.
; TODO: spirv-val %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc -o - | FileCheck --check-prefix CHECK-LLVM %s

target triple = "spir64-unknown-unknown"


; CHECK-SPIRV: TypeInt [[#I16TYPE:]] 16
; CHECK-SPIRV: TypeInt [[#I32TYPE:]] 32
; CHECK-SPIRV: TypeInt [[#I64TYPE:]] 64
; CHECK-SPIRV: TypeBool [[#BTYPE:]]
; CHECK-SPIRV: TypeStruct [[#S0TYPE:]] [[#I16TYPE]] [[#BTYPE]]
; CHECK-SPIRV: TypeStruct [[#S1TYPE:]] [[#I32TYPE]] [[#BTYPE]]
; CHECK-SPIRV: TypeStruct [[#S2TYPE:]] [[#I64TYPE]] [[#BTYPE]]
; CHECK-SPIRV: TypeVector [[#V4XI32TYPE:]] [[#I32TYPE]] 4
; CHECK-SPIRV: TypeVector [[#V4XBTYPE:]] [[#BTYPE]] 4
; CHECK-SPIRV: TypeStruct [[#S3TYPE:]] [[#V4XI32TYPE]] [[#V4XBTYPE]]
; CHECK-SPIRV: ISubBorrow [[#S0TYPE]]
; CHECK-SPIRV: ISubBorrow [[#S1TYPE]]
; CHECK-SPIRV: ISubBorrow [[#S2TYPE]]
; CHECK-SPIRV: ISubBorrow [[#S3TYPE]]
; CHECK-LLVM: call { i16, i1 } @llvm.usub.with.overflow.i16(i16 %a, i16 %b)
; CHECK-LLVM: call { i32, i1 } @llvm.usub.with.overflow.i32(i32 %a, i32 %b)
; CHECK-LLVM: call { i64, i1 } @llvm.usub.with.overflow.i64(i64 %a, i64 %b)
; CHECK-LLVM: call { <4 x i32>, <4 x i1> } @llvm.usub.with.overflow.v4i32(<4 x i32> %a, <4 x i32> %b)

define spir_func void @test_usub_with_overflow_i16(i16 %a, i16 %b) {
entry:
  %res = call {i16, i1} @llvm.usub.with.overflow.i16(i16 %a, i16 %b)
  ret void
}

define spir_func void @test_usub_with_overflow_i32(i32 %a, i32 %b) {
entry:
  %res = call {i32, i1} @llvm.usub.with.overflow.i32(i32 %a, i32 %b)
  ret void
}

define spir_func void @test_usub_with_overflow_i64(i64 %a, i64 %b) {
entry:
  %res = call {i64, i1} @llvm.usub.with.overflow.i64(i64 %a, i64 %b)
  ret void
}

define spir_func void @test_usub_with_overflow_v4i32(<4 x i32> %a, <4 x i32> %b) {
entry:
 %res = call {<4 x i32>, <4 x i1>} @llvm.usub.with.overflow.v4i32(<4 x i32> %a, <4 x i32> %b)
 ret void
}

declare {i16, i1} @llvm.usub.with.overflow.i16(i16 %a, i16 %b)
declare {i32, i1} @llvm.usub.with.overflow.i32(i32 %a, i32 %b)
declare {i64, i1} @llvm.usub.with.overflow.i64(i64 %a, i64 %b)
declare {<4 x i32>, <4 x i1>} @llvm.usub.with.overflow.v4i32(<4 x i32> %a, <4 x i32> %b)
