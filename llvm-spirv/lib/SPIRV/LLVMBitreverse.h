//===- LLVMBitreverse.h - implementation of llvm.bitreverse -===//
//
//                     The LLVM/SPIRV Translator
//
// Copyright (c) 2024 The Khronos Group Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
//
// This file implements lowering of llvm.bitreverse.* into basic LLVM
// operations.
//
//===----------------------------------------------------------------------===//

// The IR below is manually modified IR which was produced by the
// commands:
//
//   clang -emit-llvm bitreverse.c -S -O2
//   cat bitreverse.ll | sed 's/ dso_local//'                 \
//                     | sed 's/ noundef//'                   \
//                     | sed 's/zeroext %/%/'                 \
//                     | sed 's/ local_unnamed_addr #[0-9]//' \
//                     | sed 's/, !tbaa !3//'                 \
//                     | grep -v "Function Attrs:"
//
// from the C code in LLVMIntrinsicEmulation/bitreverse.c with a custom clang
// that was modified to disable llvm.bitreverse.* intrinsic generation.
//
// Manual modification was done to avoid coercing vector types into scalar
// types.  For example, the original LLVM IR:
//
//   define i32 @llvm_bitreverse_v4i8(i32 %a.coerce) {
//   entry:
//     %0 = bitcast i32 %a.coerce to <4 x i8>
//     %shl = shl <4 x i8> %0, <i8 4, i8 4, i8 4, i8 4>
//     %shr = lshr <4 x i8> %0, <i8 4, i8 4, i8 4, i8 4>
//     ...
//     %1 = bitcast <4 x i8> %or12 to i32
//     ret i32 %1
//   }
//
// was converted to:
//
//   define <4 x i8> @llvm_bitreverse_v4i8(<4 x i8> %a) {
//   entry:
//     %shl = shl <4 x i8> %a, <i8 4, i8 4, i8 4, i8 4>
//     %shr = lshr <4 x i8> %a, <i8 4, i8 4, i8 4, i8 4>
//     ...
//     ret <4 x i8> %or12
//   }

static const char LLVMBitreverseScalari8[]{R"(
define zeroext i8 @llvm_bitreverse_i8(i8 %A) {
entry:
  %and = shl i8 %A, 4
  %shr = lshr i8 %A, 4
  %or = or disjoint i8 %and, %shr
  %and5 = shl i8 %or, 2
  %shl6 = and i8 %and5, -52
  %shr8 = lshr i8 %or, 2
  %and9 = and i8 %shr8, 51
  %or10 = or disjoint i8 %shl6, %and9
  %and13 = shl i8 %or10, 1
  %shl14 = and i8 %and13, -86
  %shr16 = lshr i8 %or10, 1
  %and17 = and i8 %shr16, 85
  %or18 = or disjoint i8 %shl14, %and17
  ret i8 %or18
}
)"};

static const char LLVMBitreverseScalari16[]{R"(
define zeroext i16 @llvm_bitreverse_i16(i16 %A) {
entry:
  %and = shl i16 %A, 8
  %shr = lshr i16 %A, 8
  %or = or disjoint i16 %and, %shr
  %and5 = shl i16 %or, 4
  %shl6 = and i16 %and5, -3856
  %shr8 = lshr i16 %or, 4
  %and9 = and i16 %shr8, 3855
  %or10 = or disjoint i16 %shl6, %and9
  %and13 = shl i16 %or10, 2
  %shl14 = and i16 %and13, -13108
  %shr16 = lshr i16 %or10, 2
  %and17 = and i16 %shr16, 13107
  %or18 = or disjoint i16 %shl14, %and17
  %and21 = shl i16 %or18, 1
  %shl22 = and i16 %and21, -21846
  %shr24 = lshr i16 %or18, 1
  %and25 = and i16 %shr24, 21845
  %or26 = or disjoint i16 %shl22, %and25
  ret i16 %or26
}
)"};

static const char LLVMBitreverseScalari32[]{R"(
define i32 @llvm_bitreverse_i32(i32 %A) {
entry:
  %and = shl i32 %A, 16
  %shr = lshr i32 %A, 16
  %or = or disjoint i32 %and, %shr
  %and2 = shl i32 %or, 8
  %shl3 = and i32 %and2, -16711936
  %shr4 = lshr i32 %or, 8
  %and5 = and i32 %shr4, 16711935
  %or6 = or disjoint i32 %shl3, %and5
  %and7 = shl i32 %or6, 4
  %shl8 = and i32 %and7, -252645136
  %shr9 = lshr i32 %or6, 4
  %and10 = and i32 %shr9, 252645135
  %or11 = or disjoint i32 %shl8, %and10
  %and12 = shl i32 %or11, 2
  %shl13 = and i32 %and12, -858993460
  %shr14 = lshr i32 %or11, 2
  %and15 = and i32 %shr14, 858993459
  %or16 = or disjoint i32 %shl13, %and15
  %and17 = shl i32 %or16, 1
  %shl18 = and i32 %and17, -1431655766
  %shr19 = lshr i32 %or16, 1
  %and20 = and i32 %shr19, 1431655765
  %or21 = or disjoint i32 %shl18, %and20
  ret i32 %or21
}
)"};

static const char LLVMBitreverseScalari64[]{R"(
define i64 @llvm_bitreverse_i64(i64 %A) {
entry:
  %and = shl i64 %A, 32
  %shr = lshr i64 %A, 32
  %or = or disjoint i64 %and, %shr
  %and2 = shl i64 %or, 16
  %shl3 = and i64 %and2, -281470681808896
  %shr4 = lshr i64 %or, 16
  %and5 = and i64 %shr4, 281470681808895
  %or6 = or disjoint i64 %shl3, %and5
  %and7 = shl i64 %or6, 8
  %shl8 = and i64 %and7, -71777214294589696
  %shr9 = lshr i64 %or6, 8
  %and10 = and i64 %shr9, 71777214294589695
  %or11 = or disjoint i64 %shl8, %and10
  %and12 = shl i64 %or11, 4
  %shl13 = and i64 %and12, -1085102592571150096
  %shr14 = lshr i64 %or11, 4
  %and15 = and i64 %shr14, 1085102592571150095
  %or16 = or disjoint i64 %shl13, %and15
  %and17 = shl i64 %or16, 2
  %shl18 = and i64 %and17, -3689348814741910324
  %shr19 = lshr i64 %or16, 2
  %and20 = and i64 %shr19, 3689348814741910323
  %or21 = or disjoint i64 %shl18, %and20
  %and22 = shl i64 %or21, 1
  %shl23 = and i64 %and22, -6148914691236517206
  %shr24 = lshr i64 %or21, 1
  %and25 = and i64 %shr24, 6148914691236517205
  %or26 = or disjoint i64 %shl23, %and25
  ret i64 %or26
}
)"};

static const char LLVMBitreverseV2i8[]{R"(
define <2 x i8> @llvm_bitreverse_v2i8(<2 x i8> %A) {
entry:
  %shl = shl <2 x i8> %A, <i8 4, i8 4>
  %shr = lshr <2 x i8> %A, <i8 4, i8 4>
  %or = or disjoint <2 x i8> %shl, %shr
  %and3 = shl <2 x i8> %or, <i8 2, i8 2>
  %shl4 = and <2 x i8> %and3, <i8 -52, i8 -52>
  %shr5 = lshr <2 x i8> %or, <i8 2, i8 2>
  %and6 = and <2 x i8> %shr5, <i8 51, i8 51>
  %or7 = or disjoint <2 x i8> %shl4, %and6
  %and8 = shl <2 x i8> %or7, <i8 1, i8 1>
  %shl9 = and <2 x i8> %and8, <i8 -86, i8 -86>
  %shr10 = lshr <2 x i8> %or7, <i8 1, i8 1>
  %and11 = and <2 x i8> %shr10, <i8 85, i8 85>
  %or12 = or disjoint <2 x i8> %shl9, %and11
  ret <2 x i8> %or12
}
)"};

static const char LLVMBitreverseV2i16[]{R"(
define <2 x i16> @llvm_bitreverse_v2i16(<2 x i16> %A) {
entry:
  %shl = shl <2 x i16> %A, <i16 8, i16 8>
  %shr = lshr <2 x i16> %A, <i16 8, i16 8>
  %or = or disjoint <2 x i16> %shl, %shr
  %and3 = shl <2 x i16> %or, <i16 4, i16 4>
  %shl4 = and <2 x i16> %and3, <i16 -3856, i16 -3856>
  %shr5 = lshr <2 x i16> %or, <i16 4, i16 4>
  %and6 = and <2 x i16> %shr5, <i16 3855, i16 3855>
  %or7 = or disjoint <2 x i16> %shl4, %and6
  %and8 = shl <2 x i16> %or7, <i16 2, i16 2>
  %shl9 = and <2 x i16> %and8, <i16 -13108, i16 -13108>
  %shr10 = lshr <2 x i16> %or7, <i16 2, i16 2>
  %and11 = and <2 x i16> %shr10, <i16 13107, i16 13107>
  %or12 = or disjoint <2 x i16> %shl9, %and11
  %and13 = shl <2 x i16> %or12, <i16 1, i16 1>
  %shl14 = and <2 x i16> %and13, <i16 -21846, i16 -21846>
  %shr15 = lshr <2 x i16> %or12, <i16 1, i16 1>
  %and16 = and <2 x i16> %shr15, <i16 21845, i16 21845>
  %or17 = or disjoint <2 x i16> %shl14, %and16
  ret <2 x i16> %or17
}
)"};

static const char LLVMBitreverseV2i32[]{R"(
define <2 x i32> @llvm_bitreverse_v2i32(<2 x i32> %A) {
entry:
  %shl = shl <2 x i32> %A, <i32 16, i32 16>
  %shr = lshr <2 x i32> %A, <i32 16, i32 16>
  %or = or disjoint <2 x i32> %shl, %shr
  %and3 = shl <2 x i32> %or, <i32 8, i32 8>
  %shl4 = and <2 x i32> %and3, <i32 -16711936, i32 -16711936>
  %shr5 = lshr <2 x i32> %or, <i32 8, i32 8>
  %and6 = and <2 x i32> %shr5, <i32 16711935, i32 16711935>
  %or7 = or disjoint <2 x i32> %shl4, %and6
  %and8 = shl <2 x i32> %or7, <i32 4, i32 4>
  %shl9 = and <2 x i32> %and8, <i32 -252645136, i32 -252645136>
  %shr10 = lshr <2 x i32> %or7, <i32 4, i32 4>
  %and11 = and <2 x i32> %shr10, <i32 252645135, i32 252645135>
  %or12 = or disjoint <2 x i32> %shl9, %and11
  %and13 = shl <2 x i32> %or12, <i32 2, i32 2>
  %shl14 = and <2 x i32> %and13, <i32 -858993460, i32 -858993460>
  %shr15 = lshr <2 x i32> %or12, <i32 2, i32 2>
  %and16 = and <2 x i32> %shr15, <i32 858993459, i32 858993459>
  %or17 = or disjoint <2 x i32> %shl14, %and16
  %and18 = shl <2 x i32> %or17, <i32 1, i32 1>
  %shl19 = and <2 x i32> %and18, <i32 -1431655766, i32 -1431655766>
  %shr20 = lshr <2 x i32> %or17, <i32 1, i32 1>
  %and21 = and <2 x i32> %shr20, <i32 1431655765, i32 1431655765>
  %or22 = or disjoint <2 x i32> %shl19, %and21
  ret <2 x i32> %or22
}
)"};

static const char LLVMBitreverseV2i64[]{R"(
define <2 x i64> @llvm_bitreverse_v2i64(<2 x i64> %A) {
entry:
  %shl = shl <2 x i64> %A, <i64 32, i64 32>
  %shr = lshr <2 x i64> %A, <i64 32, i64 32>
  %or = or disjoint <2 x i64> %shl, %shr
  %and2 = shl <2 x i64> %or, <i64 16, i64 16>
  %shl3 = and <2 x i64> %and2, <i64 -281470681808896, i64 -281470681808896>
  %shr4 = lshr <2 x i64> %or, <i64 16, i64 16>
  %and5 = and <2 x i64> %shr4, <i64 281470681808895, i64 281470681808895>
  %or6 = or disjoint <2 x i64> %shl3, %and5
  %and7 = shl <2 x i64> %or6, <i64 8, i64 8>
  %shl8 = and <2 x i64> %and7, <i64 -71777214294589696, i64 -71777214294589696>
  %shr9 = lshr <2 x i64> %or6, <i64 8, i64 8>
  %and10 = and <2 x i64> %shr9, <i64 71777214294589695, i64 71777214294589695>
  %or11 = or disjoint <2 x i64> %shl8, %and10
  %and12 = shl <2 x i64> %or11, <i64 4, i64 4>
  %shl13 = and <2 x i64> %and12, <i64 -1085102592571150096, i64 -1085102592571150096>
  %shr14 = lshr <2 x i64> %or11, <i64 4, i64 4>
  %and15 = and <2 x i64> %shr14, <i64 1085102592571150095, i64 1085102592571150095>
  %or16 = or disjoint <2 x i64> %shl13, %and15
  %and17 = shl <2 x i64> %or16, <i64 2, i64 2>
  %shl18 = and <2 x i64> %and17, <i64 -3689348814741910324, i64 -3689348814741910324>
  %shr19 = lshr <2 x i64> %or16, <i64 2, i64 2>
  %and20 = and <2 x i64> %shr19, <i64 3689348814741910323, i64 3689348814741910323>
  %or21 = or disjoint <2 x i64> %shl18, %and20
  %and22 = shl <2 x i64> %or21, <i64 1, i64 1>
  %shl23 = and <2 x i64> %and22, <i64 -6148914691236517206, i64 -6148914691236517206>
  %shr24 = lshr <2 x i64> %or21, <i64 1, i64 1>
  %and25 = and <2 x i64> %shr24, <i64 6148914691236517205, i64 6148914691236517205>
  %or26 = or disjoint <2 x i64> %shl23, %and25
  ret <2 x i64> %or26
}
)"};

static const char LLVMBitreverseV3i8[]{R"(
define <3 x i8> @llvm_bitreverse_v3i8(<3 x i8> %A) {
entry:
  %shl = shl <3 x i8> %A, <i8 4, i8 4, i8 4>
  %shr = lshr <3 x i8> %A, <i8 4, i8 4, i8 4>
  %or = or disjoint <3 x i8> %shl, %shr
  %and10 = shl <3 x i8> %or, <i8 2, i8 2, i8 2>
  %shl11 = and <3 x i8> %and10, <i8 -52, i8 -52, i8 -52>
  %shr14 = lshr <3 x i8> %or, <i8 2, i8 2, i8 2>
  %and15 = and <3 x i8> %shr14, <i8 51, i8 51, i8 51>
  %or16 = or disjoint <3 x i8> %shl11, %and15
  %and20 = shl <3 x i8> %or16, <i8 1, i8 1, i8 1>
  %shl21 = and <3 x i8> %and20, <i8 -86, i8 -86, i8 -86>
  %shr24 = lshr <3 x i8> %or16, <i8 1, i8 1, i8 1>
  %and25 = and <3 x i8> %shr24, <i8 85, i8 85, i8 85>
  %or26 = or disjoint <3 x i8> %shl21, %and25
  ret <3 x i8> %or26
}
)"};

static const char LLVMBitreverseV3i16[]{R"(
define <3 x i16> @llvm_bitreverse_v3i16(<3 x i16> %A) {
entry:
  %shl = shl <3 x i16> %A, <i16 8, i16 8, i16 8>
  %shr = lshr <3 x i16> %A, <i16 8, i16 8, i16 8>
  %or = or disjoint <3 x i16> %shl, %shr
  %and10 = shl <3 x i16> %or, <i16 4, i16 4, i16 4>
  %shl11 = and <3 x i16> %and10, <i16 -3856, i16 -3856, i16 -3856>
  %shr14 = lshr <3 x i16> %or, <i16 4, i16 4, i16 4>
  %and15 = and <3 x i16> %shr14, <i16 3855, i16 3855, i16 3855>
  %or16 = or disjoint <3 x i16> %shl11, %and15
  %and20 = shl <3 x i16> %or16, <i16 2, i16 2, i16 2>
  %shl21 = and <3 x i16> %and20, <i16 -13108, i16 -13108, i16 -13108>
  %shr24 = lshr <3 x i16> %or16, <i16 2, i16 2, i16 2>
  %and25 = and <3 x i16> %shr24, <i16 13107, i16 13107, i16 13107>
  %or26 = or disjoint <3 x i16> %shl21, %and25
  %and30 = shl <3 x i16> %or26, <i16 1, i16 1, i16 1>
  %shl31 = and <3 x i16> %and30, <i16 -21846, i16 -21846, i16 -21846>
  %shr34 = lshr <3 x i16> %or26, <i16 1, i16 1, i16 1>
  %and35 = and <3 x i16> %shr34, <i16 21845, i16 21845, i16 21845>
  %or36 = or disjoint <3 x i16> %shl31, %and35
  ret <3 x i16> %or36
}
)"};

static const char LLVMBitreverseV3i32[]{R"(
define <3 x i32> @llvm_bitreverse_v3i32(<3 x i32> %A) {
entry:
  %shl = shl <3 x i32> %A, <i32 16, i32 16, i32 16>
  %shr = lshr <3 x i32> %A, <i32 16, i32 16, i32 16>
  %or = or disjoint <3 x i32> %shl, %shr
  %and8 = shl <3 x i32> %or, <i32 8, i32 8, i32 8>
  %shl9 = and <3 x i32> %and8, <i32 -16711936, i32 -16711936, i32 -16711936>
  %shr12 = lshr <3 x i32> %or, <i32 8, i32 8, i32 8>
  %and13 = and <3 x i32> %shr12, <i32 16711935, i32 16711935, i32 16711935>
  %or14 = or disjoint <3 x i32> %shl9, %and13
  %and18 = shl <3 x i32> %or14, <i32 4, i32 4, i32 4>
  %shl19 = and <3 x i32> %and18, <i32 -252645136, i32 -252645136, i32 -252645136>
  %shr22 = lshr <3 x i32> %or14, <i32 4, i32 4, i32 4>
  %and23 = and <3 x i32> %shr22, <i32 252645135, i32 252645135, i32 252645135>
  %or24 = or disjoint <3 x i32> %shl19, %and23
  %and28 = shl <3 x i32> %or24, <i32 2, i32 2, i32 2>
  %shl29 = and <3 x i32> %and28, <i32 -858993460, i32 -858993460, i32 -858993460>
  %shr32 = lshr <3 x i32> %or24, <i32 2, i32 2, i32 2>
  %and33 = and <3 x i32> %shr32, <i32 858993459, i32 858993459, i32 858993459>
  %or34 = or disjoint <3 x i32> %shl29, %and33
  %and38 = shl <3 x i32> %or34, <i32 1, i32 1, i32 1>
  %shl39 = and <3 x i32> %and38, <i32 -1431655766, i32 -1431655766, i32 -1431655766>
  %shr42 = lshr <3 x i32> %or34, <i32 1, i32 1, i32 1>
  %and43 = and <3 x i32> %shr42, <i32 1431655765, i32 1431655765, i32 1431655765>
  %or44 = or disjoint <3 x i32> %shl39, %and43
  ret <3 x i32> %or44
}
)"};

static const char LLVMBitreverseV3i64[]{R"(
define <3 x i64> @llvm_bitreverse_v3i64(<3 x i64> %A) {
entry:
  %shl = shl <3 x i64> %A, <i64 32, i64 32, i64 32>
  %shr = lshr <3 x i64> %A, <i64 32, i64 32, i64 32>
  %or = or disjoint <3 x i64> %shl, %shr
  %and9 = shl <3 x i64> %or, <i64 16, i64 16, i64 16>
  %shl10 = and <3 x i64> %and9, <i64 -281470681808896, i64 -281470681808896, i64 -281470681808896>
  %shr13 = lshr <3 x i64> %or, <i64 16, i64 16, i64 16>
  %and14 = and <3 x i64> %shr13, <i64 281470681808895, i64 281470681808895, i64 281470681808895>
  %or15 = or disjoint <3 x i64> %shl10, %and14
  %and19 = shl <3 x i64> %or15, <i64 8, i64 8, i64 8>
  %shl20 = and <3 x i64> %and19, <i64 -71777214294589696, i64 -71777214294589696, i64 -71777214294589696>
  %shr23 = lshr <3 x i64> %or15, <i64 8, i64 8, i64 8>
  %and24 = and <3 x i64> %shr23, <i64 71777214294589695, i64 71777214294589695, i64 71777214294589695>
  %or25 = or disjoint <3 x i64> %shl20, %and24
  %and29 = shl <3 x i64> %or25, <i64 4, i64 4, i64 4>
  %shl30 = and <3 x i64> %and29, <i64 -1085102592571150096, i64 -1085102592571150096, i64 -1085102592571150096>
  %shr33 = lshr <3 x i64> %or25, <i64 4, i64 4, i64 4>
  %and34 = and <3 x i64> %shr33, <i64 1085102592571150095, i64 1085102592571150095, i64 1085102592571150095>
  %or35 = or disjoint <3 x i64> %shl30, %and34
  %and39 = shl <3 x i64> %or35, <i64 2, i64 2, i64 2>
  %shl40 = and <3 x i64> %and39, <i64 -3689348814741910324, i64 -3689348814741910324, i64 -3689348814741910324>
  %shr43 = lshr <3 x i64> %or35, <i64 2, i64 2, i64 2>
  %and44 = and <3 x i64> %shr43, <i64 3689348814741910323, i64 3689348814741910323, i64 3689348814741910323>
  %or45 = or disjoint <3 x i64> %shl40, %and44
  %and49 = shl <3 x i64> %or45, <i64 1, i64 1, i64 1>
  %shl50 = and <3 x i64> %and49, <i64 -6148914691236517206, i64 -6148914691236517206, i64 -6148914691236517206>
  %shr53 = lshr <3 x i64> %or45, <i64 1, i64 1, i64 1>
  %and54 = and <3 x i64> %shr53, <i64 6148914691236517205, i64 6148914691236517205, i64 6148914691236517205>
  %or55 = or disjoint <3 x i64> %shl50, %and54
  ret <3 x i64> %or55
}
)"};

static const char LLVMBitreverseV4i8[]{R"(
define <4 x i8> @llvm_bitreverse_v4i8(<4 x i8> %A) {
entry:
  %shl = shl <4 x i8> %A, <i8 4, i8 4, i8 4, i8 4>
  %shr = lshr <4 x i8> %A, <i8 4, i8 4, i8 4, i8 4>
  %or = or disjoint <4 x i8> %shl, %shr
  %and3 = shl <4 x i8> %or, <i8 2, i8 2, i8 2, i8 2>
  %shl4 = and <4 x i8> %and3, <i8 -52, i8 -52, i8 -52, i8 -52>
  %shr5 = lshr <4 x i8> %or, <i8 2, i8 2, i8 2, i8 2>
  %and6 = and <4 x i8> %shr5, <i8 51, i8 51, i8 51, i8 51>
  %or7 = or disjoint <4 x i8> %shl4, %and6
  %and8 = shl <4 x i8> %or7, <i8 1, i8 1, i8 1, i8 1>
  %shl9 = and <4 x i8> %and8, <i8 -86, i8 -86, i8 -86, i8 -86>
  %shr10 = lshr <4 x i8> %or7, <i8 1, i8 1, i8 1, i8 1>
  %and11 = and <4 x i8> %shr10, <i8 85, i8 85, i8 85, i8 85>
  %or12 = or disjoint <4 x i8> %shl9, %and11
  ret <4 x i8> %or12
}
)"};

static const char LLVMBitreverseV4i16[]{R"(
define <4 x i16> @llvm_bitreverse_v4i16(<4 x i16> %A) {
entry:
  %shl = shl <4 x i16> %A, <i16 8, i16 8, i16 8, i16 8>
  %shr = lshr <4 x i16> %A, <i16 8, i16 8, i16 8, i16 8>
  %or = or disjoint <4 x i16> %shl, %shr
  %and3 = shl <4 x i16> %or, <i16 4, i16 4, i16 4, i16 4>
  %shl4 = and <4 x i16> %and3, <i16 -3856, i16 -3856, i16 -3856, i16 -3856>
  %shr5 = lshr <4 x i16> %or, <i16 4, i16 4, i16 4, i16 4>
  %and6 = and <4 x i16> %shr5, <i16 3855, i16 3855, i16 3855, i16 3855>
  %or7 = or disjoint <4 x i16> %shl4, %and6
  %and8 = shl <4 x i16> %or7, <i16 2, i16 2, i16 2, i16 2>
  %shl9 = and <4 x i16> %and8, <i16 -13108, i16 -13108, i16 -13108, i16 -13108>
  %shr10 = lshr <4 x i16> %or7, <i16 2, i16 2, i16 2, i16 2>
  %and11 = and <4 x i16> %shr10, <i16 13107, i16 13107, i16 13107, i16 13107>
  %or12 = or disjoint <4 x i16> %shl9, %and11
  %and13 = shl <4 x i16> %or12, <i16 1, i16 1, i16 1, i16 1>
  %shl14 = and <4 x i16> %and13, <i16 -21846, i16 -21846, i16 -21846, i16 -21846>
  %shr15 = lshr <4 x i16> %or12, <i16 1, i16 1, i16 1, i16 1>
  %and16 = and <4 x i16> %shr15, <i16 21845, i16 21845, i16 21845, i16 21845>
  %or17 = or disjoint <4 x i16> %shl14, %and16
  ret <4 x i16> %or17
}
)"};

static const char LLVMBitreverseV4i32[]{R"(
define <4 x i32> @llvm_bitreverse_v4i32(<4 x i32> %A) {
entry:
  %shl = shl <4 x i32> %A, <i32 16, i32 16, i32 16, i32 16>
  %shr = lshr <4 x i32> %A, <i32 16, i32 16, i32 16, i32 16>
  %or = or disjoint <4 x i32> %shl, %shr
  %and2 = shl <4 x i32> %or, <i32 8, i32 8, i32 8, i32 8>
  %shl3 = and <4 x i32> %and2, <i32 -16711936, i32 -16711936, i32 -16711936, i32 -16711936>
  %shr4 = lshr <4 x i32> %or, <i32 8, i32 8, i32 8, i32 8>
  %and5 = and <4 x i32> %shr4, <i32 16711935, i32 16711935, i32 16711935, i32 16711935>
  %or6 = or disjoint <4 x i32> %shl3, %and5
  %and7 = shl <4 x i32> %or6, <i32 4, i32 4, i32 4, i32 4>
  %shl8 = and <4 x i32> %and7, <i32 -252645136, i32 -252645136, i32 -252645136, i32 -252645136>
  %shr9 = lshr <4 x i32> %or6, <i32 4, i32 4, i32 4, i32 4>
  %and10 = and <4 x i32> %shr9, <i32 252645135, i32 252645135, i32 252645135, i32 252645135>
  %or11 = or disjoint <4 x i32> %shl8, %and10
  %and12 = shl <4 x i32> %or11, <i32 2, i32 2, i32 2, i32 2>
  %shl13 = and <4 x i32> %and12, <i32 -858993460, i32 -858993460, i32 -858993460, i32 -858993460>
  %shr14 = lshr <4 x i32> %or11, <i32 2, i32 2, i32 2, i32 2>
  %and15 = and <4 x i32> %shr14, <i32 858993459, i32 858993459, i32 858993459, i32 858993459>
  %or16 = or disjoint <4 x i32> %shl13, %and15
  %and17 = shl <4 x i32> %or16, <i32 1, i32 1, i32 1, i32 1>
  %shl18 = and <4 x i32> %and17, <i32 -1431655766, i32 -1431655766, i32 -1431655766, i32 -1431655766>
  %shr19 = lshr <4 x i32> %or16, <i32 1, i32 1, i32 1, i32 1>
  %and20 = and <4 x i32> %shr19, <i32 1431655765, i32 1431655765, i32 1431655765, i32 1431655765>
  %or21 = or disjoint <4 x i32> %shl18, %and20
  ret <4 x i32> %or21
}
)"};

static const char LLVMBitreverseV4i64[]{R"(
define <4 x i64> @llvm_bitreverse_v4i64(<4 x i64> %A) {
entry:
  %shl = shl <4 x i64> %A, <i64 32, i64 32, i64 32, i64 32>
  %shr = lshr <4 x i64> %A, <i64 32, i64 32, i64 32, i64 32>
  %or = or disjoint <4 x i64> %shl, %shr
  %and2 = shl <4 x i64> %or, <i64 16, i64 16, i64 16, i64 16>
  %shl3 = and <4 x i64> %and2, <i64 -281470681808896, i64 -281470681808896, i64 -281470681808896, i64 -281470681808896>
  %shr4 = lshr <4 x i64> %or, <i64 16, i64 16, i64 16, i64 16>
  %and5 = and <4 x i64> %shr4, <i64 281470681808895, i64 281470681808895, i64 281470681808895, i64 281470681808895>
  %or6 = or disjoint <4 x i64> %shl3, %and5
  %and7 = shl <4 x i64> %or6, <i64 8, i64 8, i64 8, i64 8>
  %shl8 = and <4 x i64> %and7, <i64 -71777214294589696, i64 -71777214294589696, i64 -71777214294589696, i64 -71777214294589696>
  %shr9 = lshr <4 x i64> %or6, <i64 8, i64 8, i64 8, i64 8>
  %and10 = and <4 x i64> %shr9, <i64 71777214294589695, i64 71777214294589695, i64 71777214294589695, i64 71777214294589695>
  %or11 = or disjoint <4 x i64> %shl8, %and10
  %and12 = shl <4 x i64> %or11, <i64 4, i64 4, i64 4, i64 4>
  %shl13 = and <4 x i64> %and12, <i64 -1085102592571150096, i64 -1085102592571150096, i64 -1085102592571150096, i64 -1085102592571150096>
  %shr14 = lshr <4 x i64> %or11, <i64 4, i64 4, i64 4, i64 4>
  %and15 = and <4 x i64> %shr14, <i64 1085102592571150095, i64 1085102592571150095, i64 1085102592571150095, i64 1085102592571150095>
  %or16 = or disjoint <4 x i64> %shl13, %and15
  %and17 = shl <4 x i64> %or16, <i64 2, i64 2, i64 2, i64 2>
  %shl18 = and <4 x i64> %and17, <i64 -3689348814741910324, i64 -3689348814741910324, i64 -3689348814741910324, i64 -3689348814741910324>
  %shr19 = lshr <4 x i64> %or16, <i64 2, i64 2, i64 2, i64 2>
  %and20 = and <4 x i64> %shr19, <i64 3689348814741910323, i64 3689348814741910323, i64 3689348814741910323, i64 3689348814741910323>
  %or21 = or disjoint <4 x i64> %shl18, %and20
  %and22 = shl <4 x i64> %or21, <i64 1, i64 1, i64 1, i64 1>
  %shl23 = and <4 x i64> %and22, <i64 -6148914691236517206, i64 -6148914691236517206, i64 -6148914691236517206, i64 -6148914691236517206>
  %shr24 = lshr <4 x i64> %or21, <i64 1, i64 1, i64 1, i64 1>
  %and25 = and <4 x i64> %shr24, <i64 6148914691236517205, i64 6148914691236517205, i64 6148914691236517205, i64 6148914691236517205>
  %or26 = or disjoint <4 x i64> %shl23, %and25
  ret <4 x i64> %or26
}
)"};

static const char LLVMBitreverseV8i8[]{R"(
define <8 x i8> @llvm_bitreverse_v8i8(<8 x i8> %A) {
entry:
  %shl = shl <8 x i8> %A, <i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4>
  %shr = lshr <8 x i8> %A, <i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4>
  %or = or disjoint <8 x i8> %shl, %shr
  %and3 = shl <8 x i8> %or, <i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2>
  %shl4 = and <8 x i8> %and3, <i8 -52, i8 -52, i8 -52, i8 -52, i8 -52, i8 -52, i8 -52, i8 -52>
  %shr5 = lshr <8 x i8> %or, <i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2>
  %and6 = and <8 x i8> %shr5, <i8 51, i8 51, i8 51, i8 51, i8 51, i8 51, i8 51, i8 51>
  %or7 = or disjoint <8 x i8> %shl4, %and6
  %and8 = shl <8 x i8> %or7, <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>
  %shl9 = and <8 x i8> %and8, <i8 -86, i8 -86, i8 -86, i8 -86, i8 -86, i8 -86, i8 -86, i8 -86>
  %shr10 = lshr <8 x i8> %or7, <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>
  %and11 = and <8 x i8> %shr10, <i8 85, i8 85, i8 85, i8 85, i8 85, i8 85, i8 85, i8 85>
  %or12 = or disjoint <8 x i8> %shl9, %and11
  ret <8 x i8> %or12
}
)"};

static const char LLVMBitreverseV8i16[]{R"(
define <8 x i16> @llvm_bitreverse_v8i16(<8 x i16> %A) {
entry:
  %shl = shl <8 x i16> %A, <i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8>
  %shr = lshr <8 x i16> %A, <i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8>
  %or = or disjoint <8 x i16> %shl, %shr
  %and2 = shl <8 x i16> %or, <i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4>
  %shl3 = and <8 x i16> %and2, <i16 -3856, i16 -3856, i16 -3856, i16 -3856, i16 -3856, i16 -3856, i16 -3856, i16 -3856>
  %shr4 = lshr <8 x i16> %or, <i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4>
  %and5 = and <8 x i16> %shr4, <i16 3855, i16 3855, i16 3855, i16 3855, i16 3855, i16 3855, i16 3855, i16 3855>
  %or6 = or disjoint <8 x i16> %shl3, %and5
  %and7 = shl <8 x i16> %or6, <i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2>
  %shl8 = and <8 x i16> %and7, <i16 -13108, i16 -13108, i16 -13108, i16 -13108, i16 -13108, i16 -13108, i16 -13108, i16 -13108>
  %shr9 = lshr <8 x i16> %or6, <i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2>
  %and10 = and <8 x i16> %shr9, <i16 13107, i16 13107, i16 13107, i16 13107, i16 13107, i16 13107, i16 13107, i16 13107>
  %or11 = or disjoint <8 x i16> %shl8, %and10
  %and12 = shl <8 x i16> %or11, <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
  %shl13 = and <8 x i16> %and12, <i16 -21846, i16 -21846, i16 -21846, i16 -21846, i16 -21846, i16 -21846, i16 -21846, i16 -21846>
  %shr14 = lshr <8 x i16> %or11, <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
  %and15 = and <8 x i16> %shr14, <i16 21845, i16 21845, i16 21845, i16 21845, i16 21845, i16 21845, i16 21845, i16 21845>
  %or16 = or disjoint <8 x i16> %shl13, %and15
  ret <8 x i16> %or16
}
)"};

static const char LLVMBitreverseV8i32[]{R"(
define <8 x i32> @llvm_bitreverse_v8i32(<8 x i32> %A) {
entry:
  %shl = shl <8 x i32> %A, <i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16>
  %shr = lshr <8 x i32> %A, <i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16>
  %or = or disjoint <8 x i32> %shl, %shr
  %and2 = shl <8 x i32> %or, <i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8>
  %shl3 = and <8 x i32> %and2, <i32 -16711936, i32 -16711936, i32 -16711936, i32 -16711936, i32 -16711936, i32 -16711936, i32 -16711936, i32 -16711936>
  %shr4 = lshr <8 x i32> %or, <i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8>
  %and5 = and <8 x i32> %shr4, <i32 16711935, i32 16711935, i32 16711935, i32 16711935, i32 16711935, i32 16711935, i32 16711935, i32 16711935>
  %or6 = or disjoint <8 x i32> %shl3, %and5
  %and7 = shl <8 x i32> %or6, <i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4>
  %shl8 = and <8 x i32> %and7, <i32 -252645136, i32 -252645136, i32 -252645136, i32 -252645136, i32 -252645136, i32 -252645136, i32 -252645136, i32 -252645136>
  %shr9 = lshr <8 x i32> %or6, <i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4>
  %and10 = and <8 x i32> %shr9, <i32 252645135, i32 252645135, i32 252645135, i32 252645135, i32 252645135, i32 252645135, i32 252645135, i32 252645135>
  %or11 = or disjoint <8 x i32> %shl8, %and10
  %and12 = shl <8 x i32> %or11, <i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2>
  %shl13 = and <8 x i32> %and12, <i32 -858993460, i32 -858993460, i32 -858993460, i32 -858993460, i32 -858993460, i32 -858993460, i32 -858993460, i32 -858993460>
  %shr14 = lshr <8 x i32> %or11, <i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2>
  %and15 = and <8 x i32> %shr14, <i32 858993459, i32 858993459, i32 858993459, i32 858993459, i32 858993459, i32 858993459, i32 858993459, i32 858993459>
  %or16 = or disjoint <8 x i32> %shl13, %and15
  %and17 = shl <8 x i32> %or16, <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  %shl18 = and <8 x i32> %and17, <i32 -1431655766, i32 -1431655766, i32 -1431655766, i32 -1431655766, i32 -1431655766, i32 -1431655766, i32 -1431655766, i32 -1431655766>
  %shr19 = lshr <8 x i32> %or16, <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  %and20 = and <8 x i32> %shr19, <i32 1431655765, i32 1431655765, i32 1431655765, i32 1431655765, i32 1431655765, i32 1431655765, i32 1431655765, i32 1431655765>
  %or21 = or disjoint <8 x i32> %shl18, %and20
  ret <8 x i32> %or21
}
)"};

static const char LLVMBitreverseV8i64[]{R"(
define <8 x i64> @llvm_bitreverse_v8i64(<8 x i64> %A) {
entry:
  %shl = shl <8 x i64> %A, <i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32>
  %shr = lshr <8 x i64> %A, <i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32>
  %or = or disjoint <8 x i64> %shl, %shr
  %and2 = shl <8 x i64> %or, <i64 16, i64 16, i64 16, i64 16, i64 16, i64 16, i64 16, i64 16>
  %shl3 = and <8 x i64> %and2, <i64 -281470681808896, i64 -281470681808896, i64 -281470681808896, i64 -281470681808896, i64 -281470681808896, i64 -281470681808896, i64 -281470681808896, i64 -281470681808896>
  %shr4 = lshr <8 x i64> %or, <i64 16, i64 16, i64 16, i64 16, i64 16, i64 16, i64 16, i64 16>
  %and5 = and <8 x i64> %shr4, <i64 281470681808895, i64 281470681808895, i64 281470681808895, i64 281470681808895, i64 281470681808895, i64 281470681808895, i64 281470681808895, i64 281470681808895>
  %or6 = or disjoint <8 x i64> %shl3, %and5
  %and7 = shl <8 x i64> %or6, <i64 8, i64 8, i64 8, i64 8, i64 8, i64 8, i64 8, i64 8>
  %shl8 = and <8 x i64> %and7, <i64 -71777214294589696, i64 -71777214294589696, i64 -71777214294589696, i64 -71777214294589696, i64 -71777214294589696, i64 -71777214294589696, i64 -71777214294589696, i64 -71777214294589696>
  %shr9 = lshr <8 x i64> %or6, <i64 8, i64 8, i64 8, i64 8, i64 8, i64 8, i64 8, i64 8>
  %and10 = and <8 x i64> %shr9, <i64 71777214294589695, i64 71777214294589695, i64 71777214294589695, i64 71777214294589695, i64 71777214294589695, i64 71777214294589695, i64 71777214294589695, i64 71777214294589695>
  %or11 = or disjoint <8 x i64> %shl8, %and10
  %and12 = shl <8 x i64> %or11, <i64 4, i64 4, i64 4, i64 4, i64 4, i64 4, i64 4, i64 4>
  %shl13 = and <8 x i64> %and12, <i64 -1085102592571150096, i64 -1085102592571150096, i64 -1085102592571150096, i64 -1085102592571150096, i64 -1085102592571150096, i64 -1085102592571150096, i64 -1085102592571150096, i64 -1085102592571150096>
  %shr14 = lshr <8 x i64> %or11, <i64 4, i64 4, i64 4, i64 4, i64 4, i64 4, i64 4, i64 4>
  %and15 = and <8 x i64> %shr14, <i64 1085102592571150095, i64 1085102592571150095, i64 1085102592571150095, i64 1085102592571150095, i64 1085102592571150095, i64 1085102592571150095, i64 1085102592571150095, i64 1085102592571150095>
  %or16 = or disjoint <8 x i64> %shl13, %and15
  %and17 = shl <8 x i64> %or16, <i64 2, i64 2, i64 2, i64 2, i64 2, i64 2, i64 2, i64 2>
  %shl18 = and <8 x i64> %and17, <i64 -3689348814741910324, i64 -3689348814741910324, i64 -3689348814741910324, i64 -3689348814741910324, i64 -3689348814741910324, i64 -3689348814741910324, i64 -3689348814741910324, i64 -3689348814741910324>
  %shr19 = lshr <8 x i64> %or16, <i64 2, i64 2, i64 2, i64 2, i64 2, i64 2, i64 2, i64 2>
  %and20 = and <8 x i64> %shr19, <i64 3689348814741910323, i64 3689348814741910323, i64 3689348814741910323, i64 3689348814741910323, i64 3689348814741910323, i64 3689348814741910323, i64 3689348814741910323, i64 3689348814741910323>
  %or21 = or disjoint <8 x i64> %shl18, %and20
  %and22 = shl <8 x i64> %or21, <i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1>
  %shl23 = and <8 x i64> %and22, <i64 -6148914691236517206, i64 -6148914691236517206, i64 -6148914691236517206, i64 -6148914691236517206, i64 -6148914691236517206, i64 -6148914691236517206, i64 -6148914691236517206, i64 -6148914691236517206>
  %shr24 = lshr <8 x i64> %or21, <i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1>
  %and25 = and <8 x i64> %shr24, <i64 6148914691236517205, i64 6148914691236517205, i64 6148914691236517205, i64 6148914691236517205, i64 6148914691236517205, i64 6148914691236517205, i64 6148914691236517205, i64 6148914691236517205>
  %or26 = or disjoint <8 x i64> %shl23, %and25
  ret <8 x i64> %or26
}
)"};

static const char LLVMBitreverseV16i8[]{R"(
define <16 x i8> @llvm_bitreverse_v16i8(<16 x i8> %A) {
entry:
  %shl = shl <16 x i8> %A, <i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4>
  %shr = lshr <16 x i8> %A, <i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4>
  %or = or disjoint <16 x i8> %shl, %shr
  %and2 = shl <16 x i8> %or, <i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2>
  %shl3 = and <16 x i8> %and2, <i8 -52, i8 -52, i8 -52, i8 -52, i8 -52, i8 -52, i8 -52, i8 -52, i8 -52, i8 -52, i8 -52, i8 -52, i8 -52, i8 -52, i8 -52, i8 -52>
  %shr4 = lshr <16 x i8> %or, <i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2>
  %and5 = and <16 x i8> %shr4, <i8 51, i8 51, i8 51, i8 51, i8 51, i8 51, i8 51, i8 51, i8 51, i8 51, i8 51, i8 51, i8 51, i8 51, i8 51, i8 51>
  %or6 = or disjoint <16 x i8> %shl3, %and5
  %and7 = shl <16 x i8> %or6, <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>
  %shl8 = and <16 x i8> %and7, <i8 -86, i8 -86, i8 -86, i8 -86, i8 -86, i8 -86, i8 -86, i8 -86, i8 -86, i8 -86, i8 -86, i8 -86, i8 -86, i8 -86, i8 -86, i8 -86>
  %shr9 = lshr <16 x i8> %or6, <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>
  %and10 = and <16 x i8> %shr9, <i8 85, i8 85, i8 85, i8 85, i8 85, i8 85, i8 85, i8 85, i8 85, i8 85, i8 85, i8 85, i8 85, i8 85, i8 85, i8 85>
  %or11 = or disjoint <16 x i8> %shl8, %and10
  ret <16 x i8> %or11
}
)"};

static const char LLVMBitreverseV16i16[]{R"(
define <16 x i16> @llvm_bitreverse_v16i16(<16 x i16> %A) {
entry:
  %shl = shl <16 x i16> %A, <i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8>
  %shr = lshr <16 x i16> %A, <i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8>
  %or = or disjoint <16 x i16> %shl, %shr
  %and2 = shl <16 x i16> %or, <i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4>
  %shl3 = and <16 x i16> %and2, <i16 -3856, i16 -3856, i16 -3856, i16 -3856, i16 -3856, i16 -3856, i16 -3856, i16 -3856, i16 -3856, i16 -3856, i16 -3856, i16 -3856, i16 -3856, i16 -3856, i16 -3856, i16 -3856>
  %shr4 = lshr <16 x i16> %or, <i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4>
  %and5 = and <16 x i16> %shr4, <i16 3855, i16 3855, i16 3855, i16 3855, i16 3855, i16 3855, i16 3855, i16 3855, i16 3855, i16 3855, i16 3855, i16 3855, i16 3855, i16 3855, i16 3855, i16 3855>
  %or6 = or disjoint <16 x i16> %shl3, %and5
  %and7 = shl <16 x i16> %or6, <i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2>
  %shl8 = and <16 x i16> %and7, <i16 -13108, i16 -13108, i16 -13108, i16 -13108, i16 -13108, i16 -13108, i16 -13108, i16 -13108, i16 -13108, i16 -13108, i16 -13108, i16 -13108, i16 -13108, i16 -13108, i16 -13108, i16 -13108>
  %shr9 = lshr <16 x i16> %or6, <i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2>
  %and10 = and <16 x i16> %shr9, <i16 13107, i16 13107, i16 13107, i16 13107, i16 13107, i16 13107, i16 13107, i16 13107, i16 13107, i16 13107, i16 13107, i16 13107, i16 13107, i16 13107, i16 13107, i16 13107>
  %or11 = or disjoint <16 x i16> %shl8, %and10
  %and12 = shl <16 x i16> %or11, <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
  %shl13 = and <16 x i16> %and12, <i16 -21846, i16 -21846, i16 -21846, i16 -21846, i16 -21846, i16 -21846, i16 -21846, i16 -21846, i16 -21846, i16 -21846, i16 -21846, i16 -21846, i16 -21846, i16 -21846, i16 -21846, i16 -21846>
  %shr14 = lshr <16 x i16> %or11, <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
  %and15 = and <16 x i16> %shr14, <i16 21845, i16 21845, i16 21845, i16 21845, i16 21845, i16 21845, i16 21845, i16 21845, i16 21845, i16 21845, i16 21845, i16 21845, i16 21845, i16 21845, i16 21845, i16 21845>
  %or16 = or disjoint <16 x i16> %shl13, %and15
  ret <16 x i16> %or16
}
)"};

static const char LLVMBitreverseV16i32[]{R"(
define <16 x i32> @llvm_bitreverse_v16i32(<16 x i32> %A) {
entry:
  %shl = shl <16 x i32> %A, <i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16>
  %shr = lshr <16 x i32> %A, <i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16>
  %or = or disjoint <16 x i32> %shl, %shr
  %and2 = shl <16 x i32> %or, <i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8>
  %shl3 = and <16 x i32> %and2, <i32 -16711936, i32 -16711936, i32 -16711936, i32 -16711936, i32 -16711936, i32 -16711936, i32 -16711936, i32 -16711936, i32 -16711936, i32 -16711936, i32 -16711936, i32 -16711936, i32 -16711936, i32 -16711936, i32 -16711936, i32 -16711936>
  %shr4 = lshr <16 x i32> %or, <i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8>
  %and5 = and <16 x i32> %shr4, <i32 16711935, i32 16711935, i32 16711935, i32 16711935, i32 16711935, i32 16711935, i32 16711935, i32 16711935, i32 16711935, i32 16711935, i32 16711935, i32 16711935, i32 16711935, i32 16711935, i32 16711935, i32 16711935>
  %or6 = or disjoint <16 x i32> %shl3, %and5
  %and7 = shl <16 x i32> %or6, <i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4>
  %shl8 = and <16 x i32> %and7, <i32 -252645136, i32 -252645136, i32 -252645136, i32 -252645136, i32 -252645136, i32 -252645136, i32 -252645136, i32 -252645136, i32 -252645136, i32 -252645136, i32 -252645136, i32 -252645136, i32 -252645136, i32 -252645136, i32 -252645136, i32 -252645136>
  %shr9 = lshr <16 x i32> %or6, <i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4>
  %and10 = and <16 x i32> %shr9, <i32 252645135, i32 252645135, i32 252645135, i32 252645135, i32 252645135, i32 252645135, i32 252645135, i32 252645135, i32 252645135, i32 252645135, i32 252645135, i32 252645135, i32 252645135, i32 252645135, i32 252645135, i32 252645135>
  %or11 = or disjoint <16 x i32> %shl8, %and10
  %and12 = shl <16 x i32> %or11, <i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2>
  %shl13 = and <16 x i32> %and12, <i32 -858993460, i32 -858993460, i32 -858993460, i32 -858993460, i32 -858993460, i32 -858993460, i32 -858993460, i32 -858993460, i32 -858993460, i32 -858993460, i32 -858993460, i32 -858993460, i32 -858993460, i32 -858993460, i32 -858993460, i32 -858993460>
  %shr14 = lshr <16 x i32> %or11, <i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2>
  %and15 = and <16 x i32> %shr14, <i32 858993459, i32 858993459, i32 858993459, i32 858993459, i32 858993459, i32 858993459, i32 858993459, i32 858993459, i32 858993459, i32 858993459, i32 858993459, i32 858993459, i32 858993459, i32 858993459, i32 858993459, i32 858993459>
  %or16 = or disjoint <16 x i32> %shl13, %and15
  %and17 = shl <16 x i32> %or16, <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  %shl18 = and <16 x i32> %and17, <i32 -1431655766, i32 -1431655766, i32 -1431655766, i32 -1431655766, i32 -1431655766, i32 -1431655766, i32 -1431655766, i32 -1431655766, i32 -1431655766, i32 -1431655766, i32 -1431655766, i32 -1431655766, i32 -1431655766, i32 -1431655766, i32 -1431655766, i32 -1431655766>
  %shr19 = lshr <16 x i32> %or16, <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  %and20 = and <16 x i32> %shr19, <i32 1431655765, i32 1431655765, i32 1431655765, i32 1431655765, i32 1431655765, i32 1431655765, i32 1431655765, i32 1431655765, i32 1431655765, i32 1431655765, i32 1431655765, i32 1431655765, i32 1431655765, i32 1431655765, i32 1431655765, i32 1431655765>
  %or21 = or disjoint <16 x i32> %shl18, %and20
  ret <16 x i32> %or21
}
)"};

static const char LLVMBitreverseV16i64[]{R"(
define <16 x i64> @llvm_bitreverse_v16i64(<16 x i64> %A) {
entry:
  %shl = shl <16 x i64> %A, <i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32>
  %shr = lshr <16 x i64> %A, <i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32>
  %or = or disjoint <16 x i64> %shl, %shr
  %and2 = shl <16 x i64> %or, <i64 16, i64 16, i64 16, i64 16, i64 16, i64 16, i64 16, i64 16, i64 16, i64 16, i64 16, i64 16, i64 16, i64 16, i64 16, i64 16>
  %shl3 = and <16 x i64> %and2, <i64 -281470681808896, i64 -281470681808896, i64 -281470681808896, i64 -281470681808896, i64 -281470681808896, i64 -281470681808896, i64 -281470681808896, i64 -281470681808896, i64 -281470681808896, i64 -281470681808896, i64 -281470681808896, i64 -281470681808896, i64 -281470681808896, i64 -281470681808896, i64 -281470681808896, i64 -281470681808896>
  %shr4 = lshr <16 x i64> %or, <i64 16, i64 16, i64 16, i64 16, i64 16, i64 16, i64 16, i64 16, i64 16, i64 16, i64 16, i64 16, i64 16, i64 16, i64 16, i64 16>
  %and5 = and <16 x i64> %shr4, <i64 281470681808895, i64 281470681808895, i64 281470681808895, i64 281470681808895, i64 281470681808895, i64 281470681808895, i64 281470681808895, i64 281470681808895, i64 281470681808895, i64 281470681808895, i64 281470681808895, i64 281470681808895, i64 281470681808895, i64 281470681808895, i64 281470681808895, i64 281470681808895>
  %or6 = or disjoint <16 x i64> %shl3, %and5
  %and7 = shl <16 x i64> %or6, <i64 8, i64 8, i64 8, i64 8, i64 8, i64 8, i64 8, i64 8, i64 8, i64 8, i64 8, i64 8, i64 8, i64 8, i64 8, i64 8>
  %shl8 = and <16 x i64> %and7, <i64 -71777214294589696, i64 -71777214294589696, i64 -71777214294589696, i64 -71777214294589696, i64 -71777214294589696, i64 -71777214294589696, i64 -71777214294589696, i64 -71777214294589696, i64 -71777214294589696, i64 -71777214294589696, i64 -71777214294589696, i64 -71777214294589696, i64 -71777214294589696, i64 -71777214294589696, i64 -71777214294589696, i64 -71777214294589696>
  %shr9 = lshr <16 x i64> %or6, <i64 8, i64 8, i64 8, i64 8, i64 8, i64 8, i64 8, i64 8, i64 8, i64 8, i64 8, i64 8, i64 8, i64 8, i64 8, i64 8>
  %and10 = and <16 x i64> %shr9, <i64 71777214294589695, i64 71777214294589695, i64 71777214294589695, i64 71777214294589695, i64 71777214294589695, i64 71777214294589695, i64 71777214294589695, i64 71777214294589695, i64 71777214294589695, i64 71777214294589695, i64 71777214294589695, i64 71777214294589695, i64 71777214294589695, i64 71777214294589695, i64 71777214294589695, i64 71777214294589695>
  %or11 = or disjoint <16 x i64> %shl8, %and10
  %and12 = shl <16 x i64> %or11, <i64 4, i64 4, i64 4, i64 4, i64 4, i64 4, i64 4, i64 4, i64 4, i64 4, i64 4, i64 4, i64 4, i64 4, i64 4, i64 4>
  %shl13 = and <16 x i64> %and12, <i64 -1085102592571150096, i64 -1085102592571150096, i64 -1085102592571150096, i64 -1085102592571150096, i64 -1085102592571150096, i64 -1085102592571150096, i64 -1085102592571150096, i64 -1085102592571150096, i64 -1085102592571150096, i64 -1085102592571150096, i64 -1085102592571150096, i64 -1085102592571150096, i64 -1085102592571150096, i64 -1085102592571150096, i64 -1085102592571150096, i64 -1085102592571150096>
  %shr14 = lshr <16 x i64> %or11, <i64 4, i64 4, i64 4, i64 4, i64 4, i64 4, i64 4, i64 4, i64 4, i64 4, i64 4, i64 4, i64 4, i64 4, i64 4, i64 4>
  %and15 = and <16 x i64> %shr14, <i64 1085102592571150095, i64 1085102592571150095, i64 1085102592571150095, i64 1085102592571150095, i64 1085102592571150095, i64 1085102592571150095, i64 1085102592571150095, i64 1085102592571150095, i64 1085102592571150095, i64 1085102592571150095, i64 1085102592571150095, i64 1085102592571150095, i64 1085102592571150095, i64 1085102592571150095, i64 1085102592571150095, i64 1085102592571150095>
  %or16 = or disjoint <16 x i64> %shl13, %and15
  %and17 = shl <16 x i64> %or16, <i64 2, i64 2, i64 2, i64 2, i64 2, i64 2, i64 2, i64 2, i64 2, i64 2, i64 2, i64 2, i64 2, i64 2, i64 2, i64 2>
  %shl18 = and <16 x i64> %and17, <i64 -3689348814741910324, i64 -3689348814741910324, i64 -3689348814741910324, i64 -3689348814741910324, i64 -3689348814741910324, i64 -3689348814741910324, i64 -3689348814741910324, i64 -3689348814741910324, i64 -3689348814741910324, i64 -3689348814741910324, i64 -3689348814741910324, i64 -3689348814741910324, i64 -3689348814741910324, i64 -3689348814741910324, i64 -3689348814741910324, i64 -3689348814741910324>
  %shr19 = lshr <16 x i64> %or16, <i64 2, i64 2, i64 2, i64 2, i64 2, i64 2, i64 2, i64 2, i64 2, i64 2, i64 2, i64 2, i64 2, i64 2, i64 2, i64 2>
  %and20 = and <16 x i64> %shr19, <i64 3689348814741910323, i64 3689348814741910323, i64 3689348814741910323, i64 3689348814741910323, i64 3689348814741910323, i64 3689348814741910323, i64 3689348814741910323, i64 3689348814741910323, i64 3689348814741910323, i64 3689348814741910323, i64 3689348814741910323, i64 3689348814741910323, i64 3689348814741910323, i64 3689348814741910323, i64 3689348814741910323, i64 3689348814741910323>
  %or21 = or disjoint <16 x i64> %shl18, %and20
  %and22 = shl <16 x i64> %or21, <i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1>
  %shl23 = and <16 x i64> %and22, <i64 -6148914691236517206, i64 -6148914691236517206, i64 -6148914691236517206, i64 -6148914691236517206, i64 -6148914691236517206, i64 -6148914691236517206, i64 -6148914691236517206, i64 -6148914691236517206, i64 -6148914691236517206, i64 -6148914691236517206, i64 -6148914691236517206, i64 -6148914691236517206, i64 -6148914691236517206, i64 -6148914691236517206, i64 -6148914691236517206, i64 -6148914691236517206>
  %shr24 = lshr <16 x i64> %or21, <i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1>
  %and25 = and <16 x i64> %shr24, <i64 6148914691236517205, i64 6148914691236517205, i64 6148914691236517205, i64 6148914691236517205, i64 6148914691236517205, i64 6148914691236517205, i64 6148914691236517205, i64 6148914691236517205, i64 6148914691236517205, i64 6148914691236517205, i64 6148914691236517205, i64 6148914691236517205, i64 6148914691236517205, i64 6148914691236517205, i64 6148914691236517205, i64 6148914691236517205>
  %or26 = or disjoint <16 x i64> %shl23, %and25
  ret <16 x i64> %or26
}
)"};
