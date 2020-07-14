//===- LLVMSaddWithOverflow.h - implementation of llvm.sadd.with.overflow -===//
//
//                     The LLVM/SPIRV Translator
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
// Copyright (c) 2020 Intel Corporation. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimers.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimers in the documentation
// and/or other materials provided with the distribution.
// Neither the names of Intel Corporation, nor the names of its
// contributors may be used to endorse or promote products derived from this
// Software without specific prior written permission.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
// THE SOFTWARE.
//
//===----------------------------------------------------------------------===//
//
// This file implements lowering of llvm.sadd.with.overflow.* into basic LLVM
// operations.
//
//===----------------------------------------------------------------------===//

// The IR below is slightly manually modified IR which was produced by Clang
// from the C++ code below. The modifications include:
// - adapting the return value, i.e. replacing `store` instructions for the c
//   and o arguments with `insertvalue` instructions.
// - changing type of the `phi` instruction in the last basic block from `i8`
//   to `i1`. That also requires change of the argument of the `phi` instruction
//   and allowed to remove an unnecessary `sext` instruction.
//
// #include <stdlib.h>
// #include <stdint.h>
//
// const unsigned short i16_abs_pos_max = 0x7FFF; // 32767;
// const unsigned short i16_abs_neg_max = 0x8000; // 32768;
//
// void llvm_sadd_with_overflow_i16(int16_t a, int16_t b, int16_t& c, bool& o) {
//     bool overflow = false;
//     bool both_pos = (a>=0 && b>=0);
//     bool both_neg = (a<0 && b<0);
//     if (both_pos || both_neg) {
//         // 32-bit integers are always supported in SPIR-V
//         uint32_t x = (uint32_t)abs(a) + (uint32_t)abs(b);
//         if (both_pos && x > i16_abs_pos_max ||
//             both_neg && x > i16_abs_neg_max) {
//             overflow = true;
//         }
//     }
//     c = a + b;
//     o = overflow;
// }
//
// const uint32_t i32_abs_pos_max = 0x7FFFFFFF; // 2147483647
// const uint32_t i32_abs_neg_max = 0x80000000; // 2147483648
// const int32_t i32_min = 0x80000000; // -2147483648
//
// void llvm_sadd_with_overflow_i32(int32_t a, int32_t b, int32_t& c, bool& o) {
//     bool overflow = false;
//     bool both_pos = (a>=0 && b>=0);
//     bool both_neg = (a<0 && b<0);
//     // if a or b is the most negative number we can't get its absolute value,
//     // because it is out of range.
//     if (both_neg && (a == i32_min || b == i32_min))
//         overflow = true;
//     else if (both_pos || both_neg) {
//         uint32_t x = (uint32_t)abs(a) + (uint32_t)abs(b);
//         if (both_pos && x > i32_abs_pos_max ||
//             both_neg && x > 2147483648U) {
//             overflow = true;
//         }
//     }
//     c = a + b;
//     o = overflow;
// }
//
// const uint64_t i64_abs_pos_max = 0x7fffffffffffffff; // 9223372036854775807
// const uint64_t i64_abs_neg_max = 0x8000000000000000; // 9223372036854775808
// const int64_t i64_min = 0x8000000000000000; // -9223372036854775808
//
// void llvm_sadd_with_overflow_i64(int64_t a, int64_t b, int64_t& c, bool& o) {
//     bool overflow = false;
//     bool both_pos = (a>=0 && b>=0);
//     bool both_neg = (a<0 && b<0);
//     // if a or b is the most negative number we can't get its absolute value,
//     // because it is out of range.
//     if (both_neg && (a == i64_min || b == i64_min))
//         overflow = true;
//     else if (both_pos || both_neg) {
//         uint64_t x = (uint64_t)abs(a) + (uint64_t)abs(b);
//         if (both_pos && x > i64_abs_pos_max ||
//             both_neg && x > i64_abs_neg_max) {
//             overflow = true;
//         }
//     }
//     c = a + b;
//     o = overflow;
// }
//
// const unsigned int abs_pos_max = 2147483647;
// const unsigned int abs_neg_max = 2147483648;
//
// void llvm_sadd_with_overflow_i32(int a, int b, int& c, bool& o) {
//     bool overflow = false;
//     bool both_pos = (a>=0 && b>=0);
//     bool both_neg = (a<0 && b<0);
//     if (both_pos || both_neg) {
//         unsigned int x = (unsigned int)abs(a) + (unsigned int)abs(b);
//         if (both_pos && x > abs_pos_max ||
//             both_neg && x > abs_neg_max) {
//             overflow = true;
//         }
//     }
//     c = a + b;
//     o = overflow;
// }
// Clang options: -emit-llvm -O2 -g0 -fno-discard-value-names

static const char LLVMSaddWithOverflow[]{R"(
define spir_func { i16, i1 } @llvm_sadd_with_overflow_i16(i16 %a, i16 %b) {
entry:
  %conv = sext i16 %a to i32
  %conv1 = sext i16 %b to i32
  %0 = or i16 %b, %a
  %1 = icmp sgt i16 %0, -1
  %2 = and i16 %b, %a
  %3 = icmp slt i16 %2, 0
  %brmerge = or i1 %1, %3
  br i1 %brmerge, label %if.then, label %if.end21

if.then:                                          ; preds = %entry
  %4 = icmp slt i32 %conv, 0
  %neg = sub nsw i32 0, %conv
  %5 = select i1 %4, i32 %neg, i32 %conv
  %6 = icmp slt i32 %conv1, 0
  %neg39 = sub nsw i32 0, %conv1
  %7 = select i1 %6, i32 %neg39, i32 %conv1
  %add = add nuw nsw i32 %7, %5
  %cmp15 = icmp ugt i32 %add, 32767
  %or.cond = and i1 %1, %cmp15
  %cmp19 = icmp ugt i32 %add, 32768
  %or.cond28 = and i1 %3, %cmp19
  %or.cond40 = or i1 %or.cond, %or.cond28
  br label %if.end21

if.end21:                                         ; preds = %if.then, %entry
  %overflow = phi i1 [ 0, %entry ], [ %or.cond40, %if.then ]
  %add24 = add i16 %b, %a
  %agg = insertvalue {i16, i1} undef, i16 %add24, 0
  %res = insertvalue {i16, i1} %agg, i1 %overflow, 1
  ret {i16, i1} %res
}

define spir_func { i32, i1 } @llvm_sadd_with_overflow_i32(i32 %a, i32 %b) {
entry:
  %0 = or i32 %b, %a
  %1 = icmp sgt i32 %0, -1
  %2 = and i32 %b, %a
  %3 = icmp slt i32 %2, 0
  br i1 %3, label %land.lhs.true, label %if.else

land.lhs.true:                                    ; preds = %entry
  %cmp7 = icmp eq i32 %a, -2147483648
  %cmp8 = icmp eq i32 %b, -2147483648
  %or.cond = or i1 %cmp7, %cmp8
  br i1 %or.cond, label %if.end23, label %if.then12

if.else:                                          ; preds = %entry
  br i1 %1, label %if.then12, label %if.end23

if.then12:                                        ; preds = %land.lhs.true, %if.else
  %4 = icmp slt i32 %a, 0
  %neg = sub nsw i32 0, %a
  %5 = select i1 %4, i32 %neg, i32 %a
  %6 = icmp slt i32 %b, 0
  %neg42 = sub nsw i32 0, %b
  %7 = select i1 %6, i32 %neg42, i32 %b
  %add = add nuw i32 %7, %5
  %cmp16 = icmp slt i32 %add, 0
  %or.cond27 = and i1 %1, %cmp16
  %cmp20 = icmp ugt i32 %add, -2147483648
  %or.cond28 = and i1 %3, %cmp20
  %or.cond43 = or i1 %or.cond27, %or.cond28
  br label %if.end23

if.end23:                                         ; preds = %if.then12, %if.else, %land.lhs.true
  %overflow = phi i1 [ 1, %land.lhs.true ], [ 0, %if.else ], [ %or.cond43, %if.then12 ]
  %add24 = add nsw i32 %b, %a
  %agg = insertvalue {i32, i1} undef, i32 %add24, 0
  %res = insertvalue {i32, i1} %agg, i1 %overflow, 1
  ret {i32, i1} %res
}

define spir_func { i64, i1 } @llvm_sadd_with_overflow_i64(i64 %a, i64 %b) {
entry:
  %0 = or i64 %b, %a
  %1 = icmp sgt i64 %0, -1
  %2 = and i64 %b, %a
  %3 = icmp slt i64 %2, 0
  br i1 %3, label %land.lhs.true, label %if.else

land.lhs.true:                                    ; preds = %entry
  %cmp7 = icmp eq i64 %a, -9223372036854775808
  %cmp8 = icmp eq i64 %b, -9223372036854775808
  %or.cond = or i1 %cmp7, %cmp8
  br i1 %or.cond, label %if.end23, label %if.then12

if.else:                                          ; preds = %entry
  br i1 %1, label %if.then12, label %if.end23

if.then12:                                        ; preds = %land.lhs.true, %if.else
  %neg.i = sub nsw i64 0, %a
  %abscond.i = icmp slt i64 %a, 0
  %abs.i = select i1 %abscond.i, i64 %neg.i, i64 %a
  %neg.i43 = sub nsw i64 0, %b
  %abscond.i44 = icmp slt i64 %b, 0
  %abs.i45 = select i1 %abscond.i44, i64 %neg.i43, i64 %b
  %add = add nuw i64 %abs.i45, %abs.i
  %cmp16 = icmp slt i64 %add, 0
  %or.cond27 = and i1 %1, %cmp16
  %cmp20 = icmp ugt i64 %add, -9223372036854775808
  %or.cond28 = and i1 %3, %cmp20
  %or.cond42 = or i1 %or.cond27, %or.cond28
  br label %if.end23

if.end23:                                         ; preds = %if.then12, %if.else, %land.lhs.true
  %overflow = phi i1 [ 1, %land.lhs.true ], [ 0, %if.else ], [ %or.cond42, %if.then12 ]
  %add24 = add nsw i64 %b, %a
  %agg = insertvalue {i64, i1} undef, i64 %add24, 0
  %res = insertvalue {i64, i1} %agg, i1 %overflow, 1
  ret {i64, i1} %res
}
)"};
