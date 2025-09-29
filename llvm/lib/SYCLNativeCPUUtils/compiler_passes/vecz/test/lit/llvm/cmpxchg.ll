; Copyright (C) Codeplay Software Limited
;
; Licensed under the Apache License, Version 2.0 (the "License") with LLVM
; Exceptions; you may not use this file except in compliance with the License.
; You may obtain a copy of the License at
;
;     https://github.com/uxlfoundation/oneapi-construction-kit/blob/main/LICENSE.txt
;
; Unless required by applicable law or agreed to in writing, software
; distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
; WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
; License for the specific language governing permissions and limitations
; under the License.
;
; SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

; RUN: veczc -w 4 -vecz-passes=packetizer,verify -S < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

; CHECK: define spir_kernel void @__vecz_v4_test_fn(ptr %p, ptr %q, ptr %r)
define spir_kernel void @test_fn(ptr %p, ptr %q, ptr %r) {
entry:
; CHECK: [[SPLAT_PTR_INS:%.*]] = insertelement <4 x ptr> poison, ptr %p, i64 0
; CHECK: [[SPLAT_PTR:%.*]] = shufflevector <4 x ptr> [[SPLAT_PTR_INS]], <4 x ptr> poison, <4 x i32> zeroinitializer
  %call = call i64 @__mux_get_global_id(i32 0)

; Test that this cmpxchg is packetized by generating a call to an all-true masked version.
; CHECK: [[A0:%.*]] = call { <4 x i32>, <4 x i1> } @__vecz_b_v4_masked_cmpxchg_align4_acquire_monotonic_1_Dv4_u3ptrDv4_jDv4_jDv4_b(
; CHECK-SAME: <4 x ptr> [[SPLAT_PTR]], <4 x i32> {{<(i32 1(, )?)+>|splat \(i32 1\)}},
; CHECK-SAME: <4 x i32> {{<(i32 2(, )?)+>|splat \(i32 2\)}},
; CHECK-SAME: <4 x i1> {{<(i1 true(, )?)+>|splat \(i1 true\)}}
  %old0 = cmpxchg ptr %p, i32 1, i32 2 acquire monotonic
; CHECK: [[EXT0:%.*]] = extractvalue { <4 x i32>, <4 x i1> } [[A0]], 0
  %val0 = extractvalue { i32, i1 } %old0, 0
; CHECK: [[EXT1:%.*]] = extractvalue { <4 x i32>, <4 x i1> } [[A0]], 1
  %success0 = extractvalue { i32, i1 } %old0, 1

  %out = getelementptr i32, ptr %q, i64 %call
; Stored as a vector
; CHECK: store <4 x i32> [[EXT0]], ptr
  store i32 %val0, ptr %out, align 4

; CHECK: [[PTR:%.*]] = getelementptr i8, ptr %r, i64 %call
  %outsuccess = getelementptr i8, ptr %r, i64 %call
; CHECK: [[ZEXT0:%.*]] = zext <4 x i1> [[EXT1]] to <4 x i8>
  %outbyte = zext i1 %success0 to i8
; Stored as a vector
; CHECK: store <4 x i8> [[ZEXT0]], ptr [[PTR]], align 1
  store i8 %outbyte, ptr %outsuccess, align 1

  ; Test a couple of insert/extract patterns

  ; Test inserting a uniform value into a varying literal struct
; CHECK: [[INS0:%.*]] = insertvalue { <4 x i32>, <4 x i1> } [[A0]], <4 x i1> zeroinitializer, 1
; CHECK: [[EXT2:%.*]] = extractvalue { <4 x i32>, <4 x i1> } [[INS0]], 1
; CHECK: [[ZEXT1:%.*]] = zext <4 x i1> [[EXT2]] to <4 x i8>
; CHECK: store <4 x i8> [[ZEXT1]], ptr [[PTR]], align 1
  %testinsertconst = insertvalue { i32, i1 } %old0, i1 false, 1
  %testextract0 = extractvalue { i32, i1 } %testinsertconst, 1
  %outbyte0 = zext i1 %testextract0 to i8
  store i8 %outbyte0, ptr %outsuccess, align 1

  ; Test inserting a varying value into a varying literal struct
; CHECK: [[LD:%.*]] = load <4 x i8>, ptr
; CHECK: [[VBOOL:%.*]] = trunc <4 x i8> [[LD]] to <4 x i1>
; CHECK: [[INS1:%.*]] = insertvalue { <4 x i32>, <4 x i1> } [[A0]], <4 x i1> [[VBOOL]], 1
; CHECK: [[EXT3:%.*]] = extractvalue { <4 x i32>, <4 x i1> } [[INS1]], 1
; CHECK: [[ZEXT2:%.*]] = zext <4 x i1> [[EXT3]] to <4 x i8>
; CHECK: store <4 x i8> [[ZEXT2]], ptr [[PTR]], align 1
  %byte1 = load i8, ptr %outsuccess, align 1
  %bool1 = trunc i8 %byte1 to i1
  %testinsertvarying0 = insertvalue { i32, i1 } %old0, i1 %bool1, 1
  %testextract1 = extractvalue { i32, i1 } %testinsertvarying0, 1
  %outbyte1 = zext i1 %testextract1 to i8
  store i8 %outbyte1, ptr %outsuccess, align 1

  ; Test inserting a varying value into a uniform literal struct
; CHECK: [[INS2:%.*]] = insertvalue { <4 x i32>, <4 x i1> } poison, <4 x i1> [[VBOOL]], 1
; CHECK: [[EXT4:%.*]] = extractvalue { <4 x i32>, <4 x i1> } [[INS2]], 1
; CHECK: [[ZEXT3:%.*]] = zext <4 x i1> [[EXT4]] to <4 x i8>
; CHECK: store <4 x i8> [[ZEXT3]], ptr [[PTR]], align 1
  %testinsertvarying1 = insertvalue { i32, i1 } poison, i1 %bool1, 1
  %testextract2 = extractvalue { i32, i1 } %testinsertvarying1, 1
  %outbyte2 = zext i1 %testextract2 to i8
  store i8 %outbyte2, ptr %outsuccess, align 1

  ret void
}

declare i64 @__mux_get_global_id(i32)
