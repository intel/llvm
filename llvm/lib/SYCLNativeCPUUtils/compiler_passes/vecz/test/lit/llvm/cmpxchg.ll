; Copyright (C) Codeplay Software Limited
;
; Licensed under the Apache License, Version 2.0 (the "License") with LLVM
; Exceptions; you may not use this file except in compliance with the License.
; You may obtain a copy of the License at
;
;     https://github.com/codeplaysoftware/oneapi-construction-kit/blob/main/LICENSE.txt
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
  %call = call i64 @__mux_get_global_id(i32 0)

; Test that this cmpxchg is scalarized. Not ideal, but hey.
; CHECK: [[A0:%.*]] = cmpxchg ptr %p, i32 1, i32 2 acquire monotonic, align 4
; CHECK: [[A1:%.*]] = cmpxchg ptr %p, i32 1, i32 2 acquire monotonic, align 4
; CHECK: [[A2:%.*]] = cmpxchg ptr %p, i32 1, i32 2 acquire monotonic, align 4
; CHECK: [[A3:%.*]] = cmpxchg ptr %p, i32 1, i32 2 acquire monotonic, align 4

; Then we insert the values into a strange struct
; CHECK: [[INS0:%.*]] = insertvalue [4 x { i32, i1 }] undef, { i32, i1 } [[A0]], 0
; CHECK: [[INS1:%.*]] = insertvalue [4 x { i32, i1 }] [[INS0]], { i32, i1 } [[A1]], 1
; CHECK: [[INS2:%.*]] = insertvalue [4 x { i32, i1 }] [[INS1]], { i32, i1 } [[A2]], 2
; CHECK: [[INS3:%.*]] = insertvalue [4 x { i32, i1 }] [[INS2]], { i32, i1 } [[A3]], 3
  %old0 = cmpxchg ptr %p, i32 1, i32 2 acquire monotonic

; To extract from this result, we extract each element individually then insert
; each into a vector.
; CHECK: [[ELT0_0_0:%.*]] = extractvalue [4 x { i32, i1 }] [[INS3]], 0, 0
; CHECK: [[ELT0_0_1:%.*]] = extractvalue [4 x { i32, i1 }] [[INS3]], 1, 0
; CHECK: [[ELT0_0_2:%.*]] = extractvalue [4 x { i32, i1 }] [[INS3]], 2, 0
; CHECK: [[ELT0_0_3:%.*]] = extractvalue [4 x { i32, i1 }] [[INS3]], 3, 0
; CHECK: [[INS0V0_0:%.*]] = insertelement <4 x i32> undef, i32 [[ELT0_0_0]], i32 0
; CHECK: [[INS0V0_1:%.*]] = insertelement <4 x i32> [[INS0V0_0]], i32 [[ELT0_0_1]], i32 1
; CHECK: [[INS0V0_2:%.*]] = insertelement <4 x i32> [[INS0V0_1]], i32 [[ELT0_0_2]], i32 2
; CHECK: [[INS0V0_3:%.*]] = insertelement <4 x i32> [[INS0V0_2]], i32 [[ELT0_0_3]], i32 3
  %val0 = extractvalue { i32, i1 } %old0, 0
; Same again here
; CHECK: [[ELT1_0_0:%.*]] = extractvalue [4 x { i32, i1 }] [[INS3]], 0, 1
; CHECK: [[ELT1_0_1:%.*]] = extractvalue [4 x { i32, i1 }] [[INS3]], 1, 1
; CHECK: [[ELT1_0_2:%.*]] = extractvalue [4 x { i32, i1 }] [[INS3]], 2, 1
; CHECK: [[ELT1_0_3:%.*]] = extractvalue [4 x { i32, i1 }] [[INS3]], 3, 1
; CHECK: [[INS1V0_0:%.*]] = insertelement <4 x i1> undef, i1 [[ELT1_0_0]], i32 0
; CHECK: [[INS1V0_1:%.*]] = insertelement <4 x i1> [[INS1V0_0]], i1 [[ELT1_0_1]], i32 1
; CHECK: [[INS1V0_2:%.*]] = insertelement <4 x i1> [[INS1V0_1]], i1 [[ELT1_0_2]], i32 2
; CHECK: [[INS1V0_3:%.*]] = insertelement <4 x i1> [[INS1V0_2]], i1 [[ELT1_0_3]], i32 3
  %success0 = extractvalue { i32, i1 } %old0, 1

  %out = getelementptr i32, ptr %q, i64 %call
; Stored as a vector
; CHECK: store <4 x i32> [[INS0V0_3]], ptr
  store i32 %val0, ptr %out, align 4

; CHECK: [[PTR:%.*]] = getelementptr i8, ptr %r, i64 %call
  %outsuccess = getelementptr i8, ptr %r, i64 %call
; CHECK: [[ZEXT0:%.*]] = zext <4 x i1> [[INS1V0_3]] to <4 x i8>
  %outbyte = zext i1 %success0 to i8
; Stored as a vector
; CHECK: store <4 x i8> [[ZEXT0]], ptr [[PTR]], align 1
  store i8 %outbyte, ptr %outsuccess, align 1

  ; Test a couple of insert/extract patterns

; Test inserting a uniform value into a varying literal struct
; This is very inefficient
; CHECK: [[INSS0_0:%.*]] = insertvalue { i32, i1 } [[A0]], i1 false, 1
; CHECK: [[INSS0_1:%.*]] = insertvalue { i32, i1 } [[A1]], i1 false, 1
; CHECK: [[INSS0_2:%.*]] = insertvalue { i32, i1 } [[A2]], i1 false, 1
; CHECK: [[INSS0_3:%.*]] = insertvalue { i32, i1 } [[A3]], i1 false, 1
; CHECK: [[INSS1_0:%.*]] = insertvalue [4 x { i32, i1 }] undef, { i32, i1 } [[INSS0_0]], 0
; CHECK: [[INSS1_1:%.*]] = insertvalue [4 x { i32, i1 }] [[INSS1_0]], { i32, i1 } [[INSS0_1]], 1
; CHECK: [[INSS1_2:%.*]] = insertvalue [4 x { i32, i1 }] [[INSS1_1]], { i32, i1 } [[INSS0_2]], 2
; CHECK: [[INSS1_3:%.*]] = insertvalue [4 x { i32, i1 }] [[INSS1_2]], { i32, i1 } [[INSS0_3]], 3
; CHECK: [[EXTS1_0:%.*]] = extractvalue [4 x { i32, i1 }] [[INSS1_3]], 0, 1
; CHECK: [[EXTS1_1:%.*]] = extractvalue [4 x { i32, i1 }] [[INSS1_3]], 1, 1
; CHECK: [[EXTS1_2:%.*]] = extractvalue [4 x { i32, i1 }] [[INSS1_3]], 2, 1
; CHECK: [[EXTS1_3:%.*]] = extractvalue [4 x { i32, i1 }] [[INSS1_3]], 3, 1
; CHECK: [[INS1V1_0:%.*]] = insertelement <4 x i1> undef, i1 [[EXTS1_0]], i32 0
; CHECK: [[INS1V1_1:%.*]] = insertelement <4 x i1> [[INS1V1_0]], i1 [[EXTS1_1]], i32 1
; CHECK: [[INS1V1_2:%.*]] = insertelement <4 x i1> [[INS1V1_1]], i1 [[EXTS1_2]], i32 2
; CHECK: [[INS1V1_3:%.*]] = insertelement <4 x i1> [[INS1V1_2]], i1 [[EXTS1_3]], i32 3
; CHECK: [[ZEXT1:%.*]] = zext <4 x i1> [[INS1V1_3]] to <4 x i8>
; CHECK: store <4 x i8> [[ZEXT1]], ptr [[PTR]], align 1
  %testinsertconst = insertvalue { i32, i1 } %old0, i1 false, 1
  %testextract0 = extractvalue { i32, i1 } %testinsertconst, 1
  %outbyte0 = zext i1 %testextract0 to i8
  store i8 %outbyte0, ptr %outsuccess, align 1

  ; Test inserting a varying value into a varying literal struct
; CHECK: [[V4I8_LD:%.*]] = load <4 x i8>, ptr %outsuccess, align 1
; CHECK: [[TRUNC:%.*]] = trunc <4 x i8> [[V4I8_LD]] to <4 x i1>
; CHECK: [[EXTV0_0:%.*]] = extractelement <4 x i1> [[TRUNC]], i32 0
; CHECK: [[EXTV0_1:%.*]] = extractelement <4 x i1> [[TRUNC]], i32 1
; CHECK: [[EXTV0_2:%.*]] = extractelement <4 x i1> [[TRUNC]], i32 2
; CHECK: [[EXTV0_3:%.*]] = extractelement <4 x i1> [[TRUNC]], i32 3
; CHECK: [[INSS2_0:%.*]] = insertvalue { i32, i1 } [[A0]], i1 [[EXTV0_0]], 1
; CHECK: [[INSS2_1:%.*]] = insertvalue { i32, i1 } [[A1]], i1 [[EXTV0_1]], 1
; CHECK: [[INSS2_2:%.*]] = insertvalue { i32, i1 } [[A2]], i1 [[EXTV0_2]], 1
; CHECK: [[INSS2_3:%.*]] = insertvalue { i32, i1 } [[A3]], i1 [[EXTV0_3]], 1
; CHECK: [[INSS3_0:%.*]] = insertvalue [4 x { i32, i1 }] undef, { i32, i1 } [[INSS2_0]], 0
; CHECK: [[INSS3_1:%.*]] = insertvalue [4 x { i32, i1 }] [[INSS3_0]], { i32, i1 } [[INSS2_1]], 1
; CHECK: [[INSS3_2:%.*]] = insertvalue [4 x { i32, i1 }] [[INSS3_1]], { i32, i1 } [[INSS2_2]], 2
; CHECK: [[INSS3_3:%.*]] = insertvalue [4 x { i32, i1 }] [[INSS3_2]], { i32, i1 } [[INSS2_3]], 3
; CHECK: [[EXTS3_0:%.*]] = extractvalue [4 x { i32, i1 }] [[INSS3_3]], 0, 1
; CHECK: [[EXTS3_1:%.*]] = extractvalue [4 x { i32, i1 }] [[INSS3_3]], 1, 1
; CHECK: [[EXTS3_2:%.*]] = extractvalue [4 x { i32, i1 }] [[INSS3_3]], 2, 1
; CHECK: [[EXTS3_3:%.*]] = extractvalue [4 x { i32, i1 }] [[INSS3_3]], 3, 1
; CHECK: [[INS1V2_0:%.*]] = insertelement <4 x i1> undef, i1 [[EXTS3_0]], i32 0
; CHECK: [[INS1V2_1:%.*]] = insertelement <4 x i1> [[INS1V2_0]], i1 [[EXTS3_1]], i32 1
; CHECK: [[INS1V2_2:%.*]] = insertelement <4 x i1> [[INS1V2_1]], i1 [[EXTS3_2]], i32 2
; CHECK: [[INS1V2_3:%.*]] = insertelement <4 x i1> [[INS1V2_2]], i1 [[EXTS3_3]], i32 3
; CHECK: [[ZEXT2:%.*]] = zext <4 x i1> [[INS1V2_3]] to <4 x i8>
; CHECK: store <4 x i8> [[ZEXT2]], ptr [[PTR]], align 1
  %byte1 = load i8, ptr %outsuccess, align 1
  %bool1 = trunc i8 %byte1 to i1
  %testinsertvarying0 = insertvalue { i32, i1 } %old0, i1 %bool1, 1
  %testextract1 = extractvalue { i32, i1 } %testinsertvarying0, 1
  %outbyte1 = zext i1 %testextract1 to i8
  store i8 %outbyte1, ptr %outsuccess, align 1

  ; Test inserting a varying value into a uniform literal struct
; CHECK: [[INSS4_0:%.*]] = insertvalue { i32, i1 } poison, i1 [[EXTV0_0]], 1
; CHECK: [[INSS4_1:%.*]] = insertvalue { i32, i1 } poison, i1 [[EXTV0_1]], 1
; CHECK: [[INSS4_2:%.*]] = insertvalue { i32, i1 } poison, i1 [[EXTV0_2]], 1
; CHECK: [[INSS4_3:%.*]] = insertvalue { i32, i1 } poison, i1 [[EXTV0_3]], 1
; CHECK: [[INSS5_0:%.*]] = insertvalue [4 x { i32, i1 }] undef, { i32, i1 } [[INSS4_0]], 0
; CHECK: [[INSS5_1:%.*]] = insertvalue [4 x { i32, i1 }] [[INSS5_0]], { i32, i1 } [[INSS4_1]], 1
; CHECK: [[INSS5_2:%.*]] = insertvalue [4 x { i32, i1 }] [[INSS5_1]], { i32, i1 } [[INSS4_2]], 2
; CHECK: [[INSS5_3:%.*]] = insertvalue [4 x { i32, i1 }] [[INSS5_2]], { i32, i1 } [[INSS4_3]], 3
; CHECK: [[EXTS5_0:%.*]] = extractvalue [4 x { i32, i1 }] [[INSS5_3]], 0, 1
; CHECK: [[EXTS5_1:%.*]] = extractvalue [4 x { i32, i1 }] [[INSS5_3]], 1, 1
; CHECK: [[EXTS5_2:%.*]] = extractvalue [4 x { i32, i1 }] [[INSS5_3]], 2, 1
; CHECK: [[EXTS5_3:%.*]] = extractvalue [4 x { i32, i1 }] [[INSS5_3]], 3, 1
; CHECK: [[INS2V3_0:%.*]] = insertelement <4 x i1> undef, i1 [[EXTS5_0]], i32 0
; CHECK: [[INS2V3_1:%.*]] = insertelement <4 x i1> [[INS2V3_0]], i1 [[EXTS5_1]], i32 1
; CHECK: [[INS2V3_2:%.*]] = insertelement <4 x i1> [[INS2V3_1]], i1 [[EXTS5_2]], i32 2
; CHECK: [[INS2V3_3:%.*]] = insertelement <4 x i1> [[INS2V3_2]], i1 [[EXTS5_3]], i32 3
; CHECK: [[ZEXT3:%.*]] = zext <4 x i1> [[INS2V3_3]] to <4 x i8>
; CHECK: store <4 x i8> [[ZEXT3]], ptr [[PTR]], align 1
  %testinsertvarying1 = insertvalue { i32, i1 } poison, i1 %bool1, 1
  %testextract2 = extractvalue { i32, i1 } %testinsertvarying1, 1
  %outbyte2 = zext i1 %testextract2 to i8
  store i8 %outbyte2, ptr %outsuccess, align 1

  ret void
}

declare i64 @__mux_get_global_id(i32)
