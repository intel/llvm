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

; RUN: veczc -w 4 -vecz-passes=cfg-convert,verify,packetizer,define-builtins,verify -S < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

; CHECK: define spir_kernel void @__vecz_v4_test_fn(ptr %p, ptr %q, ptr %r)
define spir_kernel void @test_fn(ptr %p, ptr %q, ptr %r) {
entry:
; CHECK: [[SPLAT_PTR_INS:%.*]] = insertelement <4 x ptr> poison, ptr %p, i64 0
; CHECK: [[SPLAT_PTR:%.*]] = shufflevector <4 x ptr> [[SPLAT_PTR_INS]], <4 x ptr> poison, <4 x i32> zeroinitializer
; CHECK: [[CMP:%.*]] = icmp sgt <4 x i64> <i64 3, i64 3, i64 3, i64 3>, 
  %call = call i64 @__mux_get_global_id(i32 0)
  %cmp = icmp sgt i64 3, %call
; CHECK: [[VEC_PTR:%.*]] = getelementptr i32, ptr %p, <4 x i64>
  %wi_p_i32 = getelementptr i32, ptr %p, i64 %call
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
; CHECK: [[CALL:%.*]] = call { <4 x i32>, <4 x i1> } @__vecz_b_v4_masked_cmpxchg_align4_acquire_monotonic_1_Dv4_u3ptrDv4_jDv4_jDv4_b(
; CHECK-SAME: <4 x ptr> [[SPLAT_PTR]], <4 x i32> <i32 1, i32 1, i32 1, i32 1>,
; CHECK-SAME: <4 x i32> <i32 2, i32 2, i32 2, i32 2>, <4 x i1> [[CMP]]
  %old0 = cmpxchg ptr %p, i32 1, i32 2 acquire monotonic
  %val0 = extractvalue { i32, i1 } %old0, 0
  %success0 = extractvalue { i32, i1 } %old0, 1

  %out = getelementptr i32, ptr %q, i64 %call
  store i32 %val0, ptr %out, align 4

  %outsuccess = getelementptr i8, ptr %r, i64 %call
  %outbyte = zext i1 %success0 to i8
  store i8 %outbyte, ptr %outsuccess, align 1

  ; Test a couple of insert/extract patterns
; CHECK: [[INS:%.*]] = insertvalue { <4 x i32>, <4 x i1> } [[CALL]], <4 x i1> [[CMP]], 1
; CHECK: [[EXT:%.*]] = extractvalue { <4 x i32>, <4 x i1> } [[INS]], 1
  %testinsert = insertvalue { i32, i1 } %old0, i1 %cmp, 1
  %testextract = extractvalue { i32, i1 } %testinsert, 1

  %outbyte0 = zext i1 %testextract to i8
  store i8 %outbyte0, ptr %outsuccess, align 1

; CHECK: = call { <4 x i32>, <4 x i1> } @__vecz_b_v4_masked_cmpxchg_weak_volatile_align8_monotonic_seqcst_0_Dv4_u3ptrDv4_jDv4_jDv4_b(
  %old1 = cmpxchg weak volatile ptr %wi_p_i32, i32 1, i32 2 syncscope("singlethread") monotonic seq_cst, align 8

  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}

; CHECK: define { <4 x i32>, <4 x i1> } @__vecz_b_v4_masked_cmpxchg_align4_acquire_monotonic_1_Dv4_u3ptrDv4_jDv4_jDv4_b(<4 x ptr> [[PTRS:%0]], <4 x i32> [[CMPS:%1]], <4 x i32> [[NEWS:%2]], <4 x i1> [[MASK:%3]]) [[ATTRS:#[0-9]+]] {
; CHECK: entry:
; CHECK: br label %loopIR

; CHECK: loopIR:
; CHECK: [[IDX:%.*]] = phi i32 [ 0, %entry ], [ [[IDX_NEXT:%.*]], %if.else ]
; CHECK: [[PREV:%.*]] = phi <4 x i32> [ poison, %entry ], [ [[MERGE:%.*]], %if.else ]
; CHECK: [[PREVSUCCESS:%.*]] = phi <4 x i1> [ poison, %entry ], [ [[MERGESUCCESS:%.*]], %if.else ]
; CHECK: [[MASKELT:%.*]] = extractelement <4 x i1> [[MASK]], i32 [[IDX]]
; CHECK: [[MASKCMP:%.*]] = icmp ne i1 [[MASKELT]], false
; CHECK: br i1 [[MASKCMP]], label %if.then, label %if.else

; CHECK: if.then:
; CHECK: [[PTR:%.*]] = extractelement <4 x ptr> [[PTRS]], i32 [[IDX]]
; CHECK: [[CMP:%.*]] = extractelement <4 x i32> [[CMPS]], i32 [[IDX]]
; CHECK: [[NEW:%.*]] = extractelement <4 x i32> [[NEWS]], i32 [[IDX]]
; CHECK: [[ATOM:%.*]] = cmpxchg ptr [[PTR]], i32 [[CMP]], i32 [[NEW]] acquire monotonic, align 4
; CHECK: [[VAL:%.*]] = extractvalue { i32, i1 } [[ATOM]], 0
; CHECK: [[RET:%.*]] = insertelement <4 x i32> [[PREV]], i32 [[VAL]], i32 [[IDX]]
; CHECK: [[SUCCESS:%.*]] = extractvalue { i32, i1 } [[ATOM]], 1
; CHECK: [[RETSUCCESS:%.*]] = insertelement <4 x i1> [[PREVSUCCESS]], i1 [[SUCCESS]], i32 [[IDX]]
; CHECK: br label %if.else

; CHECK: if.else:
; CHECK: [[MERGE]] = phi <4 x i32> [ [[PREV]], %loopIR ], [ [[RET]], %if.then ]
; CHECK: [[MERGESUCCESS]] = phi <4 x i1> [ [[PREVSUCCESS]], %loopIR ], [ [[RETSUCCESS]], %if.then ]
; CHECK: [[IDX_NEXT]] = add i32 [[IDX]], 1

; CHECK: exit:
; CHECK: [[INS0:%.*]] = insertvalue { <4 x i32>, <4 x i1> } poison, <4 x i32> [[MERGE]], 0
; CHECK: [[INS1:%.*]] = insertvalue { <4 x i32>, <4 x i1> } [[INS0]], <4 x i1> [[MERGESUCCESS]], 1
; CHECK: ret { <4 x i32>, <4 x i1> } [[INS1]]

; Assume that all masked cmpxchg operations follow the logic above. Just
; check that the right cmpxchg instruction is being generated.
; CHECK: define { <4 x i32>, <4 x i1> } @__vecz_b_v4_masked_cmpxchg_weak_volatile_align8_monotonic_seqcst_0_Dv4_u3ptrDv4_jDv4_jDv4_b(<4 x ptr> [[PTRS:%0]], <4 x i32> [[CMPS:%1]], <4 x i32> [[NEWS:%2]], <4 x i1> [[MASK:%3]]) [[ATTRS]] {
; CHECK: cmpxchg weak volatile ptr {{%.*}}, i32 {{%.*}}, i32 {{%.*}} syncscope("singlethread") monotonic seq_cst, align 8

; CHECK: attributes [[ATTRS]] = { norecurse nounwind }

declare i64 @__mux_get_global_id(i32)
