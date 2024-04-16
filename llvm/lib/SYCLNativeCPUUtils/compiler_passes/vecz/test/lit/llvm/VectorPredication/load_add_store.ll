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

; REQUIRES: llvm-13+

; RUN: veczc -k load_add_store_i32 -vecz-simd-width=4 -vecz-choices=VectorPredication -S < %s | FileCheck %s --check-prefix CHECK_4F
; RUN: veczc -k load_add_store_i32 -vecz-scalable -vecz-simd-width=4 -vecz-choices=VectorPredication -S < %s | FileCheck %s --check-prefix CHECK_1S
; RUN: veczc -k load_add_store_v4i32 -vecz-simd-width=2 -vecz-choices=VectorPredication -S < %s | FileCheck %s --check-prefix CHECK_V4_2F
; RUN: veczc -k load_add_store_v4i32 -vecz-scalable -vecz-simd-width=4 -vecz-choices=VectorPredication -S < %s | FileCheck %s --check-prefix CHECK_V4_1S

target triple = "spir64-unknown-unknown"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

declare i64 @__mux_get_global_id(i32)

define spir_kernel void @load_add_store_i32(i32* %aptr, i32* %bptr, i32* %zptr) {
entry:
  %idx = call i64 @__mux_get_global_id(i32 0)
  %arrayidxa = getelementptr inbounds i32, i32* %aptr, i64 %idx
  %arrayidxb = getelementptr inbounds i32, i32* %bptr, i64 %idx
  %arrayidxz = getelementptr inbounds i32, i32* %zptr, i64 %idx
  %a = load i32, i32* %arrayidxa, align 4
  %b = load i32, i32* %arrayidxb, align 4
  %sum = add i32 %a, %b
  store i32 %sum, i32* %arrayidxz, align 4
  ret void
}

; CHECK_4F: define spir_kernel void @__vecz_v4_vp_load_add_store_i32(
; CHECK_4F: [[LID:%.*]] = call i64 @__mux_get_local_id(i32 0)
; CHECK_4F: [[LSIZE:%.*]] = call i64 @__mux_get_local_size(i32 0)
; CHECK_4F: [[WREM:%.*]] = sub nuw nsw i64 [[LSIZE]], [[LID]]
; CHECK_4F: [[T0:%.*]] = call i64 @llvm.umin.i64(i64 [[WREM]], i64 4)
; CHECK_4F: [[VL:%.*]] = trunc {{(nuw )?(nsw )?}}i64 [[T0]] to i32
; CHECK_4F: [[LHS:%.*]] = call <4 x i32> @llvm.vp.load.v4i32.p0(ptr {{%.*}}, <4 x i1> <i1 true, i1 true, i1 true, i1 true>, i32 [[VL]])
; CHECK_4F: [[RHS:%.*]] = call <4 x i32> @llvm.vp.load.v4i32.p0(ptr {{%.*}}, <4 x i1> <i1 true, i1 true, i1 true, i1 true>, i32 [[VL]])
; CHECK_4F: [[ADD:%.*]] = call <4 x i32> @llvm.vp.add.v4i32(<4 x i32> [[LHS]], <4 x i32> [[RHS]], <4 x i1> <i1 true, i1 true, i1 true, i1 true>, i32 [[VL]])
; CHECK_4F: call void @llvm.vp.store.v4i32.p0(<4 x i32> [[ADD]], ptr {{%.*}}, <4 x i1> <i1 true, i1 true, i1 true, i1 true>, i32 [[VL]])

; CHECK_1S: define spir_kernel void @__vecz_nxv4_vp_load_add_store_i32(
; CHECK_1S: [[LID:%.*]] = call i64 @__mux_get_local_id(i32 0)
; CHECK_1S: [[LSIZE:%.*]] = call i64 @__mux_get_local_size(i32 0)
; CHECK_1S: [[WREM:%.*]] = sub nuw nsw i64 [[LSIZE]], [[LID]]
; CHECK_1S: [[T0:%.*]] = call i64 @llvm.vscale.i64()
; CHECK_1S: [[T1:%.*]] = shl i64 [[T0]], 2
; CHECK_1S: [[T2:%.*]] = call i64 @llvm.umin.i64(i64 [[WREM]], i64 [[T1]])
; CHECK_1S: [[VL:%.*]] = trunc {{(nuw )?(nsw )?}}i64 [[T2]] to i32
; CHECK_1S: [[LHS:%.*]] = call <vscale x 4 x i32> @llvm.vp.load.nxv4i32.p0(ptr {{%.*}}, [[TRUEMASK:<vscale x 4 x i1> shufflevector \(<vscale x 4 x i1> insertelement \(<vscale x 4 x i1> (undef|poison), i1 true, (i32|i64) 0\), <vscale x 4 x i1> (undef|poison), <vscale x 4 x i32> zeroinitializer\)]], i32 [[VL]])
; CHECK_1S: [[RHS:%.*]] = call <vscale x 4 x i32> @llvm.vp.load.nxv4i32.p0(ptr {{%.*}}, [[TRUEMASK]], i32 [[VL]])
; CHECK_1S: [[ADD:%.*]] = call <vscale x 4 x i32> @llvm.vp.add.nxv4i32(<vscale x 4 x i32> [[LHS]], <vscale x 4 x i32> [[RHS]], [[TRUEMASK]], i32 [[VL]])
; CHECK_1S: call void @llvm.vp.store.nxv4i32.p0(<vscale x 4 x i32> [[ADD]], ptr {{%.*}}, [[TRUEMASK]], i32 [[VL]])

define spir_kernel void @load_add_store_v4i32(<4 x i32>* %aptr, <4 x i32>* %bptr, <4 x i32>* %zptr) {
entry:
  %idx = call i64 @__mux_get_global_id(i32 0)
  %arrayidxa = getelementptr inbounds <4 x i32>, <4 x i32>* %aptr, i64 %idx
  %arrayidxb = getelementptr inbounds <4 x i32>, <4 x i32>* %bptr, i64 %idx
  %arrayidxz = getelementptr inbounds <4 x i32>, <4 x i32>* %zptr, i64 %idx
  %a = load <4 x i32>, <4 x i32>* %arrayidxa, align 16
  %b = load <4 x i32>, <4 x i32>* %arrayidxb, align 16
  %sum = add <4 x i32> %a, %b
  store <4 x i32> %sum, <4 x i32>* %arrayidxz, align 16
  ret void
}

; CHECK_V4_2F: define spir_kernel void @__vecz_v2_vp_load_add_store_v4i32(
; CHECK_V4_2F: [[LID:%.*]] = call i64 @__mux_get_local_id(i32 0)
; CHECK_V4_2F: [[LSIZE:%.*]] = call i64 @__mux_get_local_size(i32 0)
; CHECK_V4_2F: [[WREM:%.*]] = sub nuw nsw i64 [[LSIZE]], [[LID]]
; CHECK_V4_2F: [[T0:%.*]] = call i64 @llvm.umin.i64(i64 [[WREM]], i64 2)
; CHECK_V4_2F: [[VL:%.*]] = trunc {{(nuw )?(nsw )?}}i64 [[T0]] to i32
; Each WI performs 4 elements, so multiply the VL by 4
; CHECK_V4_2F: [[SVL:%.*]] = shl nuw nsw i32 [[VL]], 2
; CHECK_V4_2F: [[LHS:%.*]] = call <8 x i32> @llvm.vp.load.v8i32.p0(ptr {{%.*}}, <8 x i1> <i1 true, i1 true, i1 true, i1  true, i1 true, i1 true, i1 true, i1 true>, i32 [[SVL]])
; CHECK_V4_2F: [[RHS:%.*]] = call <8 x i32> @llvm.vp.load.v8i32.p0(ptr {{%.*}}, <8 x i1> <i1 true, i1 true, i1 true, i1  true, i1 true, i1 true, i1 true, i1 true>, i32 [[SVL]])
; CHECK_V4_2F: [[ADD:%.*]] = call <8 x i32> @llvm.vp.add.v8i32(<8 x i32> [[LHS]], <8 x i32> [[RHS]], <8 x i1> <i1 true, i1 true, i1 true, i1  true, i1 true, i1 true, i1 true, i1 true>, i32 [[SVL]])
; CHECK_V4_2F: call void @llvm.vp.store.v8i32.p0(<8 x i32> [[ADD]], ptr {{%.*}}, <8 x i1> <i1 true, i1 true, i1 true, i1  true, i1 true, i1 true, i1 true, i1 true>, i32 [[SVL]])

; CHECK_V4_1S: define spir_kernel void @__vecz_nxv4_vp_load_add_store_v4i32(
; CHECK_V4_1S: [[LID:%.*]] = call i64 @__mux_get_local_id(i32 0)
; CHECK_V4_1S: [[LSIZE:%.*]] = call i64 @__mux_get_local_size(i32 0)
; CHECK_V4_1S: [[WREM:%.*]] = sub nuw nsw i64 [[LSIZE]], [[LID]]
; CHECK_V4_1S: [[T0:%.*]] = call i64 @llvm.vscale.i64()
; CHECK_V4_1S: [[T1:%.*]] = shl i64 [[T0]], 2
; CHECK_V4_1S: [[T2:%.*]] = call i64 @llvm.umin.i64(i64 [[WREM]], i64 [[T1]])
; CHECK_V4_1S: [[VL:%.*]] = trunc {{(nuw )?(nsw )?}}i64 [[T2]] to i32
; Each WI performs 4 elements, so multiply the VL by 4
; CHECK_V4_1S: [[SVL:%.*]] = shl i32 [[VL]], 2
; CHECK_V4_1S: [[LHS:%.*]] = call <vscale x 16 x i32> @llvm.vp.load.nxv16i32.p0(ptr {{%.*}}, [[TRUEMASK:<vscale x 16 x i1> shufflevector \(<vscale x 16 x i1> insertelement \(<vscale x 16 x i1> (undef|poison), i1 true, (i32|i64) 0\), <vscale x 16 x i1> (undef|poison), <vscale x 16 x i32> zeroinitializer\)]], i32 [[SVL]])
; CHECK_V4_1S: [[RHS:%.*]] = call <vscale x 16 x i32> @llvm.vp.load.nxv16i32.p0(ptr {{%.*}}, [[TRUEMASK]], i32 [[SVL]])
; CHECK_V4_1S: [[ADD:%.*]] = call <vscale x 16 x i32> @llvm.vp.add.nxv16i32(<vscale x 16 x i32> [[LHS]], <vscale x 16 x i32> [[RHS]], [[TRUEMASK]], i32 [[SVL]])
; CHECK_V4_1S: call void @llvm.vp.store.nxv16i32.p0(<vscale x 16 x i32> [[ADD]], ptr {{%.*}}, [[TRUEMASK]], i32 [[SVL]])
