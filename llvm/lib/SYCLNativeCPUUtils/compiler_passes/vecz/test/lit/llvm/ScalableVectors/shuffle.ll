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
; RUN: veczc -vecz-scalable -vecz-simd-width=4 -S < %s | FileCheck %s

target triple = "spir64-unknown-unknown"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define spir_kernel void @do_shuffle_splat(i32* %aptr, <4 x i32>* %bptr, <4 x i32>* %zptr) {
  %idx = call i64 @__mux_get_global_id(i32 0)
  %arrayidxa = getelementptr inbounds i32, i32* %aptr, i64 %idx
  %arrayidxb = getelementptr inbounds <4 x i32>, <4 x i32>* %bptr, i64 %idx
  %a = load i32, i32* %arrayidxa, align 4
  %b = load <4 x i32>, <4 x i32>* %arrayidxb, align 16
  %insert = insertelement <4 x i32> undef, i32 %a, i32 0
  %splat = shufflevector <4 x i32> %insert, <4 x i32> undef, <4 x i32> zeroinitializer
  %arrayidxz = getelementptr inbounds <4 x i32>, <4 x i32>* %zptr, i64 %idx
  store <4 x i32> %splat, <4 x i32>* %arrayidxz
  ret void
; CHECK: define spir_kernel void @__vecz_nxv4_do_shuffle_splat
; CHECK: [[idx0:%.*]] = call <vscale x 16 x i32> @llvm.{{(experimental\.)?}}stepvector.nxv16i32()
; CHECK: [[idx1:%.*]] = lshr <vscale x 16 x i32> [[idx0]], shufflevector (<vscale x 16 x i32> insertelement (<vscale x 16 x i32> {{(undef|poison)}}, i32 2, {{(i32|i64)}} 0), <vscale x 16 x i32> {{(undef|poison)}}, <vscale x 16 x i32> zeroinitializer)

; Note that since we just did a lshr 2 on the input of the extend, it doesn't
; make any difference whether it's a zext or sext, but LLVM 16 prefers zext.
; CHECK: [[idx2:%.*]] = {{s|z}}ext{{( nneg)?}} <vscale x 16 x i32> [[idx1]] to <vscale x 16 x i64>

; CHECK: [[alloc:%.*]] = getelementptr i32, ptr %{{.*}}, <vscale x 16 x i64> [[idx2]]
; CHECK: [[splat:%.*]] = call <vscale x 16 x i32> @llvm.masked.gather.nxv16i32.nxv16p0(<vscale x 16 x ptr> [[alloc]],
; CHECK: store <vscale x 16 x i32> [[splat]], ptr
}

define spir_kernel void @do_shuffle_splat_uniform(i32 %a, <4 x i32>* %bptr, <4 x i32>* %zptr) {
  %idx = call i64 @__mux_get_global_id(i32 0)
  %arrayidxb = getelementptr inbounds <4 x i32>, <4 x i32>* %bptr, i64 %idx
  %b = load <4 x i32>, <4 x i32>* %arrayidxb, align 16
  %insert = insertelement <4 x i32> undef, i32 %a, i32 0
  %splat = shufflevector <4 x i32> %insert, <4 x i32> undef, <4 x i32> zeroinitializer
  %arrayidxz = getelementptr inbounds <4 x i32>, <4 x i32>* %zptr, i64 %idx
  store <4 x i32> %splat, <4 x i32>* %arrayidxz
  ret void
; CHECK: define spir_kernel void @__vecz_nxv4_do_shuffle_splat_uniform
; CHECK: [[ins:%.*]] = insertelement <vscale x 16 x i32> poison, i32 %a, {{(i32|i64)}} 0
; CHECK: [[splat:%.*]] = shufflevector <vscale x 16 x i32> [[ins]], <vscale x 16 x i32> {{(undef|poison)}}, <vscale x 16 x i32> zeroinitializer
; CHECK: store <vscale x 16 x i32> [[splat]], ptr
}

declare i64 @__mux_get_global_id(i32)
