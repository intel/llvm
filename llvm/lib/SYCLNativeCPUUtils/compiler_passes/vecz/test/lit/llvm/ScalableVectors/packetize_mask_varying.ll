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
; RUN: veczc -k mask_varying -vecz-scalable -vecz-simd-width=4 -S < %s | FileCheck %s

target triple = "spir64-unknown-unknown"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; A kernel which should produce a uniform masked vector load where the mask is
; a single varying splatted bit.
define spir_kernel void @mask_varying(<4 x i32>* %aptr, <4 x i32>* %zptr) {
entry:
  %idx = call i64 @__mux_get_global_id(i32 0)
  %mod_idx = urem i64 %idx, 2
  %arrayidxa = getelementptr inbounds <4 x i32>, <4 x i32>* %aptr, i64 %idx
  %ins = insertelement <4 x i1> undef, i1 true, i32 0
  %cmp = icmp slt i64 %idx, 64
  br i1 %cmp, label %if.then, label %if.end
if.then:
  %v = load <4 x i32>, <4 x i32>* %aptr
  %arrayidxz = getelementptr inbounds <4 x i32>, <4 x i32>* %zptr, i64 %idx
  store <4 x i32> %v, <4 x i32>* %arrayidxz, align 16
  br label %if.end
if.end:
  ret void
; CHECK: define spir_kernel void @__vecz_nxv4_mask_varying
; CHECK: [[idx0:%.*]] = call <vscale x 16 x i32> @llvm.{{(experimental\.)?}}stepvector.nxv16i32()
; CHECK: [[idx1:%.*]] = lshr <vscale x 16 x i32> [[idx0]], shufflevector (<vscale x 16 x i32> insertelement (<vscale x 16 x i32> {{(undef|poison)}}, i32 2, {{(i32|i64)}} 0), <vscale x 16 x i32> {{(undef|poison)}}, <vscale x 16 x i32> zeroinitializer)

; Note that since we just did a lshr 2 on the input of the extend, it doesn't
; make any difference whether it's a zext or sext, but LLVM 16 prefers zext.
; CHECK: [[idx2:%.*]] = {{s|z}}ext{{( nneg)?}} <vscale x 16 x i32> [[idx1]] to <vscale x 16 x i64>

; CHECK: [[t1:%.*]] = getelementptr i8, ptr {{.*}}, <vscale x 16 x i64> [[idx2]]
; CHECK: [[t2:%.*]] = call <vscale x 16 x i8> @llvm.masked.gather.nxv16i8.nxv16p0(<vscale x 16 x ptr> [[t1]],
; CHECK: [[splat:%.*]] = trunc <vscale x 16 x i8> [[t2]] to <vscale x 16 x i1>
; CHECK: call void @__vecz_b_masked_store16_u6nxv16ju3ptru6nxv16b(<vscale x 16 x i32> {{.*}}, ptr %arrayidxz, <vscale x 16 x i1> [[splat]])

}

declare i64 @__mux_get_global_id(i32)
declare <4 x i32> @__vecz_b_masked_load4_Dv4_jPDv4_jDv4_b(<4 x i32>*, <4 x i1>)
