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
; RUN: veczc -k store_ult -vecz-scalable -vecz-simd-width=4 -S < %s | FileCheck %s

; Check that we can scalably-vectorize a call to get_global_id by using the
; stepvector intrinsic

target triple = "spir64-unknown-unknown"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define spir_kernel void @store_ult(i32* %out, i64* %N) {
entry:
  %call = tail call i64 @__mux_get_global_id(i32 0) #2
  %0 = load i64, i64* %N, align 8
  %cmp = icmp ult i64 %call, %0
  %conv = zext i1 %cmp to i32
  %arrayidx = getelementptr inbounds i32, i32* %out, i64 %call
  store i32 %conv, i32* %arrayidx, align 4
  ret void
}

declare i64 @__mux_get_global_id(i32)

; CHECK: define spir_kernel void @__vecz_nxv4_store_ult
; CHECK:   [[step:%[0-9.a-z]+]] = call <vscale x 4 x i64> @llvm.{{(experimental\.)?}}stepvector.nxv4i64()
; CHECK:   %{{.*}} = add <vscale x 4 x i64> %{{.*}}, [[step]]
