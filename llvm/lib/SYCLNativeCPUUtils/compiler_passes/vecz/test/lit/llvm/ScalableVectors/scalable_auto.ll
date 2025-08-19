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

; RUN: veczc -k cast -vecz-scalable -S < %s | FileCheck %s

target triple = "spir64-unknown-unknown"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define spir_kernel void @cast(i32* %aptr, float* %zptr) {
entry:
  %idx = call i64 @__mux_get_global_id(i32 0)
  %arrayidxa = getelementptr inbounds i32, i32* %aptr, i64 %idx
  %arrayidxz = getelementptr inbounds float, float* %zptr, i64 %idx
  %a = load i32, i32* %arrayidxa, align 4
  %c = sitofp i32 %a to float
  store float %c, float* %arrayidxz, align 4
  ret void
}

; Check that passing -vecz-scalable with no width automatically chooses an
; appropriate scalable vectorization factor.
; CHECK: define spir_kernel void @__vecz_nxv[[VF:[0-9]+]]_cast
; CHECK: sitofp <vscale x [[VF]] x i32> {{%[0-9]+}} to <vscale x [[VF]] x float>
declare i64 @__mux_get_global_id(i32)
