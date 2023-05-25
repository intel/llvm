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

; RUN: veczc -k load_binops_store -vecz-scalable -vecz-simd-width=4 -S < %s | FileCheck %s

target triple = "spir64-unknown-unknown"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define spir_kernel void @load_binops_store(i32* %aptr, i32* %bptr, i32* %cptr, i32* %zptr) {
entry:
  %idx = call spir_func i64 @_Z13get_global_idj(i32 0)
  %arrayidxa = getelementptr inbounds i32, i32* %aptr, i64 %idx
  %arrayidxb = getelementptr inbounds i32, i32* %bptr, i64 %idx
  %arrayidxc = getelementptr inbounds i32, i32* %cptr, i64 %idx
  %arrayidxz = getelementptr inbounds i32, i32* %zptr, i64 %idx
  %a = load i32, i32* %arrayidxa, align 4
  %b = load i32, i32* %arrayidxb, align 4
  %c = load i32, i32* %arrayidxc, align 4
  %sum = add i32 %a, %b
  %mpy = mul i32 %sum, %c
  %shf = ashr i32 %mpy, 3
  %dvu = udiv i32 %shf, %sum
  store i32 %dvu, i32* %arrayidxz, align 4
  ret void
}

; CHECK: define spir_kernel void @__vecz_nxv4_load_binops_store
; CHECK: load <vscale x 4 x i32>, ptr
; CHECK: load <vscale x 4 x i32>, ptr
; CHECK: add <vscale x 4 x i32>
; CHECK: mul <vscale x 4 x i32>
; CHECK: ashr <vscale x 4 x i32>
; CHECK: store <vscale x 4 x i32>
declare spir_func i64 @_Z13get_global_idj(i32)
