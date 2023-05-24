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

; RUN: %veczc -k load_add_store -vecz-scalable -vecz-simd-width=4 -S < %s | %filecheck %s

target triple = "spir64-unknown-unknown"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define spir_kernel void @load_add_store(<4 x i32>* %aptr, <4 x i32>* %bptr, <4 x i32>* %zptr) {
entry:
  %idx = call spir_func i64 @_Z13get_global_idj(i32 0)
  %arrayidxa = getelementptr inbounds <4 x i32>, <4 x i32>* %aptr, i64 %idx
  %arrayidxb = getelementptr inbounds <4 x i32>, <4 x i32>* %bptr, i64 %idx
  %arrayidxz = getelementptr inbounds <4 x i32>, <4 x i32>* %zptr, i64 %idx
  %a = load <4 x i32>, <4 x i32>* %arrayidxa, align 4
  %b = load <4 x i32>, <4 x i32>* %arrayidxb, align 4
  %sum = add <4 x i32> %a, %b
  store <4 x i32> %sum, <4 x i32>* %arrayidxz, align 4
  ret void
}

declare spir_func i64 @_Z13get_global_idj(i32)

; CHECK: define spir_kernel void @__vecz_nxv4_load_add_store
; CHECK: [[lhs:%[0-9a-z]+]] = load <vscale x 16 x i32>, ptr
; CHECK: [[rhs:%[0-9a-z]+]] = load <vscale x 16 x i32>, ptr
; CHECK: [[sum:%[0-9a-z]+]] = add <vscale x 16 x i32> [[lhs]], [[rhs]]
; CHECK: store <vscale x 16 x i32> [[sum]],
