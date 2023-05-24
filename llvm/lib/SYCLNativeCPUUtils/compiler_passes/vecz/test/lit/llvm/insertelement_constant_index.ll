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

; RUN: %veczc -k constant_index -vecz-simd-width=4 -vecz-choices=FullScalarization -S < %s | %filecheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

declare spir_func i64 @_Z13get_global_idj(i32)

define spir_kernel void @constant_index(<4 x i32>* %in, <4 x i32>* %out) {
entry:
  %call = call spir_func i64 @_Z13get_global_idj(i32 0)
  %arrayidx = getelementptr inbounds <4 x i32>, <4 x i32>* %in, i64 %call
  %0 = load <4 x i32>, <4 x i32>* %arrayidx
  %arrayidx2 = getelementptr inbounds <4 x i32>, <4 x i32>* %out, i64 %call
  %vecins = insertelement <4 x i32> %0, i32 42, i32 2
  store <4 x i32> %vecins, <4 x i32>* %arrayidx2
  ret void
}

; CHECK: define spir_kernel void @__vecz_v4_constant_index

; We should only have 3 loads since one of the elements will be replaced
; CHECK: call <4 x i32> @__vecz_b_interleaved_load4_4_Dv4_ju3ptr
; CHECK: call <4 x i32> @__vecz_b_interleaved_load4_4_Dv4_ju3ptr
; CHECK: call <4 x i32> @__vecz_b_interleaved_load4_4_Dv4_ju3ptr
; CHECK-NOT: call <4 x i32> @__vecz_b_interleaved_load4_4_Dv4_ju3ptr

; We should have four stores, one of which would use the constant given
; CHECK: store <4 x i32>
; CHECK: store <4 x i32>
; CHECK: store <4 x i32>
; CHECK: store <4 x i32>
; CHECK: ret void
