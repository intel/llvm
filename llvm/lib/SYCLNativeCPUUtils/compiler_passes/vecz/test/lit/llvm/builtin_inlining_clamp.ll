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

; RUN: veczc -k clampkernel -vecz-passes=builtin-inlining -vecz-simd-width=4 -S < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

define spir_kernel void @clampkernel(float %a, float* %c) {
entry:
  %clmp = call spir_func float @_Z5clampfff(float %a, float 0.0, float 1.0)
  store float %clmp, float* %c, align 4
  ret void
}

define spir_func float @_Z5clampfff(float %x, float %y, float %z) {
entry:
  %call.i.i = tail call spir_func float @_Z13__abacus_fmaxff(float %x, float %y)
  %call1.i.i = tail call spir_func float @_Z13__abacus_fminff(float %call.i.i, float %z)
  ret float %call1.i.i
; CHECK-LABEL: float @_Z5clampfff(
; CHECK: [[TMP:%.*]] = call float @llvm.maxnum.f32(float %x, float %y)
; CHECK:             = call float @llvm.minnum.f32(float [[TMP]], float %z)
}

declare spir_func float @_Z13__abacus_fminff(float, float)
declare spir_func float @_Z13__abacus_fmaxff(float, float)

