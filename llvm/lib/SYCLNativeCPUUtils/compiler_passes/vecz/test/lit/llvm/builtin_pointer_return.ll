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

target triple = "spir64-unknown-unknown"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; RUN: veczc -vecz-simd-width=4 -S < %s | FileCheck %s

declare i64 @__mux_get_global_id(i32)

declare spir_func float @_Z5fractfPf(float, float*)
declare spir_func <2 x float> @_Z5fractDv2_fPS_(<2 x float>, <2 x float>*)
declare spir_func <4 x float> @_Z5fractDv4_fPS_(<4 x float>, <4 x float>*)
declare spir_func <8 x float> @_Z5fractDv8_fPS_(<8 x float>, <8 x float>*)

; FIXME: Both of these are instantiating when we have vector equivalents: see
; CA-4046.

define spir_kernel void @fract_v1(float* %xptr, float* %outptr, float* %ioutptr) {
  %iouta = alloca float
  %idx = call i64 @__mux_get_global_id(i32 0)
  %arrayidx.x = getelementptr inbounds float, float* %xptr, i64 %idx
  %x = load float, float* %arrayidx.x, align 4
  %out = call spir_func float @_Z5fractfPf(float %x, float* %iouta)
  %arrayidx.out = getelementptr inbounds float, float* %outptr, i64 %idx
  %arrayidx.iout = getelementptr inbounds float, float* %ioutptr, i64 %idx
  store float %out, float* %arrayidx.out, align 4
  %iout = load float, float* %iouta, align 4
  store float %iout, float* %arrayidx.iout, align 4
  ret void
; CHECK: call spir_func float @_Z5fractfPf(float {{%.*}}, ptr nonnull {{%.*}})
; CHECK: call spir_func float @_Z5fractfPf(float {{%.*}}, ptr nonnull {{%.*}})
; CHECK: call spir_func float @_Z5fractfPf(float {{%.*}}, ptr nonnull {{%.*}})
; CHECK: call spir_func float @_Z5fractfPf(float {{%.*}}, ptr nonnull {{%.*}})
}

define spir_kernel void @fract_v2(<2 x float>* %xptr, <2 x float>* %outptr, <2 x float>* %ioutptr) {
  %iouta = alloca <2 x float>
  %idx = call i64 @__mux_get_global_id(i32 0)
  %arrayidx.x = getelementptr inbounds <2 x float>, <2 x float>* %xptr, i64 %idx
  %x = load <2 x float>, <2 x float>* %arrayidx.x, align 8
  %out = call spir_func <2 x float> @_Z5fractDv2_fPS_(<2 x float> %x, <2 x float>* %iouta)
  %arrayidx.out = getelementptr inbounds <2 x float>, <2 x float>* %outptr, i64 %idx
  %arrayidx.iout = getelementptr inbounds <2 x float>, <2 x float>* %ioutptr, i64 %idx
  store <2 x float> %out, <2 x float>* %arrayidx.out, align 8
  %iout = load <2 x float>, <2 x float>* %iouta, align 8
  store <2 x float> %iout, <2 x float>* %arrayidx.iout, align 8
  ret void
; CHECK: call spir_func <2 x float> @_Z5fractDv2_fPS_(<2 x float> {{%.*}}, ptr nonnull {{%.*}})
; CHECK: call spir_func <2 x float> @_Z5fractDv2_fPS_(<2 x float> {{%.*}}, ptr nonnull {{%.*}})
; CHECK: call spir_func <2 x float> @_Z5fractDv2_fPS_(<2 x float> {{%.*}}, ptr nonnull {{%.*}})
; CHECK: call spir_func <2 x float> @_Z5fractDv2_fPS_(<2 x float> {{%.*}}, ptr nonnull {{%.*}})
}
