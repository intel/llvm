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

; RUN: %veczc -k test_sqrt -vecz-simd-width=4 -vecz-choices=TargetIndependentPacketization -S < %s | %filecheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

declare spir_func i64 @_Z13get_global_idj(i32)
declare spir_func float @_Z4sqrtf(float)
declare spir_func <2 x float> @_Z4sqrtDv2_f(<2 x float>)
declare spir_func <4 x float> @_Z4sqrtDv4_f(<4 x float>)
declare spir_func <8 x float> @_Z4sqrtDv8_f(<8 x float>)
declare spir_func <16 x float> @_Z4sqrtDv16_f(<16 x float>)

define spir_kernel void @test_sqrt(<2 x float> addrspace(1)* %in2, <2 x float> addrspace(1)* %out2,
                                   <4 x float> addrspace(1)* %in4, <4 x float> addrspace(1)* %out4) {
entry:
  %gid = call spir_func i64 @_Z13get_global_idj(i32 0)
  %arrayin2 = getelementptr inbounds <2 x float>, <2 x float> addrspace(1)* %in2, i64 %gid
  %arrayin4 = getelementptr inbounds <4 x float>, <4 x float> addrspace(1)* %in4, i64 %gid
  %arrayout2 = getelementptr inbounds <2 x float>, <2 x float> addrspace(1)* %out2, i64 %gid
  %arrayout4 = getelementptr inbounds <4 x float>, <4 x float> addrspace(1)* %out4, i64 %gid
  %ld2 = load <2 x float>, <2 x float> addrspace(1)* %arrayin2, align 16
  %ld4 = load <4 x float>, <4 x float> addrspace(1)* %arrayin4, align 16
  %sqrt2 = call spir_func <2 x float> @_Z4sqrtDv2_f(<2 x float> %ld2)
  %sqrt4 = call spir_func <4 x float> @_Z4sqrtDv4_f(<4 x float> %ld4)
  store <2 x float> %sqrt2, <2 x float> addrspace(1)* %arrayout2, align 16
  store <4 x float> %sqrt4, <4 x float> addrspace(1)* %arrayout4, align 16
  ret void
}

; The purpose of this test is to check that the vector context is able to
; supply the packetizer with two versions of the builtin vectorized to two
; different widths.
;
; CHECK: define spir_kernel void @__vecz_v4_test_sqrt
; CHECK: call spir_func <8 x float> @_Z4sqrtDv8_f(<8 x float> %{{.*}})
; CHECK: call spir_func <16 x float> @_Z4sqrtDv16_f(<16 x float> %{{.*}})
; CHECK: ret void
