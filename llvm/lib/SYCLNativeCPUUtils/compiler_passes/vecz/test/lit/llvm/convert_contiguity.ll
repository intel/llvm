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

; RUN: %veczc -k convert_contiguity -w 4 -S < %s | %filecheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir-unknown-unknown"

; Function Attrs: nounwind
define spir_kernel void @convert_contiguity(float addrspace(1)* %m_ptr) {
  %1 = call spir_func i64 @_Z13get_global_idj(i32 0)
  %2 = call spir_func i32 @_Z12convert_uintm(i64 %1)
  %3 = icmp slt i32 %2, 100
  %4 = select i1 %3, float 1.000000e+00, float 0.000000e+00
  %5 = call spir_func i64 @_Z12convert_longi(i32 %2)
  %6 = getelementptr inbounds float, float addrspace(1)* %m_ptr, i64 %5
  store float %4, float addrspace(1)* %6, align 4
  ret void
}

; Function Attrs: nounwind readnone
declare spir_func i32 @_Z12convert_uintm(i64)

; Function Attrs: nounwind readnone
declare spir_func i64 @_Z12convert_longi(i32)

; Function Attrs: nounwind readonly
declare spir_func i64 @_Z13get_global_idj(i32)

; It checks that the store address was identified as congituous through the
; OpenCL convert builtin function

; CHECK: void @__vecz_v4_convert_contiguity
; CHECK: store <4 x float>
