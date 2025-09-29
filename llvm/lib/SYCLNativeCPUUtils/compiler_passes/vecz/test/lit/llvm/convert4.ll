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

; RUN: veczc -k convert4 -vecz-simd-width=2 -vecz-choices=FullScalarization -S < %s | FileCheck %s

; ModuleID = 'kernel.opencl'
source_filename = "kernel.opencl"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-unknown"

; Function Attrs: convergent nofree nounwind
define spir_kernel void @convert4(<4 x i64> addrspace(1)* nocapture readonly %in, <4 x float> addrspace(1)* nocapture %out) local_unnamed_addr {
entry:
  %call = tail call i64 @__mux_get_global_id(i32 0)
  %arrayidx = getelementptr inbounds <4 x i64>, <4 x i64> addrspace(1)* %in, i64 %call
  %0 = load <4 x i64>, <4 x i64> addrspace(1)* %arrayidx, align 32
  %call1 = tail call spir_func <4 x float> @_Z14convert_float4Dv4_l(<4 x i64> %0)
  %arrayidx2 = getelementptr inbounds <4 x float>, <4 x float> addrspace(1)* %out, i64 %call
  store <4 x float> %call1, <4 x float> addrspace(1)* %arrayidx2, align 16
  ret void
}

; Function Attrs: convergent nounwind readonly
declare i64 @__mux_get_global_id(i32) local_unnamed_addr

; Function Attrs: convergent nounwind readnone
declare spir_func <4 x float> @_Z14convert_float4Dv4_l(<4 x i64>) local_unnamed_addr

; Note that we have to declare the scalar version, because when we vectorize
; an already-vector builtin, we have to scalarize it first. The scalar call
; exists during the intermediate stage between scalarization and packetization,
; and so has to exist in the module.

; Function Attrs: convergent nounwind readnone
declare spir_func float @_Z13convert_floatl(i64) local_unnamed_addr

; Function Attrs: convergent nounwind readnone
declare spir_func <2 x float> @_Z14convert_float2Dv2_l(<2 x i64>) local_unnamed_addr

; With SIMD width 2, should have 4 x convert_float2.

; CHECK: call <2 x i64> @__vecz_b_interleaved_load8_4
; CHECK: call spir_func <2 x float> @_Z14convert_float2Dv2_l
; CHECK: call spir_func <2 x float> @_Z14convert_float2Dv2_l
; CHECK: call spir_func <2 x float> @_Z14convert_float2Dv2_l
; CHECK: call spir_func <2 x float> @_Z14convert_float2Dv2_l
; CHECK-NOT: call spir_func <2 x float> @_Z14convert_float2Dv2_l
; CHECK: call void @__vecz_b_interleaved_store4_4_Dv2_fu3ptrU3AS1(<2 x float>
