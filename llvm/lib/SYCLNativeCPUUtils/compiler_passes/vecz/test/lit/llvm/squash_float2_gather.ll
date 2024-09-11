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

; RUN: veczc -k squash -vecz-passes="squash-small-vecs,packetizer" -S < %s | FileCheck %s

; ModuleID = 'kernel.opencl'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

; Function Attrs: nounwind
define spir_kernel void @squash(i64 addrspace(1)* %idx, <2 x float> addrspace(1)* %data, <2 x float> addrspace(1)* %output) #0 {
entry:
  %gid = call i64 @__mux_get_global_id(i64 0) #2
  %idx.ptr = getelementptr inbounds i64, i64 addrspace(1)* %idx, i64 %gid
  %idx.ld = load i64, i64 addrspace(1)* %idx.ptr, align 8
  %data.ptr = getelementptr inbounds <2 x float>, <2 x float> addrspace(1)* %data, i64 %idx.ld
  %data.ld = load <2 x float>, <2 x float> addrspace(1)* %data.ptr, align 8
  %output.ptr = getelementptr inbounds <2 x float>, <2 x float> addrspace(1)* %output, i64 %gid
  store <2 x float> %data.ld, <2 x float> addrspace(1)* %output.ptr, align 8
  ret void
}

declare i64 @__mux_get_global_id(i64) #1

attributes #0 = { nounwind "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "stack-protector-buffer-size"="0" "stackrealign" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "stack-protector-buffer-size"="0" "stackrealign" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nobuiltin nounwind }

; It checks that the <2 x float> is converted into a i64 for the purpose of the
; gather load
;
; CHECK: void @__vecz_v4_squash
; CHECK:  %[[GID:.+]] = call i64 @__mux_get_global_id(i64 0) #[[ATTRS:[0-9]+]]
; CHECK:  %[[IDX_PTR:.+]] = getelementptr i64, ptr addrspace(1) %idx, i64 %[[GID]]
; CHECK:  %[[WIDE_LOAD:.+]] = load <4 x i64>, ptr addrspace(1) %[[IDX_PTR]], align 8
; CHECK:  %[[DATA_PTR:.+]] = getelementptr <2 x float>, ptr addrspace(1) %data, <4 x i64> %[[WIDE_LOAD]]
; CHECK:  %[[GATHER:.+]] = call <4 x i64> @__vecz_b_gather_load8_Dv4_mDv4_u3ptrU3AS1(<4 x ptr addrspace(1)> %[[DATA_PTR]])
; CHECK:  %[[UNSQUASH:.+]] = bitcast <4 x i64> %[[GATHER]] to <8 x float>
; CHECK:  %[[OUTPUT_PTR:.+]] = getelementptr <2 x float>, ptr addrspace(1) %output, i64 %[[GID]]
; CHECK:  store <8 x float> %[[UNSQUASH]], ptr addrspace(1) %[[OUTPUT_PTR]], align 8
; CHECK:  ret void

; CHECK: attributes #[[ATTRS]] = { nobuiltin nounwind }
