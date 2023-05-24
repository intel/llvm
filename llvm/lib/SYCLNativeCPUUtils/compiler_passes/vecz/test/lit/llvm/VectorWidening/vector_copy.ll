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

; RUN: %veczc -k vector_copy -vecz-simd-width=4 -vecz-passes=packetizer -vecz-choices=TargetIndependentPacketization -S < %s | %filecheck %s

; ModuleID = 'kernel.opencl'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

; Function Attrs: nounwind
define spir_kernel void @vector_copy(<4 x i32> addrspace(1)* %out, <4 x i32> addrspace(1)* %in) #0 {
entry:
  %call = call spir_func i64 @_Z13get_global_idj(i32 0) #2
  %arrayidx = getelementptr inbounds <4 x i32>, <4 x i32> addrspace(1)* %in, i64 %call
  %0 = load <4 x i32>, <4 x i32> addrspace(1)* %arrayidx, align 16
  %arrayidx1 = getelementptr inbounds <4 x i32>, <4 x i32> addrspace(1)* %out, i64 %call
  store <4 x i32> %0, <4 x i32> addrspace(1)* %arrayidx1, align 16
  ret void
}

declare spir_func i64 @_Z13get_global_idj(i32) #1

; It makes sure the vector load and store are preserved right through to packetization
; and then widened, instead of being scalarized across work-items first
; and then getting de-interleaved by the Interleaved Group Combine Pass.
; We expect a single vector loads feeding directly into a single vector store.

; CHECK: define spir_kernel void @__vecz_v4_vector_copy
; CHECK: load <16 x i32>
; CHECK-NOT: load
; CHECK-NOT: %deinterleave{{[0-9]*}} = shufflevector
; CHECK-NOT: %interleave{{[0-9]*}} = shufflevector
; CHECK: store <16 x i32>
; CHECK-NOT: store
