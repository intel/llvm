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

; REQUIRES: llvm-13+
; RUN: veczc -vecz-target-triple="riscv64-unknown-unknown" -vecz-scalable -vecz-simd-width=4 -vecz-passes=packetizer -S < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

define spir_kernel void @f(<4 x i32> addrspace(1)* %in, <2 x i32> addrspace(1)* %out) {
entry:
  %gid = call i64 @__mux_get_global_id(i32 0)
  %in.ptr = getelementptr inbounds <4 x i32>, <4 x i32> addrspace(1)* %in, i64 %gid
  %in.data = load <4 x i32>, <4 x i32> addrspace(1)* %in.ptr
  %out.data = shufflevector <4 x i32> %in.data, <4 x i32> poison, <2 x i32> <i32 0, i32 2>
  %out.ptr = getelementptr inbounds <2 x i32>, <2 x i32> addrspace(1)* %out, i64 %gid
  store <2 x i32> %out.data, <2 x i32> addrspace(1)* %out.ptr, align 32
  ret void
}

declare i64 @__mux_get_global_id(i32) #1

; It checks that a single-operand shuffle that narrows the vector is packetized to a gather intrinsic.
; CHECK: define spir_kernel void @__vecz_nxv4_f({{.*}}) {{.*}} {
; CHECK: entry:
; CHECK:  %[[DATA:.+]] = load <vscale x 16 x i32>, {{(<vscale x 16 x i32> addrspace\(1\)\*)|(ptr addrspace\(1\))}} %{{.*}}
; CHECK:  %[[GATHER:.+]] = call <vscale x 16 x i32> @llvm.riscv.vrgather.vv.nxv16i32.i64(<vscale x 16 x i32> poison, <vscale x 16 x i32> %[[DATA]], <vscale x 16 x i32> %{{.+}}, i64 %{{.+}})
; CHECK:  %[[EXTRACT:.+]] = call <vscale x 8 x i32> @llvm.vector.extract.nxv8i32.nxv16i32(<vscale x 16 x i32> %[[GATHER]], i64 0)
; CHECK:  store <vscale x 8 x i32> %[[EXTRACT]]
; CHECK:  ret void
; CHECK: }
