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

; Test the -cl-opt-disable compile option
; RUN: %veczc -vecz-passes=scalarize -vecz-choices=FullScalarization -S < %s | %filecheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

define spir_kernel void @fmuladd(<4 x double> addrspace(1)* %a, <4 x double> addrspace(1)* %b, <4 x double> addrspace(1)* %c, <4 x double> addrspace(1)* %d, <4 x double> addrspace(1)* %e) {
entry:
  %call = call spir_func i64 @_Z13get_global_idj(i32 0)
  %arrayidx = getelementptr inbounds <4 x double>, <4 x double> addrspace(1)* %b, i64 %call
  %0 = load <4 x double>, <4 x double> addrspace(1)* %arrayidx, align 32
  %arrayidx1 = getelementptr inbounds <4 x double>, <4 x double> addrspace(1)* %c, i64 %call
  %1 = load <4 x double>, <4 x double> addrspace(1)* %arrayidx1, align 32
  %arrayidx2 = getelementptr inbounds <4 x double>, <4 x double> addrspace(1)* %d, i64 %call
  %2 = load <4 x double>, <4 x double> addrspace(1)* %arrayidx2, align 32
  %arrayidx3 = getelementptr inbounds <4 x double>, <4 x double> addrspace(1)* %e, i64 %call
  %3 = load <4 x double>, <4 x double> addrspace(1)* %arrayidx3, align 32
  %div = fdiv <4 x double> %2, %3
  %4 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %0, <4 x double> %1, <4 x double> %div)
  %arrayidx4 = getelementptr inbounds <4 x double>, <4 x double> addrspace(1)* %a, i64 %call
  %5 = load <4 x double>, <4 x double> addrspace(1)* %arrayidx4, align 32
  %sub = fsub <4 x double> %5, %4
  store <4 x double> %sub, <4 x double> addrspace(1)* %arrayidx4, align 32
  ret void
}

; CHECK: define spir_kernel void @__vecz_v[[WIDTH:[0-9]+]]_fmuladd(
; Check if the scalar fmuladd exists
; CHECK: call double @llvm.fmuladd.f64(
; Check if the vector fmuladd doesn't exist
; CHECK-NOT: call double @llvm.fmuladd.v4f64(
; CHECK: ret void

define spir_kernel void @fma(<4 x double> addrspace(1)* %a, <4 x double> addrspace(1)* %b, <4 x double> addrspace(1)* %c, <4 x double> addrspace(1)* %d, <4 x double> addrspace(1)* %e) {
entry:
  %call = call spir_func i64 @_Z13get_global_idj(i32 0)
  %arrayidx = getelementptr inbounds <4 x double>, <4 x double> addrspace(1)* %b, i64 %call
  %0 = load <4 x double>, <4 x double> addrspace(1)* %arrayidx, align 32
  %arrayidx1 = getelementptr inbounds <4 x double>, <4 x double> addrspace(1)* %c, i64 %call
  %1 = load <4 x double>, <4 x double> addrspace(1)* %arrayidx1, align 32
  %arrayidx2 = getelementptr inbounds <4 x double>, <4 x double> addrspace(1)* %d, i64 %call
  %2 = load <4 x double>, <4 x double> addrspace(1)* %arrayidx2, align 32
  %arrayidx3 = getelementptr inbounds <4 x double>, <4 x double> addrspace(1)* %e, i64 %call
  %3 = load <4 x double>, <4 x double> addrspace(1)* %arrayidx3, align 32
  %div = fdiv <4 x double> %2, %3
  %4 = call <4 x double> @llvm.fma.v4f64(<4 x double> %0, <4 x double> %1, <4 x double> %div)
  %arrayidx4 = getelementptr inbounds <4 x double>, <4 x double> addrspace(1)* %a, i64 %call
  %5 = load <4 x double>, <4 x double> addrspace(1)* %arrayidx4, align 32
  %sub = fsub <4 x double> %5, %4
  store <4 x double> %sub, <4 x double> addrspace(1)* %arrayidx4, align 32
  ret void
}

; CHECK: define spir_kernel void @__vecz_v[[WIDTH:[0-9]+]]_fma(
; Check if the scalar fma exists
; CHECK: call double @llvm.fma.f64(
; Check if the vector fma doesn't exist
; CHECK-NOT: call double @llvm.fma.v4f64(
; CHECK: ret void

declare spir_func i64 @_Z13get_global_idj(i32)

declare <4 x double> @llvm.fma.v4f64(<4 x double>, <4 x double>, <4 x double>)
declare <4 x double> @llvm.fmuladd.v4f64(<4 x double>, <4 x double>, <4 x double>)
