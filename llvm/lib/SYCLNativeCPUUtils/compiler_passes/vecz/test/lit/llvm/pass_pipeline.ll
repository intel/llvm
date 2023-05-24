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

; RUN: %veczc -k foo -w 2 -debug-vecz-pipeline -S < %s 2>&1 | %filecheck %s
; RUN: %veczc -k foo -w 2 -vecz-passes scalarize -debug-vecz-pipeline -S < %s 2>&1 | %filecheck %s --check-prefix=PASSES1
; RUN: %veczc -k foo -w 2 -vecz-passes scalarize,packetizer -debug-vecz-pipeline -S < %s 2>&1 | %filecheck %s --check-prefix=PASSES2

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

; Don't check specific passes, but assume that *some* analyses and passes are run.
; CHECK: Running analysis: {{.*}}> on __vecz_v2_foo
; CHECK: Running pass: {{.*}} on __vecz_v2_foo

; PASSES1: Running pass: RequireAnalysisPass<{{(class )?}}compiler::utils::DeviceInfoAnalysis,
; PASSES1-NOT: Running pass:
; PASSES1: Running pass: Function scalarization on __vecz_v2_foo
; PASSES1-NOT: Running pass:
; PASSES1-NOT: Running pass:

; PASSES2: Running pass: RequireAnalysisPass<{{(class )?}}compiler::utils::DeviceInfoAnalysis,
; PASSES2-NOT: Running pass:
; PASSES2: Running pass: Function scalarization on __vecz_v2_foo
; PASSES2: Running pass: Function packetization on __vecz_v2_foo
; PASSES2-NOT: Running pass:
; PASSES2-NOT: Running pass:

define spir_kernel void @foo(i32 addrspace(1)* %out) {
  %idx = call spir_func i64 @_Z13get_global_idj(i32 0)
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %idx
  store i32 0, i32 addrspace(1)* %arrayidx, align 4
  ret void
}

declare spir_func i64 @_Z13get_global_idj(i32)
