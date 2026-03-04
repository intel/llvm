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

; Check some basic properties of the veczc command line interface for multiple
; vectorizations works in various configurations. The kernel outputs here are
; not interesting, only their names.
; RUN: veczc -w 8 -k foo:4,8,16.2@32s -k bar:,64s -S < %s | FileCheck %s

; CHECK-DAG: define spir_kernel void @foo
; CHECK-DAG: define spir_kernel void @bar
; CHECK-DAG: define spir_kernel void @__vecz_v4_foo
; CHECK-DAG: define spir_kernel void @__vecz_v8_foo
; CHECK-DAG: define spir_kernel void @__vecz_nxv16_foo
; CHECK-DAG: define spir_kernel void @__vecz_v8_bar
; CHECK-DAG: define spir_kernel void @__vecz_nxv64_bar

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

define spir_kernel void @foo(i32 addrspace(1)* %in1, i32 addrspace(1)* %in2, i32 addrspace(1)* %out) {
entry:
  ret void
}

define spir_kernel void @bar(i32 addrspace(1)* %in1, i32 addrspace(1)* %in2, i32 addrspace(1)* %out) {
entry:
  ret void
}


