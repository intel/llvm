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

; RUN: veczc -k test -vecz-simd-width=4 -vecz-choices=FullScalarization -S < %s | FileCheck %s

; ModuleID = 'kernel.opencl'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

@.str = private unnamed_addr addrspace(2) constant [4 x i8] c"%p\0A\00", align 1

define spir_kernel void @test() {
entry:
  %gid = call spir_func i64 @__mux_get_global_id(i32 0)
  %printf = call spir_func i32 (ptr addrspace(2), ...) @printf(ptr addrspace(2) @.str, i64 %gid)
  ret void
}

declare spir_func i64 @__mux_get_global_id(i32)

define spir_func i32 @printf(ptr, ...) {
  ret i32 0
}

; CHECK: define spir_kernel void @__vecz_v4_test(
; CHECK: call spir_func i32 (ptr addrspace(2), ...) @printf(ptr addrspace(2) @.str, i64
; CHECK: call spir_func i32 (ptr addrspace(2), ...) @printf(ptr addrspace(2) @.str, i64
; CHECK: call spir_func i32 (ptr addrspace(2), ...) @printf(ptr addrspace(2) @.str, i64
; CHECK: call spir_func i32 (ptr addrspace(2), ...) @printf(ptr addrspace(2) @.str, i64
; CHECK: ret void
