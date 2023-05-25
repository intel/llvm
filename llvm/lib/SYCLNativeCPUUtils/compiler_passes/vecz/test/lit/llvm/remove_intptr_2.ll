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

; RUN: veczc -S < %s | FileCheck %s

target triple = "spir64-unknown-unknown"
target datalayout = "e-p:64:64:64-m:e-i64:64-f80:128-n8:16:32:64-S128"

define spir_kernel void @remove_intptr(i8 addrspace(1)* %in, i32 addrspace(1)* %out) {
entry:
  %call = tail call spir_func i64 @_Z13get_global_idj(i32 0)
  %0 = ptrtoint i8 addrspace(1)* %in to i64
  %shl = shl nuw nsw i64 %call, 2
  %add = add i64 %shl, %0
  %1 = inttoptr i64 %add to i32 addrspace(1)*
  %2 = load i32, i32 addrspace(1)* %1, align 4
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %call
  store i32 %2, i32 addrspace(1)* %arrayidx, align 4
  ret void
}

declare spir_func i64 @_Z13get_global_idj(i32)

; CHECK: spir_kernel void @__vecz_v4_remove_intptr
; CHECK-NOT: ptrtoint
; CHECK-NOT: inttoptr
; CHECK: %remove_intptr = getelementptr i8, ptr addrspace(1) %in
; CHECK: %[[LOAD:.+]] = load <4 x i32>, ptr addrspace(1) %remove_intptr, align 4
; CHECK: store <4 x i32> %[[LOAD]]
