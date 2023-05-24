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

; Check that veczc can vectorize a kernel then vectorize the vectorized kernel,
; with base mappings from 1->2 and 2->3 and derived mappings back from 2->1 and
; 3->2.
; RUN: %veczc -k add:2 -S < %s > %t2
; RUN: %veczc -k __vecz_v2_add:4 -S < %t2 | %filecheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

define spir_kernel void @add(i32 addrspace(1)* %in1, i32 addrspace(1)* %in2, i32 addrspace(1)* %out) {
entry:
  %tid = call spir_func i64 @_Z13get_global_idj(i32 0) #3
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %in1, i64 %tid
  %i1 = load i32, i32 addrspace(1)* %arrayidx, align 16
  %arrayidx1 = getelementptr inbounds i32, i32 addrspace(1)* %in2, i64 %tid
  %i2 = load i32, i32 addrspace(1)* %arrayidx1, align 16
  %add = add nsw i32 %i1, %i2
  %arrayidx2 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %tid
  store i32 %add, i32 addrspace(1)* %arrayidx2, align 16
  ret void
}

declare spir_func i64 @_Z13get_global_idj(i32) #2

; CHECK: define spir_kernel void @add(ptr addrspace(1) %in1, ptr addrspace(1) %in2, ptr addrspace(1) %out){{.*}} !codeplay_ca_vecz.base ![[BASE_1:[0-9]+]]
; CHECK: define spir_kernel void @__vecz_v2_add(ptr addrspace(1) %in1, ptr addrspace(1) %in2, ptr addrspace(1) %out){{.*}} !codeplay_ca_vecz.base ![[BASE_2:[0-9]+]] !codeplay_ca_vecz.derived ![[DERIVED_1:[0-9]+]] {
  ; CHECK: define spir_kernel void @__vecz_v4___vecz_v2_add(ptr addrspace(1) %in1, ptr addrspace(1) %in2, ptr addrspace(1) %out){{.*}} !codeplay_ca_vecz.derived ![[DERIVED_2:[0-9]+]] {

; CHECK: ![[BASE_1]] = !{![[VMD_1:[0-9]+]], {{.*}} @__vecz_v2_add}
; CHECK: ![[VMD_1]] = !{i32 2, i32 0, i32 0, i32 0}
; CHECK: ![[BASE_2]] = !{![[VMD_2:[0-9]+]], {{.*}} @__vecz_v4___vecz_v2_add}
; CHECK: ![[VMD_2]] = !{i32 4, i32 0, i32 0, i32 0}
; CHECK: ![[DERIVED_1]] = !{![[VMD_1]], {{.*}} @add}
; CHECK: ![[DERIVED_2]] = !{![[VMD_2]], {{.*}} @__vecz_v2_add}
