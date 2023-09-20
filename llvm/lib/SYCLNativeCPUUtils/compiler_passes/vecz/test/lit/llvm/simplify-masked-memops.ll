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

; RUN: veczc -k foo -vecz-passes=mask-memops -S < %s | FileCheck %s

define void @foo(i16 %x, i32 %y, ptr addrspace(1) %p) {
entry:
  call void @__vecz_b_masked_store2_tu3ptrU3AS1b(i16 %x, ptr addrspace(1) %p, i1 true)
  call void @__vecz_b_masked_store2_ju3ptrU3AS1b(i32 %y, ptr addrspace(1) %p, i1 true)
  %f = call float @__vecz_b_masked_load2_fu3ptrU3AS1b(ptr addrspace(1) %p, i1 true)
  %v4f = call <4 x float> @__vecz_b_masked_load2_Dv4_fu3ptrU3AS1Dv4_b(ptr addrspace(1) %p, <4 x i1> <i1 true, i1 true, i1 true, i1 true>)
  ret void
}

; Check we correctly set the alignment on the optimized loads and stores. The
; alignment must come from the builtin, not from the natural/preferred
; alignment for that type.
; CHECK: define void @__vecz_v4_foo(i16 %x, i32 %y, ptr addrspace(1) %p)
; CHECK: entry:
; CHECK:      store i16 %x, ptr addrspace(1) %p, align 2
; CHECK-NEXT: store i32 %y, ptr addrspace(1) %p, align 2
; CHECK-NEXT: %f = load float, ptr addrspace(1) %p, align 2
; CHECK-NEXT: %v4f = load <4 x float>, ptr addrspace(1) %p, align 2
; CHECK-NEXT: ret void

declare void @__vecz_b_masked_store2_tu3ptrU3AS1b(i16, ptr addrspace(1), i1)
declare void @__vecz_b_masked_store2_ju3ptrU3AS1b(i32, ptr addrspace(1), i1)
declare float @__vecz_b_masked_load2_fu3ptrU3AS1b(ptr addrspace(1), i1)
declare <4 x float> @__vecz_b_masked_load2_Dv4_fu3ptrU3AS1Dv4_b(ptr addrspace(1), <4 x i1>)
