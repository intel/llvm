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

; Let vecz pick the right vectorization factor for this kernel
; RUN: veczc --vecz-auto -k bar_sg8 -k foo_sg13 -S < %s | FileCheck %s
; RUN: veczc --vecz-auto -k bar_sg8:4 -k foo_sg13:8 -S < %s | FileCheck %s

; Check we auto-vectorize to 8, despite any other options telling us a
; different vectorization factor.
; CHECK: define void @__vecz_v8_bar_sg8
define void @bar_sg8(ptr addrspace(1) %in, ptr addrspace(1) %out) #0 !intel_reqd_sub_group_size !0 {
  %id = call i64 @__mux_get_global_id(i32 0)
  %in.addr = getelementptr i32, ptr addrspace(1) %in, i64 %id
  %x = load i32, ptr addrspace(1) %in.addr
; CHECK: = add <8 x i32>
  %y = add i32 %x, 1
  %out.addr = getelementptr i32, ptr addrspace(1) %out, i64 %id
  store i32 %y, ptr addrspace(1) %out.addr
  ret void
}

; Check we auto-vectorize to 13, despite any other options telling us a
; different vectorization factor. This is a silly number but it if we're told
; to do it we must obey.
; CHECK: define void @__vecz_v13_foo_sg13
define void @foo_sg13(ptr addrspace(1) %in, ptr addrspace(1) %out) #0 !intel_reqd_sub_group_size !1 {
  %id = call i64 @__mux_get_global_id(i32 0)
  %in.addr = getelementptr i32, ptr addrspace(1) %in, i64 %id
  %x = load i32, ptr addrspace(1) %in.addr
; CHECK: = add <13 x i32>
  %y = add i32 %x, 1
  %out.addr = getelementptr i32, ptr addrspace(1) %out, i64 %id
  store i32 %y, ptr addrspace(1) %out.addr
  ret void
}

declare i64 @__mux_get_global_id(i32)

attributes #0 = { "mux-kernel"="entry-point" }

!0 = !{i32 8}
!1 = !{i32 13}
