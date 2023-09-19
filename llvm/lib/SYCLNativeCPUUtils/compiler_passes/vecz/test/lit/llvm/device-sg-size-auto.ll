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
; RUN: veczc --vecz-auto -k foo -k bar --device-sg-sizes 6,7,8,9 -S < %s | FileCheck %s
; RUN: veczc --vecz-auto -k foo:4 -k bar:4 --device-sg-sizes 6,7,8,9 -S < %s | FileCheck %s

; Check we auto-vectorize to 8, despite any other options telling us a
; different vectorization factor. A factor of 8 is 'best' here because it's a
; power of two.
; CHECK: define void @__vecz_v8_foo(
define void @foo(ptr addrspace(1) %in, ptr addrspace(1) %out) #0 {
  %id = call i64 @__mux_get_global_id(i32 0)
  %in.addr = getelementptr i32, ptr addrspace(1) %in, i64 %id
  %x = load i32, ptr addrspace(1) %in.addr
  %sglid = call i32 @__mux_get_sub_group_local_id()
; CHECK: = add <8 x i32>
  %y = add i32 %x, %sglid
  %out.addr = getelementptr i32, ptr addrspace(1) %out, i64 %id
  store i32 %y, ptr addrspace(1) %out.addr
  ret void
}

; Check we auto-vectorize to 7, despite any other options telling us a
; different vectorization factor. A factor of 8 is 'best' here because it's a
; power of two, but a factor of 7 works well because it won't need a tail.
; CHECK: define void @__vecz_v7_bar(
define void @bar(ptr addrspace(1) %in, ptr addrspace(1) %out) #0 !reqd_work_group_size !0 {
  %id = call i64 @__mux_get_global_id(i32 0)
  %in.addr = getelementptr i32, ptr addrspace(1) %in, i64 %id
  %x = load i32, ptr addrspace(1) %in.addr
  %sglid = call i32 @__mux_get_sub_group_local_id()
; CHECK: = add <7 x i32>
  %y = add i32 %x, %sglid
  %out.addr = getelementptr i32, ptr addrspace(1) %out, i64 %id
  store i32 %y, ptr addrspace(1) %out.addr
  ret void
}

declare i64 @__mux_get_global_id(i32)
declare i32 @__mux_get_sub_group_local_id()

attributes #0 = { "mux-kernel"="entry-point" }

!0 = !{i64 14, i64 1, i64 1}
