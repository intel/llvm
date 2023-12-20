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

; RUN: veczc -w 4 -vecz-scalable -vecz-passes=packetizer,verify \
; RUN:   --pass-remarks-missed=vecz -S < %s 2>&1 | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

; Note: we can't currently scalably packetize this kernel, due to the struct
; type.
; CHECK: Vecz: Could not packetize %old0 = cmpxchg ptr %p, i32 1, i32 2 acquire monotonic, align 4
define spir_kernel void @test_fn(ptr %p, ptr %q, ptr %r) {
entry:
  %call = call i64 @__mux_get_global_id(i32 0)

  %old0 = cmpxchg ptr %p, i32 1, i32 2 acquire monotonic
  %val0 = extractvalue { i32, i1 } %old0, 0
  %success0 = extractvalue { i32, i1 } %old0, 1

  %out = getelementptr i32, ptr %q, i64 %call
  store i32 %val0, ptr %out, align 4

  %outsuccess = getelementptr i8, ptr %r, i64 %call
  %outbyte = zext i1 %success0 to i8
  store i8 %outbyte, ptr %outsuccess, align 1

  ; Test a couple of insert/extract patterns

  ; Test inserting a uniform value into a varying literal struct
  %testinsertconst = insertvalue { i32, i1 } %old0, i1 false, 1
  %testextract0 = extractvalue { i32, i1 } %testinsertconst, 1
  %outbyte0 = zext i1 %testextract0 to i8
  store i8 %outbyte0, ptr %outsuccess, align 1

  ; Test inserting a varying value into a varying literal struct
  %byte1 = load i8, ptr %outsuccess, align 1
  %bool1 = trunc i8 %byte1 to i1
  %testinsertvarying0 = insertvalue { i32, i1 } %old0, i1 %bool1, 1
  %testextract1 = extractvalue { i32, i1 } %testinsertvarying0, 1
  %outbyte1 = zext i1 %testextract1 to i8
  store i8 %outbyte1, ptr %outsuccess, align 1

  ; Test inserting a varying value into a uniform literal struct
  %testinsertvarying1 = insertvalue { i32, i1 } poison, i1 %bool1, 1
  %testextract2 = extractvalue { i32, i1 } %testinsertvarying1, 1
  %outbyte2 = zext i1 %testextract2 to i8
  store i8 %outbyte2, ptr %outsuccess, align 1

  ret void
}

declare i64 @__mux_get_global_id(i32)
