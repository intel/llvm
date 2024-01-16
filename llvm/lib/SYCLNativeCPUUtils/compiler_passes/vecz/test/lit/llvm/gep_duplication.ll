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

; RUN: veczc -S -vecz-passes="function(mem2reg,instcombine),cfg-convert,gvn,packetizer" < %s | FileCheck %s

target triple = "spir64-unknown-unknown"
target datalayout = "e-p:64:64:64-m:e-i64:64-f80:128-n8:16:32:64-S128"

%struct.testStruct = type { [2 x i32] }


; Check that we de-duplicate the GEPs used across this kernel (using a
; combination of instcombine and GVN).
; CHECK: spir_kernel void @__vecz_v{{[0-9]+}}_gep_duplication
; CHECK: entry:
; CHECK: getelementptr inbounds [2 x i32], ptr %myStruct, i{{32|64}} 0, i{{32|64}} 1
; CHECK-NOT: getelementptr {{.*}}%myStruct
define spir_kernel void @gep_duplication(ptr addrspace(1) align 4 %out) {
entry:
  %out.addr = alloca ptr addrspace(1), align 8
  %global_id = alloca i64, align 8
  %myStruct = alloca %struct.testStruct, align 4
  store ptr addrspace(1) %out, ptr %out.addr, align 8
  %call = call i64 @__mux_get_global_id(i32 0) #2
  store i64 %call, ptr %global_id, align 8
  %x = getelementptr inbounds %struct.testStruct, ptr %myStruct, i32 0, i32 0
  %arrayidx = getelementptr inbounds [2 x i32], ptr %x, i64 0, i64 0
  store i32 0, ptr %arrayidx, align 4
  %x1 = getelementptr inbounds %struct.testStruct, ptr %myStruct, i32 0, i32 0
  %arrayidx2 = getelementptr inbounds [2 x i32], ptr %x1, i64 0, i64 1
  store i32 1, ptr %arrayidx2, align 4
  %0 = load i64, ptr %global_id, align 8
  %and = and i64 %0, 1
  %tobool = icmp ne i64 %and, 0
  br i1 %tobool, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  %x3 = getelementptr inbounds %struct.testStruct, ptr %myStruct, i32 0, i32 0
  %arrayidx4 = getelementptr inbounds [2 x i32], ptr %x3, i64 0, i64 0
  %1 = load i32, ptr %arrayidx4, align 4
  %2 = load ptr addrspace(1), ptr %out.addr, align 8
  %3 = load i32, ptr %global_id, align 4
  %idxprom = sext i32 %3 to i64
  %arrayidx5 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 %idxprom
  store i32 %1, ptr addrspace(1) %arrayidx5, align 4
  br label %if.end

if.else:                                          ; preds = %entry
  %x6 = getelementptr inbounds %struct.testStruct, ptr %myStruct, i32 0, i32 0
  %arrayidx7 = getelementptr inbounds [2 x i32], ptr %x6, i64 0, i64 1
  %4 = load i32, ptr %arrayidx7, align 4
  %5 = load ptr addrspace(1), ptr %out.addr, align 8
  %6 = load i32, ptr %global_id, align 4
  %idxprom8 = sext i32 %6 to i64
  %arrayidx9 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 %idxprom8
  store i32 %4, ptr addrspace(1) %arrayidx9, align 4
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  ret void
}

declare i64 @__mux_get_global_id(i32)
