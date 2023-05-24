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

; TODO(CA-1981): Using `not` in qemu does not work.
; REQUIRES: native
; RUN: %not %veczc -k printf_add -vecz-simd-width=4 -S -vecz-passes=cfg-convert -vecz-choices=LinearizeBOSCC < %s 2>&1 | %filecheck %s

; This test just checks that we don't crash while converting the control flow.
; LinearizeBOSCC would leave behind an invalid function when control flow fails
; some time afterwards. This could trigger verification failures or crashes
; depending on which passes were run later.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

define spir_kernel void @printf_add(i32 addrspace(1)* %in1, i32 addrspace(1)* %in2, i32 addrspace(1)* %out, i32 addrspace(1)* %status, i8 addrspace(1)* %x) {
entry:
  %in1.addr = alloca i32 addrspace(1)*, align 8
  %in2.addr = alloca i32 addrspace(1)*, align 8
  %out.addr = alloca i32 addrspace(1)*, align 8
  %status.addr = alloca i32 addrspace(1)*, align 8
  %tid = alloca i64, align 8
  %sum = alloca i32, align 4
  store i32 addrspace(1)* %in1, i32 addrspace(1)** %in1.addr, align 8
  store i32 addrspace(1)* %in2, i32 addrspace(1)** %in2.addr, align 8
  store i32 addrspace(1)* %out, i32 addrspace(1)** %out.addr, align 8
  store i32 addrspace(1)* %status, i32 addrspace(1)** %status.addr, align 8
  %call = call spir_func i64 @_Z13get_global_idj(i32 0) #4
  store i64 %call, i64* %tid, align 8
  %0 = load i32 addrspace(1)*, i32 addrspace(1)** %in1.addr, align 8
  %1 = load i64, i64* %tid, align 8
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %0, i64 %1
  %2 = load i32, i32 addrspace(1)* %arrayidx, align 4
  %3 = load i32 addrspace(1)*, i32 addrspace(1)** %in2.addr, align 8
  %4 = load i64, i64* %tid, align 8
  %arrayidx1 = getelementptr inbounds i32, i32 addrspace(1)* %3, i64 %4
  %5 = load i32, i32 addrspace(1)* %arrayidx1, align 4
  %add = add nsw i32 %2, %5
  store i32 %add, i32* %sum, align 4
  %6 = load i32, i32* %sum, align 4
  %7 = load i32 addrspace(1)*, i32 addrspace(1)** %out.addr, align 8
  %8 = load i64, i64* %tid, align 8
  %arrayidx2 = getelementptr inbounds i32, i32 addrspace(1)* %7, i64 %8
  store i32 %6, i32 addrspace(1)* %arrayidx2, align 4
  %9 = load i64, i64* %tid, align 8
  %conv = trunc i64 %9 to i32
  %10 = load i32, i32* %sum, align 4
  %11 = call spir_func i64 @_Z14get_num_groupsj(i32 0)
  %12 = trunc i64 %11 to i32
  %13 = call spir_func i64 @_Z14get_num_groupsj(i32 1)
  %14 = trunc i64 %13 to i32
  %15 = call spir_func i64 @_Z14get_num_groupsj(i32 2)
  %16 = trunc i64 %15 to i32
  %17 = call spir_func i64 @_Z12get_group_idj(i32 0)
  %18 = trunc i64 %17 to i32
  %19 = call spir_func i64 @_Z12get_group_idj(i32 1)
  %20 = trunc i64 %19 to i32
  %21 = call spir_func i64 @_Z12get_group_idj(i32 2)
  %22 = trunc i64 %21 to i32
  %23 = mul i32 %12, %20
  %24 = mul i32 %14, %16
  %25 = mul i32 %22, %24
  %26 = add i32 %23, %25
  %27 = add i32 %18, %26
  %28 = mul i32 %14, %16
  %29 = mul i32 %12, %28
  %30 = udiv i32 1048576, %29
  %31 = and i32 %30, -4
  %32 = mul i32 %27, %31
  %33 = getelementptr i8, i8 addrspace(1)* %x, i32 %32
  %34 = bitcast i8 addrspace(1)* %33 to i32 addrspace(1)*
  %35 = bitcast i8 addrspace(1)* %33 to i32 addrspace(1)*
  %36 = atomicrmw add i32 addrspace(1)* %35, i32 12 acq_rel
  %37 = add i32 %36, 12
  %38 = icmp ugt i32 %37, %31
  br i1 %38, label %early_return.i, label %store.i

early_return.i:                                   ; preds = %entry
  %39 = bitcast i8 addrspace(1)* %33 to i32 addrspace(1)*
  %40 = getelementptr i32, i32 addrspace(1)* %39, i32 1
  %41 = atomicrmw add i32 addrspace(1)* %40, i32 12 acq_rel
  br label %.exit

store.i:                                          ; preds = %entry
  %42 = getelementptr i8, i8 addrspace(1)* %33, i32 %36
  %43 = bitcast i8 addrspace(1)* %42 to i32 addrspace(1)*
  store i32 0, i32 addrspace(1)* %43, align 1
  %44 = add i32 %36, 4
  %45 = getelementptr i8, i8 addrspace(1)* %33, i32 %44
  %46 = bitcast i8 addrspace(1)* %45 to i32 addrspace(1)*
  store i32 %conv, i32 addrspace(1)* %46, align 1
  %47 = add i32 %36, 8
  %48 = getelementptr i8, i8 addrspace(1)* %33, i32 %47
  %49 = bitcast i8 addrspace(1)* %48 to i32 addrspace(1)*
  store i32 %10, i32 addrspace(1)* %49, align 1
  br label %.exit

.exit:                                            ; preds = %store.i, %early_return.i
  %call31 = phi i32 [ -1, %early_return.i ], [ 0, %store.i ]
  %50 = load i32 addrspace(1)*, i32 addrspace(1)** %status.addr, align 8
  %51 = load i64, i64* %tid, align 8
  %arrayidx4 = getelementptr inbounds i32, i32 addrspace(1)* %50, i64 %51
  store i32 %call31, i32 addrspace(1)* %arrayidx4, align 4
  ret void
}

declare spir_func i64 @_Z13get_global_idj(i32)
declare spir_func i64 @_Z12get_group_idj(i32)
declare spir_func i64 @_Z14get_num_groupsj(i32)

; We can't vectorize this control flow
; CHECK: Error: Failed to vectorize function 'printf_add'
