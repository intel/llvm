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

; Check that we fail to vectorize but don't leave behind an invalid function.
; REQUIRES: llvm-13+
; RUN: not veczc -k regression_phis -vecz-scalable -w 1 -vecz-passes=packetizer,verify -S < %s

target triple = "spir64-unknown-unknown"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

declare spir_func i64 @_Z13get_global_idj(i32)

define spir_kernel void @regression_phis(i64 addrspace(1)* %xs, i64 addrspace(1)* %ys, i32 addrspace(1)* %out, i64 %lim) {
entry:
  %call = call spir_func i64 @_Z13get_global_idj(i32 0)
  %arrayidx.x = getelementptr inbounds i64, i64 addrspace(1)* %xs, i64 %call
  %x = load i64, i64 addrspace(1)* %arrayidx.x, align 4
  %cond = icmp eq i64 %call, 0
  br i1 %cond, label %if.then, label %exit

if.then:
  %arrayidx.y = getelementptr inbounds i64, i64 addrspace(1)* %ys, i64 %call
  %y = load i64, i64 addrspace(1)* %arrayidx.y, align 4
  br label %exit

exit:
  ; We previously left behind an invalid PHI with too few operands, owing to us
  ; bailing our while PHIs were still pending post-vectorization fixup.
  %retval = phi i64 [ %x, %entry ], [ %y, %if.then ]
  %0 = icmp eq i64 %lim, 0
  %1 = select i1 %0, i64 1, i64 %lim
  %rem = urem i64 %retval, %1
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %rem
  %2 = atomicrmw add i32 addrspace(1)* %arrayidx, i32 1 monotonic
  ret void
}
