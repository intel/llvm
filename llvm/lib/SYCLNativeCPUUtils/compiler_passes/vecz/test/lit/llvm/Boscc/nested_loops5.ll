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

; RUN: veczc -k nested_loops5 -vecz-passes=vecz-loop-rotate,cfg-convert -vecz-choices=LinearizeBOSCC -S < %s | FileCheck %s

; ModuleID = 'Unknown buffer'
source_filename = "Unknown buffer"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

declare i64 @__mux_get_local_id(i32)

declare i64 @__mux_get_local_size(i32)

define spir_kernel void @nested_loops5(float addrspace(1)*) {
entry:
  %lid = tail call i64 @__mux_get_local_id(i32 0)
  %lsize = tail call i64 @__mux_get_local_size(i32 0)
  %cmp1 = icmp ult i64 %lid, %lsize
  br i1 %cmp1, label %loop, label %end

loop:                                             ; preds = %if.end, %entry
  %livethrough = phi i64 [ %add2, %if.end ], [ %lsize, %entry ]
  %add1 = add i64 %livethrough, %lsize
  %cmp2 = icmp ult i64 %add1, %lsize
  br i1 %cmp2, label %if.then, label %if.else

if.then:                                          ; preds = %if.then, %loop
  %phi = phi i64 [ %add3, %if.then ], [ %lid, %loop ]
  %add3 = add i64 %phi, %lsize
  %cmp4 = icmp ult i64 %add3, %lsize
  br i1 %cmp4, label %if.then, label %if.end

if.else:                                          ; preds = %loop
  %gep = getelementptr inbounds float, float addrspace(1)* %0, i64 %add1
  store float 0.000000e+00, float addrspace(1)* %gep, align 4
  br label %if.end

if.end:                                           ; preds = %if.then, %if.else
  %add2 = add i64 %livethrough, %lsize
  %cmp3 = icmp ult i64 %add2, %lsize
  br i1 %cmp3, label %loop, label %end

end:                                              ; preds = %if.end, %entry
  ret void
}

; The purpose of this test is to make sure we choose the correct incoming value
; for a boscc blend instruction.

; CHECK: spir_kernel void @__vecz_v4_nested_loops5
; CHECK: entry:
; CHECK: br i1 %{{.+}}, label %loop.preheader.uniform, label %entry.boscc_indir

; CHECK: loop.preheader.uniform:
; CHECK: br label %loop.uniform

; CHECK: entry.boscc_indir:
; CHECK: br i1 %{{.+}}, label %end, label %loop.preheader

; CHECK: loop.uniform:
; CHECK: %livethrough.uniform = phi i64 [ %add2.uniform, %if.end.uniform ], [ %lsize, %loop.preheader.uniform ]
; CHECK: br i1 %{{.+}}, label %if.then.preheader.uniform, label %if.else.uniform

; CHECK: if.then.preheader.uniform:
; CHECK: br label %if.then.uniform

; CHECK: if.then.uniform:
; CHECK: br i1 %{{.+}}, label %if.then.uniform, label %if.then.uniform.boscc_indir

; CHECK: if.then.uniform.boscc_indir:
; CHECK: br i1 %{{.+}}, label %if.end.loopexit.uniform, label %if.then.uniform.boscc_store

; CHECK: if.then.uniform.boscc_store:
;    LCSSA PHI nodes got cleaned up:
; CHECK-NOT: %{{.*\.boscc_lcssa.*}}
; CHECK: br label %if.then

; CHECK: loop.preheader:
; CHECK: br label %loop

; CHECK: loop:
; CHECK: %livethrough = phi i64 [ %add2, %if.end ], [ %lsize, %loop.preheader ]
; CHECK: br i1 %{{.+}}, label %if.then.preheader, label %if.else

; CHECK: if.then.preheader:
; CHECK: br label %if.then

; CHECK: if.then:
; CHECK: %livethrough.boscc_blend = phi i64 [ %livethrough.uniform, %if.then.uniform.boscc_store ], [ %livethrough.boscc_blend, %if.then ], [ %livethrough, %if.then.preheader ]
; CHECK: br i1 %{{.+}}, label %if.then, label %if.then.pure_exit

; CHECK: if.then.pure_exit:
; CHECK: br label %if.end.loopexit

; CHECK: if.else:
; CHECK: br label %if.end

; CHECK: if.end.loopexit:
; CHECK: br label %if.end

; CHECK: if.end:
; CHECK-NOT: %livethrough.boscc_blend{{.+}}.merge = phi i64 [ %livethrough.boscc_blend, %if.end.loopexit ], [ 0, %if.else ]
; CHECK: %livethrough.boscc_blend{{.+}} = phi i64 [ %livethrough.boscc_blend, %if.end.loopexit ], [ %livethrough, %if.else ]
