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

; RUN: veczc -k boscc_merge -vecz-passes="function(instcombine),function(simplifycfg),mergereturn,vecz-loop-rotate,function(loop(indvars)),cfg-convert,cleanup-divergence" -vecz-choices=LinearizeBOSCC -S < %s | FileCheck %s

; ModuleID = 'Unknown buffer'
source_filename = "Unknown buffer"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

declare i64 @__mux_get_local_id(i32) #0
declare i64 @__mux_get_local_size(i32) #0

define spir_kernel void @boscc_merge(i32 %n, float addrspace(1)* %out, i64 %x) {
entry:
  %lid = tail call i64 @__mux_get_local_id(i32 0)
  %lsize = tail call i64 @__mux_get_local_size(i32 0)
  %out_ptr = getelementptr inbounds float, float addrspace(1)* %out, i64 %x
  %lid_sum_lsize = add i64 %lid, %lsize
  %cmp1 = icmp ult i64 %lsize, %x
  br i1 %cmp1, label %if.then, label %if.else

if.then:                                              ; preds = %entry
  %cmp2 = icmp ult i64 %lid, %x
  br i1 %cmp2, label %if.then2, label %if.else2.preheader

if.else2.preheader:                                   ; preds = %if.then
  store float 0.000000e+00, float addrspace(1)* %out_ptr, align 4 ; just so it's non-trivial for BOSCC
  br label %if.else2

if.then2:                                              ; preds = %if.then
  %cmp3 = icmp ugt i64 %lsize, %x
  br i1 %cmp3, label %if.then3.preheader, label %if.else3.preheader

if.else3.preheader:                                    ; preds = %if.then2
  br label %if.else3

if.then3.preheader:                                    ; preds = %if.then2
  br label %if.then3

if.then3:                                              ; preds = %if.then3.preheader, %if.else5
  %cmp4 = icmp ugt i64 %lid, %x
  br i1 %cmp4, label %if.then4.preheader, label %if.else4.preheader

if.else4.preheader:                                    ; preds = %if.then3
  br label %if.else4

if.then4.preheader:                                    ; preds = %if.then3
  br label %if.then4

if.else4:                                              ; preds = %if.else4.preheader, %if.else4
  %cmp5 = icmp ult i64 %lid, %x
  br i1 %cmp5, label %if.else4, label %if.else5.loopexit1

if.else5.loopexit:                                     ; preds = %if.then4
  br label %if.else5

if.else5.loopexit1:                                    ; preds = %if.else4
  br label %if.else5

if.else5:                                              ; preds = %if.else5.loopexit1, %if.else5.loopexit
  %cmp6 = icmp ult i64 %lid, %x
  br i1 %cmp6, label %if.then3, label %if.else.loopexit

if.then4:                                              ; preds = %if.then4.preheader, %if.then4
  %cmp7 = icmp ult i64 %lid_sum_lsize, %x
  br i1 %cmp7, label %if.then4, label %if.else5.loopexit

if.else3:                                              ; preds = %if.else3.preheader, %if.else3
  %cmp8 = icmp ult i64 %lid_sum_lsize, %x
  br i1 %cmp8, label %if.else3, label %if.else.loopexit2

if.else2:                                             ; preds = %if.else2.preheader, %if.else2
  %cmp9 = icmp ult i64 %lid_sum_lsize, %x
  br i1 %cmp9, label %if.else2, label %if.else.loopexit3

if.else.loopexit:                                    ; preds = %if.else5
  br label %if.else

if.else.loopexit2:                                   ; preds = %if.else3
  br label %if.else

if.else.loopexit3:                                   ; preds = %if.else2
  br label %if.else

if.else:                                             ; preds = %if.else.loopexit3, %if.else.loopexit2, %if.else.loopexit, %entry
  %cmp10 = icmp ult i64 %lid, %x
  br i1 %cmp10, label %if.then5, label %if.else6

if.then5:                                             ; preds = %if.else
  %cmp11 = icmp eq i64 %x, 0
  br i1 %cmp11, label %if.then6, label %if.else7

if.else7:                                             ; preds = %if.then5
  %load = load float, float addrspace(1)* %out, align 4
  br label %if.then6

if.then6:                                             ; preds = %if.else7, %if.then5
  %ret = phi float [ 0.000000e+00, %if.then5 ], [ %load, %if.else7 ]
  store float %ret, float addrspace(1)* %out_ptr, align 4
  br label %if.else6

if.else6:                                             ; preds = %if.then6, %if.else
  ret void
}

; CHECK: spir_kernel void @__vecz_v4_boscc_merge
; CHECK: %[[CMP1:.+]] = icmp
; CHECK:  br i1 %[[CMP1]], label %[[IFTHEN:.+]], label %[[IFELSE:.+]]

; CHECK: [[IFTHEN]]:
; CHECK: %[[CMP2:.+]] = icmp
; CHECK: br i1 %{{.+}}, label %[[IFTHEN2UNIFORM:.+]], label %[[IFTHENBOSCCINDIR:.+]]

; CHECK: [[IFELSE2PREHEADERUNIFORM:.+]]:
; CHECK: br label %[[IFELSE2UNIFORM:.+]]

; CHECK: [[IFELSE2UNIFORM]]:
; CHECK:  br i1 %{{.+}}, label %[[IFELSE2UNIFORM]], label %[[IFELSE2UNIFORMBOSCCINDIR:.+]]

; CHECK: [[IFELSE2UNIFORMBOSCCINDIR]]:
; CHECK:  br i1 %{{.+}}, label %[[IFELSELOOPEXIT3UNIFORM:.+]], label %[[IFELSE2UNIFORMBOSCCSTORE:.+]]

; CHECK: [[IFELSE2UNIFORMBOSCCSTORE]]:
; CHECK:  br label %[[IFELSE2:.+]]

; CHECK: [[IFELSELOOPEXIT3UNIFORM]]:
; CHECK: br label %[[IFELSEUNIFORM:.+]]

; CHECK: [[IFTHEN2UNIFORM]]:
; CHECK: %[[CMP3UNIFORM:.+]] = icmp
; CHECK: br i1 %[[CMP3UNIFORM]], label %[[IFTHEN3PREHEADERUNIFORM:.+]], label %[[IFELSE3PREHEADERUNIFORM:.+]]

; CHECK: [[IFTHENBOSCCINDIR]]:
; CHECK: br i1 %{{.+}}, label %[[IFELSE2PREHEADERUNIFORM]], label %[[IFELSE2PREHEADER:.+]]

; CHECK: [[IFELSE3PREHEADERUNIFORM]]:
; CHECK: br label %[[IFELSE3UNIFORM:.+]]

; CHECK: [[IFELSE3UNIFORM]]:
; CHECK: br i1 %{{.+}}, label %[[IFELSE3UNIFORM]], label %[[IFELSE3UNIFORMBOSCCINDIR:.+]]

; CHECK: [[IFELSE3UNIFORMBOSCCINDIR]]:
; CHECK: br i1 %{{.+}}, label %[[IFELSELOOPEXIT2UNIFORM:.+]], label %[[IFELSE3UNIFORMBOSCCSTORE:.+]]

; CHECK: [[IFELSE3UNIFORMBOSCCSTORE]]:
; CHECK: br label %[[IFELSE3:.+]]

; CHECK: [[IFELSELOOPEXIT2UNIFORM]]:
; CHECK: br label %[[IFELSEUNIFORM]]

; CHECK: [[IFTHEN3PREHEADERUNIFORM]]:
; CHECK: br label %[[IFTHEN3UNIFORM:.+]]

; CHECK: [[IFTHEN3UNIFORM]]:
; CHECK: br i1 %{{.+}}, label %[[IFTHEN4PREHEADERUNIFORM:.+]], label %[[IFTHEN3UNIFORMBOSCCINDIR:.+]]

; CHECK: [[IFELSE5UNIFORMBOSCCINDIR:.+]]:
; CHECK: br i1 %{{.+}}, label %[[IFELSELOOPEXITUNIFORM:.+]], label %[[IFELSE5UNIFORMBOSCCSTORE:.+]]

; CHECK: [[IFELSE5UNIFORMBOSCCSTORE]]:
; CHECK: br label %[[IFTHEN3:.+]]

; CHECK: [[IFELSE4PREHEADERUNIFORM:.+]]:
; CHECK: br label %[[IFELSE4UNIFORM:.+]]

; CHECK: [[IFELSE4UNIFORM]]:
; CHECK: br i1 %{{.+}}, label %[[IFELSE4UNIFORM]], label %[[IFELSE4UNIFORMBOSCCINDIR:.+]]

; CHECK: [[IFELSE4UNIFORMBOSCCINDIR]]:
; CHECK: br i1 %{{.+}}, label %[[IFELSE5LOOPEXIT1UNIFORM:.+]], label %[[IFELSE4UNIFORMBOSCCSTORE:.+]]

; CHECK: [[IFELSE4UNIFORMBOSCCSTORE]]:
; CHECK: br label %[[IFELSE4:.+]]

; CHECK: [[IFELSE5LOOPEXIT1UNIFORM]]:
; CHECK: br label %[[IFELSE5UNIFORM:.+]]

; CHECK: [[IFTHEN4PREHEADERUNIFORM]]:
; CHECK: br label %[[IFTHEN4UNIFORM:.+]]

; CHECK: [[IFTHEN3UNIFORMBOSCCINDIR]]:
; CHECK: br i1 %{{.+}}, label %[[IFELSE4PREHEADERUNIFORM]], label %[[IFTHEN3UNIFORMBOSCCSTORE:.+]]

; CHECK: [[IFTHEN3UNIFORMBOSCCSTORE]]:
; CHECK: br label %[[IFELSE4PREHEADER:.+]]

; CHECK: [[IFTHEN4UNIFORM]]:
; CHECK: br i1 %{{.+}}, label %[[IFTHEN4UNIFORM]], label %[[IFTHEN4UNIFORMBOSCCINDIR:.+]]

; CHECK: [[IFTHEN4UNIFORMBOSCCINDIR]]:
; CHECK: br i1 %{{.+}}, label %[[IFELSE5LOOPEXITUNIFORM:.+]], label %[[IFTHEN4UNIFORMBOSCCSTORE:.+]]

; CHECK: [[IFTHEN4UNIFORMBOSCCSTORE]]:
; CHECK: br label %[[IFTHEN4:.+]]

; CHECK: [[IFELSE5LOOPEXITUNIFORM]]:
; CHECK: br label %[[IFELSE5UNIFORM]]

; CHECK: [[IFELSE5UNIFORM]]:
; CHECK: br i1 %{{.+}}, label %[[IFTHEN3UNIFORM]], label %[[IFELSE5UNIFORMBOSCCINDIR]]

; CHECK: [[IFELSELOOPEXITUNIFORM]]:
; CHECK: br label %[[IFELSE]]

; CHECK: [[IFELSE2PREHEADER]]:
; CHECK: br label %[[IFELSE2]]

; CHECK: [[IFTHEN2:.+]]:
; CHECK: %[[CMP3:.+]] = icmp
; FIXME: We shouldn't need to mask this comparison, as it's truly uniform even
; on inactive lanes.
; CHECK: %[[CMP3_ACTIVE:.+]] = and i1 %[[CMP3]], %[[CMP2]]
; CHECK: %[[CMP3_ACTIVE_ANY:.+]] = call i1 @__vecz_b_divergence_any(i1 %[[CMP3_ACTIVE]])
; CHECK: br i1 %[[CMP3_ACTIVE_ANY]], label %[[IFTHEN3PREHEADER:.+]], label %[[IFELSE3PREHEADER:.+]]

; CHECK: [[IFELSE3PREHEADER]]:
; CHECK: br label %[[IFELSE3]]

; CHECK: [[IFTHEN3PREHEADER]]:
; CHECK: br label %[[IFTHEN3]]

; CHECK: [[IFTHEN3]]:
; CHECK: br label %[[IFELSE4PREHEADER]]

; CHECK: [[IFELSE4PREHEADER]]:
; CHECK: br label %[[IFELSE4]]

; CHECK: [[IFTHEN4PREHEADER:.+]]:
; CHECK: br label %[[IFTHEN4]]

; CHECK: [[IFELSE4]]:
; CHECK: br i1 %{{.+}}, label %[[IFELSE4]], label %[[IFELSE4PUREEXIT:.+]]

; CHECK: [[IFELSE4PUREEXIT]]:
; CHECK: br label %[[IFELSE5LOOPEXIT1:.+]]

; CHECK: [[IFELSE5LOOPEXIT:.+]]:
; CHECK: br label %[[IFELSE5:.+]]

; CHECK: [[IFELSE5LOOPEXIT1]]:
; CHECK: br label %[[IFTHEN4PREHEADER]]

; CHECK: [[IFELSE5]]:
; CHECK: br i1 %{{.+}}, label %[[IFTHEN3]], label %[[IFTHEN3PUREEXIT:.+]]

; CHECK: [[IFTHEN3PUREEXIT]]:
; CHECK: br label %[[IFELSELOOPEXIT:.+]]

; CHECK: [[IFTHEN4]]:
; CHECK: br i1 %{{.+}}, label %[[IFTHEN4]], label %[[IFTHEN4PUREEXIT:.+]]

; CHECK: [[IFTHEN4PUREEXIT]]:
; CHECK: br label %[[IFELSE5LOOPEXIT]]

; CHECK: [[IFELSE3]]:
; CHECK: br i1 %{{.+}}, label %[[IFELSE3]], label %[[IFELSE3PUREEXIT:.+]]

; CHECK: [[IFELSE3PUREEXIT]]:
; CHECK: br label %[[IFELSELOOPEXIT2:.+]]

; CHECK: [[IFELSE2]]:
; CHECK: br i1 %{{.+}}, label %[[IFELSE2]], label %[[IFELSE2PUREEXIT:.+]]

; CHECK: [[IFELSE2PUREEXIT]]:
; CHECK: br label %[[IFELSELOOPEXIT3:.+]]

; CHECK: [[IFELSELOOPEXIT]]:
; CHECK: br label %[[IFELSE]]

; CHECK: [[IFELSELOOPEXIT2]]:
; CHECK: br label %[[IFELSE]]

; CHECK: [[IFELSELOOPEXIT3]]:
; CHECK: br label %[[IFTHEN2]]

; CHECK: [[IFELSE]]:
; CHECK: br i1 %{{.+}}, label %[[IFELSE7UNIFORM:.+]], label %[[IFELSEUNIFORMBOSCCINDIR:.+]]

; CHECK: [[IFELSE7UNIFORM]]:
; CHECK: br label %[[IFELSE6:.+]]

; CHECK: [[IFELSEUNIFORMBOSCCINDIR]]:
; CHECK: br i1 %{{.+}}, label %[[IFELSE6]], label %[[IFELSE7:.+]]

; CHECK: [[IFELSE7]]:
; CHECK: br label %[[IFELSE6]]

; CHECK: [[IFELSE6]]:
; CHECK:  ret void
