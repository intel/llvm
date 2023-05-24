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

; RUN: %veczc -k nested_loops4 -vecz-passes=vecz-loop-rotate,cfg-convert -vecz-choices=LinearizeBOSCC -S < %s | %filecheck %s

source_filename = "Unknown buffer"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

; Function Attrs: nounwind readnone
declare spir_func i64 @_Z13get_global_idj(i32) #0

; Function Attrs: nounwind readnone
declare spir_func i64 @_Z15get_global_sizej(i32) #0

; Function Attrs: nounwind readnone
declare spir_func float @_Z3dotDv2_fS_(<2 x float>, <2 x float>) #0

declare spir_func <2 x float> @_Z6vload2mPU3AS1Kf(i64, float addrspace(1)*)

; Function Attrs: nounwind readnone
declare spir_func i32 @_Z6mul_hijj(i32, i32) #0

define spir_kernel void @nested_loops4(i32 %n, float addrspace(1)* %out) {
entry:
  %gid = tail call spir_func i64 @_Z13get_global_idj(i32 0) #0
  %gsize = tail call spir_func i64 @_Z15get_global_sizej(i32 0) #0
  %trunc_gid = trunc i64 %gid to i32
  %trunc_gsize = trunc i64 %gsize to i32
  %cmp1 = icmp slt i32 %trunc_gid, %n
  br i1 %cmp1, label %for.cond1, label %end

for.cond1:                                     ; preds = %entry, %for.cond1.end
  %phi_trunc_gid = phi i32 [ %trunc_gid, %entry ], [ %add2, %for.cond1.end ]
  %mul_hi = tail call spir_func i32 @_Z6mul_hijj(i32 %phi_trunc_gid, i32 %n) #0
  %wrong = sdiv i32 %mul_hi, %n
  %sext_mul_hi = sext i32 %mul_hi to i64
  %gep1 = getelementptr inbounds float, float addrspace(1)* %out, i64 %sext_mul_hi
  %cmp2 = icmp slt i32 %mul_hi, %n
  br i1 %cmp2, label %for.cond2, label %for.cond1.end

for.cond2:                                    ; preds = %for.cond1, %for.cond2.end
  %phi4_fadd = phi float [ %phi3_fadd, %for.cond2.end ], [ 0.000000e+00, %for.cond1 ]
  %cmp3 = icmp slt i32 %mul_hi, %n
  br i1 %cmp3, label %for.cond3.preheader, label %for.cond2.end

for.cond3.preheader:                                    ; preds = %for.cond2
  %add1 = add nsw i32 %mul_hi, %wrong
  br label %for.cond3

for.cond3:                                    ; preds = %for.cond3.preheader, %for.cond3.end
  %phi_wrong_correct_correct = phi i32 [ %wrong, %for.cond3.preheader ], [ %correct, %for.cond3.end ]
  %phi_add1 = phi i32 [ %add1, %for.cond3.preheader ], [ %phi_add1, %for.cond3.end ]
  %phi2_fadd = phi float [ %phi4_fadd, %for.cond3.preheader ], [ %phi1_fadd, %for.cond3.end ]
  %cmp4 = icmp slt i32 %phi_wrong_correct_correct, %n
  br i1 %cmp4, label %for.cond3.body, label %for.cond3.end

for.cond3.body:                                    ; preds = %for.cond3
  %sext_phi_add1 = sext i32 %phi_add1 to i64
  %gep2 = getelementptr inbounds float, float addrspace(1)* %gep1, i64 %sext_phi_add1
  %vload = tail call spir_func <2 x float> @_Z6vload2mPU3AS1Kf(i64 0, float addrspace(1)* %gep2)
  %dot = tail call spir_func float @_Z3dotDv2_fS_(<2 x float> %vload, <2 x float> %vload) #0
  %fadd = fadd float %phi2_fadd, %dot
  br label %for.cond3.end

for.cond3.end:                                    ; preds = %for.cond3.body, %for.cond3
  %phi1_fadd = phi float [ %phi2_fadd, %for.cond3 ], [ %fadd, %for.cond3.body ]
  %correct = add nsw i32 %phi_wrong_correct_correct, 1
  %cmp5 = icmp slt i32 %wrong, %n
  br i1 %cmp5, label %for.cond3, label %for.cond2.end

for.cond2.end:                                    ; preds = %for.cond3.end, %for.cond2
  %phi3_fadd = phi float [ %phi4_fadd, %for.cond2 ], [ %phi1_fadd, %for.cond3.end ]
  %cmp6 = icmp slt i32 %mul_hi, %n
  br i1 %cmp6, label %for.cond2, label %for.cond1.end

for.cond1.end:                                    ; preds = %for.cond2.end, %for.cond1
  %ret = phi float [ 0.000000e+00, %for.cond1 ], [ %phi3_fadd, %for.cond2.end ]
  %sext_phi_trunc_gid = sext i32 %phi_trunc_gid to i64
  %gep3 = getelementptr inbounds float, float addrspace(1)* %out, i64 %sext_phi_trunc_gid
  store float %ret, float addrspace(1)* %gep3, align 4
  %add2 = add nsw i32 %phi_trunc_gid, %trunc_gsize
  %cmp7 = icmp slt i32 %add2, %n
  br i1 %cmp7, label %for.cond1, label %end

end:                                    ; preds = %for.cond1.end, %entry
  ret void
}

attributes #0 = { nounwind readnone }

; The purpose of this test is to make sure we choose the correct incoming value
; for a boscc blend instruction.

; CHECK: spir_kernel void @__vecz_v4_nested_loops4
; CHECK: entry:
; CHECK: br i1 %{{.+}}, label %for.cond1.preheader.uniform, label %entry.boscc_indir

; CHECK: for.cond1.preheader.uniform:
; CHECK: br label %for.cond1.uniform

; CHECK: entry.boscc_indir:
; CHECK: br i1 %{{.+}}, label %end, label %for.cond1.preheader

; CHECK: for.cond1.uniform:
; CHECK: %wrong.uniform = sdiv i32 %mul_hi.uniform, %n
; CHECK: br i1 %{{.+}}, label %for.cond2.preheader.uniform, label %for.cond1.uniform.boscc_indir

; CHECK: for.cond1.end.uniform.boscc_indir:
; CHECK: br i1 %{{.+}}, label %end.loopexit.uniform, label %for.cond1.end.uniform.boscc_store

; CHECK: for.cond1.end.uniform.boscc_store:
; CHECK: br label %for.cond1

; CHECK: for.cond2.preheader.uniform:
; CHECK: br label %for.cond2.uniform

; CHECK: for.cond1.uniform.boscc_indir:
; CHECK: br i1 %{{.+}}, label %for.cond1.end.uniform, label %for.cond1.uniform.boscc_store

; CHECK: for.cond1.uniform.boscc_store:
;    LCSSA PHI nodes got cleaned up:
; CHECK-NOT: %{{.*\.boscc_lcssa.*}}
; CHECK: br label %for.cond2.preheader

; CHECK: for.cond2.uniform:
; CHECK: br i1 %{{.+}}, label %for.cond3.preheader.uniform, label %for.cond2.uniform.boscc_indir

; CHECK: for.cond2.end.uniform.boscc_indir:
; CHECK: br i1 %{{.+}}, label %for.cond1.end.loopexit.uniform, label %for.cond2.end.uniform.boscc_store

; CHECK: for.cond3.preheader.uniform:
; CHECK: br label %for.cond3.uniform

; CHECK: for.cond2.uniform.boscc_indir:
; CHECK: br i1 %{{.+}}, label %for.cond2.end.uniform, label %for.cond2.uniform.boscc_store

; CHECK: for.cond3.uniform:
; CHECK: br i1 %{{.+}}, label %for.cond3.body.uniform, label %for.cond3.uniform.boscc_indir

; CHECK: for.cond3.end.uniform.boscc_indir:
; CHECK: br i1 %{{.+}}, label %for.cond2.end.loopexit.uniform, label %for.cond3.end.uniform.boscc_store

; CHECK: for.cond3.end.uniform.boscc_store:
;    LCSSA PHI nodes got cleaned up:
; CHECK-NOT: %{{.*\.boscc_lcssa.*}}
; CHECK: br label %for.cond3

; CHECK: for.cond3.body.uniform:
; CHECK: br label %for.cond3.end.uniform

; CHECK: for.cond3.uniform.boscc_indir:
; CHECK: %[[BOSCC:.+]] = call i1 @__vecz_b_divergence_all(i1 %for.cond3.end.uniform.exit_mask)
; CHECK: br i1 %[[BOSCC]], label %for.cond3.end.uniform, label %for.cond3.uniform.boscc_store

; CHECK: for.cond3.end.uniform:
; CHECK: br i1 %{{.+}}, label %for.cond3.uniform, label %for.cond3.end.uniform.boscc_indir

; CHECK: for.cond1.preheader:
; CHECK: br label %for.cond1

; CHECK: for.cond1:
; CHECK: br label %for.cond2.preheader

; CHECK: for.cond2.preheader:
; CHECK: br label %for.cond2

; CHECK: for.cond2:
; CHECK: br label %for.cond3.preheader

; CHECK: for.cond3.preheader:
; CHECK: br label %for.cond3

; CHECK: for.cond3:

; This is the important part of the test.
; CHECK: %phi_wrong_correct_correct = phi i32 [ %wrong.boscc_blend{{.+}}, %for.cond3.preheader ], [ %correct, %for.cond3.end ], [ %correct.uniform, %for.cond3.end.uniform.boscc_store ]
