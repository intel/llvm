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

; REQUIRES: llvm-13+
; RUN: %veczc -k dummy -vecz-simd-width=4 -vecz-passes=define-builtins -S < %s | %filecheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

define spir_kernel void @dummy(i32 addrspace(2)* %in, i32 addrspace(1)* %out) {
  ; Dummy uses of the builtins, as we don't define any with zero uses.
  %a = call <4 x i32> @__vecz_b_sub_group_scan_inclusive_add_vp_Dv4_jj(<4 x i32> zeroinitializer, i32 0)
  %b = call <4 x i32> @__vecz_b_sub_group_scan_exclusive_add_vp_Dv4_jj(<4 x i32> zeroinitializer, i32 0)
  %c = call <4 x float> @__vecz_b_sub_group_scan_inclusive_add_vp_Dv4_fj(<4 x float> zeroinitializer, i32 0)
  %d = call <4 x float> @__vecz_b_sub_group_scan_exclusive_add_vp_Dv4_fj(<4 x float> zeroinitializer, i32 0)
  %e = call <4 x i32> @__vecz_b_sub_group_scan_inclusive_smin_vp_Dv4_jj(<4 x i32> zeroinitializer, i32 0)
  %f = call <4 x i32> @__vecz_b_sub_group_scan_exclusive_smin_vp_Dv4_jj(<4 x i32> zeroinitializer, i32 0)
  %g = call <4 x i32> @__vecz_b_sub_group_scan_inclusive_smax_vp_Dv4_jj(<4 x i32> zeroinitializer, i32 0)
  %h = call <4 x i32> @__vecz_b_sub_group_scan_inclusive_umin_vp_Dv4_jj(<4 x i32> zeroinitializer, i32 0)
  %i = call <4 x i32> @__vecz_b_sub_group_scan_inclusive_umax_vp_Dv4_jj(<4 x i32> zeroinitializer, i32 0)
  %j = call <4 x float> @__vecz_b_sub_group_scan_inclusive_min_vp_Dv4_fj(<4 x float> zeroinitializer, i32 0)
  %k = call <4 x float> @__vecz_b_sub_group_scan_inclusive_max_vp_Dv4_fj(<4 x float> zeroinitializer, i32 0)
  %l = call <4 x float> @__vecz_b_sub_group_scan_exclusive_min_vp_Dv4_fj(<4 x float> zeroinitializer, i32 0)
  %m = call <4 x float> @__vecz_b_sub_group_scan_exclusive_max_vp_Dv4_fj(<4 x float> zeroinitializer, i32 0)
  ret void
}

declare <4 x i32> @__vecz_b_sub_group_scan_inclusive_add_vp_Dv4_jj(<4 x i32>, i32)
; CHECK-LABEL: define <4 x i32> @__vecz_b_sub_group_scan_inclusive_add_vp_Dv4_jj(<4 x i32>{{.*}}, i32{{.*}}) {
; CHECK: entry:
; CHECK:   %[[SHUFFLE_ALLOC:.+]] = alloca <4 x i32>
; CHECK:   br label %loop
; CHECK: loop:
; CHECK:   %[[IV:.+]] = phi i32 [ 1, %entry ], [ %[[N2:.+]], %loop ]
; CHECK:   %[[VEC:.+]] = phi <4 x i32> [ %0, %entry ], [ %[[NEWVEC:.+]], %loop ]
; CHECK:   %[[MASKPHI:.+]] = phi <4 x i32> [ <i32 0, i32 1, i32 2, i32 3>, %entry ], [ %[[NEWMASK:.+]], %loop ]
; CHECK:   %[[N_INS:.+]] = insertelement <4 x i32> poison, i32 %[[IV]], {{i32|i64}} 0
; CHECK:   %[[N_SPLAT:.+]] = shufflevector <4 x i32> %[[N_INS]], <4 x i32> poison, <4 x i32> zeroinitializer
; CHECK:   %[[MASK:.+]] = xor <4 x i32> %[[MASKPHI]], %[[N_SPLAT]]

;------- target-dependent dynamic shuffle code:
; CHECK:   store <4 x i32> %[[VEC]], {{(<4 x i32>\*)|(ptr)}} %[[SHUFFLE_ALLOC]]
;------- there will be a bitcast here if pointers are typed
; CHECK:   %[[INDEX:.+]] = getelementptr inbounds i32, [[PTRTY:(i32\*)|ptr]] %{{.+}}, <4 x i32> %[[MASK]]
; CHECK:   %[[VLINS:.+]] = insertelement <4 x i32> poison, i32 %1, {{i32|i64}} 0
; CHECK:   %[[VLSPLAT:.+]] = shufflevector <4 x i32> %[[VLINS]], <4 x i32> poison, <4 x i32> zeroinitializer
; CHECK:   %[[VLMASK:.+]] = icmp ult <4 x i32> <i32 0, i32 1, i32 2, i32 3>, %[[VLSPLAT]]
; CHECK:   %[[SHUFFLE:.+]] = call <4 x i32> @llvm.masked.gather.v4i32.v4p0{{(i32)?}}(<4 x [[PTRTY]]> %[[INDEX]], i32 4, <4 x i1> %[[VLMASK]], <4 x i32> undef)

; CHECK:   %[[ACCUM:.+]] = add <4 x i32> %[[VEC]], %{{.+}}
; CHECK:   %[[BIT:.+]] = and <4 x i32> %[[MASKPHI]], %[[N_SPLAT]]
; CHECK:   %[[WHICH:.+]] = icmp ne <4 x i32> %[[BIT]], zeroinitializer
; CHECK:   %[[NEWVEC]] = select <4 x i1> %[[WHICH]], <4 x i32> %[[ACCUM]], <4 x i32> %[[VEC]]
; CHECK:   %[[NEWMASK]] = or <4 x i32> %[[MASK]], %[[N_SPLAT]]
; CHECK:   %[[N2]] = shl nuw nsw i32 %[[IV]], 1
; CHECK:   %[[CMP:.+]] = icmp ult i32 %[[N2]], %1
; CHECK:   br i1 %[[CMP]], label %loop, label %exit
; CHECK: exit:
; CHECK:   %[[RESULT:.+]] = phi <4 x i32> [ %[[NEWVEC]], %loop ]
; CHECK:   ret <4 x i32> %[[RESULT]]
; CHECK: }

declare <4 x i32> @__vecz_b_sub_group_scan_exclusive_add_vp_Dv4_jj(<4 x i32>, i32)
; CHECK-LABEL: define <4 x i32> @__vecz_b_sub_group_scan_exclusive_add_vp_Dv4_jj(<4 x i32>{{.*}}, i32{{.*}}) {
; CHECK: entry:
; CHECK:   %[[SHUFFLE_ALLOC:.+]] = alloca <4 x i32>
; CHECK:   br label %loop
; CHECK: loop:
; CHECK:   %[[IV:.+]] = phi i32 [ 1, %entry ], [ %[[N2:.+]], %loop ]
; CHECK:   %[[VEC:.+]] = phi <4 x i32> [ %0, %entry ], [ %[[NEWVEC:.+]], %loop ]
; CHECK:   %[[MASKPHI:.+]] = phi <4 x i32> [ <i32 0, i32 1, i32 2, i32 3>, %entry ], [ %[[NEWMASK:.+]], %loop ]
; CHECK:   %[[N_INS:.+]] = insertelement <4 x i32> poison, i32 %[[IV]], {{i32|i64}} 0
; CHECK:   %[[N_SPLAT:.+]] = shufflevector <4 x i32> %[[N_INS]], <4 x i32> poison, <4 x i32> zeroinitializer
; CHECK:   %[[MASK:.+]] = xor <4 x i32> %[[MASKPHI]], %[[N_SPLAT]]

;------- target-dependent dynamic shuffle code:
; CHECK:   store <4 x i32> %[[VEC]], {{(<4 x i32>\*)|(ptr)}} %[[SHUFFLE_ALLOC]]
;------- there will be a bitcast here if pointers are typed
; CHECK:   %[[INDEX:.+]] = getelementptr inbounds i32, [[PTRTY:(i32\*)|ptr]] %{{.+}}, <4 x i32> %[[MASK]]
; CHECK:   %[[VLINS:.+]] = insertelement <4 x i32> poison, i32 %1, {{i32|i64}} 0
; CHECK:   %[[VLSPLAT:.+]] = shufflevector <4 x i32> %[[VLINS]], <4 x i32> poison, <4 x i32> zeroinitializer
; CHECK:   %[[VLMASK:.+]] = icmp ult <4 x i32> <i32 0, i32 1, i32 2, i32 3>, %[[VLSPLAT]]
; CHECK:   %[[SHUFFLE:.+]] = call <4 x i32> @llvm.masked.gather.v4i32.v4p0{{(i32)?}}(<4 x [[PTRTY]]> %[[INDEX]], i32 4, <4 x i1> %[[VLMASK]], <4 x i32> undef)

; CHECK:   %[[ACCUM:.+]] = add <4 x i32> %[[VEC]], %{{.+}}
; CHECK:   %[[BIT:.+]] = and <4 x i32> %[[MASKPHI]], %[[N_SPLAT]]
; CHECK:   %[[WHICH:.+]] = icmp ne <4 x i32> %[[BIT]], zeroinitializer
; CHECK:   %[[NEWVEC]] = select <4 x i1> %[[WHICH]], <4 x i32> %[[ACCUM]], <4 x i32> %[[VEC]]
; CHECK:   %[[NEWMASK]] = or <4 x i32> %[[MASK]], %[[N_SPLAT]]
; CHECK:   %[[N2]] = shl nuw nsw i32 %[[IV]], 1
; CHECK:   %[[CMP:.+]] = icmp ult i32 %[[N2]], %1
; CHECK:   br i1 %[[CMP]], label %loop, label %exit
; CHECK: exit:
; CHECK:   %[[SCAN:.+]] = phi <4 x i32> [ %[[NEWVEC]], %loop ]

;------- target-dependent slide-up goes here
; CHECK:  %[[SLIDE:.+]] = shufflevector <4 x i32> %[[SCAN]], <4 x i32> undef, <4 x i32> <i32 {{[0-9]+}}, i32 0, i32 1, i32 2>
; CHECK:  %[[RESULT:.+]] = insertelement <4 x i32> %[[SLIDE]], i32 0, {{i32|i64}} 0

; CHECK:   ret <4 x i32> %[[RESULT]]
; CHECK: }


; We know the generated code is correct for one scan type,
; now verify that all the others use the correct binary operations.

declare <4 x float> @__vecz_b_sub_group_scan_inclusive_add_vp_Dv4_fj(<4 x float>, i32)
; CHECK-LABEL: define <4 x float> @__vecz_b_sub_group_scan_inclusive_add_vp_Dv4_fj(<4 x float>{{.*}}, i32{{.*}})
; CHECK: loop:
; CHECK:   %[[VEC:.+]] = phi <4 x float> [ %0, %entry ],
; CHECK:   %{{.+}} = fadd <4 x float> %[[VEC]], %{{.+}}

declare <4 x float> @__vecz_b_sub_group_scan_exclusive_add_vp_Dv4_fj(<4 x float>, i32)
; CHECK-LABEL: define <4 x float> @__vecz_b_sub_group_scan_exclusive_add_vp_Dv4_fj(<4 x float>{{.*}}, i32{{.*}})
; CHECK: loop:
; CHECK:   %[[VEC:.+]] = phi <4 x float> [ %0, %entry ],
; CHECK:   %{{.+}} = fadd <4 x float> %[[VEC]], %{{.+}}

declare <4 x i32> @__vecz_b_sub_group_scan_inclusive_smin_vp_Dv4_jj(<4 x i32>, i32)
; CHECK-LABEL: define <4 x i32> @__vecz_b_sub_group_scan_inclusive_smin_vp_Dv4_jj(<4 x i32>{{.*}}, i32{{.*}})
; CHECK: loop:
; CHECK:   %[[VEC:.+]] = phi <4 x i32> [ %0, %entry ],
; CHECK:   %{{.+}} = call <4 x i32> @llvm.smin.v4i32(<4 x i32> %[[VEC]], <4 x i32> %{{.+}})

declare <4 x i32> @__vecz_b_sub_group_scan_exclusive_smin_vp_Dv4_jj(<4 x i32>, i32)
; CHECK-LABEL: define <4 x i32> @__vecz_b_sub_group_scan_exclusive_smin_vp_Dv4_jj(<4 x i32>{{.*}}, i32{{.*}})
; CHECK: loop:
; CHECK:   %[[VEC:.+]] = phi <4 x i32> [ %0, %entry ],
; CHECK:   %{{.+}} = call <4 x i32> @llvm.smin.v4i32(<4 x i32> %[[VEC]], <4 x i32> %{{.+}})

declare <4 x i32> @__vecz_b_sub_group_scan_inclusive_smax_vp_Dv4_jj(<4 x i32>, i32)
; CHECK-LABEL: define <4 x i32> @__vecz_b_sub_group_scan_inclusive_smax_vp_Dv4_jj(<4 x i32>{{.*}}, i32{{.*}})
; CHECK: loop:
; CHECK:   %[[VEC:.+]] = phi <4 x i32> [ %0, %entry ],
; CHECK:   %{{.+}} = call <4 x i32> @llvm.smax.v4i32(<4 x i32> %[[VEC]], <4 x i32> %{{.+}})

declare <4 x i32> @__vecz_b_sub_group_scan_inclusive_umin_vp_Dv4_jj(<4 x i32>, i32)
; CHECK-LABEL: define <4 x i32> @__vecz_b_sub_group_scan_inclusive_umin_vp_Dv4_jj(<4 x i32>{{.*}}, i32{{.*}})
; CHECK: loop:
; CHECK:   %[[VEC:.+]] = phi <4 x i32> [ %0, %entry ],
; CHECK:   %{{.+}} = call <4 x i32> @llvm.umin.v4i32(<4 x i32> %[[VEC]], <4 x i32> %{{.+}})

declare <4 x i32> @__vecz_b_sub_group_scan_inclusive_umax_vp_Dv4_jj(<4 x i32>, i32)
; CHECK-LABEL: define <4 x i32> @__vecz_b_sub_group_scan_inclusive_umax_vp_Dv4_jj(<4 x i32>{{.*}}, i32{{.*}})
; CHECK: loop:
; CHECK:   %[[VEC:.+]] = phi <4 x i32> [ %0, %entry ],
; CHECK:   %{{.+}} = call <4 x i32> @llvm.umax.v4i32(<4 x i32> %[[VEC]], <4 x i32> %{{.+}})

declare <4 x float> @__vecz_b_sub_group_scan_inclusive_min_vp_Dv4_fj(<4 x float>, i32)
; CHECK-LABEL: define <4 x float> @__vecz_b_sub_group_scan_inclusive_min_vp_Dv4_fj(<4 x float>{{.*}}, i32{{.*}})
; CHECK: loop:
; CHECK:   %[[VEC:.+]] = phi <4 x float> [ %0, %entry ],
; CHECK:   %{{.+}} = call <4 x float> @llvm.minnum.v4f32(<4 x float> %[[VEC]], <4 x float> %{{.+}})

declare <4 x float> @__vecz_b_sub_group_scan_inclusive_max_vp_Dv4_fj(<4 x float>, i32)
; CHECK-LABEL: define <4 x float> @__vecz_b_sub_group_scan_inclusive_max_vp_Dv4_fj(<4 x float>{{.*}}, i32{{.*}})
; CHECK: loop:
; CHECK:   %[[VEC:.+]] = phi <4 x float> [ %0, %entry ],
; CHECK:   %{{.+}} = call <4 x float> @llvm.maxnum.v4f32(<4 x float> %[[VEC]], <4 x float> %{{.+}})

declare <4 x float> @__vecz_b_sub_group_scan_exclusive_min_vp_Dv4_fj(<4 x float>, i32)
; CHECK-LABEL: define <4 x float> @__vecz_b_sub_group_scan_exclusive_min_vp_Dv4_fj(<4 x float>{{.*}}, i32{{.*}})
; CHECK: loop:
; CHECK:   %[[VEC:.+]] = phi <4 x float> [ %0, %entry ],
; CHECK:   %{{.+}} = call <4 x float> @llvm.minnum.v4f32(<4 x float> %[[VEC]], <4 x float> %{{.+}})

declare <4 x float> @__vecz_b_sub_group_scan_exclusive_max_vp_Dv4_fj(<4 x float>, i32)
; CHECK-LABEL: define <4 x float> @__vecz_b_sub_group_scan_exclusive_max_vp_Dv4_fj(<4 x float>{{.*}}, i32{{.*}})
; CHECK: loop:
; CHECK:   %[[VEC:.+]] = phi <4 x float> [ %0, %entry ],
; CHECK:   %{{.+}} = call <4 x float> @llvm.maxnum.v4f32(<4 x float> %[[VEC]], <4 x float> %{{.+}})
