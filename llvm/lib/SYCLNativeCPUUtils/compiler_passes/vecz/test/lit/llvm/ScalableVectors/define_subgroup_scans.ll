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
; RUN: veczc -k dummy -vecz-scalable -vecz-simd-width=4 -vecz-passes=define-builtins -S < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

define spir_kernel void @dummy(i32 addrspace(2)* %in, i32 addrspace(1)* %out) {
  ; Dummy uses of the builtins, as we don't define any with zero uses.
  %a = call <vscale x 4 x i32> @__vecz_b_sub_group_scan_inclusive_add_u5nxv4j(<vscale x 4 x i32> zeroinitializer)
  %b = call <vscale x 4 x i32> @__vecz_b_sub_group_scan_exclusive_add_u5nxv4j(<vscale x 4 x i32> zeroinitializer)
  %c = call <vscale x 4 x float> @__vecz_b_sub_group_scan_inclusive_add_u5nxv4f(<vscale x 4 x float> zeroinitializer)
  %d = call <vscale x 4 x float> @__vecz_b_sub_group_scan_exclusive_add_u5nxv4f(<vscale x 4 x float> zeroinitializer)
  %e = call <vscale x 4 x i32> @__vecz_b_sub_group_scan_inclusive_smin_u5nxv4j(<vscale x 4 x i32> zeroinitializer)
  %f = call <vscale x 4 x i32> @__vecz_b_sub_group_scan_exclusive_smin_u5nxv4j(<vscale x 4 x i32> zeroinitializer)
  %g = call <vscale x 4 x i32> @__vecz_b_sub_group_scan_inclusive_smax_u5nxv4j(<vscale x 4 x i32> zeroinitializer)
  %h = call <vscale x 4 x i32> @__vecz_b_sub_group_scan_inclusive_umin_u5nxv4j(<vscale x 4 x i32> zeroinitializer)
  %i = call <vscale x 4 x i32> @__vecz_b_sub_group_scan_inclusive_umax_u5nxv4j(<vscale x 4 x i32> zeroinitializer)
  %j = call <vscale x 4 x float> @__vecz_b_sub_group_scan_inclusive_min_u5nxv4f(<vscale x 4 x float> zeroinitializer)
  %k = call <vscale x 4 x float> @__vecz_b_sub_group_scan_inclusive_max_u5nxv4f(<vscale x 4 x float> zeroinitializer)
  %l = call <vscale x 4 x float> @__vecz_b_sub_group_scan_exclusive_min_u5nxv4f(<vscale x 4 x float> zeroinitializer)
  %m = call <vscale x 4 x float> @__vecz_b_sub_group_scan_exclusive_max_u5nxv4f(<vscale x 4 x float> zeroinitializer)
  ret void
}

declare <vscale x 4 x i32> @__vecz_b_sub_group_scan_inclusive_add_u5nxv4j(<vscale x 4 x i32>)
; CHECK-LABEL: define <vscale x 4 x i32> @__vecz_b_sub_group_scan_inclusive_add_u5nxv4j(<vscale x 4 x i32>{{.*}}) {
; CHECK: entry:
; CHECK:   %[[SHUFFLE_ALLOC:.+]] = alloca <vscale x 4 x i32>
; CHECK:   %[[STEP:.+]] = call <vscale x 4 x i32> @llvm.{{(experimental\.)?}}stepvector.nxv4i32()
; CHECK:   %[[SCALE:.+]] = call i32 @llvm.vscale.i32()
; CHECK:   %[[SIZE:.+]] = mul i32 %[[SCALE]], 4
; CHECK:   br label %loop
; CHECK: loop:
; CHECK:   %[[IV:.+]] = phi i32 [ 1, %entry ], [ %[[N2:.+]], %loop ]
; CHECK:   %[[VEC:.+]] = phi <vscale x 4 x i32> [ %0, %entry ], [ %[[NEWVEC:.+]], %loop ]
; CHECK:   %[[MASKPHI:.+]] = phi <vscale x 4 x i32> [ %[[STEP]], %entry ], [ %[[NEWMASK:.+]], %loop ]
; CHECK:   %[[N_INS:.+]] = insertelement <vscale x 4 x i32> poison, i32 %[[IV]], {{i32|i64}} 0
; CHECK:   %[[N_SPLAT:.+]] = shufflevector <vscale x 4 x i32> %[[N_INS]], <vscale x 4 x i32> poison, <vscale x 4 x i32> zeroinitializer
; CHECK:   %[[MASK:.+]] = xor <vscale x 4 x i32> %[[MASKPHI]], %[[N_SPLAT]]

;------- target-dependent dynamic shuffle code:
; CHECK:   store <vscale x 4 x i32> %[[VEC]], {{(<vscale x 4 x i32>\*)|(ptr)}} %[[SHUFFLE_ALLOC]]
;------- there will be a bitcast here if pointers are typed
; CHECK:   %[[INDEX:.+]] = getelementptr inbounds i32, [[PTRTY:(i32\*)|ptr]] %{{.+}}, <vscale x 4 x i32> %[[MASK]]
; CHECK:   %[[SHUFFLE:.+]] = call <vscale x 4 x i32> @llvm.masked.gather.nxv4i32.nxv4p0{{(i32)?}}(<vscale x 4 x [[PTRTY]]> %[[INDEX]], i32 4, <vscale x 4 x i1> shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, {{i32|i64}} 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer), <vscale x 4 x i32> undef)

; CHECK:   %[[ACCUM:.+]] = add <vscale x 4 x i32> %[[VEC]], %{{.+}}
; CHECK:   %[[BIT:.+]] = and <vscale x 4 x i32> %[[MASKPHI]], %[[N_SPLAT]]
; CHECK:   %[[WHICH:.+]] = icmp ne <vscale x 4 x i32> %[[BIT]], zeroinitializer
; CHECK:   %[[NEWVEC]] = select <vscale x 4 x i1> %[[WHICH]], <vscale x 4 x i32> %[[ACCUM]], <vscale x 4 x i32> %[[VEC]]
; CHECK:   %[[NEWMASK]] = or <vscale x 4 x i32> %[[MASK]], %[[N_SPLAT]]
; CHECK:   %[[N2]] = shl nuw nsw i32 %[[IV]], 1
; CHECK:   %[[CMP:.+]] = icmp ult i32 %[[N2]], %[[SIZE]]
; CHECK:   br i1 %[[CMP]], label %loop, label %exit
; CHECK: exit:
; CHECK:   %[[RESULT:.+]] = phi <vscale x 4 x i32> [ %[[NEWVEC]], %loop ]
; CHECK:   ret <vscale x 4 x i32> %[[RESULT]]
; CHECK: }

declare <vscale x 4 x i32> @__vecz_b_sub_group_scan_exclusive_add_u5nxv4j(<vscale x 4 x i32>)
; CHECK-LABEL: define <vscale x 4 x i32> @__vecz_b_sub_group_scan_exclusive_add_u5nxv4j(<vscale x 4 x i32>{{.*}}) {
; CHECK: entry:
; CHECK:   %[[SHUFFLE_ALLOC:.+]] = alloca <vscale x 4 x i32>
; CHECK:   %[[STEP:.+]] = call <vscale x 4 x i32> @llvm.{{(experimental\.)?}}stepvector.nxv4i32()
; CHECK:   %[[SCALE:.+]] = call i32 @llvm.vscale.i32()
; CHECK:   %[[SIZE:.+]] = mul i32 %[[SCALE]], 4
; CHECK:   br label %loop
; CHECK: loop:
; CHECK:   %[[IV:.+]] = phi i32 [ 1, %entry ], [ %[[N2:.+]], %loop ]
; CHECK:   %[[VEC:.+]] = phi <vscale x 4 x i32> [ %0, %entry ], [ %[[NEWVEC:.+]], %loop ]
; CHECK:   %[[MASKPHI:.+]] = phi <vscale x 4 x i32> [ %[[STEP]], %entry ], [ %[[NEWMASK:.+]], %loop ]
; CHECK:   %[[N_INS:.+]] = insertelement <vscale x 4 x i32> poison, i32 %[[IV]], {{i32|i64}} 0
; CHECK:   %[[N_SPLAT:.+]] = shufflevector <vscale x 4 x i32> %[[N_INS]], <vscale x 4 x i32> poison, <vscale x 4 x i32> zeroinitializer
; CHECK:   %[[MASK:.+]] = xor <vscale x 4 x i32> %[[MASKPHI]], %[[N_SPLAT]]

;------- target-dependent dynamic shuffle code:
; CHECK:   store <vscale x 4 x i32> %[[VEC]], {{(<vscale x 4 x i32>\*)|(ptr)}} %[[SHUFFLE_ALLOC]]
;------- there will be a bitcast here if pointers are typed
; CHECK:   %[[INDEX:.+]] = getelementptr inbounds i32, [[PTRTY:(i32\*)|ptr]] %{{.+}}, <vscale x 4 x i32> %[[MASK]]
; CHECK:   %[[SHUFFLE:.+]] = call <vscale x 4 x i32> @llvm.masked.gather.nxv4i32.nxv4p0{{(i32)?}}(<vscale x 4 x [[PTRTY]]> %[[INDEX]], i32 4, <vscale x 4 x i1> shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, {{i32|i64}} 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer), <vscale x 4 x i32> undef)

; CHECK:   %[[ACCUM:.+]] = add <vscale x 4 x i32> %[[VEC]], %{{.+}}
; CHECK:   %[[BIT:.+]] = and <vscale x 4 x i32> %[[MASKPHI]], %[[N_SPLAT]]
; CHECK:   %[[WHICH:.+]] = icmp ne <vscale x 4 x i32> %[[BIT]], zeroinitializer
; CHECK:   %[[NEWVEC]] = select <vscale x 4 x i1> %[[WHICH]], <vscale x 4 x i32> %[[ACCUM]], <vscale x 4 x i32> %[[VEC]]
; CHECK:   %[[NEWMASK]] = or <vscale x 4 x i32> %[[MASK]], %[[N_SPLAT]]
; CHECK:   %[[N2]] = shl nuw nsw i32 %[[IV]], 1
; CHECK:   %[[CMP:.+]] = icmp ult i32 %[[N2]], %[[SIZE]]
; CHECK:   br i1 %[[CMP]], label %loop, label %exit
; CHECK: exit:
; CHECK:   %[[SCAN:.+]] = phi <vscale x 4 x i32> [ %[[NEWVEC]], %loop ]

;------- target-dependent slide-up code:
; CHECK:   %[[SLIDE:.+]] = call <vscale x 4 x i32> @llvm{{(\.experimental)?}}.vector.splice.nxv4i32(<vscale x 4 x i32> undef, <vscale x 4 x i32> %[[SCAN]], i32 -1)
; CHECK:   %[[RESULT:.+]] = insertelement <vscale x 4 x i32> %[[SLIDE]], i32 0, {{i32|i64}} 0

; CHECK:   ret <vscale x 4 x i32> %[[RESULT]]
; CHECK: }

; We know the generated code is correct for one scan type,
; now verify that all the others use the correct binary operations.

declare <vscale x 4 x float> @__vecz_b_sub_group_scan_inclusive_add_u5nxv4f(<vscale x 4 x float>)
; CHECK-LABEL: define <vscale x 4 x float> @__vecz_b_sub_group_scan_inclusive_add_u5nxv4f(<vscale x 4 x float>{{.*}})
; CHECK: loop:
; CHECK:   %[[VEC:.+]] = phi <vscale x 4 x float> [ %0, %entry ],
; CHECK:   %{{.+}} = fadd <vscale x 4 x float> %[[VEC]], %{{.+}}

declare <vscale x 4 x float> @__vecz_b_sub_group_scan_exclusive_add_u5nxv4f(<vscale x 4 x float>)
; CHECK-LABEL: define <vscale x 4 x float> @__vecz_b_sub_group_scan_exclusive_add_u5nxv4f(<vscale x 4 x float>{{.*}})
; CHECK: loop:
; CHECK:   %[[VEC:.+]] = phi <vscale x 4 x float> [ %0, %entry ],
; CHECK:   %{{.+}} = fadd <vscale x 4 x float> %[[VEC]], %{{.+}}

declare <vscale x 4 x i32> @__vecz_b_sub_group_scan_inclusive_smin_u5nxv4j(<vscale x 4 x i32>)
; CHECK-LABEL: define <vscale x 4 x i32> @__vecz_b_sub_group_scan_inclusive_smin_u5nxv4j(<vscale x 4 x i32>{{.*}})
; CHECK: loop:
; CHECK:   %[[VEC:.+]] = phi <vscale x 4 x i32> [ %0, %entry ],
; CHECK:   %{{.+}} = call <vscale x 4 x i32> @llvm.smin.nxv4i32(<vscale x 4 x i32> %[[VEC]], <vscale x 4 x i32> %{{.+}})

declare <vscale x 4 x i32> @__vecz_b_sub_group_scan_exclusive_smin_u5nxv4j(<vscale x 4 x i32>)
; CHECK-LABEL: define <vscale x 4 x i32> @__vecz_b_sub_group_scan_exclusive_smin_u5nxv4j(<vscale x 4 x i32>{{.*}})
; CHECK: loop:
; CHECK:   %[[VEC:.+]] = phi <vscale x 4 x i32> [ %0, %entry ],
; CHECK:   %{{.+}} = call <vscale x 4 x i32> @llvm.smin.nxv4i32(<vscale x 4 x i32> %[[VEC]], <vscale x 4 x i32> %{{.+}})

declare <vscale x 4 x i32> @__vecz_b_sub_group_scan_inclusive_smax_u5nxv4j(<vscale x 4 x i32>)
; CHECK-LABEL: define <vscale x 4 x i32> @__vecz_b_sub_group_scan_inclusive_smax_u5nxv4j(<vscale x 4 x i32>{{.*}})
; CHECK: loop:
; CHECK:   %[[VEC:.+]] = phi <vscale x 4 x i32> [ %0, %entry ],
; CHECK:   %{{.+}} = call <vscale x 4 x i32> @llvm.smax.nxv4i32(<vscale x 4 x i32> %[[VEC]], <vscale x 4 x i32> %{{.+}})

declare <vscale x 4 x i32> @__vecz_b_sub_group_scan_inclusive_umin_u5nxv4j(<vscale x 4 x i32>)
; CHECK-LABEL: define <vscale x 4 x i32> @__vecz_b_sub_group_scan_inclusive_umin_u5nxv4j(<vscale x 4 x i32>{{.*}})
; CHECK: loop:
; CHECK:   %[[VEC:.+]] = phi <vscale x 4 x i32> [ %0, %entry ],
; CHECK:   %{{.+}} = call <vscale x 4 x i32> @llvm.umin.nxv4i32(<vscale x 4 x i32> %[[VEC]], <vscale x 4 x i32> %{{.+}})

declare <vscale x 4 x i32> @__vecz_b_sub_group_scan_inclusive_umax_u5nxv4j(<vscale x 4 x i32>)
; CHECK-LABEL: define <vscale x 4 x i32> @__vecz_b_sub_group_scan_inclusive_umax_u5nxv4j(<vscale x 4 x i32>{{.*}})
; CHECK: loop:
; CHECK:   %[[VEC:.+]] = phi <vscale x 4 x i32> [ %0, %entry ],
; CHECK:   %{{.+}} = call <vscale x 4 x i32> @llvm.umax.nxv4i32(<vscale x 4 x i32> %[[VEC]], <vscale x 4 x i32> %{{.+}})

declare <vscale x 4 x float> @__vecz_b_sub_group_scan_inclusive_min_u5nxv4f(<vscale x 4 x float>)
; CHECK-LABEL: define <vscale x 4 x float> @__vecz_b_sub_group_scan_inclusive_min_u5nxv4f(<vscale x 4 x float>{{.*}})
; CHECK: loop:
; CHECK:   %[[VEC:.+]] = phi <vscale x 4 x float> [ %0, %entry ],
; CHECK:   %{{.+}} = call <vscale x 4 x float> @llvm.minnum.nxv4f32(<vscale x 4 x float> %[[VEC]], <vscale x 4 x float> %{{.+}})

declare <vscale x 4 x float> @__vecz_b_sub_group_scan_inclusive_max_u5nxv4f(<vscale x 4 x float>)
; CHECK-LABEL: define <vscale x 4 x float> @__vecz_b_sub_group_scan_inclusive_max_u5nxv4f(<vscale x 4 x float>{{.*}})
; CHECK: loop:
; CHECK:   %[[VEC:.+]] = phi <vscale x 4 x float> [ %0, %entry ],
; CHECK:   %{{.+}} = call <vscale x 4 x float> @llvm.maxnum.nxv4f32(<vscale x 4 x float> %[[VEC]], <vscale x 4 x float> %{{.+}})

declare <vscale x 4 x float> @__vecz_b_sub_group_scan_exclusive_min_u5nxv4f(<vscale x 4 x float>)
; CHECK-LABEL: define <vscale x 4 x float> @__vecz_b_sub_group_scan_exclusive_min_u5nxv4f(<vscale x 4 x float>{{.*}})
; CHECK: loop:
; CHECK:   %[[VEC:.+]] = phi <vscale x 4 x float> [ %0, %entry ],
; CHECK:   %{{.+}} = call <vscale x 4 x float> @llvm.minnum.nxv4f32(<vscale x 4 x float> %[[VEC]], <vscale x 4 x float> %{{.+}})

declare <vscale x 4 x float> @__vecz_b_sub_group_scan_exclusive_max_u5nxv4f(<vscale x 4 x float>)
; CHECK-LABEL: define <vscale x 4 x float> @__vecz_b_sub_group_scan_exclusive_max_u5nxv4f(<vscale x 4 x float>{{.*}})
; CHECK: loop:
; CHECK:   %[[VEC:.+]] = phi <vscale x 4 x float> [ %0, %entry ],
; CHECK:   %{{.+}} = call <vscale x 4 x float> @llvm.maxnum.nxv4f32(<vscale x 4 x float> %[[VEC]], <vscale x 4 x float> %{{.+}})
