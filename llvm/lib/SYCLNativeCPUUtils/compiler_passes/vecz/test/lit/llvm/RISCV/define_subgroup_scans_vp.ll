; Copyright (C) Codeplay Software Limited
;
; Licensed under the Apache License, Version 2.0 (the "License") with LLVM
; Exceptions; you may not use this file except in compliance with the License.
; You may obtain a copy of the License at
;
;     https://github.com/uxlfoundation/oneapi-construction-kit/blob/main/LICENSE.txt
;
; Unless required by applicable law or agreed to in writing, software
; distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
; WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
; License for the specific language governing permissions and limitations
; under the License.
;
; SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

; REQUIRES: llvm-13+
; RUN: veczc -k dummy -vecz-target-triple="riscv64-unknown-unknown" -vecz-scalable -vecz-simd-width=4 -vecz-passes=define-builtins -S < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "riscv64-unknown-unknown"

define spir_kernel void @dummy(i32 addrspace(2)* %in, i32 addrspace(1)* %out) {
  ; Dummy uses of the builtins, as we don't define any with zero uses.
  %a = call <vscale x 4 x i32> @__vecz_b_sub_group_scan_inclusive_add_vp_u5nxv4jj(<vscale x 4 x i32> zeroinitializer, i32 0)
  %b = call <vscale x 4 x i32> @__vecz_b_sub_group_scan_exclusive_add_vp_u5nxv4jj(<vscale x 4 x i32> zeroinitializer, i32 0)
  %c = call <vscale x 4 x float> @__vecz_b_sub_group_scan_inclusive_add_vp_u5nxv4fj(<vscale x 4 x float> zeroinitializer, i32 0)
  %d = call <vscale x 4 x float> @__vecz_b_sub_group_scan_exclusive_add_vp_u5nxv4fj(<vscale x 4 x float> zeroinitializer, i32 0)
  %e = call <vscale x 4 x i32> @__vecz_b_sub_group_scan_inclusive_smin_vp_u5nxv4jj(<vscale x 4 x i32> zeroinitializer, i32 0)
  %f = call <vscale x 4 x i32> @__vecz_b_sub_group_scan_exclusive_smin_vp_u5nxv4jj(<vscale x 4 x i32> zeroinitializer, i32 0)
  %g = call <vscale x 4 x i32> @__vecz_b_sub_group_scan_inclusive_smax_vp_u5nxv4jj(<vscale x 4 x i32> zeroinitializer, i32 0)
  %h = call <vscale x 4 x i32> @__vecz_b_sub_group_scan_inclusive_umin_vp_u5nxv4jj(<vscale x 4 x i32> zeroinitializer, i32 0)
  %i = call <vscale x 4 x i32> @__vecz_b_sub_group_scan_inclusive_umax_vp_u5nxv4jj(<vscale x 4 x i32> zeroinitializer, i32 0)
  %j = call <vscale x 4 x float> @__vecz_b_sub_group_scan_inclusive_min_vp_u5nxv4fj(<vscale x 4 x float> zeroinitializer, i32 0)
  %k = call <vscale x 4 x float> @__vecz_b_sub_group_scan_inclusive_max_vp_u5nxv4fj(<vscale x 4 x float> zeroinitializer, i32 0)
  %l = call <vscale x 4 x float> @__vecz_b_sub_group_scan_exclusive_min_vp_u5nxv4fj(<vscale x 4 x float> zeroinitializer, i32 0)
  %m = call <vscale x 4 x float> @__vecz_b_sub_group_scan_exclusive_max_vp_u5nxv4fj(<vscale x 4 x float> zeroinitializer, i32 0)
  %n = call <vscale x 4 x double> @__vecz_b_sub_group_scan_exclusive_min_vp_u5nxv4dj(<vscale x 4 x double> zeroinitializer, i32 0)
  ret void
}

declare <vscale x 4 x i32> @__vecz_b_sub_group_scan_inclusive_add_vp_u5nxv4jj(<vscale x 4 x i32>, i32)
; CHECK-LABEL: define <vscale x 4 x i32> @__vecz_b_sub_group_scan_inclusive_add_vp_u5nxv4jj(<vscale x 4 x i32>{{.*}}, i32{{.*}}) {
; CHECK: entry:
; CHECK:   %[[STEP:.+]] = call <vscale x 4 x i32> @llvm.experimental.stepvector.nxv4i32()
; CHECK:   br label %loop
; CHECK: loop:
; CHECK:   %[[IV:.+]] = phi i32 [ 1, %entry ], [ %[[N2:.+]], %loop ]
; CHECK:   %[[VEC:.+]] = phi <vscale x 4 x i32> [ %0, %entry ], [ %[[NEWVEC:.+]], %loop ]
; CHECK:   %[[MASKPHI:.+]] = phi <vscale x 4 x i32> [ %[[STEP]], %entry ], [ %[[NEWMASK:.+]], %loop ]
; CHECK:   %[[N_INS:.+]] = insertelement <vscale x 4 x i32> poison, i32 %[[IV]], {{i32|i64}} 0
; CHECK:   %[[N_SPLAT:.+]] = shufflevector <vscale x 4 x i32> %[[N_INS]], <vscale x 4 x i32> poison, <vscale x 4 x i32> zeroinitializer
; CHECK:   %[[MASK:.+]] = xor <vscale x 4 x i32> %[[MASKPHI]], %[[N_SPLAT]]

;------- target-dependent dynamic shuffle code:
; CHECK:   %[[VL:.+]] = zext i32 %1 to i64
; CHECK:   %[[SHUFFLE:.+]] = call <vscale x 4 x i32> @llvm.riscv.vrgather.vv.nxv4i32.i64({{(<vscale x 4 x i32> poison, )?}}<vscale x 4 x i32> %[[VEC]], <vscale x 4 x i32> %[[MASK]], i64 %[[VL]])

; CHECK:   %[[ACCUM:.+]] = add <vscale x 4 x i32> %[[VEC]], %[[SHUFFLE]]
; CHECK:   %[[BIT:.+]] = and <vscale x 4 x i32> %[[MASKPHI]], %[[N_SPLAT]]
; CHECK:   %[[WHICH:.+]] = icmp ne <vscale x 4 x i32> %[[BIT]], zeroinitializer
; CHECK:   %[[NEWVEC]] = select <vscale x 4 x i1> %[[WHICH]], <vscale x 4 x i32> %[[ACCUM]], <vscale x 4 x i32> %[[VEC]]
; CHECK:   %[[NEWMASK]] = or <vscale x 4 x i32> %[[MASK]], %[[N_SPLAT]]
; CHECK:   %[[N2]] = shl nuw nsw i32 %[[IV]], 1
; CHECK:   %[[CMP:.+]] = icmp ult i32 %[[N2]], %1
; CHECK:   br i1 %[[CMP]], label %loop, label %exit
; CHECK: exit:
; CHECK:   %[[RESULT:.+]] = phi <vscale x 4 x i32> [ %[[NEWVEC]], %loop ]
; CHECK:   ret <vscale x 4 x i32> %[[RESULT]]
; CHECK: }

declare <vscale x 4 x i32> @__vecz_b_sub_group_scan_exclusive_add_vp_u5nxv4jj(<vscale x 4 x i32>, i32)
; CHECK-LABEL: define <vscale x 4 x i32> @__vecz_b_sub_group_scan_exclusive_add_vp_u5nxv4jj(<vscale x 4 x i32>{{.*}}, i32{{.*}}) {
; CHECK: entry:
; CHECK:   %[[STEP:.+]] = call <vscale x 4 x i32> @llvm.experimental.stepvector.nxv4i32()
; CHECK:   br label %loop
; CHECK: loop:
; CHECK:   %[[IV:.+]] = phi i32 [ 1, %entry ], [ %[[N2:.+]], %loop ]
; CHECK:   %[[VEC:.+]] = phi <vscale x 4 x i32> [ %0, %entry ], [ %[[NEWVEC:.+]], %loop ]
; CHECK:   %[[MASKPHI:.+]] = phi <vscale x 4 x i32> [ %[[STEP]], %entry ], [ %[[NEWMASK:.+]], %loop ]
; CHECK:   %[[N_INS:.+]] = insertelement <vscale x 4 x i32> poison, i32 %[[IV]], {{i32|i64}} 0
; CHECK:   %[[N_SPLAT:.+]] = shufflevector <vscale x 4 x i32> %[[N_INS]], <vscale x 4 x i32> poison, <vscale x 4 x i32> zeroinitializer
; CHECK:   %[[MASK:.+]] = xor <vscale x 4 x i32> %[[MASKPHI]], %[[N_SPLAT]]

;------- target-dependent dynamic shuffle code:
; CHECK:   %[[VL:.+]] = zext i32 %1 to i64
; CHECK:   %[[SHUFFLE:.+]] = call <vscale x 4 x i32> @llvm.riscv.vrgather.vv.nxv4i32.i64({{(<vscale x 4 x i32> poison, )?}}<vscale x 4 x i32> %[[VEC]], <vscale x 4 x i32> %[[MASK]], i64 %[[VL]])

; CHECK:   %[[ACCUM:.+]] = add <vscale x 4 x i32> %[[VEC]], %{{.+}}
; CHECK:   %[[BIT:.+]] = and <vscale x 4 x i32> %[[MASKPHI]], %[[N_SPLAT]]
; CHECK:   %[[WHICH:.+]] = icmp ne <vscale x 4 x i32> %[[BIT]], zeroinitializer
; CHECK:   %[[NEWVEC]] = select <vscale x 4 x i1> %[[WHICH]], <vscale x 4 x i32> %[[ACCUM]], <vscale x 4 x i32> %[[VEC]]
; CHECK:   %[[NEWMASK]] = or <vscale x 4 x i32> %[[MASK]], %[[N_SPLAT]]
; CHECK:   %[[N2]] = shl nuw nsw i32 %[[IV]], 1
; CHECK:   %[[CMP:.+]] = icmp ult i32 %[[N2]], %1
; CHECK:   br i1 %[[CMP]], label %loop, label %exit
; CHECK: exit:
; CHECK:   %[[SCAN:.+]] = phi <vscale x 4 x i32> [ %[[NEWVEC]], %loop ]

;------- target-dependent slide-up code:
; CHECK:   %[[VL2:.+]] = zext i32 %1 to i64
; CHECK:   %[[RESULT:.+]] = call <vscale x 4 x i32> @llvm.riscv.vslide1up.nxv4i32.i32.i64({{(<vscale x 4 x i32> poison, )?}}<vscale x 4 x i32> %[[SCAN]], i32 0, i64 %[[VL2]])

; CHECK:   ret <vscale x 4 x i32> %[[RESULT]]
; CHECK: }

; We know the generated code is correct for one scan type,
; now verify that all the others use the correct binary operations.

declare <vscale x 4 x float> @__vecz_b_sub_group_scan_inclusive_add_vp_u5nxv4fj(<vscale x 4 x float>, i32)
; CHECK-LABEL: define <vscale x 4 x float> @__vecz_b_sub_group_scan_inclusive_add_vp_u5nxv4fj(<vscale x 4 x float>{{.*}}, i32{{.*}})
; CHECK: loop:
; CHECK:   %[[VEC:.+]] = phi <vscale x 4 x float> [ %0, %entry ],
; CHECK:   %{{.+}} = fadd <vscale x 4 x float> %[[VEC]], %{{.+}}

declare <vscale x 4 x float> @__vecz_b_sub_group_scan_exclusive_add_vp_u5nxv4fj(<vscale x 4 x float>, i32)
; CHECK-LABEL: define <vscale x 4 x float> @__vecz_b_sub_group_scan_exclusive_add_vp_u5nxv4fj(<vscale x 4 x float>{{.*}}, i32{{.*}})
; CHECK: loop:
; CHECK:   %[[VEC:.+]] = phi <vscale x 4 x float> [ %0, %entry ],
; CHECK:   %{{.+}} = fadd <vscale x 4 x float> %[[VEC]], %{{.+}}

; Make sure the floating point version of the slide1up intrinsic is created
; CHECK:   call <vscale x 4 x float> @llvm.riscv.vfslide1up.nxv4f32.f32.i64({{(<vscale x 4 x float> poison, )?}}<vscale x 4 x float> %{{.+}}, float 0.000000e+00, i64 %{{.+}})

declare <vscale x 4 x i32> @__vecz_b_sub_group_scan_inclusive_smin_vp_u5nxv4jj(<vscale x 4 x i32>, i32)
; CHECK-LABEL: define <vscale x 4 x i32> @__vecz_b_sub_group_scan_inclusive_smin_vp_u5nxv4jj(<vscale x 4 x i32>{{.*}}, i32{{.*}})
; CHECK: loop:
; CHECK:   %[[VEC:.+]] = phi <vscale x 4 x i32> [ %0, %entry ],
; CHECK:   %{{.+}} = call <vscale x 4 x i32> @llvm.smin.nxv4i32(<vscale x 4 x i32> %[[VEC]], <vscale x 4 x i32> %{{.+}})

declare <vscale x 4 x i32> @__vecz_b_sub_group_scan_exclusive_smin_vp_u5nxv4jj(<vscale x 4 x i32>, i32)
; CHECK-LABEL: define <vscale x 4 x i32> @__vecz_b_sub_group_scan_exclusive_smin_vp_u5nxv4jj(<vscale x 4 x i32>{{.*}}, i32{{.*}})
; CHECK: loop:
; CHECK:   %[[VEC:.+]] = phi <vscale x 4 x i32> [ %0, %entry ],
; CHECK:   %{{.+}} = call <vscale x 4 x i32> @llvm.smin.nxv4i32(<vscale x 4 x i32> %[[VEC]], <vscale x 4 x i32> %{{.+}})

declare <vscale x 4 x i32> @__vecz_b_sub_group_scan_inclusive_smax_vp_u5nxv4jj(<vscale x 4 x i32>, i32)
; CHECK-LABEL: define <vscale x 4 x i32> @__vecz_b_sub_group_scan_inclusive_smax_vp_u5nxv4jj(<vscale x 4 x i32>{{.*}}, i32{{.*}})
; CHECK: loop:
; CHECK:   %[[VEC:.+]] = phi <vscale x 4 x i32> [ %0, %entry ],
; CHECK:   %{{.+}} = call <vscale x 4 x i32> @llvm.smax.nxv4i32(<vscale x 4 x i32> %[[VEC]], <vscale x 4 x i32> %{{.+}})

declare <vscale x 4 x i32> @__vecz_b_sub_group_scan_inclusive_umin_vp_u5nxv4jj(<vscale x 4 x i32>, i32)
; CHECK-LABEL: define <vscale x 4 x i32> @__vecz_b_sub_group_scan_inclusive_umin_vp_u5nxv4jj(<vscale x 4 x i32>{{.*}}, i32{{.*}})
; CHECK: loop:
; CHECK:   %[[VEC:.+]] = phi <vscale x 4 x i32> [ %0, %entry ],
; CHECK:   %{{.+}} = call <vscale x 4 x i32> @llvm.umin.nxv4i32(<vscale x 4 x i32> %[[VEC]], <vscale x 4 x i32> %{{.+}})

declare <vscale x 4 x i32> @__vecz_b_sub_group_scan_inclusive_umax_vp_u5nxv4jj(<vscale x 4 x i32>, i32)
; CHECK-LABEL: define <vscale x 4 x i32> @__vecz_b_sub_group_scan_inclusive_umax_vp_u5nxv4jj(<vscale x 4 x i32>{{.*}}, i32{{.*}})
; CHECK: loop:
; CHECK:   %[[VEC:.+]] = phi <vscale x 4 x i32> [ %0, %entry ],
; CHECK:   %{{.+}} = call <vscale x 4 x i32> @llvm.umax.nxv4i32(<vscale x 4 x i32> %[[VEC]], <vscale x 4 x i32> %{{.+}})

declare <vscale x 4 x float> @__vecz_b_sub_group_scan_inclusive_min_vp_u5nxv4fj(<vscale x 4 x float>, i32)
; CHECK-LABEL: define <vscale x 4 x float> @__vecz_b_sub_group_scan_inclusive_min_vp_u5nxv4fj(<vscale x 4 x float>{{.*}}, i32{{.*}})
; CHECK: loop:
; CHECK:   %[[VEC:.+]] = phi <vscale x 4 x float> [ %0, %entry ],
; CHECK:   %{{.+}} = call <vscale x 4 x float> @llvm.minnum.nxv4f32(<vscale x 4 x float> %[[VEC]], <vscale x 4 x float> %{{.+}})

declare <vscale x 4 x float> @__vecz_b_sub_group_scan_inclusive_max_vp_u5nxv4fj(<vscale x 4 x float>, i32)
; CHECK-LABEL: define <vscale x 4 x float> @__vecz_b_sub_group_scan_inclusive_max_vp_u5nxv4fj(<vscale x 4 x float>{{.*}}, i32{{.*}})
; CHECK: loop:
; CHECK:   %[[VEC:.+]] = phi <vscale x 4 x float> [ %0, %entry ],
; CHECK:   %{{.+}} = call <vscale x 4 x float> @llvm.maxnum.nxv4f32(<vscale x 4 x float> %[[VEC]], <vscale x 4 x float> %{{.+}})

declare <vscale x 4 x float> @__vecz_b_sub_group_scan_exclusive_min_vp_u5nxv4fj(<vscale x 4 x float>, i32)
; CHECK-LABEL: define <vscale x 4 x float> @__vecz_b_sub_group_scan_exclusive_min_vp_u5nxv4fj(<vscale x 4 x float>{{.*}}, i32{{.*}})
; CHECK: loop:
; CHECK:   %[[VEC:.+]] = phi <vscale x 4 x float> [ %0, %entry ],
; CHECK:   %{{.+}} = call <vscale x 4 x float> @llvm.minnum.nxv4f32(<vscale x 4 x float> %[[VEC]], <vscale x 4 x float> %{{.+}})

declare <vscale x 4 x float> @__vecz_b_sub_group_scan_exclusive_max_vp_u5nxv4fj(<vscale x 4 x float>, i32)
; CHECK-LABEL: define <vscale x 4 x float> @__vecz_b_sub_group_scan_exclusive_max_vp_u5nxv4fj(<vscale x 4 x float>{{.*}}, i32{{.*}})
; CHECK: loop:
; CHECK:   %[[VEC:.+]] = phi <vscale x 4 x float> [ %0, %entry ],
; CHECK:   %{{.+}} = call <vscale x 4 x float> @llvm.maxnum.nxv4f32(<vscale x 4 x float> %[[VEC]], <vscale x 4 x float> %{{.+}})

declare <vscale x 4 x double> @__vecz_b_sub_group_scan_exclusive_min_vp_u5nxv4dj(<vscale x 4 x double>, i32)
; CHECK-LABEL: define <vscale x 4 x double> @__vecz_b_sub_group_scan_exclusive_min_vp_u5nxv4dj(<vscale x 4 x double>{{.*}}, i32{{.*}})
; CHECK: loop:
; CHECK:   %[[VEC:.+]] = phi <vscale x 4 x double> [ %0, %entry ],
; CHECK:   %{{.+}} = call <vscale x 4 x double> @llvm.minnum.nxv4f64(<vscale x 4 x double> %[[VEC]], <vscale x 4 x double> %{{.+}})
; CHECK:   call <vscale x 4 x double> @llvm.riscv.vfslide1up.nxv4f64.f64.i64({{(<vscale x 4 x double> poison, )?}}<vscale x 4 x double> %{{.+}}, double 0x7FF0000000000000, i64 %{{.+}})
