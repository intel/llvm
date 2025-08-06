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

; RUN: veczc -k entry -vecz-passes="builtin-inlining,function(instcombine,early-cse),cfg-convert,packetizer" -S < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

; Laid out, this struct is 80 bytes
%struct.S2 = type { i16, [7 x i32], i32, <16 x i8>, [4 x i32] }

; Function Attrs: norecurse nounwind
define spir_kernel void @entry(i64 addrspace(1)* %result, %struct.S2* %result2) {
entry:
  %gid = call i64 @__mux_get_local_id(i32 0)
  %sa = alloca %struct.S2, align 16
  %sb = alloca %struct.S2, align 16
  %sa_i8 = bitcast %struct.S2* %sa to i8*
  %sb_i8 = bitcast %struct.S2* %sb to i8*
  %sb_i8as = addrspacecast i8* %sb_i8 to i8 addrspace(1)*
  %rsi = ptrtoint i64 addrspace(1)* %result to i64
  %rsit = trunc i64 %rsi to i8
  call void @llvm.memset.p0i8.i64(i8* %sa_i8, i8 %rsit, i64 80, i32 16, i1 false)
  call void @llvm.memset.p1i8.i64(i8 addrspace(1)* %sb_i8as, i8 0, i64 80, i32 16, i1 false)
  %lr = addrspacecast %struct.S2* %result2 to %struct.S2 addrspace(1)*
  %lri = bitcast %struct.S2 addrspace(1)* %lr to i64 addrspace(1)*
  %cond = icmp eq i64 addrspace(1)* %result, %lri
  br i1 %cond, label %middle, label %end

middle:
  call void @llvm.memcpy.p1i8.p0i8.i64(i8 addrspace(1)* %sb_i8as, i8* %sa_i8, i64 80, i32 16, i1 false)
  br label %end

end:
  %g_343 = getelementptr inbounds %struct.S2, %struct.S2* %sa, i64 0, i32 0
  %g_343_load = load i16, i16* %g_343
  %g_343_zext = zext i16 %g_343_load to i64
  %resp = getelementptr i64, i64 addrspace(1)* %result, i64 %gid
  store i64 %g_343_zext, i64 addrspace(1)* %resp, align 8
  %result2_i8 = bitcast %struct.S2* %result2 to i8*
  call void @llvm.memcpy.p0i8.p1i8.i64(i8* %result2_i8, i8 addrspace(1)* %sb_i8as, i64 80, i32 16, i1 false)
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.memset.p0i8.i64(i8* nocapture, i8, i64, i32, i1)
declare void @llvm.memset.p1i8.i64(i8 addrspace(1)* nocapture, i8, i64, i32, i1)

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p1i8.p0i8.i64(i8 addrspace(1)* nocapture, i8* nocapture readonly, i64, i32, i1)
declare void @llvm.memcpy.p0i8.p1i8.i64(i8* nocapture, i8 addrspace(1)* nocapture readonly, i64, i32, i1)

declare i64 @__mux_get_local_id(i32)

; Note: Between LLVM 17 and LLVM 18, optimizations to alignments were moved to
; their own pass. We don't run that pass here, resulting in a difference in
; alignment values between LLVM versions. Because of that, we don't check
; alignment of any loads or stores

; Sanity checks: Make sure the non-vecz entry function is still in place and
; contains memset and memcpy. This is done in order to prevent future bafflement
; in case some pass optimizes them out.
; CHECK: define spir_kernel void @entry
; CHECK: entry:
; CHECK: call void @llvm.memset
; CHECK: call void @llvm.memset
; CHECK: middle:
; CHECK: call void @llvm.memcpy
; CHECK: end:
; CHECK: call void @llvm.memcpy

; And now for the actual checks

; Check if the kernel was vectorized
; CHECK: define spir_kernel void @__vecz_v{{[0-9]+}}_entry
; CHECK: %[[SB_I8AS:.*]] = addrspacecast ptr %sb to ptr addrspace(1)

; Check if the memset and memcpy calls have been removed
; CHECK-NOT: call void @llvm.memset
; CHECK-NOT: call void @llvm.memcpy

; Check if the calculation of the stored value for the second memset is in place
; CHECK: %ms64val

; Check if the generated loads and stores are in place
; Check the stores for the first memset
; CHECK: store i64 %ms64val, ptr %sa
; CHECK: %[[V14:[0-9]+]] = getelementptr inbounds {{(nuw )?}}i8, ptr %sa, i64 8
; CHECK: store i64 %ms64val, ptr %[[V14]]
; CHECK: %[[V15:[0-9]+]] = getelementptr inbounds {{(nuw )?}}i8, ptr %sa, i64 16
; CHECK: store i64 %ms64val, ptr %[[V15]]
; CHECK: %[[V16:[0-9]+]] = getelementptr inbounds {{(nuw )?}}i8, ptr %sa, i64 24
; CHECK: store i64 %ms64val, ptr %[[V16]]
; CHECK: %[[V17:[0-9]+]] = getelementptr inbounds {{(nuw )?}}i8, ptr %sa, i64 32
; CHECK: store i64 %ms64val, ptr %[[V17]]
; CHECK: %[[V18:[0-9]+]] = getelementptr inbounds {{(nuw )?}}i8, ptr %sa, i64 40
; CHECK: store i64 %ms64val, ptr %[[V18]]
; CHECK: %[[V19:[0-9]+]] = getelementptr inbounds {{(nuw )?}}i8, ptr %sa, i64 48
; CHECK: store i64 %ms64val, ptr %[[V19]]
; CHECK: %[[V20:[0-9]+]] = getelementptr inbounds {{(nuw )?}}i8, ptr %sa, i64 56
; CHECK-EQ14: %[[V20:[0-9]+]] = getelementptr inbounds {{(nuw )?}}%struct.S2, %struct.S2* %sa, i64 0, i32 3, i64 8
; CHECK: %[[V21:[0-9]+]] = getelementptr inbounds {{(nuw )?}}i8, ptr %sa, i64 64
; CHECK: %[[V22:[0-9]+]] = getelementptr inbounds {{(nuw )?}}i8, ptr %sa, i64 72

; Check the stores for the second memset
; CHECK: store i64 0, ptr addrspace(1) %[[SB_I8AS]]
; CHECK: %[[V24:[0-9]+]] = getelementptr inbounds {{(nuw )?}}i8, ptr addrspace(1) %[[SB_I8AS]], i64 8
; CHECK: store i64 0, ptr addrspace(1) %[[V24]]
; CHECK: %[[V26:[0-9]+]] = getelementptr inbounds {{(nuw )?}}i8, ptr addrspace(1) %[[SB_I8AS]], i64 16
; CHECK: store i64 0, ptr addrspace(1) %[[V26]]
; CHECK: %[[V28:[0-9]+]] = getelementptr inbounds {{(nuw )?}}i8, ptr addrspace(1) %[[SB_I8AS]], i64 24
; CHECK: store i64 0, ptr addrspace(1) %[[V28]]
; CHECK: %[[V30:[0-9]+]] = getelementptr inbounds {{(nuw )?}}i8, ptr addrspace(1) %[[SB_I8AS]], i64 32
; CHECK: store i64 0, ptr addrspace(1) %[[V30]]
; CHECK: %[[V32:[0-9]+]] = getelementptr inbounds {{(nuw )?}}i8, ptr addrspace(1) %[[SB_I8AS]], i64 40
; CHECK: store i64 0, ptr addrspace(1) %[[V32]]
; CHECK: %[[V33:[0-9]+]] = getelementptr inbounds {{(nuw )?}}i8, ptr addrspace(1) %[[SB_I8AS]], i64 48
; CHECK: store i64 0, ptr addrspace(1) %[[V33]]
; CHECK: %[[V35T:[0-9]+]] = getelementptr inbounds {{(nuw )?}}i8, ptr addrspace(1) %[[SB_I8AS]], i64 56
; CHECK-EQ14: %[[V35T:[0-9]+]] = getelementptr inbounds {{(nuw )?}}%struct.S2, %struct.S2* %sb, i64 0, i32 3, i64 8
; CHECK-EQ14: %[[V35:[0-9]+]] = bitcast i8* %[[V35T]] to i64*
; CHECK-EQ14: %[[SB_I8AS18:.+]] = addrspacecast i64* %[[V35]] to i64 addrspace(1)*
; CHECK: store i64 0, ptr addrspace(1) %[[V35T]]
; CHECK: %[[V36:[0-9]+]] = getelementptr inbounds {{(nuw )?}}i8, ptr addrspace(1) %[[SB_I8AS]], i64 64
; CHECK: store i64 0, ptr addrspace(1) %[[V36]]
; CHECK: %[[V38:[0-9]+]] = getelementptr inbounds {{(nuw )?}}i8, ptr addrspace(1) %[[SB_I8AS]], i64 72
; CHECK: store i64 0, ptr addrspace(1) %[[V38]]


; Check the loads and stores for the first memcpy
; CHECK:middle:                                           ; preds = %entry
; CHECK: %[[SA_I822:.+]] = load i64, ptr %sa
; CHECK: store i64 %[[SA_I822]], ptr addrspace(1) %[[SB_I8AS]]
; CHECK: %[[SA_I824:.+]] = load i64, ptr %[[V14]]
; CHECK: store i64 %[[SA_I824]], ptr addrspace(1) %[[V24]]
; CHECK: %[[SA_I826:.+]] = load i64, ptr %[[V15]]
; CHECK: store i64 %[[SA_I826]], ptr addrspace(1) %[[V26]]
; CHECK: %[[SA_I828:.+]] = load i64, ptr %[[V16]]
; CHECK: store i64 %[[SA_I828]], ptr addrspace(1) %[[V28]]
; CHECK: %[[SA_I830:.+]] = load i64, ptr %[[V17]]
; CHECK: store i64 %[[SA_I830]], ptr addrspace(1) %[[V30]]
; CHECK: %[[SA_I832:.+]] = load i64, ptr %[[V18]]
; CHECK: store i64 %[[SA_I832]], ptr addrspace(1) %[[V32]]
; CHECK: %[[SA_I834:.+]] = load i64, ptr %[[V19]]
; CHECK: store i64 %[[SA_I834]], ptr addrspace(1) %[[V33]]
; CHECK: %[[SA_I836:.+]] = load i64, ptr %[[V20]]
; CHECK: store i64 %[[SA_I836]], ptr addrspace(1) %[[V35T]]
; CHECK: %[[SA_I838:.+]] = load i64, ptr %[[V21]]
; CHECK: store i64 %[[SA_I838]], ptr addrspace(1) %[[V36]]
; CHECK: %[[SA_I840:.+]] = load i64, ptr %[[V22]]
; CHECK: store i64 %[[SA_I840]], ptr addrspace(1) %[[V38]]

; Check the loads and stores for the second memcpy
; CHECK:end:                                              ; preds = %middle, %entry
; CHECK: %[[SB_I8AS42:.+]] = load i64, ptr addrspace(1) %[[SB_I8AS]]
; CHECK: store i64 %[[SB_I8AS42]], ptr %result2
; CHECK: %[[V42:[0-9]+]] = getelementptr inbounds {{(nuw )?}}i8, ptr %result2, i64 8
; CHECK: %[[SB_I8AS44:.+]] = load i64, ptr addrspace(1) %[[V24]]
; CHECK: store i64 %[[SB_I8AS44]], ptr %[[V42]]
; CHECK: %[[V43:[0-9]+]] = getelementptr inbounds {{(nuw )?}}i8, ptr %result2, i64 16
; CHECK: %[[SB_I8AS46:.+]] = load i64, ptr addrspace(1) %[[V26]]
; CHECK: store i64 %[[SB_I8AS46]], ptr %[[V43]]
; CHECK: %[[V44:[0-9]+]] = getelementptr inbounds {{(nuw )?}}i8, ptr %result2, i64 24
; CHECK: %[[SB_I8AS48:.+]] = load i64, ptr addrspace(1) %[[V28]]
; CHECK: store i64 %[[SB_I8AS48]], ptr %[[V44]]
; CHECK: %[[V45:[0-9]+]] = getelementptr inbounds {{(nuw )?}}i8, ptr %result2, i64 32
; CHECK: %[[SB_I8AS50:.+]] = load i64, ptr addrspace(1) %[[V30]]
; CHECK: store i64 %[[SB_I8AS50]], ptr %[[V45]]
; CHECK: %[[V46:[0-9]+]] = getelementptr inbounds {{(nuw )?}}i8, ptr %result2, i64 40
; CHECK: %[[SB_I8AS52:.+]] = load i64, ptr addrspace(1) %[[V32]]
; CHECK: store i64 %[[SB_I8AS52]], ptr %[[V46]]
; CHECK: %[[V47:[0-9]+]] = getelementptr inbounds {{(nuw )?}}i8, ptr %result2, i64 48
; CHECK: %[[SB_I8AS54:.+]] = load i64, ptr addrspace(1) %[[V33]]
; CHECK: store i64 %[[SB_I8AS54]], ptr %[[V47]]
; CHECK: %[[V48:[0-9]+]] = getelementptr inbounds {{(nuw )?}}i8, ptr %result2, i64 56
; CHECK-EQ14: %[[V48:[0-9]+]] = getelementptr inbounds {{(nuw )?}}%struct.S2, %struct.S2* %result2, i64 0, i32 3, i64 8
; CHECK: %[[SB_I8AS56:.+]] = load i64, ptr addrspace(1) %[[V35T]]
; CHECK: store i64 %[[SB_I8AS56]], ptr %[[V48]]
; CHECK: %[[V49:[0-9]+]] = getelementptr inbounds {{(nuw )?}}i8, ptr %result2, i64 64
; CHECK: %[[SB_I8AS58:.+]] = load i64, ptr addrspace(1) %[[V36]]
; CHECK: store i64 %[[SB_I8AS58]], ptr %[[V49]]
; CHECK: %[[V50:[0-9]+]] = getelementptr inbounds {{(nuw )?}}i8, ptr %result2, i64 72
; CHECK: %[[SB_I8AS60:.+]] = load i64, ptr addrspace(1) %[[V38]]
; CHECK: store i64 %[[SB_I8AS60]], ptr %[[V50]]

; End of function
; CHECK: ret void
