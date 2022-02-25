;;===----------------------------------------------------------------------===//
;;
;; Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
;; See https://llvm.org/LICENSE.txt for license information.
;; SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
;;
;;===----------------------------------------------------------------------===//

#if __clang_major__ >= 7
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7"
#else
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7"
#endif

; Function Attrs: nounwind readnone speculatable
declare i32 addrspace(5)* @llvm.amdgcn.implicit.offset()

define hidden i64 @_Z22__spirv_GlobalOffset_xv() nounwind alwaysinline {
entry:
  %0 = tail call i32 addrspace(5)* @llvm.amdgcn.implicit.offset()
  %1 = load i32, i32 addrspace(5)* %0, align 4
  %zext = zext i32 %1 to i64
  ret i64 %zext
}

define hidden i64 @_Z22__spirv_GlobalOffset_yv() nounwind alwaysinline {
entry:
  %0 = tail call i32 addrspace(5)* @llvm.amdgcn.implicit.offset()
  %arrayidx = getelementptr inbounds i32, i32 addrspace(5)* %0, i64 1
  %1 = load i32, i32 addrspace(5)* %arrayidx, align 4
  %zext = zext i32 %1 to i64
  ret i64 %zext
}

define hidden i64 @_Z22__spirv_GlobalOffset_zv() nounwind alwaysinline {
entry:
  %0 = tail call i32 addrspace(5)* @llvm.amdgcn.implicit.offset()
  %arrayidx = getelementptr inbounds i32, i32 addrspace(5)* %0, i64 2
  %1 = load i32, i32 addrspace(5)* %arrayidx, align 4
  %zext = zext i32 %1 to i64
  ret i64 %zext
}
