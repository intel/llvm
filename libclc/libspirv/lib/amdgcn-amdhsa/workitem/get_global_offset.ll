;;===----------------------------------------------------------------------===//
;;
;; Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
;; See https://llvm.org/LICENSE.txt for license information.
;; SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
;;
;;===----------------------------------------------------------------------===//

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
