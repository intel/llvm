;;===----------------------------------------------------------------------===//
;;
;; Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
;; See https://llvm.org/LICENSE.txt for license information.
;; SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
;;
;;===----------------------------------------------------------------------===//

; Function Attrs: nounwind readnone speculatable
declare i32 addrspace(5)* @llvm.amdgcn.implicit.offset()

define hidden i64 @_Z27__spirv_BuiltInGlobalOffseti(i32 %dim) nounwind alwaysinline {
entry:
  switch i32 %dim, label %return [
    i32 0, label %sw.bb
    i32 1, label %sw.bb1
    i32 2, label %sw.bb3
  ]

sw.bb:                                            ; preds = %entry
  %0 = tail call i32 addrspace(5)* @llvm.amdgcn.implicit.offset()
  %1 = load i32, i32 addrspace(5)* %0, align 4
  %zext = zext i32 %1 to i64
  br label %return

sw.bb1:                                           ; preds = %entry
  %2 = tail call i32 addrspace(5)* @llvm.amdgcn.implicit.offset()
  %arrayidx = getelementptr inbounds i32, i32 addrspace(5)* %2, i64 1
  %3 = load i32, i32 addrspace(5)* %arrayidx, align 4
  %zext2 = zext i32 %3 to i64
  br label %return

sw.bb3:                                           ; preds = %entry
  %4 = tail call i32 addrspace(5)* @llvm.amdgcn.implicit.offset()
  %arrayidx2 = getelementptr inbounds i32, i32 addrspace(5)* %4, i64 2
  %5 = load i32, i32 addrspace(5)* %arrayidx2, align 4
  %zext3 = zext i32 %5 to i64
  br label %return

return:                                           ; preds = %entry, %sw.bb3, %sw.bb1, %sw.bb
  %retval = phi i64 [ %zext, %sw.bb ], [ %zext2, %sw.bb1 ], [ %zext3, %sw.bb3 ], [ 0, %entry ]
  ret i64 %retval
}
