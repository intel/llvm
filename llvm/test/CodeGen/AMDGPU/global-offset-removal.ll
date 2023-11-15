; RUN: opt -bugpoint-enable-legacy-pm -globaloffset -enable-global-offset=false %s -S -o - | FileCheck %s

; This test checks that the implicit offset intrinsic is correctly removed

declare ptr addrspace(5) @llvm.amdgcn.implicit.offset()
; CHECK-NOT: llvm.amdgcn.implicit.offset

define weak_odr dso_local i64 @_ZTS14example_kernel() {
entry:
; CHECK-NOT: @llvm.amdgcn.implicit.offset()
; CHECK-NOT: getelementptr
; CHECK-NOT: load
; CHECK: [[REG:%[0-9]+]] = zext i{{[0-9]+}} 0 to i{{[0-9]+}}
; CHECK: ret i{{[0-9]+}} [[REG]]
  %0 = tail call ptr addrspace(5) @llvm.amdgcn.implicit.offset()
  %1 = getelementptr inbounds i32, ptr addrspace(5) %0, i64 1
  %2 = load i32, ptr addrspace(5) %1, align 4
  %3 = zext i32 %2 to i64
  ret i64 %3
}
