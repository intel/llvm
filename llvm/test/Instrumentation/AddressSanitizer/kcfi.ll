;; Test that we emit kcfi_type metadata for asan.module_ctor with KCFI.

; Added -opaque-pointers.
; FIXME: align with the community code when project is ready to enable opaque
; pointers by default.
; RUN: opt < %s -passes=asan -opaque-pointers -S | FileCheck %s

; CHECK: @llvm.global_ctors = {{.*}}{ i32 1, ptr @asan.module_ctor, ptr @asan.module_ctor }

; CHECK: define internal void @asan.module_ctor()
; CHECK-SAME: !kcfi_type

!llvm.module.flags = !{!0}
!0 = !{i32 4, !"kcfi", i32 1}
