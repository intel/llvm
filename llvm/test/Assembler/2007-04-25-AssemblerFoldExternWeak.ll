; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: llvm-as < %s | llvm-dis -opaque-pointers | FileCheck %s
; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: verify-uselistorder -opaque-pointers %s
; PR1358

; CHECK: icmp ne (ptr @test_weak, ptr null)
@G = global i1 icmp ne (ptr @test_weak, ptr null)

declare extern_weak i32 @test_weak(...)

