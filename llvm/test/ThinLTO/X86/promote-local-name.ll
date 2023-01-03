; Test the "-use-source-filename-for-promoted-locals" flag.

; Do setup work for all below tests: generate bitcode and combined index
; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: opt -opaque-pointers -module-summary -module-hash %s -o %t.bc
; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: opt -opaque-pointers -module-summary -module-hash %p/Inputs/promote-local-name-1.ll -o %t1.bc
; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: llvm-lto -opaque-pointers -thinlto-action=thinlink -o %t2.bc %t.bc %t1.bc

; This module will import b() which should cause the copy of foo and baz from
; that module (%t1.bc) to be imported. Check that the imported reference's
; promoted name matches the imported copy. We check for a specific name,
; because the result of the "-use-source-filename-for-promoted-locals" flag
; should be deterministic.

; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: llvm-lto -opaque-pointers -use-source-filename-for-promoted-locals -thinlto-action=import %t.bc -thinlto-index=%t2.bc -o - | llvm-dis -opaque-pointers -o - | FileCheck %s

; CHECK: @baz.llvm.llvm_test_LTO_X86_promote_local_name_1_ll = internal constant i32 10, align 4
; CHECK: call i32 @foo.llvm.llvm_test_LTO_X86_promote_local_name_1_ll
; CHECK: define available_externally hidden i32 @foo.llvm.llvm_test_LTO_X86_promote_local_name_1_ll()

source_filename = "llvm/test/ThinLTO/X86/promote-local-name.ll"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noinline nounwind uwtable
define i32 @main() {
entry:
  %retval = alloca i32, align 4
  store i32 0, ptr %retval, align 4
  %call = call i32 (...) @b()
  ret i32 %call
}

declare i32 @b(...)
