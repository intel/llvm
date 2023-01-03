; Test to ensure that the LTO API (legacy and new) lowers @llvm.public.type.test.

; RUN: llvm-as < %s > %t1
; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: llvm-lto -opaque-pointers -exported-symbol=_main %t1 -o %t2 --lto-save-before-opt -opaque-pointers --whole-program-visibility
; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: llvm-dis -opaque-pointers -o - %t2.0.preopt.bc | FileCheck %s --check-prefix=HIDDEN
; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: llvm-lto -opaque-pointers -exported-symbol=_main %t1 -o %t2 --lto-save-before-opt
; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: llvm-dis -opaque-pointers -o - %t2.0.preopt.bc | FileCheck %s --check-prefix=PUBLIC

; RUN: llvm-lto2 run -opaque-pointers %t1 -save-temps -pass-remarks=. \
; RUN:   -whole-program-visibility \
; RUN:   -o %t2 \
; RUN:   -r=%t1,_main,px
; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: llvm-dis -opaque-pointers %t2.0.0.preopt.bc -o - | FileCheck %s --check-prefix=HIDDEN
; RUN: llvm-lto2 run -opaque-pointers %t1 -save-temps -pass-remarks=. \
; RUN:   -o %t2 \
; RUN:   -r=%t1,_main,px
; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: llvm-dis -opaque-pointers %t2.0.0.preopt.bc -o - | FileCheck %s --check-prefix=PUBLIC

; PUBLIC-NOT: call {{.*}}@llvm.public.type.test
; PUBLIC-NOT: call {{.*}}@llvm.type.test
; HIDDEN-NOT: call {{.*}}@llvm.public.type.test
; HIDDEN: call {{.*}}@llvm.type.test

target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.9"

define i32 @main(ptr %vtable) {
entry:
  %p = call i1 @llvm.public.type.test(ptr %vtable, metadata !"_ZTS1A")
  call void @llvm.assume(i1 %p)
  ret i32 0
}

declare void @llvm.assume(i1)
declare i1 @llvm.public.type.test(ptr, metadata)
