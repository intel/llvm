; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: opt -opaque-pointers -module-summary %s -o %t1.bc
; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: opt -opaque-pointers -module-summary %p/Inputs/index-const-prop-gvref.ll -o %t2.bc
; RUN: llvm-lto2 run -relocation-model=static -opaque-pointers -save-temps %t2.bc -r=%t2.bc,b,pl -r=%t2.bc,a,pl \
; RUN:   %t1.bc -r=%t1.bc,main,plx -r=%t1.bc,a, -r=%t1.bc,b, -o %t3
; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: llvm-dis -opaque-pointers %t3.1.3.import.bc -o - | FileCheck %s --check-prefix=SRC
; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: llvm-dis -opaque-pointers %t3.2.3.import.bc -o - | FileCheck %s --check-prefix=DEST

;; When producing an ELF DSO, clear dso_local for declarations to avoid direct access.
; RUN: llvm-lto2 run -relocation-model=pic -opaque-pointers -save-temps %t2.bc -r=%t2.bc,b,pl -r=%t2.bc,a,pl \
; RUN:   %t1.bc -r=%t1.bc,main,plx -r=%t1.bc,a, -r=%t1.bc,b, -o %t4
; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: llvm-dis -opaque-pointers %t4.1.3.import.bc -o - | FileCheck %s --check-prefix=SRC
; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: llvm-dis -opaque-pointers %t4.2.3.import.bc -o - | FileCheck %s --check-prefix=DEST_DSO

; No variable in the source module should have been internalized
; SRC:      @b = dso_local global ptr @a
; SRC-NEXT: @a = dso_local global i32 42

; We can't internalize globals referenced by other live globals
; DEST:          @b = external dso_local global ptr
; DEST-NEXT:     @a = available_externally dso_local global i32 42, align 4
; DEST_DSO:      @b = external global ptr
; DEST_DSO-NEXT: @a = available_externally global i32 42, align 4

;; Test old API.
;; When producing an ELF DSO, clear dso_local for declarations to avoid direct access.
; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: llvm-lto -opaque-pointers -thinlto-action=run %t2.bc %t1.bc -relocation-model=static -thinlto-save-temps=%t5.
; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: llvm-dis -opaque-pointers < %t5.0.3.imported.bc | FileCheck %s --check-prefix=OLDAPI_SRC
; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: llvm-dis -opaque-pointers < %t5.1.3.imported.bc | FileCheck %s --check-prefix=OLDAPI_DST
; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: llvm-lto -opaque-pointers -thinlto-action=run %t2.bc %t1.bc -relocation-model=pic -thinlto-save-temps=%t6.
; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: llvm-dis -opaque-pointers < %t6.0.3.imported.bc | FileCheck %s --check-prefix=OLDAPI_SRC
; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: llvm-dis -opaque-pointers < %t6.1.3.imported.bc | FileCheck %s --check-prefix=OLDAPI_DST_DSO

; OLDAPI_SRC:      @b = internal global ptr @a, align 8
; OLDAPI_SRC-NEXT: @a = dso_local global i32 42, align 4
; OLDAPI_DST:      @b = external dso_local global ptr
; OLDAPI_DST-NEXT: @a = available_externally dso_local global i32 42, align 4
; OLDAPI_DST_DSO:      @b = external global ptr
; OLDAPI_DST_DSO-NEXT: @a = available_externally global i32 42, align 4

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@a = external global i32
@b = external global ptr

define i32 @main() {
  %p = load ptr, ptr @b, align 8  
  store i32 33, ptr %p, align 4
  %v = load i32, ptr @a, align 4
  ret i32 %v
}
