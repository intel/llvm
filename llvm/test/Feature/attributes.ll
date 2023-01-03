; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: llvm-as < %s | llvm-dis -opaque-pointers > %t1.ll
; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: llvm-as %t1.ll -o - | llvm-dis -opaque-pointers > %t2.ll
; RUN: diff %t1.ll %t2.ll

@.str = private unnamed_addr constant [14 x i8] c"hello world!\0A\00", align 1

define void @foo() #0 {
entry:
  %call = call i32 (ptr, ...) @printf(ptr @.str)
  ret void
}

declare i32 @printf(ptr, ...)

attributes #0 = { nounwind ssp uwtable }
