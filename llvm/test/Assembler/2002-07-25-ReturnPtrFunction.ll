; Test that returning a pointer to a function causes the disassembler to print 
; the right thing.
;
; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: llvm-as < %s | llvm-dis -opaque-pointers | llvm-as
; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: verify-uselistorder -opaque-pointers %s

declare ptr @foo()

define void @test() {
        call ptr () @foo( )           ; <ptr>:1 [#uses=0]
        ret void
}


