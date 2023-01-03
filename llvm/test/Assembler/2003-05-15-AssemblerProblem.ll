; This bug was caused by two CPR's existing for the same global variable, 
; colliding in the Module level CPR map.
; RUN: llvm-as %s -o /dev/null
; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: verify-uselistorder -opaque-pointers %s

define void @test() {
        call void (...) @AddString( ptr null, i32 0 )
        ret void
}

define void @AddString(ptr %tmp.124, i32 %tmp.127) {
        call void (...) @AddString( ptr %tmp.124, i32 %tmp.127 )
        ret void
}

