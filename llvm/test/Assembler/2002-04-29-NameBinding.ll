; There should be NO references to the global v1.  The local v1 should
; have all of the references!
;
; Check by running globaldce, which will remove the constant if there are
; no references to it!
; 
; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: opt -opaque-pointers < %s -passes=globaldce -S | \
; RUN:   not grep constant
;
; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: verify-uselistorder -opaque-pointers %s

@v1 = internal constant i32 5           

define i32 @createtask() {
        %v1 = alloca i32                ;; Alloca should have one use! 
        %reg112 = load i32, ptr %v1         ;; This load should not use the global!
        ret i32 %reg112
}

