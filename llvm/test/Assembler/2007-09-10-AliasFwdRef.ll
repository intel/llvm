; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: llvm-as < %s | llvm-dis -opaque-pointers
; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: verify-uselistorder -opaque-pointers %s
; PR1645

@__gthread_active_ptr.5335 = internal constant ptr @__gthrw_pthread_cancel    
@__gthrw_pthread_cancel = weak alias i32 (i32), ptr @pthread_cancel



define weak i32 @pthread_cancel(i32) {
  ret i32 0
}
