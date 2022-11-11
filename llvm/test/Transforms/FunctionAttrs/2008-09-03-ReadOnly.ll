; First test run uses module pass that is not designed for the old pass manager.
; Expect different input till the moment when NewPM is enabled by default.
; RUN: opt < %s -function-attrs -S -enable-new-pm=0 | FileCheck %s --check-prefix=CHECK-OLD-PM
; RUN: opt < %s -passes=function-attrs -S | FileCheck %s --check-prefix=CHECK-FUNC-PASS

define i32 @f() {
; CHECK: Function Attrs: nofree readonly
; CHECK-LABEL: @f(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TMP:%.*]] = call i32 @e()
; CHECK-NEXT:    ret i32 [[TMP]]
;
entry:
  %tmp = call i32 @e( )
  ret i32 %tmp
}

; CHECK-OLD-PM: declare i32 @e() #0
; CHECK-FUNC-PASS: declare i32 @e() #1
declare i32 @e() readonly

; CHECK: attributes #0 = { nofree readonly }
; CHECK-FUNC-PASS: attributes #1 = { readonly }
