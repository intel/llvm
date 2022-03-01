; First test run uses module pass that is not designed for the old pass manager.
; Expect different input till the moment when NewPM is enabled by default.
; RUN: opt < %s -basic-aa -function-attrs -S -enable-new-pm=0 | FileCheck %s --check-prefix=CHECK-OLD-PM
; RUN: opt < %s -aa-pipeline=basic-aa -passes=function-attrs -S | FileCheck %s --check-prefix=CHECK-FUNC-PASS

; CHECK: define i32 @f() #0
define i32 @f() {
entry:
  %tmp = call i32 @e( )
  ret i32 %tmp
}

; CHECK: declare i32 @e() #1
declare i32 @e() readonly

; CHECK: attributes #0 = { nofree readonly }
; CHECK: attributes #1 = { readonly }
