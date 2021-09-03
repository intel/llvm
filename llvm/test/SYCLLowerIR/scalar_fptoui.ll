; This is a regression test for LowerESIMD crashing on a scalar fptoui
; instruction.
;
; RUN: opt < %s -LowerESIMD -S | FileCheck %s

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

; Function Attrs: convergent norecurse
define dso_local spir_func i32 @foo(float %x) {
  %y = fptoui float %x to i32
; check that the scalar float to unsigned int conversion is left intact
; CHECK: %y = fptoui float %x to i32
  ret i32 %y
}

!1 = !{i32 0, i32 0}
