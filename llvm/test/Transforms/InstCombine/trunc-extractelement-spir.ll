; RUN: opt < %s -instcombine -S | FileCheck %s

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir"

define i32 @shrinkExtractElt_i64_to_i32_0(<3 x i64> %x) {
; CHECK-LABEL: @shrinkExtractElt_i64_to_i32_0(
; CHECK-NOT:    {{%.+}} = bitcast <3 x i64> {{%.+}} to <6 x i32>
; CHECK-NOT:    {{%.+}} = extractelement <6 x i32> {{%.+}}, i32 0

  %e = extractelement <3 x i64> %x, i32 0
  %t = trunc i64 %e to i32
  ret i32 %t
}
