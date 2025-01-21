; RUN: llvm-as %s -o %t.bc
; RUN: not llvm-spirv %t.bc 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR

; CHECK-ERROR: InvalidBitWidth: Invalid bit width in input: 16

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-unknown"

declare { <2 x float>, <2 x i16> } @llvm.frexp.v2f32.v2i16(<2 x float>)

define { <2 x float>, <2 x i16> } @frexp_zero_vector() {
  %ret = call { <2 x float>, <2 x i16> } @llvm.frexp.v2f32.v2i16(<2 x float> zeroinitializer)
  ret { <2 x float>, <2 x i16> } %ret
}
