; RUN: llvm-spirv -spirv-text %s -o - | FileCheck %s
; RUN: llvm-spirv %s -o %t.spv
; RUN: spirv-val %t.spv
;
; The fcmp is decorated with metadata besides the fast-math flags.
; The metadata is always ignored since we expresses the fast math mode
; through the regular LLVM flags.
;
; CHECK-DAG: Name [[#OLT:]] "oltRes"
; CHECK-DAG: Name [[#OGT:]] "ogtRes"

; CHECK-NOT: Decorate [[#OGT]] FPFastMathMode {{[0-9]+}}
; CHECK-NOT: Decorate [[#OLT]] FPFastMathMode 16

; CHECK-DAG: FOrdLessThan [[#BOOL:]] [[#OLT]]
; CHECK-DAG: Decorate [[#OLT]] FPFastMathMode 1

; CHECK-DAG: FOrdGreaterThan [[#BOOL:]] [[#OGT]]

target triple = "spirv-unknown-unknown"

define void @foo(float %1, float %2) {
  entry:
    %oltRes = fcmp nnan olt float %1,  %2, !spirv.Decorations !1
    ret void
}

define void @bar(float %1, float %2) {
  entry:
    %ogtRes = fcmp ogt float %1,  %2, !spirv.Decorations !1
    ret void
}

!1 = !{!2}
!2 = !{i32 40, i32 16} ; 40 is FPFastMathMode
