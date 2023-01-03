; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: llvm-as -opaque-pointers < %s | llvm-dis -opaque-pointers -show-annotations | FileCheck -check-prefix=ANNOT %s
; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: llvm-as -opaque-pointers < %s | llvm-dis -opaque-pointers | FileCheck -check-prefix=BARE %s
; RUN: verify-uselistorder %s

; The bare version of this file should not have any #uses lines.
; BARE: @B =
; BARE-NOT: #uses
; BARE: }

@B = external global i32
; ANNOT: @B = external global i32   ; [#uses=0 type=ptr]

define <4 x i1> @foo(<4 x float> %a, <4 x float> %b) nounwind {
entry:
  %cmp = fcmp olt <4 x float> %a, %b              ; [#uses=1]
  ret <4 x i1> %cmp
}

; ANNOT: %cmp = fcmp olt <4 x float> %a, %b              ; [#uses=1 type=<4 x i1>]

