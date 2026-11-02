; RUN: llvm-spirv %s -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv -to-text %t.spv -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.spv -o %t.rev.bc -r --spirv-target-env=SPV-IR
; RUN: llvm-dis %t.rev.bc -o %t.rev.ll
; RUN: FileCheck --input-file=%t.rev.ll %s --check-prefix=CHECK-LLVM

; Branch weight metadata (as produced by __builtin_expect, [[likely]]/
; [[unlikely]], or PGO) is translated to and from the optional Branch
; Weights operands of OpBranchConditional.

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-G1"
target triple = "spir64-unknown-unknown"

; Function result IDs, named in a block separate from the function bodies.
; CHECK-SPIRV: Name [[#WEIGHTED:]] "weighted_branch"
; CHECK-SPIRV: Name [[#UNWEIGHTED:]] "unweighted_branch"
; CHECK-SPIRV: Name [[#LARGE:]] "large_weights_branch"
; CHECK-SPIRV: Name [[#RATIO:]] "large_weights_ratio_branch"
; CHECK-SPIRV: Name [[#UMAX:]] "uint32_max_sum_branch"
; CHECK-SPIRV: Name [[#ONEZERO:]] "one_zero_weight_branch"
; CHECK-SPIRV: Name [[#ALLZERO:]] "zero_weights_branch"

; CHECK-SPIRV: Function [[#]] [[#WEIGHTED]]
; CHECK-SPIRV: BranchConditional [[#]] [[#]] [[#]] 2000 1
define spir_func i32 @weighted_branch(i32 %x) {
entry:
  %cmp = icmp sgt i32 %x, 10
  br i1 %cmp, label %if.then, label %if.else, !prof !0

if.then:
  ret i32 %x

if.else:
  ret i32 %x
}

; CHECK-SPIRV: Function [[#]] [[#UNWEIGHTED]]
; CHECK-SPIRV: BranchConditional [[#]] [[#]] [[#]] {{$}}
define spir_func i32 @unweighted_branch(i32 %x) {
entry:
  %cmp = icmp sgt i32 %x, 10
  br i1 %cmp, label %if.then, label %if.else

if.then:
  ret i32 %x

if.else:
  ret i32 %x
}

; Sum overflows 32 bits, so weights are scaled down.
; CHECK-SPIRV: Function [[#]] [[#LARGE]]
; CHECK-SPIRV: BranchConditional [[#]] [[#]] [[#]] 1500000000 1500000000
define spir_func i32 @large_weights_branch(i32 %x) {
entry:
  %cmp = icmp sgt i32 %x, 10
  br i1 %cmp, label %if.then, label %if.else, !prof !1

if.then:
  ret i32 %x

if.else:
  ret i32 %x
}

; Downscaling preserves the ratio between the two weights: 3000000000:1500000000
; is 2:1, same as the scaled-down 1500000000:750000000.
; CHECK-SPIRV: Function [[#]] [[#RATIO]]
; CHECK-SPIRV: BranchConditional [[#]] [[#]] [[#]] 1500000000 750000000
define spir_func i32 @large_weights_ratio_branch(i32 %x) {
entry:
  %cmp = icmp sgt i32 %x, 10
  br i1 %cmp, label %if.then, label %if.else, !prof !4

if.then:
  ret i32 %x

if.else:
  ret i32 %x
}

; NOTE: sum is exactly UINT32_MAX, so it already fits, but
; llvm::calculateCountScale (ProfDataUtils.h) downscales it anyway - a
; deliberate conservative bound inherited from clang, harmless for a hint.
; CHECK-SPIRV: Function [[#]] [[#UMAX]]
; CHECK-SPIRV: BranchConditional [[#]] [[#]] [[#]] 2147483647 0
define spir_func i32 @uint32_max_sum_branch(i32 %x) {
entry:
  %cmp = icmp sgt i32 %x, 10
  br i1 %cmp, label %if.then, label %if.else, !prof !5

if.then:
  ret i32 %x

if.else:
  ret i32 %x
}

; A single zero weight is valid and preserved as-is.
; CHECK-SPIRV: Function [[#]] [[#ONEZERO]]
; CHECK-SPIRV: BranchConditional [[#]] [[#]] [[#]] 0 5
define spir_func i32 @one_zero_weight_branch(i32 %x) {
entry:
  %cmp = icmp sgt i32 %x, 10
  br i1 %cmp, label %if.then, label %if.else, !prof !2

if.then:
  ret i32 %x

if.else:
  ret i32 %x
}

; All-zero weights are dropped instead of emitting OpBranchConditional 0 0.
; CHECK-SPIRV: Function [[#]] [[#ALLZERO]]
; CHECK-SPIRV: BranchConditional [[#]] [[#]] [[#]] {{$}}
define spir_func i32 @zero_weights_branch(i32 %x) {
entry:
  %cmp = icmp sgt i32 %x, 10
  br i1 %cmp, label %if.then, label %if.else, !prof !3

if.then:
  ret i32 %x

if.else:
  ret i32 %x
}

; CHECK-LLVM-LABEL: define spir_func i32 @weighted_branch
; CHECK-LLVM: br i1 %{{.*}}, label %{{.*}}, label %{{.*}}, !prof ![[#PROF:]]

; CHECK-LLVM-LABEL: define spir_func i32 @unweighted_branch
; CHECK-LLVM-NOT: !prof

; CHECK-LLVM-LABEL: define spir_func i32 @large_weights_branch
; CHECK-LLVM: br i1 %{{.*}}, label %{{.*}}, label %{{.*}}, !prof ![[#PROF2:]]

; CHECK-LLVM-LABEL: define spir_func i32 @large_weights_ratio_branch
; CHECK-LLVM: br i1 %{{.*}}, label %{{.*}}, label %{{.*}}, !prof ![[#PROF4:]]

; CHECK-LLVM-LABEL: define spir_func i32 @uint32_max_sum_branch
; CHECK-LLVM: br i1 %{{.*}}, label %{{.*}}, label %{{.*}}, !prof ![[#PROF5:]]

; CHECK-LLVM-LABEL: define spir_func i32 @one_zero_weight_branch
; CHECK-LLVM: br i1 %{{.*}}, label %{{.*}}, label %{{.*}}, !prof ![[#PROF3:]]

; CHECK-LLVM-LABEL: define spir_func i32 @zero_weights_branch
; CHECK-LLVM-NOT: !prof

; CHECK-LLVM: ![[#PROF]] = !{!"branch_weights", i32 2000, i32 1}
; CHECK-LLVM: ![[#PROF2]] = !{!"branch_weights", i32 1500000000, i32 1500000000}
; CHECK-LLVM: ![[#PROF4]] = !{!"branch_weights", i32 1500000000, i32 750000000}
; CHECK-LLVM: ![[#PROF5]] = !{!"branch_weights", i32 2147483647, i32 0}
; CHECK-LLVM: ![[#PROF3]] = !{!"branch_weights", i32 0, i32 5}

!0 = !{!"branch_weights", i32 2000, i32 1}
!1 = !{!"branch_weights", i32 3000000000, i32 3000000000}
!2 = !{!"branch_weights", i32 0, i32 5}
!3 = !{!"branch_weights", i32 0, i32 0}
!4 = !{!"branch_weights", i32 3000000000, i32 1500000000}
!5 = !{!"branch_weights", i32 4294967294, i32 1}
