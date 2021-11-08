; TODO: remove -enable-new-pm from RUN-lines once we move to new pass manager
; by default. This test was designed to test NewPM only.

; RUN: not opt -enable-new-pm -O1 -O2 < %s 2>&1 | FileCheck %s --check-prefix=MULTIPLE
; RUN: not opt -enable-new-pm -O1 -passes='no-op-module' < %s 2>&1 | FileCheck %s --check-prefix=BOTH
; RUN: opt -enable-new-pm -O0 < %s -S 2>&1 | FileCheck %s --check-prefix=OPT
; RUN: opt -enable-new-pm -O1 < %s -S 2>&1 | FileCheck %s --check-prefix=OPT
; RUN: opt -enable-new-pm -O2 < %s -S 2>&1 | FileCheck %s --check-prefix=OPT
; RUN: opt -enable-new-pm -O3 < %s -S 2>&1 | FileCheck %s --check-prefix=OPT
; RUN: opt -enable-new-pm -Os < %s -S 2>&1 | FileCheck %s --check-prefix=OPT
; RUN: opt -enable-new-pm -Oz < %s -S 2>&1 | FileCheck %s --check-prefix=OPT
; RUN: opt -enable-new-pm -O2 -debug-pass-manager -disable-output < %s 2>&1 | FileCheck %s --check-prefix=AA

; MULTIPLE: Cannot specify multiple -O#
; BOTH: Cannot specify -O# and --passes=
; OPT: define void @f
; Make sure we run the default AA pipeline with `opt -O#`
; AA: Running analysis: ScopedNoAliasAA

define void @f() {
  unreachable
}
