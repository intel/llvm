; Check SYCL LLVM optimization passes

; RUN: opt -sycl-opt -enable-new-pm=0 -debug-pass-manager -passes='default<O2>' %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-O2
; CHECK-O2: Running pass: DeadArgumentEliminationSYCLPass
;
; RUN: opt -sycl-opt -enable-new-pm=0 -debug-pass-manager -passes='default<O0>' %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-O0
; CHECK-O0-NOT: Running pass: DeadArgumentEliminationSYCLPass


; New pass manager


; RUN: opt -sycl-opt -enable-new-pm=1 -debug-pass-manager -passes='default<O2>' %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-NEWPM-O2
; CHECK-NEWPM-O2: Running pass: DeadArgumentEliminationSYCLPass
;
; RUN: opt -sycl-opt -enable-new-pm=1 -debug-pass-manager -passes='default<O0>' %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-NEWPM-O0
; CHECK-NEWPM-O0-NOT: Running pass: DeadArgumentEliminationSYCLPass
