; Prepare bitcode files.
; RUN: rm -rf %t && mkdir -p %t
; RUN: llvm-as %s -o %t/main.bc
; RUN: llvm-as %p/Inputs/libspirv-native_cpu.ll -o %t/libspirv-native_cpu.bc

; No warnings expected for linking in libspirv-native_cpu.bc.
; RUN: llvm-link %t/main.bc %t/libspirv-native_cpu.bc -S 2>&1 | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK-NOT: warning:
; CHECK: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
; CHECK: target triple = "x86_64-unknown-linux-gnu"
; CHECK-NOT: warning:
