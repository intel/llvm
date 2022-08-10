; RUN: llvm-as -opaque-pointers=0 %s -o %t.bc

; RUN: not llvm-spirv %t.bc -spirv-max-version=1.0 -o - -spirv-text 2>&1 | FileCheck %s --check-prefix=CHECK-INVALID
; RUN: not llvm-spirv %t.bc -spirv-max-version=1.1 -o - -spirv-text 2>&1 | FileCheck %s --check-prefix=CHECK-INVALID

; RUN: llvm-spirv %t.bc -o - -spirv-text | FileCheck %s --check-prefix=CHECK12
; RUN: llvm-spirv %t.bc -spirv-max-version=1.2 -o - -spirv-text | FileCheck %s --check-prefix=CHECK12
; RUN: llvm-spirv %t.bc -spirv-max-version=1.3 -o - -spirv-text | FileCheck %s --check-prefix=CHECK12
; RUN: llvm-spirv %t.bc -spirv-max-version=1.4 -o - -spirv-text | FileCheck %s --check-prefix=CHECK12

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spirv64v1.2-unknown-unknown"

; This is a module with a SPIR-V subarch.  Ensure that the SPIR-V version specified as the subarch is taken into account by llvm-spirv.

; CHECK-INVALID: TripleMaxVersionIncompatible: Triple version and maximum version are incompatible.

; 66048 = 0x10200, i.e. version 1.2
; CHECK12: 119734787 66048
