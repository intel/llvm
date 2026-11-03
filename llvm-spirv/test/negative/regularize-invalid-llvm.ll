; RUN: not llvm-spirv -s %s -o %t.bc 2>&1 | FileCheck %s
; CHECK: expected top-level entity

invalid
