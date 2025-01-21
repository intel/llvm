; RUN: not llvm-spirv %S/empty-file.bc -o - 2>&1 | FileCheck %s

; CHECK: Can't translate, file is empty
