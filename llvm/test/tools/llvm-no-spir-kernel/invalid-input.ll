; RUN: echo garbage > garbage.ll
; RUN: not llvm-no-spir-kernel garbage.ll
